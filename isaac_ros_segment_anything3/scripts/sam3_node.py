#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
SAM3 (Segment Anything 3) ROS2 node for text-prompted segmentation.

Pipeline:
  1. Image -> preprocess (1008x1008) -> vision_encoder (Triton) -> FPN features [cached per frame]
  2. Text prompt -> tokenize (input_ids + attention_mask, max_len=32)
     -> text_encoder (Triton) -> text_features + text_mask [cached]
  3. FPN features + prompt_features + prompt_mask -> decoder (Triton)
     -> pred_masks (288x288), pred_boxes (cx,cy,w,h normalized), pred_logits, presence_logits
  4. Post-process -> publish Detection2DArray + segmentation mask

Topics:
  Subscribed:
    /image_raw (sensor_msgs/Image)
    /sam3/text_prompt (std_msgs/String)
  Published:
    /sam3/raw_segmentation_mask (sensor_msgs/Image, mono8)
    /sam3/detections (vision_msgs/Detection2DArray)
  Services:
    /sam3/set_text_prompt (SetTextPrompt)
"""

import threading

import cv2
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from isaac_ros_segment_anything3_interfaces.srv import SetTextPrompt

# Fixed constants from the ONNX model signatures
_IMAGE_SIZE = 1008
_MAX_SEQ_LEN = 32
_MASK_SIZE = 288
_NUM_QUERIES = 200
_MAX_PROMPTS = 3


class Sam3Node(Node):
    """ROS2 node for SAM3 text-prompted segmentation via Triton."""

    def __init__(self):
        super().__init__('sam3_node')

        # Declare parameters
        self.declare_parameter('triton_server_url', 'localhost:8001')
        self.declare_parameter('model_repository_path', '/tmp/models')
        self.declare_parameter(
            'vision_encoder_model_name', 'sam3_vision_encoder')
        self.declare_parameter(
            'text_encoder_model_name', 'sam3_text_encoder')
        self.declare_parameter('decoder_model_name', 'sam3_decoder')
        self.declare_parameter('tokenizer_path', '/tmp/models/tokenizer.json')
        self.declare_parameter('image_size', _IMAGE_SIZE)
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('max_prompts', _MAX_PROMPTS)

        # Read parameters
        self._triton_url = self.get_parameter(
            'triton_server_url').get_parameter_value().string_value
        self._model_repo = self.get_parameter(
            'model_repository_path').get_parameter_value().string_value
        self._vision_model = self.get_parameter(
            'vision_encoder_model_name').get_parameter_value().string_value
        self._text_model = self.get_parameter(
            'text_encoder_model_name').get_parameter_value().string_value
        self._decoder_model = self.get_parameter(
            'decoder_model_name').get_parameter_value().string_value
        self._tokenizer_path = self.get_parameter(
            'tokenizer_path').get_parameter_value().string_value
        self._image_size = self.get_parameter(
            'image_size').get_parameter_value().integer_value
        self._confidence_threshold = self.get_parameter(
            'confidence_threshold').get_parameter_value().double_value
        self._max_prompts = self.get_parameter(
            'max_prompts').get_parameter_value().integer_value

        # Initialize tokenizer
        self._tokenizer = None
        self._init_tokenizer()

        # Initialize Triton client
        self._triton_client = None
        self._init_triton_client()

        # State (protected by lock for thread safety)
        self._lock = threading.Lock()
        self._current_prompts = []
        self._text_features_cache = None   # (text_features, text_mask) tuple
        self._bridge = CvBridge()

        # Image normalization (ImageNet mean/std, same as SAM2)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Subscriber: input image
        image_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._image_sub = self.create_subscription(
            Image, 'image_raw', self._image_callback, image_qos)

        # Subscriber: text prompt topic
        self._prompt_sub = self.create_subscription(
            String, 'sam3/text_prompt', self._text_prompt_callback, 10)

        # Publishers
        self._mask_pub = self.create_publisher(
            Image, 'sam3/raw_segmentation_mask', 10)
        self._detection_pub = self.create_publisher(
            Detection2DArray, 'sam3/detections', 10)

        # Service: set text prompt
        self._set_prompt_srv = self.create_service(
            SetTextPrompt, 'sam3/set_text_prompt',
            self._set_text_prompt_callback)

        self.get_logger().info(
            f'SAM3 node initialized. Triton: {self._triton_url}, '
            f'Models: {self._vision_model}, {self._text_model}, '
            f'{self._decoder_model}, image_size={self._image_size}')

    def _init_tokenizer(self):
        """Initialize the HuggingFace tokenizer."""
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(self._tokenizer_path)
            # Enable padding and truncation to fixed length
            self._tokenizer.enable_padding(
                length=_MAX_SEQ_LEN, pad_id=0, pad_token='[PAD]')
            self._tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)
            self.get_logger().info(
                f'Tokenizer loaded from {self._tokenizer_path} '
                f'(max_seq_len={_MAX_SEQ_LEN})')
        except FileNotFoundError:
            self.get_logger().error(
                f'Tokenizer file not found: {self._tokenizer_path}. '
                f'Run download_models.py first.')
        except ImportError:
            self.get_logger().error(
                'tokenizers library not installed. '
                'Install with: pip install tokenizers')

    def _init_triton_client(self):
        """Initialize the Triton gRPC client."""
        try:
            import tritonclient.grpc as grpc_client
            self._grpc_client_module = grpc_client
            self._triton_client = grpc_client.InferenceServerClient(
                url=self._triton_url)

            # Verify models are ready
            for model_name in [self._vision_model, self._text_model,
                               self._decoder_model]:
                if self._triton_client.is_model_ready(model_name):
                    self.get_logger().info(f'Model ready: {model_name}')
                else:
                    self.get_logger().warn(
                        f'Model NOT ready: {model_name}. '
                        f'Ensure Triton server is running with the '
                        f'SAM3 model repository.')
        except ImportError:
            self.get_logger().error(
                'tritonclient not installed. '
                'Install with: pip install tritonclient[all]')
        except Exception as e:
            self.get_logger().warn(
                f'Cannot connect to Triton at {self._triton_url}: {e}. '
                f'Will retry on first inference request.')

    # ----------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------

    def _preprocess_image(self, cv_image):
        """
        Preprocess image for SAM3 vision encoder.

        Following the SAM2 pipeline pattern:
          resize (keep aspect ratio) -> pad (bottom-right) -> normalize -> CHW

        Args:
            cv_image: RGB uint8 image (H, W, 3).

        Returns:
            np.ndarray of shape (1, 3, image_size, image_size), float32.
        """
        h, w = cv_image.shape[:2]
        scale = self._image_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = cv2.resize(cv_image, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        # Pad to image_size x image_size (bottom-right padding)
        padded = np.zeros(
            (self._image_size, self._image_size, 3), dtype=np.float32)
        padded[:new_h, :new_w] = resized.astype(np.float32) / 255.0

        # Normalize with ImageNet mean/std
        padded = (padded - self._mean) / self._std

        # HWC -> CHW -> NCHW
        return padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def _tokenize_text(self, prompts):
        """
        Tokenize text prompts using HuggingFace tokenizer.

        Produces both input_ids and attention_mask with fixed sequence
        length of _MAX_SEQ_LEN (32).

        Args:
            prompts: List of strings.

        Returns:
            Tuple of (input_ids, attention_mask):
              - input_ids: np.ndarray of shape (num_prompts, 32), int64
              - attention_mask: np.ndarray of shape (num_prompts, 32), int64
            Returns (None, None) on failure.
        """
        if self._tokenizer is None:
            self.get_logger().error('Tokenizer not initialized')
            return None, None

        encodings = self._tokenizer.encode_batch(prompts)
        num_prompts = len(prompts)
        input_ids = np.zeros(
            (num_prompts, _MAX_SEQ_LEN), dtype=np.int64)
        attention_mask = np.zeros(
            (num_prompts, _MAX_SEQ_LEN), dtype=np.int64)

        for i, enc in enumerate(encodings):
            seq_len = min(len(enc.ids), _MAX_SEQ_LEN)
            input_ids[i, :seq_len] = enc.ids[:seq_len]
            attention_mask[i, :seq_len] = enc.attention_mask[:seq_len]

        return input_ids, attention_mask

    # ----------------------------------------------------------------
    # Triton inference
    # ----------------------------------------------------------------

    def _run_vision_encoder(self, image):
        """
        Run vision encoder on Triton.

        Args:
            image: np.ndarray (1, 3, 1008, 1008), float32.

        Returns:
            Tuple of (fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2)
            as np.ndarrays, or None on failure.
        """
        if self._triton_client is None:
            self.get_logger().error('Triton client not initialized')
            return None

        grpc = self._grpc_client_module
        inputs = [grpc.InferInput('images', list(image.shape), 'FP32')]
        inputs[0].set_data_from_numpy(image)

        outputs = [
            grpc.InferRequestedOutput('fpn_feat_0'),
            grpc.InferRequestedOutput('fpn_feat_1'),
            grpc.InferRequestedOutput('fpn_feat_2'),
            grpc.InferRequestedOutput('fpn_pos_2'),
        ]

        try:
            result = self._triton_client.infer(
                model_name=self._vision_model,
                inputs=inputs,
                outputs=outputs)
            return (
                result.as_numpy('fpn_feat_0'),
                result.as_numpy('fpn_feat_1'),
                result.as_numpy('fpn_feat_2'),
                result.as_numpy('fpn_pos_2'),
            )
        except Exception as e:
            self.get_logger().error(
                f'Vision encoder inference failed: {e}')
            return None

    def _run_text_encoder(self, input_ids, attention_mask):
        """
        Run text encoder on Triton.

        Args:
            input_ids: np.ndarray (num_prompts, 32), int64.
            attention_mask: np.ndarray (num_prompts, 32), int64.

        Returns:
            Tuple of (text_features, text_mask):
              - text_features: np.ndarray (num_prompts, 32, 256), float32
              - text_mask: np.ndarray (num_prompts, 32), bool
            Returns None on failure.
        """
        if self._triton_client is None:
            self.get_logger().error('Triton client not initialized')
            return None

        grpc = self._grpc_client_module
        inputs = [
            grpc.InferInput(
                'input_ids', list(input_ids.shape), 'INT64'),
            grpc.InferInput(
                'attention_mask', list(attention_mask.shape), 'INT64'),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [
            grpc.InferRequestedOutput('text_features'),
            grpc.InferRequestedOutput('text_mask'),
        ]

        try:
            result = self._triton_client.infer(
                model_name=self._text_model,
                inputs=inputs,
                outputs=outputs)
            return (
                result.as_numpy('text_features'),
                result.as_numpy('text_mask'),
            )
        except Exception as e:
            self.get_logger().error(
                f'Text encoder inference failed: {e}')
            return None

    def _run_decoder(self, fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2,
                     prompt_features, prompt_mask):
        """
        Run decoder on Triton.

        Args:
            fpn_feat_0: np.ndarray (-1, 256, 288, 288), float32.
            fpn_feat_1: np.ndarray (-1, 256, 144, 144), float32.
            fpn_feat_2: np.ndarray (-1, 256, 72, 72), float32.
            fpn_pos_2:  np.ndarray (-1, 256, 72, 72), float32.
            prompt_features: np.ndarray (-1, seq_len, 256), float32.
            prompt_mask: np.ndarray (-1, seq_len), bool.

        Returns:
            Tuple of (pred_masks, pred_boxes, pred_logits, presence_logits)
            as np.ndarrays, or None on failure.
        """
        if self._triton_client is None:
            self.get_logger().error('Triton client not initialized')
            return None

        grpc = self._grpc_client_module
        inputs = [
            grpc.InferInput(
                'fpn_feat_0', list(fpn_feat_0.shape), 'FP32'),
            grpc.InferInput(
                'fpn_feat_1', list(fpn_feat_1.shape), 'FP32'),
            grpc.InferInput(
                'fpn_feat_2', list(fpn_feat_2.shape), 'FP32'),
            grpc.InferInput(
                'fpn_pos_2', list(fpn_pos_2.shape), 'FP32'),
            grpc.InferInput(
                'prompt_features', list(prompt_features.shape), 'FP32'),
            grpc.InferInput(
                'prompt_mask', list(prompt_mask.shape), 'BOOL'),
        ]
        inputs[0].set_data_from_numpy(fpn_feat_0)
        inputs[1].set_data_from_numpy(fpn_feat_1)
        inputs[2].set_data_from_numpy(fpn_feat_2)
        inputs[3].set_data_from_numpy(fpn_pos_2)
        inputs[4].set_data_from_numpy(prompt_features)
        inputs[5].set_data_from_numpy(prompt_mask)

        outputs = [
            grpc.InferRequestedOutput('pred_masks'),
            grpc.InferRequestedOutput('pred_boxes'),
            grpc.InferRequestedOutput('pred_logits'),
            grpc.InferRequestedOutput('presence_logits'),
        ]

        try:
            result = self._triton_client.infer(
                model_name=self._decoder_model,
                inputs=inputs,
                outputs=outputs)
            return (
                result.as_numpy('pred_masks'),
                result.as_numpy('pred_boxes'),
                result.as_numpy('pred_logits'),
                result.as_numpy('presence_logits'),
            )
        except Exception as e:
            self.get_logger().error(f'Decoder inference failed: {e}')
            return None

    # ----------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------

    def _text_prompt_callback(self, msg):
        """Handle text prompt topic messages (std_msgs/String)."""
        prompts = [p.strip() for p in msg.data.split(',') if p.strip()]
        if not prompts:
            return

        if len(prompts) > self._max_prompts:
            self.get_logger().warn(
                f'Too many prompts ({len(prompts)}), '
                f'truncating to {self._max_prompts}. '
                f'See CONTRIBUTING.md for details.')
            prompts = prompts[:self._max_prompts]

        with self._lock:
            self._current_prompts = prompts
            self._text_features_cache = None
            self.get_logger().info(
                f'Text prompts updated via topic: {prompts}')

    def _set_text_prompt_callback(self, request, response):
        """Handle SetTextPrompt service calls."""
        prompts = list(request.text_prompts)

        if len(prompts) > self._max_prompts:
            prompts = prompts[:self._max_prompts]
            response.success = True
            response.message = (
                f'Truncated to {self._max_prompts} prompts '
                f'(decoder runs per-prompt, see CONTRIBUTING.md)')
        else:
            response.success = True
            response.message = \
                f'Set {len(prompts)} text prompts'

        with self._lock:
            self._current_prompts = prompts

            if request.confidence_threshold > 0.0:
                self._confidence_threshold = request.confidence_threshold

            self._text_features_cache = None
            response.active_prompts = self._current_prompts

            self.get_logger().info(
                f'Text prompts set via service: {self._current_prompts}')

        return response

    def _image_callback(self, msg):
        """Process incoming image through the SAM3 pipeline."""
        self.get_logger().debug(
            f'Image received: {msg.width}x{msg.height}, '
            f'encoding={msg.encoding}')

        # Snapshot shared state under lock (keep lock scope minimal
        # to avoid blocking prompt updates during inference).
        with self._lock:
            if not self._current_prompts:
                self.get_logger().debug('No prompts set, skipping')
                return
            if self._triton_client is None or self._tokenizer is None:
                self.get_logger().warn(
                    'Triton client or tokenizer not ready, skipping')
                return
            current_prompts = list(self._current_prompts)
            text_cache = self._text_features_cache
            confidence_threshold = self._confidence_threshold

        self.get_logger().info(
            f'Processing {msg.width}x{msg.height} image with '
            f'prompts={current_prompts}')

        # Convert ROS image to OpenCV (no lock needed)
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        orig_h, orig_w = cv_image.shape[:2]

        # 1. Preprocess image
        preprocessed = self._preprocess_image(cv_image)

        # 2. Vision encoder (always run for each new frame)
        vision_result = self._run_vision_encoder(preprocessed)
        if vision_result is None:
            return
        fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2 = vision_result

        # 3. Text encoder (cached until prompt changes)
        if text_cache is not None:
            text_features, text_mask = text_cache
        else:
            input_ids, attention_mask = self._tokenize_text(current_prompts)
            if input_ids is None:
                return
            text_result = self._run_text_encoder(input_ids, attention_mask)
            if text_result is None:
                return
            text_features, text_mask = text_result
            # Store back under lock
            with self._lock:
                # Only update if prompts haven't changed meanwhile
                if self._current_prompts == current_prompts:
                    self._text_features_cache = (text_features, text_mask)

        # 4. Decoder: run per-prompt (batch=1) since ONNX model
        #    expects batch=1 for prompt inputs.
        all_masks = []
        all_boxes = []
        all_logits = []
        all_presence = []

        for p_idx in range(text_features.shape[0]):
            pf = text_features[p_idx:p_idx + 1].astype(np.float32)
            pm = text_mask[p_idx:p_idx + 1].astype(bool)

            decoder_result = self._run_decoder(
                fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2,
                pf, pm)
            if decoder_result is None:
                continue
            masks, boxes, logits, presence = decoder_result
            all_masks.append(masks[0])       # (200, 288, 288)
            all_boxes.append(boxes[0])       # (200, 4)
            all_logits.append(logits[0])     # (200,)
            all_presence.append(presence[0])  # (1,)

        if not all_masks:
            return

        pred_masks = np.stack(all_masks)       # (N, 200, 288, 288)
        pred_boxes = np.stack(all_boxes)       # (N, 200, 4)
        pred_logits = np.stack(all_logits)     # (N, 200)
        presence_logits = np.stack(all_presence)  # (N, 1)

        # 5. Post-process and publish
        self._publish_results(
            msg.header, orig_h, orig_w,
            pred_masks, pred_boxes, pred_logits, presence_logits,
            current_prompts, confidence_threshold)

    # ----------------------------------------------------------------
    # Output publishing
    # ----------------------------------------------------------------

    @staticmethod
    def _cxcywh_to_xyxy(boxes, img_w, img_h):
        """
        Convert normalized [cx, cy, w, h] boxes to pixel [x1, y1, x2, y2].

        Args:
            boxes: np.ndarray (..., 4) with values in [0, 1].
            img_w: Original image width in pixels.
            img_h: Original image height in pixels.

        Returns:
            np.ndarray (..., 4) in pixel coordinates [x1, y1, x2, y2].
        """
        cx = boxes[..., 0] * img_w
        cy = boxes[..., 1] * img_h
        w = boxes[..., 2] * img_w
        h = boxes[..., 3] * img_h
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _publish_results(self, header, orig_h, orig_w,
                         pred_masks, pred_boxes, pred_logits,
                         presence_logits, prompts, confidence_threshold):
        """
        Filter by confidence threshold and publish results.

        Decoder outputs (batch dimension = number of text prompts):
          - pred_masks: (batch, 200, 288, 288) float32
          - pred_boxes: (batch, 200, 4) float32, normalized [cx, cy, w, h]
          - pred_logits: (batch, 200) float32 (raw logits, apply sigmoid)
          - presence_logits: (batch, 1) float32

        Args:
            header: ROS message header.
            orig_h, orig_w: Original image dimensions.
            pred_masks, pred_boxes, pred_logits, presence_logits: Decoder outputs.
            prompts: List of text prompt strings (snapshot from callback).
            confidence_threshold: Threshold for filtering (snapshot).

        Publishes:
          - Detection2DArray on /sam3/detections
          - Combined segmentation mask on /sam3/raw_segmentation_mask
        """
        detection_array = Detection2DArray()
        detection_array.header = header

        # Combined mask for all detections
        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Convert logits to confidence scores via sigmoid
        scores = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        presence_scores = 1.0 / (
            1.0 + np.exp(-presence_logits.astype(np.float64)))

        # pred_masks, pred_boxes, scores shape: (batch, 200, ...)
        # presence_scores shape: (batch, 1)
        num_batches = scores.shape[0] if scores.ndim >= 2 else 1

        for b in range(num_batches):
            # Check presence: skip this batch entry if presence is low
            batch_presence = float(
                presence_scores[b, 0]
                if presence_scores.ndim >= 2
                else presence_scores[b])
            if batch_presence < confidence_threshold:
                continue

            batch_scores = scores[b] if scores.ndim >= 2 else scores
            batch_boxes = pred_boxes[b] if pred_boxes.ndim >= 3 else pred_boxes
            batch_masks = pred_masks[b] if pred_masks.ndim >= 4 else pred_masks

            # Convert boxes from normalized [cx,cy,w,h] to pixel [x1,y1,x2,y2]
            boxes_xyxy = self._cxcywh_to_xyxy(batch_boxes, orig_w, orig_h)

            # Determine prompt label for this batch entry
            prompt_label = prompts[b] if b < len(prompts) else f'class_{b}'

            num_detections = batch_scores.shape[0] if batch_scores.ndim >= 1 else 1

            for j in range(num_detections):
                score = float(
                    batch_scores[j]
                    if batch_scores.ndim >= 1
                    else batch_scores)
                if score < confidence_threshold:
                    continue

                # Build Detection2D
                det = Detection2D()
                det.header = header

                # Bounding box from xyxy
                box = boxes_xyxy[j]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), \
                    float(box[2]), float(box[3])
                det.bbox.center.position.x = (x1 + x2) / 2.0
                det.bbox.center.position.y = (y1 + y2) / 2.0
                det.bbox.size_x = x2 - x1
                det.bbox.size_y = y2 - y1

                # Hypothesis with class label and score
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = prompt_label
                hyp.hypothesis.score = score
                det.results.append(hyp)

                detection_array.detections.append(det)

                # Accumulate mask: masks are 288x288, resize to original
                mask_data = batch_masks[j] if batch_masks.ndim >= 3 \
                    else batch_masks
                binary_mask = (mask_data > 0).astype(np.uint8) * 255
                resized_mask = cv2.resize(
                    binary_mask, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST)

                # Each prompt gets a unique label value in the mask
                label_value = min(
                    255, (b + 1) * (255 // max(num_batches, 1)))
                combined_mask = np.maximum(
                    combined_mask,
                    (resized_mask > 0).astype(np.uint8) * label_value)

        # Publish detections
        self._detection_pub.publish(detection_array)

        # Publish combined mask
        try:
            mask_msg = self._bridge.cv2_to_imgmsg(combined_mask, 'mono8')
            mask_msg.header = header
            self._mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().error(f'Mask publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = Sam3Node()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
