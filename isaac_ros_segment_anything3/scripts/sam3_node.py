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
SAM3 / EfficientSAM3 ROS2 node for text-prompted segmentation.

Supports both SAM3 (full) and EfficientSAM3 (distilled) models via
model_type parameter. Both share the same 3-model pipeline:

Pipeline:
  1. Image -> preprocess -> vision_encoder (Triton) -> FPN features [cached per frame]
  2. Text prompt -> tokenize (input_ids + attention_mask)
     -> text_encoder (Triton) -> text_features + text_mask [cached]
  3. FPN features + prompt_features + prompt_mask -> decoder (Triton)
     -> pred_masks, pred_boxes (cx,cy,w,h normalized), pred_logits, presence_logits
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

import os
import threading
import time as _time

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

from isaac_ros_segment_anything3_interfaces.msg import Sam3Timing
from isaac_ros_segment_anything3_interfaces.srv import SetTextPrompt

_MAX_PROMPTS = 3

# Model profiles: model-specific constants for SAM3 and EfficientSAM3.
# Both share the same 3-model pipeline (vision_encoder + text_encoder + decoder)
# but differ in model sizes, image resolution, and default Triton model names.
MODEL_PROFILES = {
    'sam3': {
        'image_size': 1008,
        'max_seq_len': 32,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'stretch_resize': True,
        'default_vision_model': 'sam3_vision_encoder',
        'default_text_model': 'sam3_text_encoder',
        'default_decoder_model': 'sam3_decoder',
        'pytorch_builder': 'build_sam3_image_model',
        'pytorch_builder_kwargs': {},
    },
    'efficient_sam3': {
        'image_size': 1008,
        'max_seq_len': 32,       # Same CLIP BPE tokenizer
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'stretch_resize': True,  # Upstream uses stretch, not aspect-pad
        'default_vision_model': 'esam3_vision_encoder',
        'default_text_model': 'esam3_text_encoder',
        'default_decoder_model': 'esam3_decoder',
        'pytorch_builder': 'build_efficientsam3_image_model',
        'pytorch_builder_kwargs': {
            'backbone_type': 'tinyvit',
            'model_name': '11m',
            'text_encoder_type': 'MobileCLIP-S1',
        },
    },
}


class Sam3Node(Node):
    """ROS2 node for SAM3/EfficientSAM3 text-prompted segmentation.

    Supports two inference backends:
      - 'triton': ONNX models via Triton Inference Server (default)
      - 'pytorch': Direct PyTorch CUDA inference (~16x faster)
    """

    def __init__(self):
        super().__init__('sam3_node')

        # Load model profile first (determines defaults for other params)
        self.declare_parameter('model_type', 'sam3')
        model_type = self.get_parameter(
            'model_type').get_parameter_value().string_value
        if model_type not in MODEL_PROFILES:
            self.get_logger().error(
                f'Unknown model_type: {model_type}. '
                f'Valid options: {list(MODEL_PROFILES.keys())}. '
                f'Falling back to sam3.')
            model_type = 'sam3'
        self._model_type = model_type
        profile = MODEL_PROFILES[model_type]

        # Inference backend: 'triton' (ONNX via Triton) or 'pytorch' (direct CUDA)
        self.declare_parameter('inference_backend', 'triton')
        self.declare_parameter('pytorch_checkpoint', '')
        self.declare_parameter('pytorch_device', 'cuda')

        self._backend = self.get_parameter(
            'inference_backend').get_parameter_value().string_value
        self._pytorch_checkpoint = self.get_parameter(
            'pytorch_checkpoint').get_parameter_value().string_value
        self._pytorch_device = self.get_parameter(
            'pytorch_device').get_parameter_value().string_value

        # Optional TensorRT compiled vision encoder (.ep file).
        # When provided, replaces the PyTorch vision backbone for ~4x speedup.
        # Compile with: scripts/compile_sam3_trt.py
        self.declare_parameter('pytorch_trt_vision_engine', '')
        self._pytorch_trt_engine = self.get_parameter(
            'pytorch_trt_vision_engine').get_parameter_value().string_value

        # Optional TensorRT compiled decoder (.ep file).
        # When provided, replaces model.forward_grounding for ~3x speedup.
        # Compile with: scripts/compile_sam3_trt_decoder.py
        self.declare_parameter('pytorch_trt_decoder_engine', '')
        self._pytorch_trt_decoder_engine = self.get_parameter(
            'pytorch_trt_decoder_engine').get_parameter_value().string_value

        # Apply torch.compile() + AMP FP16 to decoder (alternative to TRT .ep).
        # First call is slow (~30s compilation). Gives ~3.4x speedup
        # via kernel fusion + FP16 mixed precision (no .ep file needed).
        self.declare_parameter('pytorch_compile_decoder', False)
        self._pytorch_compile_decoder = self.get_parameter(
            'pytorch_compile_decoder').get_parameter_value().bool_value

        # Use AMP FP16 for decoder when pytorch_compile_decoder=True.
        # Combined with torch.compile gives ~3.4x decoder speedup.
        self.declare_parameter('pytorch_amp_decoder', True)
        self._pytorch_amp_decoder = self.get_parameter(
            'pytorch_amp_decoder').get_parameter_value().bool_value

        # Declare parameters (defaults from profile)
        self.declare_parameter('triton_server_url', 'localhost:8001')
        self.declare_parameter('model_repository_path', '/tmp/models')
        self.declare_parameter(
            'vision_encoder_model_name',
            profile['default_vision_model'])
        self.declare_parameter(
            'text_encoder_model_name',
            profile['default_text_model'])
        self.declare_parameter(
            'decoder_model_name', profile['default_decoder_model'])
        self.declare_parameter('tokenizer_path', '/tmp/models/tokenizer.json')
        self.declare_parameter('image_size', profile['image_size'])
        self.declare_parameter('confidence_threshold', 0.5)
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
        self._max_seq_len = profile['max_seq_len']

        # Initialize tokenizer
        self._tokenizer = None
        self._init_tokenizer()

        # Initialize inference backend
        self._triton_client = None
        self._pytorch_model = None
        self._trt_decoder = None   # set in _init_pytorch_backend if .ep found
        if self._backend == 'pytorch':
            self._init_pytorch_backend()
        else:
            self._init_triton_client()

        # State (protected by lock for thread safety)
        self._lock = threading.Lock()
        self._current_prompts = []
        self._text_features_cache = None   # (text_features, text_mask) tuple
        self._bridge = CvBridge()

        # Image normalization from model profile
        self._mean = np.array(profile['mean'], dtype=np.float32)
        self._std = np.array(profile['std'], dtype=np.float32)
        self._stretch_resize = profile.get('stretch_resize', False)

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
        self._timing_pub = self.create_publisher(
            Sam3Timing, 'sam3/timing', 10)

        # Service: set text prompt
        self._set_prompt_srv = self.create_service(
            SetTextPrompt, 'sam3/set_text_prompt',
            self._set_text_prompt_callback)

        if self._backend == 'pytorch':
            self.get_logger().info(
                f'SAM3 node initialized (model_type={self._model_type}, '
                f'backend=pytorch, device={self._pytorch_device}, '
                f'image_size={self._image_size})')
        else:
            self.get_logger().info(
                f'SAM3 node initialized (model_type={self._model_type}, '
                f'backend=triton, url={self._triton_url}, '
                f'models=[{self._vision_model}, {self._text_model}, '
                f'{self._decoder_model}], image_size={self._image_size})')

    def _init_tokenizer(self):
        """Initialize the HuggingFace tokenizer."""
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(self._tokenizer_path)
            # Enable padding and truncation to fixed length
            self._tokenizer.enable_padding(
                length=self._max_seq_len, pad_id=0, pad_token='[PAD]')
            self._tokenizer.enable_truncation(
                max_length=self._max_seq_len)
            self.get_logger().info(
                f'Tokenizer loaded from {self._tokenizer_path} '
                f'(max_seq_len={self._max_seq_len})')
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

    def _init_pytorch_backend(self):
        """Initialize PyTorch models for direct CUDA inference.

        Uses the model's own forward_grounding() method (matching upstream
        Sam3Processor) instead of our ONNX export wrappers. This ensures
        the geometry encoder properly processes prompts with cross-attention
        to image features before the main decoder runs.

        Supports both full SAM3 and EfficientSAM3 via the pytorch_builder
        field in MODEL_PROFILES.
        """
        profile = MODEL_PROFILES[self._model_type]
        builder_name = profile.get('pytorch_builder')
        builder_kwargs = profile.get('pytorch_builder_kwargs')
        if builder_name is None:
            self.get_logger().error(
                f'PyTorch backend not supported for model_type='
                f'{self._model_type}. Use inference_backend:=triton.')
            return

        try:
            import torch
            self._torch = torch
        except ImportError:
            self.get_logger().error(
                'PyTorch not installed. '
                'Install with: pip install torch')
            return

        # Resolve checkpoint path
        checkpoint = self._pytorch_checkpoint
        if not checkpoint:
            default_name = ('efficient_sam3.pth'
                            if 'efficient' in self._model_type
                            else 'sam3.pt')
            checkpoint = os.path.join(self._model_repo, default_name)
        if not os.path.isfile(checkpoint):
            self.get_logger().error(
                f'PyTorch checkpoint not found: {checkpoint}. '
                f'Run download_models.py --inference-backend pytorch')
            return

        # Import builder function from sam3 package
        try:
            import sam3.model_builder as sam3_builder
            from sam3.model.data_misc import FindStage
            build_fn = getattr(sam3_builder, builder_name)
        except (ImportError, AttributeError) as e:
            self.get_logger().error(
                f'Cannot import {builder_name} from sam3 package: {e}. '
                f'Install with: pip install git+https://github.com/'
                f'SimonZeng7108/efficientsam3.git')
            return

        try:
            self.get_logger().info(
                f'Loading PyTorch model ({builder_name}) '
                f'from {checkpoint} ...')
            device = self._pytorch_device

            # Build model: kwargs differ per model type
            if builder_name == 'build_sam3_image_model':
                model = build_fn(
                    checkpoint_path=checkpoint,
                    device=device,
                    load_from_HF=False,
                )
            else:
                model = build_fn(
                    checkpoint_path=checkpoint,
                    device=device,
                    **builder_kwargs,
                )

            self._pytorch_model = model

            # Optionally load TRT-compiled vision encoder
            trt_engine_path = self._pytorch_trt_engine
            if not trt_engine_path:
                # Auto-detect: look for vision_encoder_trt_fp16.ep next to checkpoint
                default_trt = os.path.join(
                    os.path.dirname(checkpoint),
                    'vision_encoder_trt_fp16.ep')
                if os.path.isfile(default_trt):
                    trt_engine_path = default_trt

            if trt_engine_path and os.path.isfile(trt_engine_path):
                try:
                    self.get_logger().info(
                        f'Loading TRT vision engine from {trt_engine_path} ...')
                    # torch_tensorrt must be imported to register custom TRT ops
                    # before torch.export.load() can deserialize the engine
                    import torch_tensorrt  # noqa: F401
                    loaded_ep = torch.export.load(trt_engine_path)
                    trt_vision = loaded_ep.module()
                    # Drop-in replace: preserves (list[4], list[4], None, None) output
                    model.backbone.vision_backbone = trt_vision
                    self.get_logger().info(
                        'TRT vision engine loaded — vision encoder ~4x faster (FP16)')
                except Exception as e:
                    self.get_logger().warn(
                        f'Failed to load TRT engine ({e}), using PyTorch FP32')

            # Optionally load TRT-compiled decoder
            trt_decoder_path = self._pytorch_trt_decoder_engine
            if not trt_decoder_path:
                # Auto-detect: look for decoder_trt_fp16.ep next to checkpoint
                default_dec = os.path.join(
                    os.path.dirname(checkpoint),
                    'decoder_trt_fp16.ep')
                if os.path.isfile(default_dec):
                    trt_decoder_path = default_dec

            self._trt_decoder = None
            if trt_decoder_path and os.path.isfile(trt_decoder_path):
                try:
                    self.get_logger().info(
                        f'Loading TRT decoder from {trt_decoder_path} ...')
                    import torch_tensorrt  # noqa: F401  — registers TRT ops
                    loaded_ep = torch.export.load(trt_decoder_path)
                    self._trt_decoder = loaded_ep.module()
                    self.get_logger().info(
                        'TRT decoder loaded — decoder ~3x faster (FP16)')
                except Exception as e:
                    self.get_logger().warn(
                        f'Failed to load TRT decoder ({e}), using PyTorch')

            # Apply torch.compile() + AMP FP16 to decoder (alternative to TRT .ep).
            # Skipped if TRT decoder (.ep) already loaded.
            # Strategy: disable tracing into geometry_encoder._encode_boxes
            # (contains pin_memory() that torch.compile cannot trace), then
            # compile the full forward_grounding. At inference time, use
            # autocast FP16 for ~3.4x speedup.
            if self._trt_decoder is None and self._pytorch_compile_decoder:
                try:
                    import types
                    import torch._dynamo
                    # Pin geometry_encoder._encode_boxes to run eagerly
                    # (trivial op with empty dummy prompts, ~0ms)
                    orig_fn = model.geometry_encoder._encode_boxes.__func__
                    model.geometry_encoder._encode_boxes = types.MethodType(
                        torch._dynamo.disable(orig_fn),
                        model.geometry_encoder,
                    )
                    model.forward_grounding = torch.compile(
                        model.forward_grounding,
                        mode='default',
                    )
                    amp_note = ' + AMP FP16' if self._pytorch_amp_decoder else ''
                    self.get_logger().info(
                        f'Decoder torch.compile(){amp_note} applied '
                        f'(first call ~30s compile, then ~3.4x faster)')
                except Exception as e:
                    self.get_logger().warn(
                        f'torch.compile() failed ({e}), using eager mode')

            # Pre-create FindStage for single-image inference
            # (reused across frames, only text_ids changes per prompt)
            self._find_stage = FindStage(
                img_ids=self._torch.tensor(
                    [0], device=device, dtype=self._torch.long),
                text_ids=self._torch.tensor(
                    [0], device=device, dtype=self._torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            # Warmup: run a dummy forward pass to trigger torch.compile()
            # compilation at startup so the first real inference is fast.
            if self._pytorch_compile_decoder:
                self.get_logger().info(
                    'Running decoder warmup (torch.compile, ~30s)...')
                dummy_img = torch.zeros(
                    1, 3, self._image_size, self._image_size, device=device)
                with torch.inference_mode():
                    bb = model.backbone.forward_image(dummy_img)
                    to = model.backbone.forward_text(
                        ['warmup'], device=device)
                    bb.update(to)
                    geo = model._get_dummy_prompt()
                    import contextlib
                    amp_ctx = (
                        torch.autocast(device_type='cuda',
                                       dtype=torch.float16)
                        if self._pytorch_amp_decoder
                        else contextlib.nullcontext()
                    )
                    with amp_ctx:
                        model.forward_grounding(
                            backbone_out=bb,
                            find_input=self._find_stage,
                            geometric_prompt=geo,
                            find_target=None,
                        )
                self.get_logger().info('Decoder warmup complete.')

            self.get_logger().info(
                f'PyTorch model loaded on {device} '
                f'(using model.forward_grounding directly)')

        except Exception as e:
            self.get_logger().error(
                f'Failed to load PyTorch model: {e}')
            import traceback
            traceback.print_exc()

    # ----------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------

    def _preprocess_image(self, cv_image):
        """
        Preprocess image for SAM3 vision encoder.

        Two modes depending on model profile:
          stretch: resize to square (upstream EfficientSAM3 convention)
          pad:     aspect-preserving resize + bottom-right padding (SAM2)

        Args:
            cv_image: RGB uint8 image (H, W, 3).

        Returns:
            np.ndarray of shape (1, 3, image_size, image_size), float32.
        """
        sz = self._image_size

        if self._stretch_resize:
            # Upstream EfficientSAM3: stretch to square, normalize to [-1, 1]
            resized = cv2.resize(cv_image, (sz, sz),
                                 interpolation=cv2.INTER_LINEAR)
            img = resized.astype(np.float32) / 255.0
        else:
            # SAM2/SAM3 convention: aspect-preserving + bottom-right pad
            h, w = cv_image.shape[:2]
            scale = sz / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(cv_image, (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)
            img = np.zeros((sz, sz, 3), dtype=np.float32)
            img[:new_h, :new_w] = resized.astype(np.float32) / 255.0

        # Normalize (mean/std from model profile)
        img = (img - self._mean) / self._std

        # HWC -> CHW -> NCHW
        return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def _tokenize_text(self, prompts):
        """
        Tokenize text prompts using HuggingFace tokenizer.

        Produces both input_ids and attention_mask with fixed sequence
        length of max_seq_len (from model profile).

        Args:
            prompts: List of strings.

        Returns:
            Tuple of (input_ids, attention_mask):
              - input_ids: np.ndarray of shape (num_prompts, max_seq_len), int64
              - attention_mask: np.ndarray of shape (num_prompts, max_seq_len), int64
            Returns (None, None) on failure.
        """
        if self._tokenizer is None:
            self.get_logger().error('Tokenizer not initialized')
            return None, None

        encodings = self._tokenizer.encode_batch(prompts)
        num_prompts = len(prompts)
        max_seq_len = self._max_seq_len
        input_ids = np.zeros(
            (num_prompts, max_seq_len), dtype=np.int64)
        attention_mask = np.zeros(
            (num_prompts, max_seq_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            seq_len = min(len(enc.ids), max_seq_len)
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
    # PyTorch inference
    # ----------------------------------------------------------------

    def _run_pytorch_forward(self, image_np, prompts, text_cache):
        """
        Run the full SAM3/EfficientSAM3 pipeline via PyTorch model directly.

        Uses model.forward_grounding() which matches upstream Sam3Processor
        exactly, including proper geometry encoder CLS processing with
        cross-attention to image features.

        Optionally uses TRT-compiled decoder (self._trt_decoder) or
        torch.compile()-decorated forward_grounding for additional speedup.

        Args:
            image_np: np.ndarray (1, 3, 1008, 1008), float32.
            prompts: List of text prompt strings.
            text_cache: Cached backbone_out text fields, or None.

        Returns:
            Tuple of (all_results, text_cache_new, stage_times):
              all_results: List of dicts per prompt with pred_masks/boxes/logits/presence
              text_cache_new: Updated text cache dict for reuse.
              stage_times: dict with keys 'vision_ms', 'text_ms', 'decoder_ms'
            Returns (None, text_cache, {}) on failure.
        """
        try:
            torch = self._torch
            device = self._pytorch_device
            model = self._pytorch_model

            image = torch.from_numpy(image_np).to(device)

            with torch.inference_mode():
                # 1. Vision encoder (run once per frame)
                backbone_out = model.backbone.forward_image(image)
                torch.cuda.synchronize()
                t_vision_done = _time.perf_counter()

                # 2. Text encoder (cached until prompts change)
                if text_cache is not None:
                    backbone_out.update(text_cache)
                    text_cache_new = text_cache
                else:
                    text_outputs = model.backbone.forward_text(
                        prompts, device=device)
                    backbone_out.update(text_outputs)
                    text_cache_new = text_outputs
                torch.cuda.synchronize()
                t_text_done = _time.perf_counter()

                # 3. Run decoder per prompt
                all_results = []

                if self._trt_decoder is not None:
                    # TRT decoder path: extract tensors from backbone_out
                    # and call the compiled DecoderWrapper.
                    # Wrapper inputs: fpn_0, fpn_1, fpn_2, fpn_pos_2,
                    #                  lang_feat, lang_mask, lang_embeds
                    fpn = backbone_out['backbone_fpn']
                    pos = backbone_out['vision_pos_enc']
                    lang_feat = backbone_out['language_features']
                    lang_mask = backbone_out['language_mask']
                    lang_embeds = backbone_out['language_embeds']
                    fpn_pos_2 = pos[-1]

                    for p_idx in range(len(prompts)):
                        # Select single-prompt text features (seq_first)
                        pf = lang_feat[:, p_idx:p_idx+1, :]   # (seq, 1, 256)
                        pm = lang_mask[p_idx:p_idx+1, :]       # (1, seq)
                        pe = lang_embeds[p_idx:p_idx+1, :, :]  # (1, 1, 256)

                        out = self._trt_decoder(
                            fpn[0], fpn[1], fpn[2], fpn_pos_2,
                            pf, pm, pe)

                        all_results.append({
                            'pred_masks': out[0].cpu().numpy(),
                            'pred_boxes': out[1].cpu().numpy(),
                            'pred_logits': out[2].cpu().numpy(),
                            'presence': out[3].cpu().numpy(),
                        })
                else:
                    # PyTorch (eager or torch.compile) decoder path.
                    # Use AMP FP16 if enabled (gives ~3.4x speedup when
                    # combined with torch.compile).
                    import contextlib
                    amp_ctx = (
                        torch.autocast(device_type='cuda', dtype=torch.float16)
                        if self._pytorch_amp_decoder and self._pytorch_compile_decoder
                        else contextlib.nullcontext()
                    )
                    for p_idx in range(len(prompts)):
                        find_stage = self._find_stage
                        find_stage.text_ids = torch.tensor(
                            [p_idx], device=device, dtype=torch.long)

                        geo_prompt = model._get_dummy_prompt()

                        with amp_ctx:
                            outputs = model.forward_grounding(
                                backbone_out=backbone_out,
                                find_input=find_stage,
                                geometric_prompt=geo_prompt,
                                find_target=None,
                            )

                        all_results.append({
                            'pred_masks': outputs['pred_masks'].float().cpu().numpy(),
                            'pred_boxes': outputs['pred_boxes'].float().cpu().numpy(),
                            'pred_logits': outputs['pred_logits'].float().cpu().numpy(),
                            'presence': outputs['presence_logit_dec'].float().cpu().numpy(),
                        })

                torch.cuda.synchronize()
                t_decoder_done = _time.perf_counter()

            return all_results, text_cache_new, {
                'vision_ms': 0.0,    # filled in by caller with wall-clock split
                'text_ms': 0.0,
                'decoder_ms': 0.0,
                '_t_vision': t_vision_done,
                '_t_text': t_text_done,
                '_t_decoder': t_decoder_done,
            }

        except Exception as e:
            self.get_logger().error(
                f'PyTorch forward failed: {e}')
            import traceback
            traceback.print_exc()
            return None, text_cache, {}

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
            backend_ready = (
                self._pytorch_model is not None
                if self._backend == 'pytorch'
                else self._triton_client is not None)
            # PyTorch uses model.backbone.forward_text() (has own tokenizer).
            # Triton needs our external HuggingFace tokenizer.
            tokenizer_ready = (
                self._backend == 'pytorch' or self._tokenizer is not None)
            if not backend_ready or not tokenizer_ready:
                self.get_logger().warn(
                    'Inference backend or tokenizer not ready, skipping')
                return
            current_prompts = list(self._current_prompts)
            text_cache = self._text_features_cache
            confidence_threshold = self._confidence_threshold

        _t_start = _time.perf_counter()

        self.get_logger().debug(
            f'Processing {msg.width}x{msg.height} image with '
            f'prompts={current_prompts}')

        # Convert ROS image to OpenCV (no lock needed)
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return
        _t_cvbridge = _time.perf_counter()

        orig_h, orig_w = cv_image.shape[:2]

        # 1. Preprocess image
        preprocessed = self._preprocess_image(cv_image)
        _t_preproc = _time.perf_counter()

        if self._backend == 'pytorch':
            # PyTorch: use model.forward_grounding() directly.
            # This matches upstream Sam3Processor exactly, including
            # proper geometry encoder CLS processing.
            text_cache_hit = (text_cache is not None)

            results, text_cache_new, stage_times = self._run_pytorch_forward(
                preprocessed, current_prompts, text_cache)
            if results is None:
                return

            _t_model = _time.perf_counter()

            # Update text cache under lock
            if not text_cache_hit and text_cache_new is not None:
                with self._lock:
                    if self._current_prompts == current_prompts:
                        self._text_features_cache = text_cache_new

            # Stack results across prompts
            num_prompts = len(results)
            all_masks = []
            all_boxes = []
            all_logits = []
            all_presence = []
            for r in results:
                # forward_grounding output shapes:
                #   pred_masks: [1, 200, 288, 288]
                #   pred_boxes: [1, 200, 4]
                #   pred_logits: [1, 200, 1]
                #   presence: [1, 1] (per-text-prompt, not per-query)
                all_masks.append(r['pred_masks'][0])       # (200, 288, 288)
                all_boxes.append(r['pred_boxes'][0])       # (200, 4)
                all_logits.append(r['pred_logits'][0])     # (200, 1)
                all_presence.append(r['presence'][0])      # (1,)

            pred_masks = np.stack(all_masks)       # (N, 200, 288, 288)
            pred_boxes = np.stack(all_boxes)       # (N, 200, 4)
            pred_logits = np.stack(all_logits)     # (N, 200, 1)
            presence_logits = np.stack(all_presence)  # (N, 1)

            # 5. Post-process and publish
            self._publish_results(
                msg.header, orig_h, orig_w,
                pred_masks, pred_boxes, pred_logits, presence_logits,
                current_prompts, confidence_threshold)
            _t_end = _time.perf_counter()

            # Per-stage timing from _run_pytorch_forward CUDA sync points.
            # stage_times['_t_vision/_t_text/_t_decoder'] are absolute timestamps.
            _t_preproc_end = _t_preproc  # alias for clarity
            t_vis = stage_times.get('_t_vision', _t_model)
            t_txt = stage_times.get('_t_text', _t_model)
            t_dec = stage_times.get('_t_decoder', _t_model)

            timing_msg = Sam3Timing()
            timing_msg.header.stamp = self.get_clock().now().to_msg()
            timing_msg.cvbridge_ms = (_t_cvbridge - _t_start) * 1000.0
            timing_msg.preprocess_ms = (_t_preproc_end - _t_cvbridge) * 1000.0
            # vision_encoder_ms: from end of preproc to end of forward_image
            timing_msg.vision_encoder_ms = (t_vis - _t_preproc_end) * 1000.0
            # text_encoder_ms: from end of vision to end of forward_text
            timing_msg.text_encoder_ms = (t_txt - t_vis) * 1000.0
            timing_msg.text_encoder_cache_hit = text_cache_hit
            # decoder_ms: from end of text to end of all decoder calls
            timing_msg.decoder_ms = (t_dec - t_txt) * 1000.0
            timing_msg.num_prompts = num_prompts
            timing_msg.postprocess_ms = (_t_end - _t_model) * 1000.0
            timing_msg.total_ms = (_t_end - _t_start) * 1000.0
            timing_msg.backend = self._backend
            timing_msg.model_type = self._model_type

        else:
            # Triton: 3-stage pipeline (vision → text → decoder)
            # 2. Vision encoder
            vision_result = self._run_vision_encoder(preprocessed)
            if vision_result is None:
                return
            _t_vision = _time.perf_counter()

            # 3. Text encoder (cached until prompt changes)
            _t_text_start = _time.perf_counter()
            text_cache_hit = (text_cache is not None)

            if text_cache is not None:
                text_features, text_mask = text_cache
            else:
                input_ids, attention_mask = self._tokenize_text(
                    current_prompts)
                if input_ids is None:
                    return
                text_result = self._run_text_encoder(
                    input_ids, attention_mask)
                if text_result is None:
                    return
                text_features, text_mask = text_result
                with self._lock:
                    if self._current_prompts == current_prompts:
                        self._text_features_cache = (
                            text_features, text_mask)

            _t_text_end = _time.perf_counter()

            # 4. Decoder: run per-prompt (batch=1).
            all_masks = []
            all_boxes = []
            all_logits = []
            all_presence = []

            num_prompts = text_features.shape[0]
            fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2 = vision_result
            for p_idx in range(num_prompts):
                pf = text_features[p_idx:p_idx + 1].astype(np.float32)
                pm = text_mask[p_idx:p_idx + 1].astype(bool)
                decoder_result = self._run_decoder(
                    fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2,
                    pf, pm)
                if decoder_result is None:
                    continue
                masks, boxes, logits, presence = decoder_result
                all_masks.append(masks[0])
                all_boxes.append(boxes[0])
                all_logits.append(logits[0])
                all_presence.append(presence[0])

            _t_decoder = _time.perf_counter()

            if not all_masks:
                return

            pred_masks = np.stack(all_masks)
            pred_boxes = np.stack(all_boxes)
            pred_logits = np.stack(all_logits)
            presence_logits = np.stack(all_presence)

            # 5. Post-process and publish
            self._publish_results(
                msg.header, orig_h, orig_w,
                pred_masks, pred_boxes, pred_logits, presence_logits,
                current_prompts, confidence_threshold)
            _t_end = _time.perf_counter()

            timing_msg = Sam3Timing()
            timing_msg.header.stamp = self.get_clock().now().to_msg()
            timing_msg.cvbridge_ms = (_t_cvbridge - _t_start) * 1000.0
            timing_msg.preprocess_ms = (_t_preproc - _t_cvbridge) * 1000.0
            timing_msg.vision_encoder_ms = (
                _t_vision - _t_preproc) * 1000.0
            timing_msg.text_encoder_ms = (
                _t_text_end - _t_text_start) * 1000.0
            timing_msg.text_encoder_cache_hit = text_cache_hit
            timing_msg.decoder_ms = (
                _t_decoder - _t_vision
                - (_t_text_end - _t_text_start)) * 1000.0
            timing_msg.num_prompts = num_prompts
            timing_msg.postprocess_ms = (_t_end - _t_decoder) * 1000.0
            timing_msg.total_ms = (_t_end - _t_start) * 1000.0
            timing_msg.backend = self._backend
            timing_msg.model_type = self._model_type

        self._timing_pub.publish(timing_msg)

        _text_hit = 'HIT' if timing_msg.text_encoder_cache_hit else 'MISS'
        self.get_logger().info(
            f'Timing: cvbridge={timing_msg.cvbridge_ms:.1f}ms '
            f'preproc={timing_msg.preprocess_ms:.1f}ms '
            f'vision={timing_msg.vision_encoder_ms:.1f}ms '
            f'text={timing_msg.text_encoder_ms:.1f}ms({_text_hit}) '
            f'decoder={timing_msg.decoder_ms:.1f}ms '
            f'postproc={timing_msg.postprocess_ms:.1f}ms '
            f'total={timing_msg.total_ms:.1f}ms')

    # ----------------------------------------------------------------
    # Output publishing
    # ----------------------------------------------------------------

    @staticmethod
    def _cxcywh_to_xyxy(boxes, orig_w, orig_h, image_size, stretch):
        """
        Convert normalized [cx, cy, w, h] boxes to pixel [x1, y1, x2, y2].

        Args:
            boxes: np.ndarray (..., 4) with values in [0, 1].
            orig_w: Original image width in pixels.
            orig_h: Original image height in pixels.
            image_size: Model input square size used in preprocessing.
            stretch: If True, image was stretched to square (scale by w/h).
                     If False, aspect-preserving pad was used.

        Returns:
            np.ndarray (..., 4) in pixel coordinates [x1, y1, x2, y2].
        """
        if stretch:
            # Upstream: boxes normalized to [0,1], scale directly by image dims
            cx = boxes[..., 0] * float(orig_w)
            cy = boxes[..., 1] * float(orig_h)
            w = boxes[..., 2] * float(orig_w)
            h = boxes[..., 3] * float(orig_h)
        else:
            # Pad mode: boxes in model-input space, undo padding scale
            max_side = float(max(orig_w, orig_h))
            scale_back = max_side / float(image_size)
            cx = boxes[..., 0] * float(image_size) * scale_back
            cy = boxes[..., 1] * float(image_size) * scale_back
            w = boxes[..., 2] * float(image_size) * scale_back
            h = boxes[..., 3] * float(image_size) * scale_back

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1 = np.clip(x1, 0.0, float(orig_w))
        y1 = np.clip(y1, 0.0, float(orig_h))
        x2 = np.clip(x2, 0.0, float(orig_w))
        y2 = np.clip(y2, 0.0, float(orig_h))

        return np.stack([x1, y1, x2, y2], axis=-1)

    def _publish_results(self, header, orig_h, orig_w,
                         pred_masks, pred_boxes, pred_logits,
                         presence_logits, prompts, confidence_threshold):
        """
        Filter by confidence threshold and publish results.

        Decoder outputs (batch dimension = number of text prompts):
          - pred_masks: (batch, 200, 288, 288) float32
          - pred_boxes: (batch, 200, 4) float32, normalized [cx, cy, w, h]
          - pred_logits: (batch, 200, 1) or (batch, 200) float32 (raw logits)
          - presence_logits: (batch, 1) float32 (per-text-prompt gate)

        Scoring: final = sigmoid(pred_logits) * sigmoid(presence_logits)
        Matching upstream Sam3Processor._forward_grounding().
        """
        detection_array = Detection2DArray()
        detection_array.header = header

        # Combined mask for all detections
        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Convert logits to confidence scores via sigmoid.
        # Following upstream: final_score = sigmoid(logit) * sigmoid(presence)
        det_scores = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        presence_scores = 1.0 / (
            1.0 + np.exp(-presence_logits.astype(np.float64)))

        # Handle shape variations between PyTorch and Triton backends:
        # PyTorch (forward_grounding): logits (B,200,1), presence (B,1)
        # Triton (DecoderWrapper):     logits (B,200),   presence (B,1)
        if det_scores.ndim == 3:
            det_scores = det_scores.squeeze(-1)  # (B, 200, 1) -> (B, 200)
        # presence (B, 1) broadcasts with det_scores (B, 200) -> (B, 200)
        scores = det_scores * presence_scores

        self.get_logger().debug(
            f'Scores: max_det={float(det_scores.max()):.4f}, '
            f'max_pres={float(presence_scores.max()):.4f}, '
            f'max_final={float(scores.max()):.4f}, '
            f'above_thresh={(scores > confidence_threshold).sum()}')

        # pred_masks, pred_boxes, scores shape: (batch, 200, ...)
        num_batches = scores.shape[0] if scores.ndim >= 2 else 1

        for b in range(num_batches):
            batch_scores = scores[b] if scores.ndim >= 2 else scores
            batch_boxes = pred_boxes[b] if pred_boxes.ndim >= 3 else pred_boxes
            batch_masks = pred_masks[b] if pred_masks.ndim >= 4 else pred_masks

            # Convert boxes from normalized [cx,cy,w,h] to original pixels
            boxes_xyxy = self._cxcywh_to_xyxy(
                batch_boxes, orig_w, orig_h, self._image_size,
                self._stretch_resize)

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
                if (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
                    continue
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

                # Accumulate mask: decoder outputs (288, 288) logits.
                # Following upstream: interpolate -> sigmoid -> threshold.
                mask_data = batch_masks[j] if batch_masks.ndim >= 3 \
                    else batch_masks

                if self._stretch_resize:
                    # Stretch mode: bilinear upscale full mask to orig size,
                    # apply sigmoid, then threshold at 0.5
                    mask_float = mask_data.astype(np.float32)
                    resized_logits = cv2.resize(
                        mask_float, (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR)
                    # sigmoid -> threshold
                    resized_mask = (1.0 / (1.0 + np.exp(-resized_logits))
                                    > 0.5).astype(np.uint8) * 255
                else:
                    # Pad mode: crop valid region, then resize
                    binary_mask = (mask_data > 0).astype(np.uint8) * 255
                    scale = self._image_size / float(max(orig_h, orig_w))
                    valid_h = int(orig_h * scale)
                    valid_w = int(orig_w * scale)
                    mask_h, mask_w = binary_mask.shape[:2]
                    crop_h = max(1, min(mask_h, int(round(
                        valid_h * (mask_h / float(self._image_size))))))
                    crop_w = max(1, min(mask_w, int(round(
                        valid_w * (mask_w / float(self._image_size))))))
                    resized_mask = cv2.resize(
                        binary_mask[:crop_h, :crop_w], (orig_w, orig_h),
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
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
