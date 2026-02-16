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
Proof-of-Life Test for Isaac ROS Segment Anything 3.

Tests:
  1. SAM3 node launches successfully.
  2. SetTextPrompt service is advertised and responds.
  3. Detection output is published when image + prompt are provided.

Prerequisites:
  - Triton server running with SAM3 models loaded.
  - Models downloaded via download_models.py.
"""

import os
import time

from isaac_ros_segment_anything3_interfaces.srv import SetTextPrompt
from isaac_ros_test import IsaacROSBaseTest
import launch
from launch_ros.actions import Node
import pytest
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for SAM3 POL test."""
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    namespace = IsaacROSSegmentAnything3Test.generate_namespace()

    model_repo = os.environ.get('SAM3_MODEL_REPO', '/tmp/models')
    tokenizer_path = os.environ.get(
        'SAM3_TOKENIZER_PATH', '/tmp/models/tokenizer.json')
    triton_url = os.environ.get('SAM3_TRITON_URL', 'localhost:8001')

    sam3_node = Node(
        package='isaac_ros_segment_anything3',
        executable='sam3_node.py',
        name='sam3_node',
        namespace=namespace,
        parameters=[{
            'triton_server_url': triton_url,
            'model_repository_path': model_repo,
            'tokenizer_path': tokenizer_path,
            'image_size': 1024,
            'confidence_threshold': 0.1,
        }],
        output='screen',
    )

    return IsaacROSSegmentAnything3Test.generate_test_description([
        sam3_node,
    ])


class IsaacROSSegmentAnything3Test(IsaacROSBaseTest):
    """
    Proof-of-Life Test for Isaac ROS SAM3 pipeline.

    1. Verifies node starts and service is available.
    2. Sets text prompts via service.
    3. Publishes test image and verifies detection output.
    """

    SUBSCRIBER_CHANNEL = 'sam3/detections'
    MASK_CHANNEL = 'sam3/raw_segmentation_mask'
    TEST_DURATION = 200.0

    def test_sam3_service_and_detection(self) -> None:
        self.node._logger.info('Starting Isaac ROS SAM3 POL Test')

        received_messages = {}

        subscriber_topic_namespace = self.generate_namespace(
            self.SUBSCRIBER_CHANNEL)
        mask_topic_namespace = self.generate_namespace(
            self.MASK_CHANNEL)

        test_subscribers = [
            (subscriber_topic_namespace, Detection2DArray),
            (mask_topic_namespace, Image),
        ]

        subs = self.create_logging_subscribers(
            subscription_requests=test_subscribers,
            received_messages=received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            add_received_message_timestamps=True
        )

        # Create service client
        service_name = self.generate_namespace('sam3/set_text_prompt')
        self.cli = self.node.create_client(SetTextPrompt, service_name)

        # Create image publisher
        image_topic = self.generate_namespace('image_raw')
        image_pub = self.node.create_publisher(Image, image_topic, 10)

        try:
            # Wait for service
            self.node._logger.info('Waiting for set_text_prompt service...')
            timeout_count = 0
            while not self.cli.wait_for_service(timeout_sec=1.0):
                timeout_count += 1
                if timeout_count > 30:
                    self.fail('set_text_prompt service not available')

            # Set text prompts
            req = SetTextPrompt.Request()
            req.request_header = Header()
            req.text_prompts = ['person']
            req.confidence_threshold = 0.1

            self.node._logger.info('Calling set_text_prompt service...')
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)

            if future.done():
                response = future.result()
                self.assertTrue(response.success,
                                f'Service failed: {response.message}')
                self.node._logger.info(
                    f'Prompts set: {response.active_prompts}')
            else:
                self.fail('Service call timed out')

            # Publish test images and wait for detections
            self.node._logger.info(
                'Publishing test images and waiting for detections...')

            end_time = time.time() + self.TEST_DURATION
            while time.time() < end_time:
                # Create a dummy test image (640x480 RGB)
                test_img = Image()
                test_img.header = Header()
                test_img.header.stamp = self.node.get_clock().now().to_msg()
                test_img.height = 480
                test_img.width = 640
                test_img.encoding = 'rgb8'
                test_img.step = 640 * 3
                test_img.data = bytes(640 * 480 * 3)
                image_pub.publish(test_img)

                time.sleep(1.0)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if len(received_messages.get(
                        subscriber_topic_namespace, [])) > 0:
                    break

            # Verify detections received
            num_detections_received = len(
                received_messages.get(subscriber_topic_namespace, []))
            self.assertGreater(
                num_detections_received, 0,
                'No detection messages received')

            # Log detection info
            det_array, _ = received_messages[subscriber_topic_namespace][-1]
            self.node._logger.info(
                f'Received {len(det_array.detections)} detections '
                f'in {num_detections_received} messages')

            for det in det_array.detections:
                for result in det.results:
                    self.node._logger.info(
                        f'  Detection: class={result.hypothesis.class_id}, '
                        f'score={result.hypothesis.score:.3f}, '
                        f'bbox=({det.bbox.center.position.x:.0f}, '
                        f'{det.bbox.center.position.y:.0f}, '
                        f'{det.bbox.size_x:.0f}, {det.bbox.size_y:.0f})')

            # Verify masks received
            num_masks_received = len(
                received_messages.get(mask_topic_namespace, []))
            self.assertGreater(
                num_masks_received, 0,
                'No mask messages received')

            mask_msg, _ = received_messages[mask_topic_namespace][-1]
            self.assertEqual(mask_msg.encoding, 'mono8')
            self.node._logger.info(
                f'Received mask: {mask_msg.width}x{mask_msg.height} '
                f'{mask_msg.encoding}')

            self.node._logger.info(
                'Finished Isaac ROS SAM3 POL Test')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
