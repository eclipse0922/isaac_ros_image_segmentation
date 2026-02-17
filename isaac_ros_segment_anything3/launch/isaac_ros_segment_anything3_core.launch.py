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
IsaacROSLaunchFragment for SAM3 text-prompted segmentation.

This launch fragment integrates with the Isaac ROS Examples framework.
SAM3 uses a standalone Python node (not ComposableNode) because it
orchestrates three separate Triton models with tokenization logic.
"""

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


class IsaacROSSegmentAnything3LaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) \
            -> Dict[str, Any]:
        # SAM3 uses a standalone Python node, not ComposableNodes,
        # because it orchestrates three Triton models with text tokenization.
        return {}

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) \
            -> Dict[str, launch.actions.OpaqueFunction]:

        sam3_model_type = LaunchConfiguration('sam3_model_type')
        sam3_triton_url = LaunchConfiguration('sam3_triton_url')
        sam3_model_repo = LaunchConfiguration('sam3_model_repo')
        sam3_tokenizer_path = LaunchConfiguration('sam3_tokenizer_path')
        sam3_confidence_threshold = LaunchConfiguration(
            'sam3_confidence_threshold')
        sam3_image_size = LaunchConfiguration('sam3_image_size')
        sam3_inference_backend = LaunchConfiguration(
            'sam3_inference_backend')
        sam3_pytorch_checkpoint = LaunchConfiguration(
            'sam3_pytorch_checkpoint')
        sam3_pytorch_device = LaunchConfiguration(
            'sam3_pytorch_device')

        img_topic = interface_specs.get(
            'subscribed_topics', {}).get('image', 'image_rect')

        return {
            'sam3_model_type': DeclareLaunchArgument(
                'sam3_model_type',
                default_value='sam3',
                description='Model type: sam3 or efficient_sam3'),
            'sam3_triton_url': DeclareLaunchArgument(
                'sam3_triton_url',
                default_value='localhost:8001',
                description='Triton gRPC server URL for SAM3'),
            'sam3_model_repo': DeclareLaunchArgument(
                'sam3_model_repo',
                default_value='/tmp/models',
                description='Triton model repository path for SAM3'),
            'sam3_tokenizer_path': DeclareLaunchArgument(
                'sam3_tokenizer_path',
                default_value='/tmp/models/tokenizer.json',
                description='Path to SAM3 tokenizer.json'),
            'sam3_confidence_threshold': DeclareLaunchArgument(
                'sam3_confidence_threshold',
                default_value='0.3',
                description='Confidence threshold for SAM3 detections'),
            'sam3_image_size': DeclareLaunchArgument(
                'sam3_image_size',
                default_value='1008',
                description='SAM3 model input image size'),
            'sam3_inference_backend': DeclareLaunchArgument(
                'sam3_inference_backend',
                default_value='triton',
                description='Inference backend: triton or pytorch'),
            'sam3_pytorch_checkpoint': DeclareLaunchArgument(
                'sam3_pytorch_checkpoint',
                default_value='',
                description='PyTorch checkpoint path (empty=auto)'),
            'sam3_pytorch_device': DeclareLaunchArgument(
                'sam3_pytorch_device',
                default_value='cuda',
                description='PyTorch device (cuda or cpu)'),
            'sam3_node': Node(
                package='isaac_ros_segment_anything3',
                executable='sam3_node.py',
                name='sam3_node',
                parameters=[{
                    'model_type': sam3_model_type,
                    'triton_server_url': sam3_triton_url,
                    'model_repository_path': sam3_model_repo,
                    'tokenizer_path': sam3_tokenizer_path,
                    'confidence_threshold': sam3_confidence_threshold,
                    'image_size': sam3_image_size,
                    'inference_backend': sam3_inference_backend,
                    'pytorch_checkpoint': sam3_pytorch_checkpoint,
                    'pytorch_device': sam3_pytorch_device,
                }],
                remappings=[
                    ('image_raw', img_topic),
                ],
                output='screen',
            ),
        }
