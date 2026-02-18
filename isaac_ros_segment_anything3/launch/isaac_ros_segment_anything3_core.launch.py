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

Integrates with the Isaac ROS Examples framework.
SAM3 uses a standalone Python node (not ComposableNode) with direct
PyTorch CUDA inference and optional TensorRT-compiled vision encoder.
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
        # SAM3 uses a standalone Python node, not ComposableNodes.
        return {}

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) \
            -> Dict[str, launch.actions.OpaqueFunction]:

        sam3_pytorch_checkpoint = LaunchConfiguration('sam3_pytorch_checkpoint')
        sam3_pytorch_device = LaunchConfiguration('sam3_pytorch_device')
        sam3_pytorch_compile_decoder = LaunchConfiguration(
            'sam3_pytorch_compile_decoder')
        sam3_pytorch_amp_decoder = LaunchConfiguration(
            'sam3_pytorch_amp_decoder')
        sam3_confidence_threshold = LaunchConfiguration(
            'sam3_confidence_threshold')

        img_topic = interface_specs.get(
            'subscribed_topics', {}).get('image', 'image_rect')

        return {
            'sam3_pytorch_checkpoint': DeclareLaunchArgument(
                'sam3_pytorch_checkpoint',
                default_value='/tmp/models/sam3.pt',
                description='Path to SAM3 checkpoint (sam3.pt, ~3.3GB)'),
            'sam3_pytorch_device': DeclareLaunchArgument(
                'sam3_pytorch_device',
                default_value='cuda',
                description='PyTorch device (cuda or cpu)'),
            'sam3_pytorch_compile_decoder': DeclareLaunchArgument(
                'sam3_pytorch_compile_decoder',
                default_value='True',
                description='Apply torch.compile to decoder (~30s startup, ~3x faster)'),
            'sam3_pytorch_amp_decoder': DeclareLaunchArgument(
                'sam3_pytorch_amp_decoder',
                default_value='True',
                description='Use AMP FP16 for decoder (requires torch.compile)'),
            'sam3_confidence_threshold': DeclareLaunchArgument(
                'sam3_confidence_threshold',
                default_value='0.3',
                description='Confidence threshold for SAM3 detections'),
            'sam3_node': Node(
                package='isaac_ros_segment_anything3',
                executable='sam3_node.py',
                name='sam3_node',
                parameters=[{
                    'pytorch_checkpoint': sam3_pytorch_checkpoint,
                    'pytorch_device': sam3_pytorch_device,
                    'pytorch_compile_decoder': sam3_pytorch_compile_decoder,
                    'pytorch_amp_decoder': sam3_pytorch_amp_decoder,
                    'confidence_threshold': sam3_confidence_threshold,
                }],
                remappings=[
                    ('image_raw', img_topic),
                ],
                output='screen',
            ),
        }
