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
Standalone launch file for SAM3 text-prompted segmentation.

Launches the SAM3 node with PyTorch direct inference (+ optional TRT).
TRT vision engine is auto-detected from the checkpoint directory.

Usage:
    # Basic launch (auto-detects TRT engine in checkpoint directory)
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3.launch.py \\
        pytorch_checkpoint:=/ws/models/sam3/sam3.pt

    # Custom topic
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3.launch.py \\
        pytorch_checkpoint:=/ws/models/sam3/sam3.pt \\
        input_image_topic:=/camera/color/image_raw

    # Set text prompt after ~40s node startup:
    ros2 service call /sam3/set_text_prompt \\
        isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \\
        "{text_prompts: ['robot arm']}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pytorch_checkpoint_arg = DeclareLaunchArgument(
        'pytorch_checkpoint',
        default_value='/tmp/models/sam3.pt',
        description='Path to SAM3 checkpoint (sam3.pt, ~3.3GB)')

    pytorch_device_arg = DeclareLaunchArgument(
        'pytorch_device',
        default_value='cuda',
        description='PyTorch device (cuda or cpu)')

    pytorch_compile_decoder_arg = DeclareLaunchArgument(
        'pytorch_compile_decoder',
        default_value='True',
        description='Apply torch.compile to decoder (~30s startup, ~3x faster)')

    pytorch_amp_decoder_arg = DeclareLaunchArgument(
        'pytorch_amp_decoder',
        default_value='True',
        description='Use AMP FP16 for decoder (combined with torch.compile)')

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='Confidence threshold for filtering detections')

    input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic',
        default_value='image_raw',
        description='Input image topic name')

    sam3_node = Node(
        package='isaac_ros_segment_anything3',
        executable='sam3_node.py',
        name='sam3_node',
        parameters=[{
            'pytorch_checkpoint':
                LaunchConfiguration('pytorch_checkpoint'),
            'pytorch_device':
                LaunchConfiguration('pytorch_device'),
            'pytorch_compile_decoder':
                LaunchConfiguration('pytorch_compile_decoder'),
            'pytorch_amp_decoder':
                LaunchConfiguration('pytorch_amp_decoder'),
            'confidence_threshold':
                LaunchConfiguration('confidence_threshold'),
        }],
        remappings=[
            ('image_raw', LaunchConfiguration('input_image_topic')),
        ],
        output='screen',
    )

    return LaunchDescription([
        pytorch_checkpoint_arg,
        pytorch_device_arg,
        pytorch_compile_decoder_arg,
        pytorch_amp_decoder_arg,
        confidence_threshold_arg,
        input_image_topic_arg,
        sam3_node,
    ])
