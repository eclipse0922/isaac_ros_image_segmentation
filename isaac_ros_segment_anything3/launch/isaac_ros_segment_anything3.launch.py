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
Standalone launch file for SAM3/EfficientSAM3 text-prompted segmentation.

Usage:
    # SAM3 (default)
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3.launch.py

    # EfficientSAM3
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3.launch.py \
        model_type:=efficient_sam3

    # EfficientSAM3 with PyTorch backend (~16x faster)
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3.launch.py \
        model_type:=efficient_sam3 inference_backend:=pytorch

    # Custom configuration
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3.launch.py \
        triton_server_url:=localhost:8001 \
        model_repository_path:=/tmp/models \
        input_image_topic:=/camera/image_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='sam3',
        description='Model type: sam3 or efficient_sam3')

    triton_server_url_arg = DeclareLaunchArgument(
        'triton_server_url',
        default_value='localhost:8001',
        description='Triton gRPC server URL')

    model_repository_path_arg = DeclareLaunchArgument(
        'model_repository_path',
        default_value='/tmp/models',
        description='Path to Triton model repository')

    vision_encoder_model_name_arg = DeclareLaunchArgument(
        'vision_encoder_model_name',
        default_value='sam3_vision_encoder',
        description='Triton model name for vision encoder')

    text_encoder_model_name_arg = DeclareLaunchArgument(
        'text_encoder_model_name',
        default_value='sam3_text_encoder',
        description='Triton model name for text encoder')

    decoder_model_name_arg = DeclareLaunchArgument(
        'decoder_model_name',
        default_value='sam3_decoder',
        description='Triton model name for decoder')

    tokenizer_path_arg = DeclareLaunchArgument(
        'tokenizer_path',
        default_value='/tmp/models/tokenizer.json',
        description='Path to tokenizer.json file')

    image_size_arg = DeclareLaunchArgument(
        'image_size',
        default_value='1024',
        description='Model input image size')

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='Confidence threshold for filtering detections')

    inference_backend_arg = DeclareLaunchArgument(
        'inference_backend',
        default_value='triton',
        description='Inference backend: triton or pytorch')

    pytorch_checkpoint_arg = DeclareLaunchArgument(
        'pytorch_checkpoint',
        default_value='',
        description='PyTorch checkpoint path (empty=auto)')

    pytorch_device_arg = DeclareLaunchArgument(
        'pytorch_device',
        default_value='cuda',
        description='PyTorch device (cuda or cpu)')

    input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic',
        default_value='image_raw',
        description='Input image topic name')

    # SAM3 node
    sam3_node = Node(
        package='isaac_ros_segment_anything3',
        executable='sam3_node.py',
        name='sam3_node',
        parameters=[{
            'model_type':
                LaunchConfiguration('model_type'),
            'triton_server_url':
                LaunchConfiguration('triton_server_url'),
            'model_repository_path':
                LaunchConfiguration('model_repository_path'),
            'vision_encoder_model_name':
                LaunchConfiguration('vision_encoder_model_name'),
            'text_encoder_model_name':
                LaunchConfiguration('text_encoder_model_name'),
            'decoder_model_name':
                LaunchConfiguration('decoder_model_name'),
            'tokenizer_path':
                LaunchConfiguration('tokenizer_path'),
            'image_size':
                LaunchConfiguration('image_size'),
            'confidence_threshold':
                LaunchConfiguration('confidence_threshold'),
            'inference_backend':
                LaunchConfiguration('inference_backend'),
            'pytorch_checkpoint':
                LaunchConfiguration('pytorch_checkpoint'),
            'pytorch_device':
                LaunchConfiguration('pytorch_device'),
        }],
        remappings=[
            ('image_raw', LaunchConfiguration('input_image_topic')),
        ],
        output='screen',
    )

    return LaunchDescription([
        model_type_arg,
        triton_server_url_arg,
        model_repository_path_arg,
        vision_encoder_model_name_arg,
        text_encoder_model_name_arg,
        decoder_model_name_arg,
        tokenizer_path_arg,
        image_size_arg,
        confidence_threshold_arg,
        inference_backend_arg,
        pytorch_checkpoint_arg,
        pytorch_device_arg,
        input_image_topic_arg,
        sam3_node,
    ])
