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
Isaac ROS SAM3 Benchmark Graph

Performance benchmark for SAM3/EfficientSAM3 text-prompted segmentation.

This benchmark script launches the SAM3 node, monitor node, and plays back
rosbag data to measure throughput and latency.

Usage:
  ros2 launch isaac_ros_segment_anything3_benchmark \
    isaac_ros_segment_anything3_graph.py \
    backend:=pytorch \
    model_type:=efficient_sam3 \
    num_prompts:=1 \
    rosbag_path:=datasets/sam3_test_720p
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for SAM3 benchmark."""

    # Declare launch arguments
    backend_arg = DeclareLaunchArgument(
        'backend',
        default_value='pytorch',
        description='Inference backend: triton or pytorch'
    )

    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='efficient_sam3',
        description='Model type: sam3 or efficient_sam3'
    )

    num_prompts_arg = DeclareLaunchArgument(
        'num_prompts',
        default_value='1',
        description='Number of text prompts (1-3)'
    )

    text_prompts_arg = DeclareLaunchArgument(
        'text_prompts',
        default_value='person',
        description='Comma-separated text prompts'
    )

    rosbag_path_arg = DeclareLaunchArgument(
        'rosbag_path',
        default_value='datasets/sam3_test_720p',
        description='Path to rosbag dataset'
    )

    test_duration_arg = DeclareLaunchArgument(
        'test_duration',
        default_value='30.0',
        description='Test duration in seconds'
    )

    output_path_arg = DeclareLaunchArgument(
        'output_path',
        default_value='results/benchmark.json',
        description='Output JSON path for results'
    )

    triton_url_arg = DeclareLaunchArgument(
        'triton_url',
        default_value='localhost:8001',
        description='Triton server URL (for triton backend)'
    )

    model_repo_arg = DeclareLaunchArgument(
        'model_repository',
        default_value='/tmp/models',
        description='Model repository path'
    )

    # Launch configurations
    backend = LaunchConfiguration('backend')
    model_type = LaunchConfiguration('model_type')
    num_prompts = LaunchConfiguration('num_prompts')
    text_prompts = LaunchConfiguration('text_prompts')
    rosbag_path = LaunchConfiguration('rosbag_path')
    test_duration = LaunchConfiguration('test_duration')
    output_path = LaunchConfiguration('output_path')
    triton_url = LaunchConfiguration('triton_url')
    model_repo = LaunchConfiguration('model_repository')

    # SAM3 Node
    sam3_node = Node(
        package='isaac_ros_segment_anything3',
        executable='sam3_node.py',
        name='sam3_node',
        parameters=[{
            'model_type': model_type,
            'inference_backend': backend,
            'triton_server_url': triton_url,
            'model_repository_path': model_repo,
            'tokenizer_path': [model_repo, '/tokenizer.json'],
            'image_size': 1008,
            'confidence_threshold': 0.3,
            'pytorch_device': 'cuda',
        }],
        remappings=[
            ('image_raw', '/image_raw'),
        ],
        output='screen',
    )

    # Monitor Node
    monitor_node = Node(
        package='isaac_ros_segment_anything3_benchmark',
        executable='sam3_monitor_node.py',
        name='sam3_monitor_node',
        parameters=[{
            'test_duration_sec': test_duration,
            'output_path': output_path,
            'warmup_frames': 10,
        }],
        output='screen',
    )

    # Rosbag Playback
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--loop'],
        output='screen',
    )

    # Set Text Prompt (after brief delay for node startup)
    set_prompt_cmd = ExecuteProcess(
        cmd=[
            'bash', '-c',
            'sleep 3 && ros2 service call /sam3/set_text_prompt '
            'isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt '
            '"{text_prompts: [\'' + str(text_prompts.perform(None)) + '\']}"'
        ],
        output='screen',
    )

    return LaunchDescription([
        # Arguments
        backend_arg,
        model_type_arg,
        num_prompts_arg,
        text_prompts_arg,
        rosbag_path_arg,
        test_duration_arg,
        output_path_arg,
        triton_url_arg,
        model_repo_arg,
        # Nodes
        sam3_node,
        monitor_node,
        rosbag_play,
        set_prompt_cmd,
    ])
