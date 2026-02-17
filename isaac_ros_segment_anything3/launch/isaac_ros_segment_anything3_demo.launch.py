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
Unified Foxglove demo launch for EfficientSAM3 text-prompted segmentation.

Supports both MP4 video files and rosbag files as input.
Connect Foxglove Studio to ws://localhost:<foxglove_port> to visualize.

Usage:
    # Video input (default)
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3_demo.launch.py \
        input_type:=video input_path:=/path/to/video.mp4

    # Rosbag input
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3_demo.launch.py \
        input_type:=bag input_path:=/path/to/bag_folder
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # --- Launch arguments ---
    input_type_arg = DeclareLaunchArgument(
        'input_type', default_value='video',
        description="Input type: 'video' (mp4) or 'bag' (rosbag)")

    input_path_arg = DeclareLaunchArgument(
        'input_path',
        description='Path to the input video file (mp4) or rosbag folder')

    input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic', default_value='image_raw',
        description='Input image topic for SAM3/overlay (for bag use this '
                    'to match the bag image topic)')

    fps_arg = DeclareLaunchArgument(
        'fps', default_value='30.0',
        description='Video publishing rate in Hz (only for input_type:=video)')

    loop_arg = DeclareLaunchArgument(
        'loop', default_value='True',
        description='Loop video or bag file')

    text_prompt_arg = DeclareLaunchArgument(
        'text_prompt', default_value='sky',
        description='Initial text prompt (comma-separated for multiple)')

    model_type_arg = DeclareLaunchArgument(
        'model_type', default_value='efficient_sam3',
        description='Model type: sam3 or efficient_sam3')

    inference_backend_arg = DeclareLaunchArgument(
        'inference_backend', default_value='pytorch',
        description='Inference backend: triton or pytorch')

    model_repository_path_arg = DeclareLaunchArgument(
        'model_repository_path', default_value='/tmp/models',
        description='Path to model repository')

    tokenizer_path_arg = DeclareLaunchArgument(
        'tokenizer_path', default_value='/tmp/models/tokenizer.json',
        description='Path to tokenizer.json')

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.3',
        description='Detection confidence threshold')

    foxglove_port_arg = DeclareLaunchArgument(
        'foxglove_port', default_value='8765',
        description='Foxglove bridge WebSocket port')

    overlay_alpha_arg = DeclareLaunchArgument(
        'overlay_alpha', default_value='0.45',
        description='Overlay mask opacity (0.0-1.0)')

    # --- Derived configurations ---
    is_video = PythonExpression(
        ["'", LaunchConfiguration('input_type'), "' == 'video'"])
    is_bag_loop = PythonExpression(
        [
            "'", LaunchConfiguration('input_type'), "' == 'bag' and '",
            LaunchConfiguration('loop'), "' == 'True'",
        ])
    is_bag_no_loop = PythonExpression(
        [
            "'", LaunchConfiguration('input_type'), "' == 'bag' and '",
            LaunchConfiguration('loop'), "' != 'True'",
        ])

    # --- Input publishers ---
    video_publisher = Node(
        condition=IfCondition(is_video),
        package='isaac_ros_segment_anything3',
        executable='video_publisher.py',
        name='video_publisher',
        parameters=[{
            'video_path': LaunchConfiguration('input_path'),
            'fps': LaunchConfiguration('fps'),
            'loop': LaunchConfiguration('loop'),
        }],
        remappings=[
            ('image_raw', LaunchConfiguration('input_image_topic')),
        ],
        output='screen',
    )

    bag_player_loop = ExecuteProcess(
        condition=IfCondition(is_bag_loop),
        cmd=[
            'ros2', 'bag', 'play',
            LaunchConfiguration('input_path'),
            '--loop',
        ],
        output='screen',
    )

    bag_player_once = ExecuteProcess(
        condition=IfCondition(is_bag_no_loop),
        cmd=[
            'ros2', 'bag', 'play',
            LaunchConfiguration('input_path'),
        ],
        output='screen',
    )

    # --- Segmentation pipeline ---
    sam3_node = Node(
        package='isaac_ros_segment_anything3',
        executable='sam3_node.py',
        name='sam3_node',
        parameters=[{
            'model_type': LaunchConfiguration('model_type'),
            'inference_backend': LaunchConfiguration('inference_backend'),
            'model_repository_path':
                LaunchConfiguration('model_repository_path'),
            'tokenizer_path': LaunchConfiguration('tokenizer_path'),
            'confidence_threshold':
                LaunchConfiguration('confidence_threshold'),
        }],
        remappings=[
            ('image_raw', LaunchConfiguration('input_image_topic')),
        ],
        output='screen',
    )

    overlay_node = Node(
        package='isaac_ros_segment_anything3',
        executable='overlay_node.py',
        name='overlay_node',
        parameters=[{
            'alpha': LaunchConfiguration('overlay_alpha'),
        }],
        remappings=[
            ('image_raw', LaunchConfiguration('input_image_topic')),
        ],
        output='screen',
    )

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': LaunchConfiguration('foxglove_port'),
            'send_buffer_limit': 100000000,
        }],
        output='screen',
    )

    # Publish initial text prompt after delay to allow model loading.
    initial_prompt = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg=[
                'Publishing initial prompt: ',
                LaunchConfiguration('text_prompt'),
            ]),
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '--once',
                    '/sam3/text_prompt', 'std_msgs/msg/String',
                    ['{data: "', LaunchConfiguration('text_prompt'), '"}'],
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        input_type_arg,
        input_path_arg,
        input_image_topic_arg,
        fps_arg,
        loop_arg,
        text_prompt_arg,
        model_type_arg,
        inference_backend_arg,
        model_repository_path_arg,
        tokenizer_path_arg,
        confidence_threshold_arg,
        foxglove_port_arg,
        overlay_alpha_arg,
        video_publisher,
        bag_player_loop,
        bag_player_once,
        sam3_node,
        overlay_node,
        foxglove_bridge,
        initial_prompt,
    ])
