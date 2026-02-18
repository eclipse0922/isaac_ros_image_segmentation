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
Foxglove demo launch for SAM3 text-prompted segmentation.

Supports MP4 video files and rosbag files as input.
Connect Foxglove Studio to ws://localhost:<foxglove_port> to visualize.

Pipeline:
  video/bag -> sam3_node (PyTorch + TRT) -> overlay_node -> foxglove_bridge

Usage:
    # Video input
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3_demo.launch.py \\
        input_type:=video input_path:=/path/to/video.mp4 \\
        pytorch_checkpoint:=/ws/models/sam3/sam3.pt

    # Rosbag input
    ros2 launch isaac_ros_segment_anything3 isaac_ros_segment_anything3_demo.launch.py \\
        input_type:=bag input_path:=/path/to/bag_folder \\
        input_image_topic:=camera/color/image_raw \\
        pytorch_checkpoint:=/ws/models/sam3/sam3.pt

    # Change text prompt after launch (~90s for model to load + torch.compile):
    ros2 service call /sam3/set_text_prompt \\
        isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \\
        "{text_prompts: ['robot arm']}"
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
    # --- Launch arguments: input source ---
    input_type_arg = DeclareLaunchArgument(
        'input_type', default_value='video',
        description="Input type: 'video' (mp4) or 'bag' (rosbag)")

    input_path_arg = DeclareLaunchArgument(
        'input_path',
        description='Path to input video file (mp4) or rosbag folder')

    input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic', default_value='image_raw',
        description='Image topic name (match rosbag topic for bag input)')

    fps_arg = DeclareLaunchArgument(
        'fps', default_value='30.0',
        description='Video publishing rate in Hz (video input only)')

    loop_arg = DeclareLaunchArgument(
        'loop', default_value='True',
        description='Loop video or bag file')

    # --- Launch arguments: model ---
    pytorch_checkpoint_arg = DeclareLaunchArgument(
        'pytorch_checkpoint', default_value='/tmp/models/sam3.pt',
        description='Path to SAM3 checkpoint (sam3.pt, ~3.3GB)')

    pytorch_device_arg = DeclareLaunchArgument(
        'pytorch_device', default_value='cuda',
        description='PyTorch device (cuda or cpu)')

    pytorch_compile_decoder_arg = DeclareLaunchArgument(
        'pytorch_compile_decoder', default_value='True',
        description='Apply torch.compile to decoder (~30s startup, ~3x faster)')

    pytorch_amp_decoder_arg = DeclareLaunchArgument(
        'pytorch_amp_decoder', default_value='True',
        description='Use AMP FP16 for decoder (with torch.compile)')

    # --- Launch arguments: inference ---
    text_prompt_arg = DeclareLaunchArgument(
        'text_prompt', default_value='person',
        description='Initial text prompt (comma-separated for multiple)')

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.3',
        description='Detection confidence threshold')

    # --- Launch arguments: visualization ---
    foxglove_port_arg = DeclareLaunchArgument(
        'foxglove_port', default_value='8765',
        description='Foxglove bridge WebSocket port')

    overlay_alpha_arg = DeclareLaunchArgument(
        'overlay_alpha', default_value='0.45',
        description='Overlay mask opacity (0.0-1.0)')

    # --- Derived conditions ---
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

    # --- SAM3 segmentation node (PyTorch + TRT) ---
    sam3_node = Node(
        package='isaac_ros_segment_anything3',
        executable='sam3_node.py',
        name='sam3_node',
        parameters=[{
            'pytorch_checkpoint': LaunchConfiguration('pytorch_checkpoint'),
            'pytorch_device': LaunchConfiguration('pytorch_device'),
            'pytorch_compile_decoder':
                LaunchConfiguration('pytorch_compile_decoder'),
            'pytorch_amp_decoder': LaunchConfiguration('pytorch_amp_decoder'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
        }],
        remappings=[
            ('image_raw', LaunchConfiguration('input_image_topic')),
        ],
        output='screen',
    )

    # --- Overlay visualization node ---
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

    # --- Foxglove WebSocket bridge ---
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

    # Set text prompt via service after model finishes loading.
    # SAM3 with torch.compile takes ~40s (model load ~10s + compile ~30s).
    # 90s delay is a safe upper bound; if the service isn't ready yet,
    # run manually: ros2 service call /sam3/set_text_prompt ...
    initial_prompt = TimerAction(
        period=90.0,
        actions=[
            LogInfo(msg=[
                'Setting initial text prompt via service: ',
                LaunchConfiguration('text_prompt'),
            ]),
            ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call',
                    '/sam3/set_text_prompt',
                    'isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt',
                    [
                        '{text_prompts: ["',
                        LaunchConfiguration('text_prompt'),
                        '"]}',
                    ],
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
        pytorch_checkpoint_arg,
        pytorch_device_arg,
        pytorch_compile_decoder_arg,
        pytorch_amp_decoder_arg,
        text_prompt_arg,
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
