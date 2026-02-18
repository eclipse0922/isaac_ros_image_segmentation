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
Video/image file publisher for ROS2 demos.

Reads an MP4 (or any OpenCV-supported video) file or a single image
(JPEG, PNG, BMP, etc.) and publishes frames as sensor_msgs/Image (rgb8)
at a configurable FPS. Single images are looped indefinitely when
loop=True.

Parameters:
    video_path (str): Path to the video or image file. Required.
    fps (float): Publishing rate in Hz. Default: 5.0
    loop (bool): Loop the video when it reaches the end. Default: True

Published Topics:
    image_raw (sensor_msgs/Image): Video frames as rgb8 images.
"""

import os

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class VideoPublisher(Node):

    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('video_path', '')
        self.declare_parameter('fps', 5.0)
        self.declare_parameter('loop', True)

        video_path = self.get_parameter(
            'video_path').get_parameter_value().string_value
        fps = self.get_parameter(
            'fps').get_parameter_value().double_value
        self._loop = self.get_parameter(
            'loop').get_parameter_value().bool_value

        if not video_path:
            self.get_logger().error(
                'video_path parameter is required. '
                'Set it via: video_path:=/path/to/video.mp4')
            return

        # Check if input is a static image (OpenCV VideoCapture doesn't
        # support single image files reliably on all platforms).
        ext = os.path.splitext(video_path)[1].lower()
        self._static_frame = None
        self._cap = None

        if ext in _IMAGE_EXTS:
            frame = cv2.imread(video_path)
            if frame is None:
                self.get_logger().error(f'Cannot open image: {video_path}')
                return
            self._static_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            self.get_logger().info(
                f'Image: {video_path} ({width}x{height}), '
                f'publishing at {fps:.1f} Hz, loop={self._loop}')
        else:
            self._cap = cv2.VideoCapture(video_path)
            if not self._cap.isOpened():
                self.get_logger().error(f'Cannot open video: {video_path}')
                return

            native_fps = self._cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0:
                fps = native_fps if native_fps > 0 else 30.0

            self.get_logger().info(
                f'Video: {video_path} ({width}x{height}, '
                f'{native_fps:.1f}fps, {total_frames} frames)')
            self.get_logger().info(
                f'Publishing at {fps:.1f} Hz, loop={self._loop}')

        self._bridge = CvBridge()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Image, 'image_raw', qos)

        period = 1.0 / fps
        self._timer = self.create_timer(period, self._timer_callback)
        self._frame_count = 0

    def _timer_callback(self):
        if self._static_frame is not None:
            # Static image: publish the same frame repeatedly
            msg = self._bridge.cv2_to_imgmsg(self._static_frame, 'rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            self._pub.publish(msg)
            self._frame_count += 1
            return

        if self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if ret:
                    self._frame_count = 0
                    self.get_logger().info('Video looped')
            if not ret:
                self.get_logger().info('Video ended')
                self._timer.cancel()
                return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        msg = self._bridge.cv2_to_imgmsg(rgb, 'rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self._pub.publish(msg)
        self._frame_count += 1

    def destroy_node(self):
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
