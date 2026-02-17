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
Overlay visualization node for SAM3/EfficientSAM3 demos.

Subscribes to the original image and segmentation mask, blends them
with per-prompt colors, and publishes the colored overlay.

Subscribed Topics:
    image_raw (sensor_msgs/Image): Original image (rgb8).
    sam3/raw_segmentation_mask (sensor_msgs/Image): Segmentation mask (mono8).

Published Topics:
    sam3/overlay (sensor_msgs/Image): Colored overlay (rgb8).

Parameters:
    alpha (float): Mask overlay opacity. Default: 0.45
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Color palette for up to 3 prompts (RGB)
_COLORS = np.array([
    [255, 50, 50],     # Red
    [50, 200, 50],     # Green
    [80, 80, 255],     # Blue
], dtype=np.uint8)


class OverlayNode(Node):

    def __init__(self):
        super().__init__('overlay_node')

        self.declare_parameter('alpha', 0.45)
        self._alpha = self.get_parameter(
            'alpha').get_parameter_value().double_value

        self._bridge = CvBridge()

        # Publisher
        self._pub = self.create_publisher(Image, 'sam3/overlay', 10)

        # Synchronized subscribers
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        image_sub = message_filters.Subscriber(
            self, Image, 'image_raw', qos_profile=qos)
        mask_sub = message_filters.Subscriber(
            self, Image, 'sam3/raw_segmentation_mask', qos_profile=qos)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, mask_sub], queue_size=5, slop=0.5)
        self._sync.registerCallback(self._callback)

        self.get_logger().info(
            f'Overlay node ready (alpha={self._alpha:.2f})')

    def _callback(self, image_msg, mask_msg):
        image = self._bridge.imgmsg_to_cv2(image_msg, 'rgb8')
        mask = self._bridge.imgmsg_to_cv2(mask_msg, 'mono8')

        # Resize mask to image size if needed
        if mask.shape[:2] != image.shape[:2]:
            import cv2
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        # Find unique nonzero labels (each prompt has a distinct value)
        labels = np.unique(mask)
        labels = labels[labels > 0]

        if len(labels) == 0:
            # No detections: publish original image
            self._pub.publish(image_msg)
            return

        # Build colored overlay
        overlay = image.copy().astype(np.float32)
        alpha = self._alpha

        for i, label in enumerate(labels):
            color = _COLORS[i % len(_COLORS)].astype(np.float32)
            region = mask == label
            overlay[region] = (
                alpha * color + (1.0 - alpha) * overlay[region])

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        out_msg = self._bridge.cv2_to_imgmsg(overlay, 'rgb8')
        out_msg.header = image_msg.header
        self._pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OverlayNode()
    try:
        rclpy.spin(node)
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
