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
CLI utility to set SAM3 text prompts via the SetTextPrompt service.

Usage:
    ros2 run isaac_ros_segment_anything3 set_text_prompt.py "person" "car"
    ros2 run isaac_ros_segment_anything3 set_text_prompt.py --threshold 0.5 "dog"
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from isaac_ros_segment_anything3_interfaces.srv import SetTextPrompt


class TextPromptSetter(Node):
    """Node to call the SetTextPrompt service."""

    def __init__(self):
        super().__init__('text_prompt_setter')
        self.client = self.create_client(
            SetTextPrompt, 'sam3/set_text_prompt')

        timeout_count = 0
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                'Waiting for sam3/set_text_prompt service...')
            timeout_count += 1
            if timeout_count > 10:
                self.get_logger().error(
                    'Service not available after 10 seconds. Exiting.')
                return

        self.get_logger().info('Connected to sam3/set_text_prompt service!')

    def set_prompts(self, prompts, threshold=0.0):
        """Call the SetTextPrompt service."""
        request = SetTextPrompt.Request()
        request.request_header = Header()
        request.text_prompts = prompts
        request.confidence_threshold = threshold

        try:
            self.get_logger().info('Calling set_text_prompt service...')
            future = self.client.call_async(request)

            self.get_logger().info(
                'Waiting for service response (5 second timeout)...')
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            if future.done():
                self.get_logger().info('Service call completed')
                try:
                    response = future.result()
                    if response.success:
                        self.get_logger().info(
                            f'Success: {response.message}')
                        self.get_logger().info(
                            f'Active prompts: {response.active_prompts}')
                        return True
                    else:
                        self.get_logger().error(
                            f'Failed: {response.message}')
                        return False
                except Exception as result_error:
                    self.get_logger().error(
                        f'Error getting result: {str(result_error)}')
                    return False
            else:
                self.get_logger().error(
                    'Service call timed out after 5 seconds')
                future.cancel()
                return False

        except Exception as e:
            self.get_logger().error(f'Service call error: {str(e)}')
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Set SAM3 text prompts for segmentation')
    parser.add_argument(
        'prompts', nargs='+',
        help='Text prompts (e.g., "person" "car" "dog")')
    parser.add_argument(
        '--threshold', type=float, default=0.0,
        help='Confidence threshold (0.0 to keep current, default: 0.0)')

    args = parser.parse_args()

    rclpy.init()

    try:
        setter = TextPromptSetter()
        print(f'Setting prompts: {args.prompts}')
        if args.threshold > 0:
            print(f'Confidence threshold: {args.threshold}')

        success = setter.set_prompts(args.prompts, args.threshold)

        if success:
            print('Prompts set successfully!')
            return 0
        else:
            print('Failed to set prompts (check logs for details)')
            return 1

    except KeyboardInterrupt:
        print('\nInterrupted by user')
        return 1
    except Exception as e:
        print(f'Error: {e}')
        return 1
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
