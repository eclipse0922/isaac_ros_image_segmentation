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
Convert video files to ROS2 rosbag format for benchmarking.

Usage:
  python3 generate_test_rosbag.py \
    --input /path/to/video.mp4 \
    --output datasets/sam3_test_720p \
    --fps 30 \
    --duration 30
"""

import argparse
import hashlib
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class VideoToRosbag:
    """Convert video file to rosbag with configurable FPS and duration."""

    def __init__(self, input_path, output_path, fps=30, duration=30):
        self.input_path = input_path
        self.output_path = output_path
        self.target_fps = fps
        self.duration = duration

    def convert(self):
        """Perform the conversion."""
        print(f'Converting {self.input_path} to rosbag at {self.output_path}')
        print(f'Target: {self.target_fps} fps, {self.duration} seconds')

        # Open video
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            raise ValueError(f'Failed to open video: {self.input_path}')

        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f'Source video: {width}x{height}, {source_fps:.1f} fps, {total_frames} frames')

        # Calculate frame interval for target FPS
        frame_interval = max(1, int(round(source_fps / self.target_fps)))
        target_frame_count = min(
            int(self.target_fps * self.duration),
            total_frames // frame_interval
        )

        print(f'Sampling every {frame_interval} frames to achieve ~{self.target_fps} fps')
        print(f'Will extract {target_frame_count} frames')

        # Setup rosbag writer
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        storage_options = StorageOptions(
            uri=str(self.output_path),
            storage_id='sqlite3'
        )
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        writer = SequentialWriter()
        writer.open(storage_options, converter_options)

        # Create topic
        topic_metadata = TopicMetadata(
            name='/image_raw',
            type='sensor_msgs/msg/Image',
            serialization_format='cdr'
        )
        writer.create_topic(topic_metadata)

        # Write frames
        frame_idx = 0
        written_count = 0
        start_timestamp_ns = 0

        while written_count < target_frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to match target FPS
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create ROS Image message
            msg = Image()
            msg.header = Header()
            msg.header.frame_id = 'camera'

            # Calculate timestamp (nanoseconds)
            timestamp_ns = start_timestamp_ns + int(
                (written_count * 1e9) / self.target_fps)
            msg.header.stamp.sec = int(timestamp_ns // 1e9)
            msg.header.stamp.nanosec = int(timestamp_ns % 1e9)

            msg.height = height
            msg.width = width
            msg.encoding = 'rgb8'
            msg.is_bigendian = 0
            msg.step = width * 3
            msg.data = frame_rgb.tobytes()

            # Write to rosbag
            writer.write(
                '/image_raw',
                serialize_message(msg),
                timestamp_ns
            )

            written_count += 1
            frame_idx += 1

            if written_count % 30 == 0:
                print(f'  Written {written_count}/{target_frame_count} frames...')

        cap.release()
        print(f'Successfully wrote {written_count} frames to {self.output_path}')

        # Calculate rosbag hash for reproducibility
        self._calculate_hash()

    def _calculate_hash(self):
        """Calculate SHA256 hash of rosbag for verification."""
        db_path = Path(self.output_path) / 'rosbag2.db3'
        if not db_path.exists():
            print('Warning: rosbag database file not found for hashing')
            return

        sha256 = hashlib.sha256()
        with open(db_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        hash_hex = sha256.hexdigest()
        hash_file = Path(self.output_path) / 'rosbag.sha256'
        hash_file.write_text(hash_hex)
        print(f'Rosbag hash: sha256:{hash_hex[:16]}...')
        print(f'Hash saved to {hash_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert video to ROS2 rosbag for SAM3 benchmarking')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output', required=True,
                        help='Output rosbag directory path')
    parser.add_argument('--fps', type=float, default=30,
                        help='Target frames per second (default: 30)')
    parser.add_argument('--duration', type=float, default=30,
                        help='Duration in seconds (default: 30)')

    args = parser.parse_args()

    converter = VideoToRosbag(
        input_path=args.input,
        output_path=args.output,
        fps=args.fps,
        duration=args.duration
    )
    converter.convert()


if __name__ == '__main__':
    main()
