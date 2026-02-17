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
SAM3 Benchmark Monitor Node

Subscribes to /sam3/timing topic, aggregates performance statistics over
a configured test duration, and outputs results to JSON file.

Usage:
  ros2 run isaac_ros_segment_anything3_benchmark sam3_monitor_node.py \
    --ros-args -p test_duration_sec:=30.0 -p output_path:=results/benchmark.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from isaac_ros_segment_anything3_interfaces.msg import Sam3Timing


class Sam3MonitorNode(Node):
    """Monitor node for SAM3 benchmark timing collection."""

    def __init__(self):
        super().__init__('sam3_monitor_node')

        # Parameters
        self.declare_parameter('test_duration_sec', 30.0)
        self.declare_parameter('output_path', 'results/benchmark.json')
        self.declare_parameter('warmup_frames', 10)  # Skip first N frames

        self._test_duration = self.get_parameter(
            'test_duration_sec').get_parameter_value().double_value
        self._output_path = self.get_parameter(
            'output_path').get_parameter_value().string_value
        self._warmup_frames = self.get_parameter(
            'warmup_frames').get_parameter_value().integer_value

        # Data storage
        self._timing_data = []
        self._start_time = None
        self._test_complete = False
        self._frame_count = 0

        # Subscribe to timing topic
        self._timing_sub = self.create_subscription(
            Sam3Timing, 'sam3/timing', self._timing_callback, 10)

        # Timer to check completion
        self._check_timer = self.create_timer(1.0, self._check_completion)

        self.get_logger().info(
            f'SAM3 Monitor initialized: duration={self._test_duration}s, '
            f'output={self._output_path}, warmup={self._warmup_frames} frames')

    def _timing_callback(self, msg):
        """Collect timing data from SAM3 node."""
        if self._test_complete:
            return

        if self._start_time is None:
            self._start_time = self.get_clock().now()
            self.get_logger().info('Benchmark started, collecting timing data...')

        self._frame_count += 1

        # Skip warmup frames
        if self._frame_count <= self._warmup_frames:
            return

        # Store timing data
        self._timing_data.append({
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'cvbridge_ms': msg.cvbridge_ms,
            'preprocess_ms': msg.preprocess_ms,
            'vision_encoder_ms': msg.vision_encoder_ms,
            'text_encoder_ms': msg.text_encoder_ms,
            'text_cache_hit': msg.text_encoder_cache_hit,
            'decoder_ms': msg.decoder_ms,
            'num_prompts': msg.num_prompts,
            'postprocess_ms': msg.postprocess_ms,
            'total_ms': msg.total_ms,
            'backend': msg.backend,
            'model_type': msg.model_type,
        })

    def _check_completion(self):
        """Check if test duration has elapsed."""
        if self._test_complete or self._start_time is None:
            return

        elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
        if elapsed >= self._test_duration:
            self._test_complete = True
            self.get_logger().info(
                f'Benchmark complete. Collected {len(self._timing_data)} frames '
                f'(after {self._warmup_frames} warmup frames).')
            self._compute_and_save_results()
            rclpy.shutdown()

    def _compute_and_save_results(self):
        """Compute statistics and save results to JSON."""
        if not self._timing_data:
            self.get_logger().error('No timing data collected!')
            return

        # Extract metrics
        total_ms = np.array([d['total_ms'] for d in self._timing_data])
        cvbridge_ms = np.array([d['cvbridge_ms'] for d in self._timing_data])
        preprocess_ms = np.array([d['preprocess_ms'] for d in self._timing_data])
        vision_ms = np.array([d['vision_encoder_ms'] for d in self._timing_data])
        text_ms = np.array([d['text_encoder_ms'] for d in self._timing_data])
        decoder_ms = np.array([d['decoder_ms'] for d in self._timing_data])
        postprocess_ms = np.array([d['postprocess_ms'] for d in self._timing_data])

        # Cache hit rate
        cache_hits = sum(1 for d in self._timing_data if d['text_cache_hit'])
        cache_hit_rate = cache_hits / len(self._timing_data)

        # Get configuration from first message
        backend = self._timing_data[0]['backend']
        model_type = self._timing_data[0]['model_type']
        num_prompts = self._timing_data[0]['num_prompts']

        # Compute throughput
        timestamps = np.array([d['timestamp'] for d in self._timing_data])
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            throughput_fps = len(timestamps) / time_span if time_span > 0 else 0.0
        else:
            throughput_fps = 0.0

        # Build results dictionary
        results = {
            'benchmark_metadata': {
                'name': 'Isaac ROS SAM3 Graph Benchmark',
                'test_datetime': datetime.now().isoformat(),
                'test_duration_sec': self._test_duration,
                'warmup_frames': self._warmup_frames,
                'config': {
                    'model_type': model_type,
                    'backend': backend,
                    'num_prompts': num_prompts,
                }
            },
            'performance_metrics': {
                'num_frames': len(self._timing_data),
                'throughput_fps': float(throughput_fps),
                'latency_ms': {
                    'mean': float(np.mean(total_ms)),
                    'std_dev': float(np.std(total_ms)),
                    'min': float(np.min(total_ms)),
                    'max': float(np.max(total_ms)),
                    'p50': float(np.percentile(total_ms, 50)),
                    'p95': float(np.percentile(total_ms, 95)),
                    'p99': float(np.percentile(total_ms, 99)),
                },
                'stage_breakdown_ms': {
                    'cvbridge': {
                        'mean': float(np.mean(cvbridge_ms)),
                        'std_dev': float(np.std(cvbridge_ms)),
                    },
                    'preprocess': {
                        'mean': float(np.mean(preprocess_ms)),
                        'std_dev': float(np.std(preprocess_ms)),
                    },
                    'vision_encoder': {
                        'mean': float(np.mean(vision_ms)),
                        'std_dev': float(np.std(vision_ms)),
                    },
                    'text_encoder': {
                        'mean': float(np.mean(text_ms)),
                        'std_dev': float(np.std(text_ms)),
                        'cache_hit_rate': float(cache_hit_rate),
                    },
                    'decoder': {
                        'mean': float(np.mean(decoder_ms)),
                        'std_dev': float(np.std(decoder_ms)),
                    },
                    'postprocess': {
                        'mean': float(np.mean(postprocess_ms)),
                        'std_dev': float(np.std(postprocess_ms)),
                    },
                },
            },
        }

        # Save to JSON
        output_path = Path(self._output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.get_logger().info(f'Results saved to {output_path}')
        self.get_logger().info(
            f'Performance Summary:\n'
            f'  Throughput: {throughput_fps:.2f} fps\n'
            f'  Latency (mean): {results["performance_metrics"]["latency_ms"]["mean"]:.1f}ms\n'
            f'  Vision Encoder: {results["performance_metrics"]["stage_breakdown_ms"]["vision_encoder"]["mean"]:.1f}ms\n'
            f'  Text Encoder: {results["performance_metrics"]["stage_breakdown_ms"]["text_encoder"]["mean"]:.1f}ms '
            f'(cache hit rate: {cache_hit_rate*100:.1f}%)\n'
            f'  Decoder: {results["performance_metrics"]["stage_breakdown_ms"]["decoder"]["mean"]:.1f}ms\n'
        )


def main(args=None):
    rclpy.init(args=args)
    node = Sam3MonitorNode()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
