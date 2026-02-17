#!/bin/bash
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Automated test script for EfficientSAM3 pipeline alignment.
# Run inside the sam3_pytorch:latest Docker container.
#
# Usage:
#   docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
#     -v /media/sewon/Dev/isaac_ros_image_segmentation:/ws \
#     -v /tmp:/tmp sam3_pytorch:latest \
#     bash /ws/isaac_ros_segment_anything3/scripts/test_esam3_docker.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="/ws"
CHECKPOINT="/tmp/models/efficient_sam3.pth"
TOKENIZER="/tmp/models/tokenizer.json"
RESULTS_DIR="/ws/isaac_ros_segment_anything3/test_results"

echo "============================================"
echo "EfficientSAM3 Pipeline Alignment Test"
echo "============================================"
echo ""

# --- Step 0: Setup ---
mkdir -p "$RESULTS_DIR"

# Download a real COCO test image if none exists
TEST_IMAGE="$RESULTS_DIR/test_coco.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    echo "Downloading COCO test image..."
    python3 -c "
import urllib.request
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # cats on couch
dst = '$TEST_IMAGE'
try:
    urllib.request.urlretrieve(url, dst)
    import cv2
    img = cv2.imread(dst)
    print(f'Downloaded: {img.shape[1]}x{img.shape[0]}')
except Exception as e:
    print(f'Download failed: {e}')
    # Fallback: generate gradient image
    import numpy as np, cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)
    img[:, :, 1] = np.linspace(0, 200, 480, dtype=np.uint8)[:, np.newaxis]
    img[:240, :, 2] = 200
    cv2.imwrite(dst, img)
    print(f'Fallback: created synthetic test image')
"
fi

# Check for rosbag test frame
ROSBAG_FRAME="$RESULTS_DIR/robotarm_frame.jpg"
if [ ! -f "$ROSBAG_FRAME" ]; then
    echo "Extracting frame from r2b_robotarm dataset..."
    python3 -c "
import sys
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import cv2

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri='/ws/datasets/r2bdataset2024_v1/r2b_robotarm/r2b_robotarm_0.mcap',
        storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if 'image_raw' in topic or 'color' in topic:
            msg = deserialize_message(data, Image)
            cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite('$ROSBAG_FRAME', cv_img)
            print(f'Saved frame: {cv_img.shape[1]}x{cv_img.shape[0]}')
            break
except Exception as e:
    print(f'Could not extract rosbag frame: {e}')
    print('Skipping rosbag frame extraction.')
" 2>/dev/null || echo "No rosbag available, skipping."
fi

# --- Step 1: Check model ---
echo ""
echo "--- Step 1: Check model and tokenizer ---"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found at $CHECKPOINT"
    echo "Run: python3 $SCRIPT_DIR/download_models.py --model-type efficient_sam3 --inference-backend pytorch --pytorch-checkpoint <path>"
    exit 1
fi
echo "Checkpoint: $CHECKPOINT ($(du -h "$CHECKPOINT" | cut -f1))"

if [ ! -f "$TOKENIZER" ]; then
    echo "Tokenizer not found at $TOKENIZER"
    exit 1
fi
echo "Tokenizer: $TOKENIZER"

# --- Step 2: Upstream ground-truth test ---
echo ""
echo "--- Step 2: Upstream Sam3Processor ground truth ---"

if [ -f "$ROSBAG_FRAME" ]; then
    echo "Testing on robot arm frame..."
    python3 "$SCRIPT_DIR/test_upstream_esam3.py" \
        --checkpoint "$CHECKPOINT" \
        --image "$ROSBAG_FRAME" \
        --prompt "robot arm" \
        --prompt "person" \
        --output "$RESULTS_DIR/upstream_robotarm.jpg" \
        2>&1 | tee "$RESULTS_DIR/upstream_robotarm.log"
fi

if [ -f "$TEST_IMAGE" ]; then
    echo ""
    echo "Testing on COCO image..."
    python3 "$SCRIPT_DIR/test_upstream_esam3.py" \
        --checkpoint "$CHECKPOINT" \
        --image "$TEST_IMAGE" \
        --prompt "cat" \
        --prompt "couch" \
        --output "$RESULTS_DIR/upstream_coco.jpg" \
        2>&1 | tee "$RESULTS_DIR/upstream_coco.log"
fi

# --- Step 3: Pipeline comparison ---
echo ""
echo "--- Step 3: Pipeline comparison (upstream vs wrapper) ---"

COMPARE_IMAGE="$ROSBAG_FRAME"
COMPARE_PROMPT="robot arm"
if [ ! -f "$COMPARE_IMAGE" ]; then
    COMPARE_IMAGE="$TEST_IMAGE"
    COMPARE_PROMPT="cat"
fi

python3 "$SCRIPT_DIR/test_pipeline_comparison.py" \
    --checkpoint "$CHECKPOINT" \
    --image "$COMPARE_IMAGE" \
    --prompt "$COMPARE_PROMPT" \
    2>&1 | tee "$RESULTS_DIR/comparison.log"

echo ""
echo "============================================"
echo "Results saved to: $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR/"
