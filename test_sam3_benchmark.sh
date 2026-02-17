#!/bin/bash
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

echo "=========================================="
echo "SAM3 Benchmark Test Suite"
echo "=========================================="
echo ""

# Detect workspace directory (works both in Docker and on host)
if [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
elif [ -d "/media/sewon/Dev/isaac_ros_image_segmentation" ]; then
    WORKSPACE_DIR="/media/sewon/Dev/isaac_ros_image_segmentation"
else
    # Use current directory as fallback
    WORKSPACE_DIR="$(pwd)"
fi

cd "$WORKSPACE_DIR" || { echo "Failed to change to workspace directory"; exit 1; }

# Source ROS2 early (needed for rosbag generation and building)
if [ -z "$ROS_DISTRO" ]; then
    echo "Sourcing ROS2 Jazzy..."
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    else
        echo "✗ ROS2 Jazzy not found. Are you running inside Docker?"
        exit 1
    fi
fi

# Step 1: Generate test rosbag dataset
echo "[1/5] Preparing test dataset..."
DATASET_DIR="$WORKSPACE_DIR/datasets/sam3_benchmark_test"

if [ -d "$DATASET_DIR" ]; then
    echo "✓ Test dataset already exists at $DATASET_DIR"
else
    echo "  Generating test rosbag (720p, 300 frames, /image_raw topic)..."
    python3 -c "
import numpy as np
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from sensor_msgs.msg import Image

writer = SequentialWriter()
writer.open(
    StorageOptions(uri='$DATASET_DIR', storage_id='sqlite3'),
    ConverterOptions('cdr', 'cdr')
)
writer.create_topic(TopicMetadata(
    id=0, name='/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr'))

for i in range(300):
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    msg = Image()
    msg.header.stamp.sec = i // 30
    msg.header.stamp.nanosec = (i % 30) * 33333333
    msg.header.frame_id = 'camera'
    msg.height, msg.width = 720, 1280
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = 1280 * 3
    msg.data = img.tobytes()
    ts = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
    writer.write('/image_raw', serialize_message(msg), ts)
    if i % 60 == 0: print(f'    Frame {i}/300')

# Explicitly close writer to ensure metadata.yaml is written correctly
del writer
print('  ✓ Test rosbag created')
" || { echo "✗ Failed to generate test dataset"; exit 1; }
fi
echo ""

# Install jq for results formatting (if not available)
if ! command -v jq &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq jq >/dev/null 2>&1 || true
fi

# Step 2: Build packages
echo "[2/5] Building ROS2 packages..."

echo "Building isaac_ros_segment_anything3_interfaces..."
colcon build --packages-select isaac_ros_segment_anything3_interfaces \
    --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "(^Starting|^Finished|error|ERROR)" || true

echo "Building isaac_ros_segment_anything3..."
colcon build --packages-select isaac_ros_segment_anything3 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "(^Starting|^Finished|error|ERROR)" || true

echo "Building isaac_ros_segment_anything3_benchmark..."
colcon build --packages-select isaac_ros_segment_anything3_benchmark \
    --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "(^Starting|^Finished|error|ERROR)" || true

# Source workspace
source install/setup.bash
echo "✓ Packages built and sourced successfully"
echo ""

# Step 3: Verify Sam3Timing message
echo "[3/5] Verifying Sam3Timing message interface..."
if ros2 interface show isaac_ros_segment_anything3_interfaces/msg/Sam3Timing >/dev/null 2>&1; then
    echo "✓ Sam3Timing message available"
    ros2 interface show isaac_ros_segment_anything3_interfaces/msg/Sam3Timing 2>/dev/null || true
else
    echo "✗ Sam3Timing message not found. Build may have failed."
    exit 1
fi
echo ""

# Step 4: Check models
echo "[4/5] Checking model availability..."
MODEL_REPO="/tmp/esam3_models"

if [ -d "$MODEL_REPO" ]; then
    echo "✓ Model repository found at $MODEL_REPO"
    ls -lh "$MODEL_REPO"/*.onnx 2>/dev/null || echo "  (ONNX models for Triton backend)"
else
    echo "⚠ Model repository not found at $MODEL_REPO"
    echo "  For PyTorch backend, models will be downloaded automatically."
    echo "  For Triton backend, run download_models.py first."
fi
echo ""

# Step 5: Run benchmark
echo "[5/5] Running SAM3 benchmark..."
echo "=========================================="
echo ""

RESULTS_DIR="$WORKSPACE_DIR/isaac_ros_segment_anything3_benchmark/results"
mkdir -p "$RESULTS_DIR"

BACKEND="pytorch"
MODEL_TYPE="efficient_sam3"
ROSBAG_REMAP=""

# Prefer real r2b_robotarm data if available, otherwise use dummy data
# Check both NGC download path and symlink path
R2B_DIR=""
for candidate in \
    "$WORKSPACE_DIR/datasets/r2bdataset2024_v1/r2b_robotarm" \
    "$WORKSPACE_DIR/datasets/r2b_dataset/r2b_robotarm"; do
    if [ -f "$candidate/r2b_robotarm_0.mcap" ]; then
        R2B_DIR="$candidate"
        break
    fi
done

if [ -n "$R2B_DIR" ] && ros2 bag info "$R2B_DIR" >/dev/null 2>&1; then
    BENCHMARK_DATASET="$R2B_DIR"
    DATASET_LABEL="r2b_robotarm (real)"
    TEXT_PROMPT="robot arm"
    OUTPUT_FILE="$RESULTS_DIR/esam3_pytorch_r2b_robotarm.json"
    # r2b uses /camera_1/color/image_raw, SAM3 subscribes to /image_raw
    ROSBAG_REMAP="--remap /camera_1/color/image_raw:=/image_raw"
else
    BENCHMARK_DATASET="$DATASET_DIR"
    DATASET_LABEL="dummy 720p (synthetic)"
    TEXT_PROMPT="person"
    OUTPUT_FILE="$RESULTS_DIR/esam3_pytorch_dummy_test.json"
fi

echo "Configuration:"
echo "  Backend: $BACKEND"
echo "  Model: $MODEL_TYPE"
echo "  Prompt: $TEXT_PROMPT"
echo "  Dataset: $DATASET_LABEL"
echo "  Output: $OUTPUT_FILE"
echo "  Duration: 30 seconds"
echo ""

# Set up model repository
MODEL_REPO="/tmp/models"
mkdir -p "$MODEL_REPO"

# Download tokenizer if missing
TOKENIZER_PATH="$MODEL_REPO/tokenizer.json"
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "Downloading tokenizer.json..."
    wget -q -O "$TOKENIZER_PATH" \
        'https://github.com/jamjamjon/assets/releases/download/sam3/tokenizer.json' \
        || { echo "✗ Failed to download tokenizer"; exit 1; }
    echo "✓ Tokenizer downloaded"
else
    echo "✓ Tokenizer exists at $TOKENIZER_PATH"
fi

# For PyTorch backend: check/download checkpoint
if [ "$BACKEND" = "pytorch" ]; then
    CHECKPOINT_PATH="$MODEL_REPO/efficient_sam3.pth"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Downloading EfficientSAM3 checkpoint from HuggingFace..."
        pip install -q huggingface_hub 2>/dev/null || true
        python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='Simon7108528/EfficientSAM3',
    filename='stage1_all_converted/efficient_sam3_tinyvit_11m_mobileclip_s1.pth',
    local_dir='/tmp/hf_esam3',
)
shutil.copy2(path, '$CHECKPOINT_PATH')
print('✓ Checkpoint downloaded')
" || {
            echo "⚠ HuggingFace download failed. Checking local paths..."
            for p in /tmp/esam3_repo/checkpoints/*.pth /tmp/esam3_checkpoints/*.pth; do
                if [ -f "\$p" ]; then
                    cp "\$p" "$CHECKPOINT_PATH"
                    echo "✓ Found checkpoint at \$p"
                    break
                fi
            done
        }
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "✗ Cannot find EfficientSAM3 checkpoint."
            echo "  Download manually: https://huggingface.co/Simon7108528/EfficientSAM3"
            echo "  Place at: $CHECKPOINT_PATH"
            exit 1
        fi
    else
        echo "✓ Checkpoint exists at $CHECKPOINT_PATH"
    fi
fi

echo ""
echo "Starting benchmark (this will take ~40 seconds)..."
echo "  - 5 sec: Node startup + warmup"
echo "  - 30 sec: Benchmark data collection"
echo "  - 5 sec: Shutdown + JSON write"
echo ""

# Cleanup function to kill background processes
cleanup() {
    echo "Cleaning up background processes..."
    kill $SAM3_PID $MONITOR_PID $ROSBAG_PID 2>/dev/null || true
    wait $SAM3_PID $MONITOR_PID $ROSBAG_PID 2>/dev/null || true
}
trap cleanup EXIT

# 1. Start SAM3 node in background
echo "  Starting SAM3 node ($MODEL_TYPE, $BACKEND)..."
ros2 run isaac_ros_segment_anything3 sam3_node.py \
    --ros-args \
    -p model_type:=$MODEL_TYPE \
    -p inference_backend:=$BACKEND \
    -p model_repository_path:=$MODEL_REPO \
    -p tokenizer_path:=$TOKENIZER_PATH \
    -p image_size:=1008 \
    -p confidence_threshold:=0.3 \
    -p pytorch_device:=cuda \
    &
SAM3_PID=$!

# 2. Start monitor node in background
echo "  Starting monitor node (duration=30s)..."
ros2 run isaac_ros_segment_anything3_benchmark sam3_monitor_node.py \
    --ros-args \
    -p test_duration_sec:=30.0 \
    -p output_path:="$OUTPUT_FILE" \
    -p warmup_frames:=10 \
    &
MONITOR_PID=$!

# 3. Wait for SAM3 node to initialize (model loading can take 10-30s on first run)
echo "  Waiting for SAM3 node to initialize..."
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ros2 service list 2>/dev/null | grep -q '/sam3/set_text_prompt'; then
        echo "  ✓ SAM3 node ready (${WAITED}s)"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "    Still waiting... (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  ✗ SAM3 node did not start within ${MAX_WAIT}s"
    echo "    Check logs above for errors"
    exit 1
fi

# 4. Set text prompt
echo "  Setting text prompt: '$TEXT_PROMPT'..."
ros2 service call /sam3/set_text_prompt \
    isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
    "{text_prompts: ['$TEXT_PROMPT']}" || {
    echo "  ⚠ Failed to set prompt. Retrying in 3s..."
    sleep 3
    ros2 service call /sam3/set_text_prompt \
        isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
        "{text_prompts: ['$TEXT_PROMPT']}" || true
}

# 5. Start rosbag playback
echo "  Starting rosbag playback ($DATASET_LABEL)..."
ros2 bag play "$BENCHMARK_DATASET" --loop $ROSBAG_REMAP &
ROSBAG_PID=$!

# 6. Wait for monitor to complete (it shuts down after test_duration)
echo ""
echo "  Benchmark running... (30 seconds)"
echo "  Waiting for monitor node to complete..."
wait $MONITOR_PID 2>/dev/null || true

# 7. Stop remaining processes
echo "  Stopping nodes..."
kill $SAM3_PID $ROSBAG_PID 2>/dev/null || true
wait $SAM3_PID $ROSBAG_PID 2>/dev/null || true
trap - EXIT

# Step 6: Verify results
echo ""
echo "=========================================="
echo "Benchmark Results"
echo "=========================================="
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Results file created: $OUTPUT_FILE"
    echo ""

    # Display summary
    if command -v jq &> /dev/null; then
        echo "Performance Summary:"
        echo "-------------------"
        jq -r '
            "Throughput: \(.performance_metrics.throughput_fps | tonumber | . * 100 | round / 100) fps",
            "Frames Collected: \(.performance_metrics.num_frames)",
            "",
            "Latency (E2E):",
            "  Mean: \(.performance_metrics.latency_ms.mean | tonumber | . * 10 | round / 10) ms",
            "  Std Dev: \(.performance_metrics.latency_ms.std_dev | tonumber | . * 10 | round / 10) ms",
            "  Min: \(.performance_metrics.latency_ms.min | tonumber | . * 10 | round / 10) ms",
            "  Max: \(.performance_metrics.latency_ms.max | tonumber | . * 10 | round / 10) ms",
            "  P95: \(.performance_metrics.latency_ms.p95 | tonumber | . * 10 | round / 10) ms",
            "",
            "Stage Breakdown:",
            "  CV Bridge: \(.performance_metrics.stage_breakdown_ms.cvbridge.mean | tonumber | . * 10 | round / 10) ms",
            "  Preprocess: \(.performance_metrics.stage_breakdown_ms.preprocess.mean | tonumber | . * 10 | round / 10) ms",
            "  Vision Encoder: \(.performance_metrics.stage_breakdown_ms.vision_encoder.mean | tonumber | . * 10 | round / 10) ms",
            "  Text Encoder: \(.performance_metrics.stage_breakdown_ms.text_encoder.mean | tonumber | . * 10 | round / 10) ms (cache hit: \(.performance_metrics.stage_breakdown_ms.text_encoder.cache_hit_rate | tonumber | . * 1000 | round / 10)%)",
            "  Decoder: \(.performance_metrics.stage_breakdown_ms.decoder.mean | tonumber | . * 10 | round / 10) ms",
            "  Postprocess: \(.performance_metrics.stage_breakdown_ms.postprocess.mean | tonumber | . * 10 | round / 10) ms"
        ' "$OUTPUT_FILE"

        echo ""
        echo "Expected vs Actual (PyTorch backend):"
        echo "-------------------------------------"

        ACTUAL_E2E=$(jq -r '.performance_metrics.latency_ms.mean' "$OUTPUT_FILE" | awk '{printf "%.1f", $1}')
        ACTUAL_VISION=$(jq -r '.performance_metrics.stage_breakdown_ms.vision_encoder.mean' "$OUTPUT_FILE" | awk '{printf "%.1f", $1}')
        ACTUAL_DECODER=$(jq -r '.performance_metrics.stage_breakdown_ms.decoder.mean' "$OUTPUT_FILE" | awk '{printf "%.1f", $1}')
        ACTUAL_CACHE=$(jq -r '.performance_metrics.stage_breakdown_ms.text_encoder.cache_hit_rate' "$OUTPUT_FILE" | awk '{printf "%.1f", $1 * 100}')

        echo "  E2E Latency: ${ACTUAL_E2E}ms (expected: ~131ms)"
        echo "  Vision Encoder: ${ACTUAL_VISION}ms (expected: ~20ms)"
        echo "  Decoder: ${ACTUAL_DECODER}ms (expected: ~70ms)"
        echo "  Text Cache Hit: ${ACTUAL_CACHE}% (expected: >95%)"

    else
        echo "Install jq for formatted output: apt-get install -y jq"
        echo ""
        echo "Raw JSON:"
        cat "$OUTPUT_FILE"
    fi

    echo ""
    echo "✓ Benchmark test completed successfully!"
    echo ""
    echo "Full results saved to: $OUTPUT_FILE"

else
    echo "✗ Results file not found: $OUTPUT_FILE"
    echo ""
    echo "Possible issues:"
    echo "  1. Monitor node didn't receive timing messages"
    echo "  2. SAM3 node failed to start"
    echo "  3. Rosbag playback failed"
    echo ""
    echo "Try running manually:"
    echo "  ros2 topic echo /sam3/timing"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test suite complete!"
echo "=========================================="
