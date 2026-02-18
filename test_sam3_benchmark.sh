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
if [ -d "/ws" ]; then
    WORKSPACE_DIR="/ws"
elif [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
elif [ -d "/media/sewon/Dev/isaac_ros_image_segmentation" ]; then
    WORKSPACE_DIR="/media/sewon/Dev/isaac_ros_image_segmentation"
else
    # Use current directory as fallback
    WORKSPACE_DIR="$(pwd)"
fi

cd "$WORKSPACE_DIR" || { echo "Failed to change to workspace directory"; exit 1; }

# Source ROS2 early (needed for rosbag generation, ros2 CLI, and ROS packages)
# Always source /opt/ros/jazzy/setup.bash — the Docker image may set ROS_DISTRO
# via env variables but not add ros2 to PATH (only sourced in interactive shells)
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -z "$ROS_DISTRO" ]; then
    echo "✗ ROS2 Jazzy not found. Are you running inside Docker?"
    exit 1
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

# Step 2: Source packages
# Strategy:
#   - Source pre-built install/ (avoids rosidl rebuild failures in Docker)
#   - Sync latest Python scripts from source to install via direct copy
#     (safe for pure-Python packages; no compilation needed)
echo "[2/5] Setting up ROS2 packages..."

if [ ! -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    echo "✗ Pre-built install/ not found at $WORKSPACE_DIR/install/"
    echo "  Run once inside Docker to build: colcon build --packages-select"
    echo "  isaac_ros_segment_anything3_interfaces isaac_ros_segment_anything3"
    echo "  isaac_ros_segment_anything3_benchmark"
    exit 1
fi

# Sync latest Python scripts (bypasses colcon rebuild; instant update)
SAM3_INSTALL_SCRIPTS="$WORKSPACE_DIR/install/isaac_ros_segment_anything3/lib/isaac_ros_segment_anything3"
SAM3_SOURCE_SCRIPTS="$WORKSPACE_DIR/isaac_ros_segment_anything3/scripts"
if [ -d "$SAM3_INSTALL_SCRIPTS" ] && [ -d "$SAM3_SOURCE_SCRIPTS" ]; then
    cp "$SAM3_SOURCE_SCRIPTS/sam3_node.py" "$SAM3_INSTALL_SCRIPTS/sam3_node.py"
    chmod +x "$SAM3_INSTALL_SCRIPTS/sam3_node.py"
    echo "✓ Synced sam3_node.py to install/"
fi

BENCH_INSTALL_SCRIPTS="$WORKSPACE_DIR/install/isaac_ros_segment_anything3_benchmark/lib/isaac_ros_segment_anything3_benchmark"
BENCH_SOURCE_SCRIPTS="$WORKSPACE_DIR/isaac_ros_segment_anything3_benchmark/scripts"
if [ -d "$BENCH_INSTALL_SCRIPTS" ] && [ -d "$BENCH_SOURCE_SCRIPTS" ]; then
    cp "$BENCH_SOURCE_SCRIPTS/"*.py "$BENCH_INSTALL_SCRIPTS/" 2>/dev/null || true
    chmod +x "$BENCH_INSTALL_SCRIPTS/"*.py 2>/dev/null || true
    echo "✓ Synced benchmark scripts to install/"
fi

# Source pre-built workspace
source "$WORKSPACE_DIR/install/setup.bash"
echo "✓ Packages sourced"
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
RESULTS_DIR="$WORKSPACE_DIR/isaac_ros_segment_anything3_benchmark/results"
mkdir -p "$RESULTS_DIR"

ROSBAG_REMAP=""

# Prefer real r2b_robotarm data if available, otherwise use dummy data
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
    OUTPUT_FILE="$RESULTS_DIR/sam3_pytorch_r2b_robotarm.json"
    # r2b uses /camera_1/color/image_raw, SAM3 subscribes to /image_raw
    ROSBAG_REMAP="--remap /camera_1/color/image_raw:=/image_raw"
else
    BENCHMARK_DATASET="$DATASET_DIR"
    DATASET_LABEL="dummy 720p (synthetic)"
    TEXT_PROMPT="person"
    OUTPUT_FILE="$RESULTS_DIR/sam3_pytorch_dummy_test.json"
fi

# Resolve checkpoint path
MODEL_REPO="/tmp/models"
mkdir -p "$MODEL_REPO" 2>/dev/null || true
CHECKPOINT_PATH="$MODEL_REPO/sam3.pt"
# Fall back to workspace-local checkpoint if /tmp/models/sam3.pt not found
if [ ! -f "$CHECKPOINT_PATH" ]; then
    WS_CHECKPOINT="$WORKSPACE_DIR/models/sam3/sam3.pt"
    if [ -f "$WS_CHECKPOINT" ]; then
        CHECKPOINT_PATH="$WS_CHECKPOINT"
    fi
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "✗ SAM3 checkpoint not found."
    echo "  Expected at: $MODEL_REPO/sam3.pt or $WORKSPACE_DIR/models/sam3/sam3.pt"
    echo "  Download from: https://huggingface.co/facebook/sam3"
    exit 1
fi
size_mb=$(du -m "$CHECKPOINT_PATH" | cut -f1)
echo "✓ SAM3 checkpoint: $CHECKPOINT_PATH (${size_mb}MB)"
echo ""

# Step 5: Run benchmark
echo "[5/5] Running SAM3 benchmark..."
echo "=========================================="
echo ""

echo "Configuration:"
echo "  Model: SAM3 (PyTorch)"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Prompt: $TEXT_PROMPT"
echo "  Dataset: $DATASET_LABEL"
echo "  Output: $OUTPUT_FILE"
echo "  Duration: 30 seconds"
echo ""

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

# Check for optional TRT vision engine (.pt2 = PyTorch 2.10+, .ep = legacy)
TRT_VISION_ARG=""
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
for TRT_EXT in pt2 ep; do
    TRT_PATH="$CHECKPOINT_DIR/vision_encoder_trt_fp16.$TRT_EXT"
    if [ -f "$TRT_PATH" ]; then
        TRT_VISION_ARG="-p pytorch_trt_vision_engine:=$TRT_PATH"
        echo "  TRT vision engine: $TRT_PATH"
        break
    fi
done
if [ -z "$TRT_VISION_ARG" ]; then
    echo "  No TRT vision engine found (will use PyTorch FP32)"
fi

# 1. Start SAM3 node in background
echo "  Starting SAM3 node (pytorch, compile_decoder=True)..."
ros2 run isaac_ros_segment_anything3 sam3_node.py \
    --ros-args \
    -p pytorch_checkpoint:=$CHECKPOINT_PATH \
    -p image_size:=1008 \
    -p confidence_threshold:=0.3 \
    -p pytorch_device:=cuda \
    -p pytorch_compile_decoder:=True \
    -p pytorch_amp_decoder:=True \
    $TRT_VISION_ARG \
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

# 3. Wait for SAM3 node to initialize (model loading + torch.compile warmup = up to 90s)
echo "  Waiting for SAM3 node to initialize (includes torch.compile warmup ~30s)..."
MAX_WAIT=120
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
        echo "Expected vs Actual (SAM3 PyTorch + TRT backend, RTX 4090):"
        echo "-----------------------------------------------------------"

        ACTUAL_E2E=$(jq -r '.performance_metrics.latency_ms.mean' "$OUTPUT_FILE" | awk '{printf "%.1f", $1}')
        ACTUAL_VISION=$(jq -r '.performance_metrics.stage_breakdown_ms.vision_encoder.mean' "$OUTPUT_FILE" | awk '{printf "%.1f", $1}')
        ACTUAL_DECODER=$(jq -r '.performance_metrics.stage_breakdown_ms.decoder.mean' "$OUTPUT_FILE" | awk '{printf "%.1f", $1}')
        ACTUAL_CACHE=$(jq -r '.performance_metrics.stage_breakdown_ms.text_encoder.cache_hit_rate' "$OUTPUT_FILE" | awk '{printf "%.1f", $1 * 100}')

        echo "  E2E Latency: ${ACTUAL_E2E}ms (expected: ~85-95ms with TRT+compile+AMP)"
        echo "  Vision Encoder: ${ACTUAL_VISION}ms (expected: ~30ms TRT FP16 / ~128ms PyTorch FP32)"
        echo "  Decoder: ${ACTUAL_DECODER}ms (expected: ~35ms with compile+AMP FP16)"
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
