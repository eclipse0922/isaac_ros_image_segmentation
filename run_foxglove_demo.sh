#!/bin/bash
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SAM3 Foxglove Demo
# Launches SAM3 node + overlay + foxglove_bridge inside Docker.
# Connect Foxglove Studio to ws://localhost:8765 to visualize.
#
# Usage (from host):
#   # Video file input:
#   ./run_foxglove_demo.sh --video /path/to/video.mp4
#
#   # Rosbag input:
#   ./run_foxglove_demo.sh --bag /path/to/bag_folder [--topic /image_raw]
#
#   # Rosbag with non-default image topic (e.g. r2b_robotarm):
#   ./run_foxglove_demo.sh --bag datasets/r2bdataset2024_v1/r2b_robotarm \
#       --topic /camera_1/color/image_raw --prompt "robot arm"
#
# After the node is ready (~40s), set/change the text prompt via:
#   docker exec sam3_foxglove \
#     ros2 service call /sam3/set_text_prompt \
#     isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
#     "{text_prompts: ['robot arm']}"

set -e

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="sam3_foxglove"

# Defaults
INPUT_TYPE=""
INPUT_PATH=""
IMAGE_TOPIC="image_raw"
TEXT_PROMPT="person"
FOXGLOVE_PORT=8765
CHECKPOINT_PATH="$WORKSPACE_DIR/models/sam3/sam3.pt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            INPUT_TYPE="video"
            INPUT_PATH="$2"
            shift 2 ;;
        --bag)
            INPUT_TYPE="bag"
            INPUT_PATH="$2"
            shift 2 ;;
        --topic)
            IMAGE_TOPIC="$2"
            shift 2 ;;
        --prompt)
            TEXT_PROMPT="$2"
            shift 2 ;;
        --port)
            FOXGLOVE_PORT="$2"
            shift 2 ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if [ -z "$INPUT_TYPE" ]; then
    echo "Error: specify --video <path> or --bag <path>"
    echo ""
    echo "Usage:"
    echo "  $0 --video /path/to/video.mp4 [--prompt 'cat']"
    echo "  $0 --bag /path/to/bag [--topic /camera/image_raw] [--prompt 'robot arm']"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: SAM3 checkpoint not found at: $CHECKPOINT_PATH"
    echo "Download from: https://huggingface.co/facebook/sam3"
    exit 1
fi

# Convert relative path to absolute
INPUT_PATH="$(realpath "$INPUT_PATH" 2>/dev/null || echo "$INPUT_PATH")"

echo "=========================================="
echo "SAM3 Foxglove Demo"
echo "=========================================="
echo "  Input: $INPUT_TYPE → $INPUT_PATH"
echo "  Topic: $IMAGE_TOPIC"
echo "  Prompt: $TEXT_PROMPT"
echo "  Foxglove port: $FOXGLOVE_PORT"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo ""
echo "  Connect Foxglove Studio to: ws://localhost:$FOXGLOVE_PORT"
echo "  Recommended panels: Image (sam3/overlay), Raw Messages (sam3/detections)"
echo "=========================================="
echo ""

# Stop existing container if running
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Launch Docker container with the demo
docker run --rm --runtime=nvidia \
    --name "$CONTAINER_NAME" \
    --network host \
    -v "$WORKSPACE_DIR:/ws" \
    -v /tmp:/tmp \
    sam3_pytorch:latest \
    bash -c "
        set -e

        # Install sam3 package
        pip install -q git+https://github.com/SimonZeng7108/efficientsam3.git@77e830355cb > /dev/null 2>&1

        # Source ROS2 and workspace
        source /opt/ros/jazzy/setup.bash
        source /ws/install/setup.bash 2>/dev/null || true

        # Sync latest scripts
        cp /ws/isaac_ros_segment_anything3/scripts/sam3_node.py \
           /ws/install/isaac_ros_segment_anything3/lib/isaac_ros_segment_anything3/sam3_node.py 2>/dev/null || true
        cp /ws/isaac_ros_segment_anything3/scripts/overlay_node.py \
           /ws/install/isaac_ros_segment_anything3/lib/isaac_ros_segment_anything3/overlay_node.py 2>/dev/null || true

        echo '[demo] Starting SAM3 node...'
        ros2 run isaac_ros_segment_anything3 sam3_node.py \
            --ros-args \
            -p pytorch_checkpoint:=/ws/models/sam3/sam3.pt \
            -p pytorch_compile_decoder:=True \
            -p pytorch_amp_decoder:=True \
            -p confidence_threshold:=0.3 \
            --remap image_raw:=$IMAGE_TOPIC &
        SAM3_PID=\$!

        echo '[demo] Starting overlay node...'
        ros2 run isaac_ros_segment_anything3 overlay_node.py \
            --ros-args \
            --remap image_raw:=$IMAGE_TOPIC &
        OVERLAY_PID=\$!

        echo '[demo] Starting foxglove_bridge on port $FOXGLOVE_PORT...'
        ros2 run foxglove_bridge foxglove_bridge \
            --ros-args \
            -p port:=$FOXGLOVE_PORT \
            -p send_buffer_limit:=100000000 &
        FOXGLOVE_PID=\$!

        # Cleanup function
        cleanup() {
            echo '[demo] Shutting down...'
            kill \$SAM3_PID \$OVERLAY_PID \$FOXGLOVE_PID \$INPUT_PID 2>/dev/null || true
            wait 2>/dev/null || true
        }
        trap cleanup EXIT

        # Wait for SAM3 to be ready (polls service endpoint)
        echo '[demo] Waiting for SAM3 node to be ready (~40s for torch.compile)...'
        MAX_WAIT=120
        WAITED=0
        while [ \$WAITED -lt \$MAX_WAIT ]; do
            if ros2 service list 2>/dev/null | grep -q '/sam3/set_text_prompt'; then
                echo \"[demo] SAM3 ready after \${WAITED}s\"
                break
            fi
            sleep 2
            WAITED=\$((WAITED + 2))
            if [ \$((WAITED % 20)) -eq 0 ]; then
                echo \"[demo] Still waiting... (\${WAITED}s)\"
            fi
        done

        # Set initial text prompt
        echo \"[demo] Setting text prompt: '$TEXT_PROMPT'\"
        ros2 service call /sam3/set_text_prompt \
            isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
            \"{text_prompts: ['$TEXT_PROMPT']}\" || true

        # Start input source
        if [ '$INPUT_TYPE' = 'video' ]; then
            echo \"[demo] Starting video publisher: $INPUT_PATH\"
            ros2 run isaac_ros_segment_anything3 video_publisher.py \
                --ros-args \
                -p video_path:=$INPUT_PATH \
                -p fps:=10.0 \
                -p loop:=True \
                --remap image_raw:=$IMAGE_TOPIC &
            INPUT_PID=\$!
        else
            echo \"[demo] Starting bag playback: $INPUT_PATH\"
            ros2 bag play '$INPUT_PATH' --loop &
            INPUT_PID=\$!
        fi

        echo '[demo] Pipeline running. Connect Foxglove Studio to ws://localhost:$FOXGLOVE_PORT'
        echo '[demo] Subscribe to: sam3/overlay (Image), sam3/detections (Detection2DArray)'
        echo '[demo] Press Ctrl+C to stop.'
        echo ''

        # Wait for any process to exit
        wait \$SAM3_PID
    "
