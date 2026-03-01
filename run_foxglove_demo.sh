#!/bin/bash
# SAM3 Foxglove Demo
#
# Launches SAM3 node + overlay + foxglove_bridge inside Docker.
# Connect Foxglove Studio to ws://localhost:8765 to visualize.
#
# Usage:
#   ./run_foxglove_demo.sh                          # auto-downloads r2b_robotarm, runs with "robot arm" prompt
#   ./run_foxglove_demo.sh --prompt "person"        # same dataset, different prompt
#   ./run_foxglove_demo.sh --video /path/to/vid.mp4 # use a video file instead
#   ./run_foxglove_demo.sh --bag /path/to/bag       # use a custom rosbag
#
# After the node is ready (~40s), change the text prompt live:
#   docker exec sam3_foxglove \
#     ros2 service call /sam3/set_text_prompt \
#     isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
#     "{text_prompts: ['robot arm']}"

set -e

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="sam3_foxglove"

# Defaults
INPUT_TYPE="bag"
IMAGE_TOPIC="/camera_1/color/image_raw"
TEXT_PROMPT="robot arm"
FOXGLOVE_PORT=8765
CHECKPOINT_PATH="$WORKSPACE_DIR/models/sam3/sam3.pt"
DATASET_DIR="$WORKSPACE_DIR/datasets/r2bdataset2024_v1/r2b_robotarm"
CUSTOM_BAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            INPUT_TYPE="video"
            CUSTOM_BAG="$2"
            IMAGE_TOPIC="image_raw"
            shift 2 ;;
        --bag)
            INPUT_TYPE="bag"
            CUSTOM_BAG="$2"
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

# ── Checkpoint check ──────────────────────────────────────────────────────────
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: SAM3 checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Download it from HuggingFace (requires access request):"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli login"
    echo "  huggingface-cli download facebook/sam3 sam3.pt --local-dir models/sam3"
    exit 1
fi

# ── Dataset: auto-download r2b_robotarm if no custom input specified ──────────
if [ "$INPUT_TYPE" = "bag" ] && [ -z "$CUSTOM_BAG" ]; then
    MCAP_FILE="$DATASET_DIR/r2b_robotarm_0.mcap"

    if [ ! -f "$MCAP_FILE" ]; then
        echo "r2b_robotarm dataset not found. Downloading (~1.4 GB)..."
        echo ""
        mkdir -p "$DATASET_DIR"

        BASE_URL="https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2024/versions/1/files/r2b_robotarm"

        wget -q --show-progress -O "$DATASET_DIR/metadata.yaml" \
            "$BASE_URL/metadata.yaml" || {
            echo "Error: failed to download metadata.yaml"
            echo "Check your internet connection or download manually."
            exit 1
        }

        wget -q --show-progress -O "$DATASET_DIR/r2b_robotarm_0.mcap" \
            "$BASE_URL/r2b_robotarm_0.mcap" || {
            echo "Error: failed to download r2b_robotarm_0.mcap"
            rm -f "$DATASET_DIR/r2b_robotarm_0.mcap"
            exit 1
        }

        echo ""
        echo "Download complete."
    fi

    INPUT_PATH="$DATASET_DIR"
else
    INPUT_PATH="$CUSTOM_BAG"
fi

# ── Resolve Docker-internal path ──────────────────────────────────────────────
INPUT_PATH="$(realpath "$INPUT_PATH" 2>/dev/null || echo "$INPUT_PATH")"
if [[ "$INPUT_PATH" == "$WORKSPACE_DIR"* ]]; then
    DOCKER_INPUT_PATH="/ws${INPUT_PATH#$WORKSPACE_DIR}"
else
    echo "Warning: $INPUT_PATH is outside the workspace — mounting it as read-only at /data"
    DOCKER_INPUT_PATH="/data"
    EXTRA_MOUNT="-v ${INPUT_PATH}:/data:ro"
fi

echo "=========================================="
echo "SAM3 Foxglove Demo"
echo "=========================================="
echo "  Input:    $INPUT_TYPE → $INPUT_PATH"
echo "  Topic:    $IMAGE_TOPIC"
echo "  Prompt:   $TEXT_PROMPT"
echo "  Port:     $FOXGLOVE_PORT"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo ""
echo "  Connect Foxglove Studio to: ws://localhost:$FOXGLOVE_PORT"
echo "  Panels: Image (sam3/overlay)  |  Raw Messages (sam3/detections)"
echo "=========================================="
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run --rm --runtime=nvidia \
    --name "$CONTAINER_NAME" \
    --network host \
    -v "$WORKSPACE_DIR:/ws" \
    -v /tmp:/tmp \
    ${EXTRA_MOUNT:-} \
    sam3_pytorch:latest \
    bash -c "
        set -e

        # Install sam3 package
        pip install -q git+https://github.com/SimonZeng7108/efficientsam3.git@77e830355cb > /dev/null 2>&1

        # Source ROS2 and workspace
        source /opt/ros/jazzy/setup.bash

        # Build packages if install/ is missing or stale
        if [ ! -f /ws/install/isaac_ros_segment_anything3/lib/isaac_ros_segment_anything3/sam3_node.py ]; then
            echo '[demo] Building ROS 2 packages (first run)...'
            cd /ws && colcon build \
                --packages-select \
                    isaac_ros_segment_anything3_interfaces \
                    isaac_ros_segment_anything3 \
                --cmake-args -DCMAKE_BUILD_TYPE=Release \
                --event-handlers console_direct+ 2>&1 | grep -E '(Starting|Finished|Failed|Error)'
            echo '[demo] Build complete.'
        fi

        source /ws/install/setup.bash

        # Sync latest scripts (dev convenience — picks up edits without rebuild)
        for _script in sam3_node.py overlay_node.py video_publisher.py; do
            cp /ws/isaac_ros_segment_anything3/scripts/\$_script \
               /ws/install/isaac_ros_segment_anything3/lib/isaac_ros_segment_anything3/\$_script 2>/dev/null || true
        done

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

        cleanup() {
            echo '[demo] Shutting down...'
            kill \$SAM3_PID \$OVERLAY_PID \$FOXGLOVE_PID \$INPUT_PID 2>/dev/null || true
            wait 2>/dev/null || true
        }
        trap cleanup EXIT

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

        echo \"[demo] Setting text prompt: '$TEXT_PROMPT'\"
        ros2 service call /sam3/set_text_prompt \
            isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
            \"{text_prompts: ['$TEXT_PROMPT']}\" 2>/dev/null || true

        if [ '$INPUT_TYPE' = 'video' ]; then
            echo \"[demo] Starting video publisher: $DOCKER_INPUT_PATH\"
            ros2 run isaac_ros_segment_anything3 video_publisher.py \
                --ros-args \
                -p video_path:=$DOCKER_INPUT_PATH \
                -p fps:=10.0 \
                -p loop:=True \
                --remap image_raw:=$IMAGE_TOPIC &
            INPUT_PID=\$!
        else
            echo \"[demo] Starting bag playback: $DOCKER_INPUT_PATH\"
            ros2 bag play '$DOCKER_INPUT_PATH' --loop &
            INPUT_PID=\$!
        fi

        echo '[demo] Pipeline running. Connect Foxglove Studio to ws://localhost:$FOXGLOVE_PORT'
        echo '[demo] Subscribe to: sam3/overlay (Image), sam3/detections (Detection2DArray)'
        echo '[demo] Press Ctrl+C to stop.'
        echo ''

        wait \$SAM3_PID
    "
