# SAM3 Benchmark Manual Test Guide

자동 스크립트 실패 시 각 단계를 수동으로 실행하는 가이드입니다.

## Prerequisites

Docker 컨테이너 안에서 실행:
```bash
docker run -it --runtime=nvidia --rm \
  -v /media/sewon/Dev/isaac_ros_image_segmentation:/workspace \
  -w /workspace \
  sam3_pytorch:latest \
  bash
```

## Step 1: Download Dataset

```bash
mkdir -p datasets/r2b_dataset/r2b_robotarm
cd datasets/r2b_dataset/r2b_robotarm

# Download metadata
wget 'https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2024/versions/1/files/r2b_robotarm/metadata.yaml'

# Download rosbag (1.4GB)
wget 'https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2024/versions/1/files/r2b_robotarm/r2b_robotarm_0.db3'

cd /workspace
```

**Verify:**
```bash
ls -lh datasets/r2b_dataset/r2b_robotarm/
# Should show: metadata.yaml (1.5K), r2b_robotarm_0.db3 (1.4G)
```

## Step 2: Build Packages

```bash
# Source ROS2
source /opt/ros/jazzy/setup.bash

# Build interfaces (Sam3Timing message)
colcon build --packages-select isaac_ros_segment_anything3_interfaces

# Build SAM3 node (updated with timing publisher)
colcon build --packages-select isaac_ros_segment_anything3

# Build benchmark package
colcon build --packages-select isaac_ros_segment_anything3_benchmark

# Source workspace
source install/setup.bash
```

**Verify:**
```bash
ros2 interface show isaac_ros_segment_anything3_interfaces/msg/Sam3Timing
# Should display message definition with fields like cvbridge_ms, vision_encoder_ms, etc.
```

## Step 3: Test Components Individually

### 3a. Test SAM3 Node (Manual)

Terminal 1 - SAM3 Node:
```bash
source install/setup.bash

ros2 run isaac_ros_segment_anything3 sam3_node.py \
  --ros-args \
  -p inference_backend:=pytorch \
  -p model_type:=efficient_sam3 \
  -p image_size:=1008
```

Terminal 2 - Monitor Timing:
```bash
source install/setup.bash

ros2 topic echo /sam3/timing
```

Terminal 3 - Play Rosbag + Set Prompt:
```bash
source install/setup.bash

# Play rosbag
ros2 bag play datasets/r2b_dataset/r2b_robotarm --loop &

# Set text prompt (after 5 sec)
sleep 5
ros2 service call /sam3/set_text_prompt \
  isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
  "{text_prompts: ['person']}"
```

**Expected:** Terminal 2 should show timing messages every frame.

### 3b. Test Monitor Node

Terminal 1 - SAM3 + Rosbag (from 3a):
```bash
# Keep running from previous test
```

Terminal 2 - Monitor Node:
```bash
source install/setup.bash

ros2 run isaac_ros_segment_anything3_benchmark sam3_monitor_node.py \
  --ros-args \
  -p test_duration_sec:=10.0 \
  -p output_path:=results/manual_test.json
```

**Expected:** After 10 seconds, results/manual_test.json should be created with performance metrics.

## Step 4: Run Full Benchmark

```bash
source install/setup.bash

ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  backend:=pytorch \
  model_type:=efficient_sam3 \
  text_prompts:=person \
  rosbag_path:=datasets/r2b_dataset/r2b_robotarm \
  test_duration:=30.0 \
  output_path:=results/esam3_pytorch_robotarm.json
```

**Expected:** Runs for ~35 seconds, creates JSON results file.

## Step 5: Verify Results

```bash
# Check file exists
ls -lh results/esam3_pytorch_robotarm.json

# View results (with jq)
apt-get update && apt-get install -y jq
jq '.performance_metrics' results/esam3_pytorch_robotarm.json

# Check key metrics
jq -r '
  "E2E Latency: \(.performance_metrics.latency_ms.mean)ms",
  "Throughput: \(.performance_metrics.throughput_fps)fps",
  "Vision Encoder: \(.performance_metrics.stage_breakdown_ms.vision_encoder.mean)ms",
  "Cache Hit Rate: \(.performance_metrics.stage_breakdown_ms.text_encoder.cache_hit_rate * 100)%"
' results/esam3_pytorch_robotarm.json
```

## Troubleshooting

### Issue: "Sam3Timing message not found"

**Cause:** Interfaces package not built or not sourced.

**Fix:**
```bash
colcon build --packages-select isaac_ros_segment_anything3_interfaces
source install/setup.bash
ros2 interface list | grep Sam3Timing
```

### Issue: "No timing messages received"

**Cause:** SAM3 node not publishing timing data.

**Fix:**
```bash
# Check if timing topic exists
ros2 topic list | grep timing

# Check if SAM3 node is running
ros2 node list

# Check SAM3 node publishers
ros2 node info /sam3_node | grep Publishers
```

### Issue: "Rosbag not found"

**Cause:** Dataset not downloaded or wrong path.

**Fix:**
```bash
# Verify rosbag path
ros2 bag info datasets/r2b_dataset/r2b_robotarm

# Check metadata
ls -l datasets/r2b_dataset/r2b_robotarm/
```

### Issue: "PyTorch models not loading"

**Cause:** Models not downloaded.

**Fix:**
```bash
# Download PyTorch models manually
python3 isaac_ros_segment_anything3/scripts/download_models.py \
  --model_type efficient_sam3 \
  --backend pytorch

# Check model files
ls -lh /tmp/esam3_checkpoints/
```

### Issue: "Low performance / High latency"

**Check:**
```bash
# GPU utilization
nvidia-smi -l 1

# CPU usage
top

# Check if running on CPU instead of GPU
# In SAM3 node logs, should see: backend=pytorch, device=cuda
```

## Testing Different Scenarios

### Test Multi-Prompt Scaling

```bash
# 2 prompts
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  backend:=pytorch \
  text_prompts:="person,cup" \
  output_path:=results/esam3_pytorch_2prompts.json

# Expected: Decoder time ~140ms (2× 70ms)
```

### Test Triton Backend (for comparison)

```bash
# Start Triton server first
docker run --runtime=nvidia -d -p 8001:8001 \
  -v /tmp/esam3_models:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models --grpc-port=8001

# Run benchmark
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  backend:=triton \
  triton_url:=localhost:8001 \
  model_repository:=/tmp/esam3_models \
  output_path:=results/esam3_triton_robotarm.json

# Expected: E2E ~849ms (much slower than PyTorch)
```

### Test SAM3 vs EfficientSAM3

```bash
# EfficientSAM3 (default)
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  model_type:=efficient_sam3 \
  output_path:=results/esam3.json

# Full SAM3
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  model_type:=sam3 \
  output_path:=results/sam3.json

# Compare
jq -r '.performance_metrics.latency_ms.mean' results/esam3.json
jq -r '.performance_metrics.latency_ms.mean' results/sam3.json
```

## Expected Performance Baselines (RTX 4090)

| Configuration | E2E Latency | Vision | Decoder | Cache Hit |
|--------------|-------------|--------|---------|-----------|
| ESAM3 PyTorch (1p) | ~131ms | ~20ms | ~70ms | >95% |
| ESAM3 Triton (1p) | ~849ms | ~322ms | ~536ms | >95% |
| SAM3 PyTorch (1p) | ~145ms | ~25ms | ~75ms | >95% |

Deviations >10% indicate potential issues with GPU, model loading, or configuration.
