# SAM3 Benchmark Quick Start

벤치마크를 5분 안에 실행하는 빠른 시작 가이드입니다.

## 🚀 한 줄로 실행

```bash
# Docker 컨테이너 안에서
./test_sam3_benchmark.sh
```

이것만 실행하면 자동으로:
1. ✅ r2b_robotarm 데이터셋 다운로드 (1.4GB)
2. ✅ 패키지 빌드
3. ✅ 벤치마크 실행 (30초)
4. ✅ 결과 분석 및 출력

## 📦 Prerequisites

### Docker 컨테이너 실행

```bash
# 호스트 터미널에서
docker run -it --runtime=nvidia --rm \
  -v /media/sewon/Dev/isaac_ros_image_segmentation:/workspace \
  -w /workspace \
  sam3_pytorch:latest \
  bash
```

## 📊 예상 결과 (RTX 4090)

```
Performance Summary:
-------------------
Throughput: 7.6 fps
Frames Collected: 228

Latency (E2E):
  Mean: 131.5 ms
  Std Dev: 12.3 ms
  P95: 145.2 ms

Stage Breakdown:
  Vision Encoder: 19.8 ms
  Text Encoder: 0.1 ms (cache hit: 97.0%)
  Decoder: 70.4 ms

Expected vs Actual:
  E2E Latency: 131.5ms (expected: ~131ms) ✓
  Vision Encoder: 19.8ms (expected: ~20ms) ✓
  Decoder: 70.4ms (expected: ~70ms) ✓
  Text Cache Hit: 97.0% (expected: >95%) ✓
```

## 🔧 다른 시나리오 테스트

### Multi-Prompt (2개 프롬프트)

```bash
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  backend:=pytorch \
  text_prompts:="person,cup" \
  output_path:=results/esam3_2prompts.json

# 예상: Decoder ~140ms (2× 70ms)
```

### Triton Backend (비교용)

```bash
# Triton 서버 시작 (별도 터미널)
docker run --runtime=nvidia -d -p 8001:8001 \
  -v /tmp/esam3_models:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models

# 벤치마크 실행
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  backend:=triton \
  output_path:=results/esam3_triton.json

# 예상: E2E ~849ms (PyTorch보다 6.5× 느림)
```

### SAM3 vs EfficientSAM3 비교

```bash
# EfficientSAM3 (기본)
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  model_type:=efficient_sam3 \
  output_path:=results/esam3.json

# 전체 SAM3
ros2 launch isaac_ros_segment_anything3_benchmark/scripts/isaac_ros_segment_anything3_graph.py \
  model_type:=sam3 \
  output_path:=results/sam3.json

# 비교
echo "EfficientSAM3: $(jq -r '.performance_metrics.latency_ms.mean' results/esam3.json)ms"
echo "SAM3: $(jq -r '.performance_metrics.latency_ms.mean' results/sam3.json)ms"
```

## 📁 생성되는 파일

```
isaac_ros_image_segmentation/
├── datasets/r2b_dataset/r2b_robotarm/     # 다운로드된 데이터셋
│   ├── metadata.yaml
│   └── r2b_robotarm_0.db3 (1.4GB)
│
├── isaac_ros_segment_anything3_benchmark/results/
│   └── esam3_pytorch_robotarm_test.json   # 벤치마크 결과
│
└── install/                                # 빌드된 ROS2 패키지
    ├── isaac_ros_segment_anything3_interfaces/
    ├── isaac_ros_segment_anything3/
    └── isaac_ros_segment_anything3_benchmark/
```

## ❓ 문제 해결

### "colcon: command not found"

ROS2가 sourced 안됨:
```bash
source /opt/ros/jazzy/setup.bash
```

### "Sam3Timing message not found"

인터페이스 빌드 후 source 필요:
```bash
colcon build --packages-select isaac_ros_segment_anything3_interfaces
source install/setup.bash
```

### "No timing messages"

SAM3 노드가 제대로 시작 안됨:
```bash
ros2 topic list | grep timing
ros2 node list
```

### 자세한 디버깅

[MANUAL_TEST_GUIDE.md](MANUAL_TEST_GUIDE.md) 참조

## 📚 상세 문서

- **벤치마크 방법론**: [isaac_ros_segment_anything3_benchmark/README.md](isaac_ros_segment_anything3_benchmark/README.md)
- **수동 테스트 가이드**: [MANUAL_TEST_GUIDE.md](MANUAL_TEST_GUIDE.md)
- **테스트 시나리오**: [isaac_ros_segment_anything3_benchmark/config/test_scenarios.yaml](isaac_ros_segment_anything3_benchmark/config/test_scenarios.yaml)

## 🎯 다음 단계

벤치마크 성공 후:

1. **다른 데이터셋 테스트**
   ```bash
   # r2b_galileo (실내 장면, 471MB)
   wget -P datasets/r2b_dataset/r2b_galileo/ \
     'https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2024/versions/1/files/r2b_galileo/metadata.yaml'
   wget -P datasets/r2b_dataset/r2b_galileo/ \
     'https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2024/versions/1/files/r2b_galileo/r2b_galileo_0.db3'
   ```

2. **결과 비교 & 시각화**
   ```bash
   # Python으로 결과 비교 스크립트 작성 가능
   python3 -c "
   import json
   with open('results/esam3_pytorch_robotarm_test.json') as f:
       data = json.load(f)
       print(f'Mean latency: {data[\"performance_metrics\"][\"latency_ms\"][\"mean\"]:.1f}ms')
   "
   ```

3. **성능 최적화**
   - TensorRT 변환 시도 (목표: ~40ms E2E)
   - Batch decoder 구현 (multi-prompt 병렬화)
   - Vision caching (비디오 use case)
