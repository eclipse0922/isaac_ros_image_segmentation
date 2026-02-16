# Contributing to isaac_ros_segment_anything3

## Architecture Overview

SAM3 uses a **Python ROS2 node + tritonclient** approach, unlike SAM1/SAM2 which use C++ ComposableNodes.

```
image -> vision_encoder (Triton) -> FPN features
text  -> tokenizer -> text_encoder (Triton) -> text features
                   -> decoder (Triton, per-prompt) -> masks, boxes, scores
```

### Why Python instead of C++ ComposableNode

SAM3 requires 3 separate Triton models with control flow that doesn't fit a fixed ComposableNode graph:

- **Fan-in**: decoder takes both vision and text encoder outputs
- **Conditional execution**: text encoder is cached until prompt changes
- **Per-prompt loop**: decoder runs once per prompt (batch=1 ONNX constraint)
- **Tokenizer**: HuggingFace `tokenizers` library (Python-native)

## Known Limitations

### 1. Decoder batch=1 constraint

The ONNX decoder model expects `batch=1` for prompt inputs. Each prompt requires a separate decoder call (~500-700ms on desktop GPU).

| Prompts | Decoder time | Total pipeline |
|---------|-------------|----------------|
| 1 | ~600ms | ~1.3s |
| 2 | ~1200ms | ~1.9s |
| 3 | ~1800ms | ~2.5s |

**Current mitigation**: `max_prompts` parameter (default: 3).

**How to improve**:
- Re-export the ONNX decoder with dynamic batch support
- Source: [jamjamjon/YOLO-SAM3](https://github.com/jamjamjon/YOLO-SAM3) export scripts
- Modify `_run_decoder()` to pass all prompts in a single call

### 2. Vision encoder runs every frame

No frame-level caching for the vision encoder (~660ms per frame).

**How to improve**:
- Add frame hash or content-based change detection
- Skip vision encoder when the image hasn't changed (e.g., static camera)
- Consider TensorRT conversion for faster inference

### 3. Python GIL

Python's GIL can cause callback delays under heavy load. The current implementation uses `MultiThreadedExecutor` with minimal lock scope.

**How to improve**:
- Move Triton inference to a separate process (`multiprocessing`)
- Migrate to C++ node with `tritonclient` C++ API

### 4. No NITROS zero-copy

Python node cannot use NITROS GPU-to-GPU zero-copy transfers.

**How to improve**:
- Migrate to C++ ComposableNode with ManagedNitros pub/sub
- Requires C++ tokenizer implementation (e.g., sentencepiece or custom)

## Improvement Roadmap

Contributions are welcome in these areas, ordered by impact:

1. **TensorRT conversion** - Convert ONNX models to TensorRT engines for 2-5x speedup
2. **Batch decoder support** - Re-export ONNX with dynamic batch to eliminate per-prompt loop
3. **Vision encoder caching** - Skip re-encoding when frame content hasn't changed
4. **C++ migration** - Full C++ ComposableNode for NITROS zero-copy and GIL-free execution

## Model Files

Three ONNX models from [jamjamjon/assets](https://github.com/jamjamjon/assets/releases):

| Model | Size | Input | Output |
|-------|------|-------|--------|
| `sam3_vision_encoder` | ~1.7GB | images [1,3,1008,1008] | 4 FPN features |
| `sam3_text_encoder` | ~1.3GB | input_ids + attention_mask [N,32] | text_features + text_mask |
| `sam3_decoder` | ~92MB | 4 FPN + prompt_features [1,32,256] + prompt_mask [1,32] | masks, boxes, logits |

Use `scripts/download_models.py` to download and auto-verify.

## Development Setup

```bash
# Install Python dependencies
pip install tritonclient[all] onnxruntime opencv-python-headless tokenizers numpy

# Download models
python3 scripts/download_models.py --output /tmp/sam3_models

# Start Triton server
docker run --rm --gpus all -p 8001:8001 \
  -v /tmp/sam3_models:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models

# Build ROS package
colcon build --packages-select isaac_ros_segment_anything3_interfaces isaac_ros_segment_anything3
```
