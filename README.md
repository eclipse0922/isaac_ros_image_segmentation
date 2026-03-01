# Isaac ROS Image Segmentation (+ SAM3)

> Fork of [NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation).
> Adds **SAM3** (Meta, 2025) as a ROS 2 node with a PyTorch-native backend.

## SAM3 on ROS 2

[SAM3](https://ai.meta.com/blog/sam3/) is Meta's open-vocabulary segmentation model released in 2025 that handles grounded segmentation via text prompts. This fork integrates it into the Isaac ROS image segmentation ecosystem as a Python-based ROS 2 node.

Key characteristics:
- **Open-vocabulary** — segment by text prompt ("robot arm", "person", etc.)
- **Full SAM3** — ViT-H backbone, FPN neck, grounding decoder (not a distilled variant)
- **PyTorch-native** — no ONNX or Triton; runs directly with `torch.inference_mode()`

### Why PyTorch-native?

The natural path for Isaac ROS is to export models to ONNX → TensorRT for optimized inference. That didn't work here.

`torch.export` (required for `torch_tensorrt`) **fails on the SAM3 decoder** due to `pin_memory` not being supported in export mode. ONNX export hits the same wall. TensorRT works fine for the vision encoder only, which is why the optional TRT vision path exists — but the decoder stays in PyTorch.

Instead, the optimizations applied are:
- **BF16 autocast** (`pytorch_amp_bf16`, default `True`) — wraps the full pipeline, ~2× vision encoder speedup with no quality loss
- **`torch.compile` decoder** (`pytorch_compile_decoder`, default `True`) — ~3× decoder speedup via inductor
- **`torch.compile` vision encoder** (`pytorch_compile_vision`, optional) — additional speedup if TRT not used

### Performance (RTX 4090, r2b_robotarm dataset)

| Stage | Time |
|---|---|
| Vision Encoder (BF16) | ~56 ms |
| Text Encoder (cached) | ~0 ms |
| Decoder (compile + BF16) | ~34 ms |
| Preprocess + postprocess | ~30 ms |
| **ROS 2 E2E** | **~130 ms (~7.5 fps)** |

---

## Quickstart

### Prerequisites

- NVIDIA GPU with CUDA 12+
- Docker with `--runtime=nvidia`
- [HuggingFace account with access to `facebook/sam3`](https://huggingface.co/facebook/sam3) (gated repo — request access first)

### 1. Clone & Download Checkpoint

```bash
git clone https://github.com/eclipse0922/isaac_ros_image_segmentation.git
cd isaac_ros_image_segmentation

# Login to HuggingFace and download sam3.pt (~3.3 GB)
pip install huggingface_hub
huggingface-cli login
mkdir -p models/sam3
huggingface-cli download facebook/sam3 sam3.pt --local-dir models/sam3
```

### 2. Build Docker Image

```bash
docker build \
  -f isaac_ros_segment_anything3/docker/Dockerfile.pytorch \
  -t sam3_pytorch:latest .
```

Base: NGC PyTorch 26.01 (Ubuntu 24.04, CUDA 13.1, PyTorch 2.10, TensorRT 10.14) + ROS 2 Jazzy.

### 3. Run Foxglove Demo (with robot arm video)

```bash
# On host — streams segmentation overlay to Foxglove Studio on port 8765
./run_foxglove_demo.sh \
  --bag datasets/r2bdataset2024_v1/r2b_robotarm \
  --topic /camera_1/color/image_raw \
  --prompt "robot arm"
```

Open [Foxglove Studio](https://foxglove.dev/studio) → connect to `ws://localhost:8765` → subscribe to `/sam3/overlay`.

> The r2b_robotarm dataset (~1.4 GB MCAP) can be downloaded via NGC:
> ```bash
> ./download_dataset_ngc.sh
> ```

### Standalone Test (no ROS 2)

```bash
docker run --runtime=nvidia --rm \
  -v $(pwd):/ws -w /ws \
  sam3_pytorch:latest \
  python3 isaac_ros_segment_anything3/scripts/test_sam3_pytorch.py \
    --checkpoint models/sam3/sam3.pt \
    --image /path/to/image.jpg \
    --prompt "cat" \
    --precision bf16
```

---

## ROS 2 Node

**Package:** `isaac_ros_segment_anything3`
**Node:** `sam3_node.py`

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_checkpoint_path` | `/tmp/models/sam3.pt` | Path to `sam3.pt` |
| `text_prompts` | `""` | Comma-separated prompts (also settable at runtime via service) |
| `confidence_threshold` | `0.3` | Detection score threshold |
| `pytorch_amp_bf16` | `True` | BF16 autocast for full pipeline |
| `pytorch_compile_decoder` | `True` | `torch.compile` on decoder |
| `pytorch_compile_vision` | `False` | `torch.compile` on vision encoder |
| `image_size` | `1008` | Input resolution (SAM3 native: 1008×1008) |

### Runtime Prompt Update

```bash
ros2 service call /sam3/set_text_prompt \
  isaac_ros_segment_anything3_interfaces/srv/SetTextPrompt \
  "{text_prompts: ['robot arm', 'person']}"
```

### Topics

| Topic | Type | Description |
|---|---|---|
| `/image_raw` (sub) | `sensor_msgs/Image` | Input RGB image |
| `/sam3/raw_segmentation_mask` (pub) | `sensor_msgs/Image` | Binary mask |
| `/sam3/overlay` (pub) | `sensor_msgs/Image` | Visualization overlay |
| `/sam3/timing` (pub) | `Sam3Timing` | Per-stage latency |

---

## Architecture Notes

SAM3 uses a **ViT-H vision backbone** with an FPN neck (4 feature scales) and a grounding decoder that takes text embeddings as queries. Unlike SAM1/2, there is no point/box prompt — everything goes through text.

Key implementation details:
- Image resolution: 1008×1008 (stretch resize, no letterbox)
- Normalization: mean=std=[0.5, 0.5, 0.5]
- Text encoding via built-in CLIP tokenizer (`model.backbone.forward_text()`)
- Decoder outputs 200 query slots with normalized [cx, cy, w, h] boxes
- Scoring: `sigmoid(pred_logits) × sigmoid(presence_logits) > threshold`
- Geometry encoder CLS token requires cross-attention with image features before the main decoder

---

## Original NVIDIA Packages

The packages below are from the original NVIDIA Isaac ROS release and remain unchanged in this fork. Refer to the [upstream documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/index.html) for usage.

| Package | Model | Description |
|---|---|---|
| `isaac_ros_unet` | U-Net | Semantic segmentation via TensorRT |
| `isaac_ros_segformer` | Segformer | Transformer-based segmentation |
| `isaac_ros_segment_anything` | SAM | Prompt-based segmentation (SAM1) |
| `isaac_ros_segment_anything2` | SAM2 | Video object segmentation |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
SAM3 model weights are subject to [Meta's model license](https://huggingface.co/facebook/sam3).
