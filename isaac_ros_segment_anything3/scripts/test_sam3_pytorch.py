#!/usr/bin/env python3
"""
Standalone SAM3 PyTorch inference test.

Loads the SAM3 model, runs text-prompted grounding on a single image,
and saves a visualization with detection boxes and masks.

Usage (inside Docker):
    python3 /ws/isaac_ros_segment_anything3/scripts/test_sam3_pytorch.py \
        --checkpoint /ws/models/sam3/sam3.pt \
        --image /ws/datasets/cat.jpg \
        --prompt "cat" \
        --output /ws/datasets/cat_result.jpg
"""

import argparse
import os
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='SAM3 PyTorch inference test')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to sam3.pt checkpoint')
    parser.add_argument('--image', required=True,
                        help='Path to input image (JPEG, PNG, etc.)')
    parser.add_argument('--prompt', default='cat',
                        help='Text prompt (comma-separated for multiple)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--output', default='result.jpg',
                        help='Output visualization path')
    parser.add_argument('--device', default='cuda',
                        help='PyTorch device')
    parser.add_argument('--trt', nargs='?', const='auto', default=None,
                        help='TRT vision engine: "auto" (detect in ckpt dir) or path to .pt2')
    parser.add_argument('--n-warmup', type=int, default=0,
                        help='Number of warmup runs before timed inference')
    parser.add_argument('--n-runs', type=int, default=1,
                        help='Number of timed runs (reports average)')
    args = parser.parse_args()

    import torch
    import sam3.model_builder as sam3_builder
    from sam3.model.data_misc import FindStage

    # Load model
    print(f'Loading SAM3 from {args.checkpoint} ...')
    t0 = time.perf_counter()
    model = sam3_builder.build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
        load_from_HF=False,
    )
    print(f'Model loaded in {time.perf_counter() - t0:.1f}s')

    # Load TRT vision engine (auto-detect or explicit path)
    if args.trt is not None:
        if args.trt == 'auto':
            ckpt_dir = os.path.dirname(args.checkpoint)
            trt_path = None
            for name in ('vision_encoder_trt_fp16.pt2',
                         'vision_encoder_trt_fp16.ep',
                         'vision_encoder_trt_fp32.pt2'):
                candidate = os.path.join(ckpt_dir, name)
                if os.path.isfile(candidate):
                    trt_path = candidate
                    break
        else:
            trt_path = args.trt
        if trt_path and os.path.isfile(trt_path):
            print(f'Loading TRT vision engine: {trt_path} ...')
            import torch_tensorrt  # noqa: F401
            loaded_ep = torch.export.load(trt_path)
            model.backbone.vision_backbone = loaded_ep.module()
            print('TRT vision engine loaded')
        else:
            print(f'WARNING: No TRT engine found ({trt_path})')

    # Load and preprocess image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f'ERROR: Cannot open image: {args.image}')
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f'Image: {args.image} ({orig_w}x{orig_h})')

    IMAGE_SIZE = 1008
    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    img_f = resized.astype(np.float32) / 255.0
    img_f = (img_f - MEAN) / STD
    img_tensor = torch.from_numpy(
        img_f.transpose(2, 0, 1)[np.newaxis]).to(args.device)

    # Parse prompts
    prompts = [p.strip() for p in args.prompt.split(',')]
    print(f'Prompts: {prompts}')

    # Run inference
    device = args.device

    def run_once():
        """Single forward pass, returns (all_results, t_vision, t_text, t_decoder)."""
        with torch.inference_mode():
            t1 = time.perf_counter()
            backbone_out = model.backbone.forward_image(img_tensor)
            torch.cuda.synchronize()
            t_vision = time.perf_counter() - t1

            t2 = time.perf_counter()
            text_out = model.backbone.forward_text(prompts, device=device)
            backbone_out.update(text_out)
            torch.cuda.synchronize()
            t_text = time.perf_counter() - t2

            find_stage = FindStage(
                img_ids=torch.tensor([0], device=device, dtype=torch.long),
                text_ids=torch.tensor([0], device=device, dtype=torch.long),
                input_boxes=None, input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None, input_points_mask=None,
            )
            geo_prompt = model._get_dummy_prompt()

            results = []
            t3 = time.perf_counter()
            for p_idx in range(len(prompts)):
                find_stage.text_ids = torch.tensor(
                    [p_idx], device=device, dtype=torch.long)
                outputs = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_stage,
                    geometric_prompt=geo_prompt,
                    find_target=None,
                )
                results.append({
                    'pred_masks': outputs['pred_masks'].float().cpu().numpy(),
                    'pred_boxes': outputs['pred_boxes'].float().cpu().numpy(),
                    'pred_logits': outputs['pred_logits'].float().cpu().numpy(),
                    'presence': outputs['presence_logit_dec'].float().cpu().numpy(),
                })
            torch.cuda.synchronize()
            t_decoder = time.perf_counter() - t3
        return results, t_vision, t_text, t_decoder

    # Warmup
    for i in range(args.n_warmup):
        run_once()
        if i == 0:
            print(f'Warmup 1/{args.n_warmup} done')
    if args.n_warmup:
        print(f'Warmup complete ({args.n_warmup} runs)')

    # Timed runs
    timings_v, timings_t, timings_d = [], [], []
    all_results = None
    for i in range(args.n_runs):
        all_results, tv, tt, td = run_once()
        timings_v.append(tv)
        timings_t.append(tt)
        timings_d.append(td)

    avg_v = np.mean(timings_v) * 1000
    avg_t = np.mean(timings_t) * 1000
    avg_d = np.mean(timings_d) * 1000
    avg_total = avg_v + avg_t + avg_d
    print(f'\nTiming ({args.n_runs} runs, {args.n_warmup} warmup): '
          f'vision={avg_v:.1f}ms, text={avg_t:.1f}ms, '
          f'decoder={avg_d:.1f}ms, total={avg_total:.1f}ms')

    # Post-process and visualize
    vis = img_bgr.copy()
    total_dets = 0

    for b, r in enumerate(all_results):
        logits = r['pred_logits'][0]     # (200, 1) or (200,)
        boxes = r['pred_boxes'][0]       # (200, 4)
        masks = r['pred_masks'][0]       # (200, 288, 288)
        presence = r['presence'][0]      # (1,)

        det_scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        if det_scores.ndim == 2:
            det_scores = det_scores.squeeze(-1)
        pres_score = 1.0 / (1.0 + np.exp(-presence.astype(np.float64)))

        scores = det_scores  # SAM3: don't multiply by presence

        top5_idx = np.argsort(scores)[::-1][:5]
        print(f'\nPrompt "{prompts[b]}":')
        print(f'  presence_logit={float(presence[0]):.3f} '
              f'(sigmoid={float(pres_score[0]):.4f})')
        print(f'  Top-5 det_scores: '
              f'{[f"{scores[i]:.3f}" for i in top5_idx]}')
        print(f'  Above threshold ({args.threshold}): '
              f'{int((scores > args.threshold).sum())}')

        # Draw detections
        colors = [(0, 0, 255), (0, 200, 0), (255, 100, 0)]
        color = colors[b % len(colors)]

        for j in range(len(scores)):
            if scores[j] < args.threshold:
                continue
            total_dets += 1

            # Box: normalized [cx, cy, w, h] -> pixel [x1, y1, x2, y2]
            cx, cy, w, h = boxes[j]
            x1 = int((cx - w / 2) * orig_w)
            y1 = int((cy - h / 2) * orig_h)
            x2 = int((cx + w / 2) * orig_w)
            y2 = int((cy + h / 2) * orig_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f'{prompts[b]} {scores[j]:.2f}'
            cv2.putText(vis, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Overlay mask
            mask_logit = masks[j]  # (288, 288)
            mask_full = cv2.resize(mask_logit, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_full > 0).astype(np.uint8)
            overlay = np.zeros_like(vis)
            overlay[:] = color
            vis = np.where(
                mask_bin[:, :, None],
                (0.55 * vis.astype(np.float32) +
                 0.45 * overlay.astype(np.float32)).astype(np.uint8),
                vis)

    print(f'\nTotal detections: {total_dets}')

    cv2.imwrite(args.output, vis)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
