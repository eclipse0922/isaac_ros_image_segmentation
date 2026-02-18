#!/usr/bin/env python3
"""Benchmark SAM3 vision encoder with various PyTorch optimization strategies."""

import time
import numpy as np
import torch

def benchmark(fn, name, n_warmup=5, n_runs=20):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean = np.mean(times)
    std = np.std(times)
    print(f'  {name:40s} {mean:7.1f}ms  std={std:.1f}ms  '
          f'(min={min(times):.1f}, max={max(times):.1f})')
    return mean


def check_quality(model, run_fn, img_t, device, label, amp_dtype=None):
    """Run full pipeline and report quality metrics.

    If amp_dtype is set, wraps the entire pipeline (including forward_grounding)
    in torch.autocast to avoid dtype mismatches between backbone output and decoder.
    """
    from sam3.model.data_misc import FindStage
    from contextlib import nullcontext
    ctx = torch.autocast('cuda', dtype=amp_dtype) if amp_dtype else nullcontext()
    with torch.inference_mode(), ctx:
        bb = run_fn(img_t)
        txt = model.backbone.forward_text(['cat'], device=device)
        bb.update(txt)
        fs = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None,
            input_boxes_label=None, input_points=None, input_points_mask=None)
        geo = model._get_dummy_prompt()
        out = model.forward_grounding(
            backbone_out=bb, find_input=fs,
            geometric_prompt=geo, find_target=None)
        logits = out['pred_logits'][0].float().cpu().numpy()
        presence = out['presence_logit_dec'][0].float().cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        if scores.ndim == 2:
            scores = scores.squeeze(-1)
        pres = 1.0 / (1.0 + np.exp(-presence.astype(np.float64)))
        best = float(scores.max())
        above = int((scores > 0.3).sum())
        ok = 'OK' if best > 0.9 and float(pres[0]) > 0.5 else 'DEGRADED'
        print(f'    Quality [{ok}]: best_det={best:.3f}, '
              f'presence={float(pres[0]):.4f}, above_0.3={above}')
        return best, float(pres[0])


def main():
    import cv2
    import sam3.model_builder as sam3_builder

    ckpt = '/ws/models/sam3/sam3.pt'
    device = 'cuda'
    image_size = 1008

    print('Loading SAM3 model...')
    model = sam3_builder.build_sam3_image_model(
        checkpoint_path=ckpt, device=device, load_from_HF=False)
    model.eval()

    # Real image for quality check
    img_bgr = cv2.imread('/ws/datasets/cat.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(img_rgb, (image_size, image_size)).astype(np.float32) / 255.0
    r = (r - 0.5) / 0.5
    img_t = torch.from_numpy(r.transpose(2, 0, 1)[None]).to(device)

    forward_image = model.backbone.forward_image
    results = {}

    # ============================================================
    print('\n=== 1. PyTorch FP32 (eager) ===')
    with torch.inference_mode():
        ms = benchmark(lambda: forward_image(img_t), 'forward_image FP32 eager')
    results['1. FP32 eager'] = ms
    check_quality(model, lambda x: forward_image(x), img_t, device, 'FP32')

    # ============================================================
    print('\n=== 2. AMP BF16 (eager) ===')
    @torch.inference_mode()
    def run_bf16():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return forward_image(img_t)
    ms = benchmark(run_bf16, 'forward_image BF16 eager')
    results['2. BF16 eager'] = ms
    check_quality(model, lambda x: forward_image(x), img_t, device, 'BF16',
                  amp_dtype=torch.bfloat16)

    # ============================================================
    print('\n=== 3. torch.compile FP32 (default) ===')
    torch._dynamo.reset()
    compiled_fi = torch.compile(forward_image, mode='default')
    with torch.inference_mode():
        print('  Compiling (~30-60s)...')
        t0 = time.perf_counter()
        compiled_fi(img_t)
        torch.cuda.synchronize()
        print(f'  First call: {(time.perf_counter()-t0)*1000:.0f}ms')
        ms = benchmark(lambda: compiled_fi(img_t), 'forward_image compiled FP32')
    results['3. compiled FP32'] = ms
    check_quality(model, lambda x: compiled_fi(x), img_t, device, 'compiled FP32')

    # ============================================================
    print('\n=== 4. torch.compile + AMP BF16 ===')
    @torch.inference_mode()
    def run_compiled_bf16():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return compiled_fi(img_t)
    print('  Compiling BF16 path...')
    t0 = time.perf_counter()
    run_compiled_bf16()
    torch.cuda.synchronize()
    print(f'  First call: {(time.perf_counter()-t0)*1000:.0f}ms')
    ms = benchmark(run_compiled_bf16, 'forward_image compiled+BF16')
    results['4. compiled+BF16'] = ms
    check_quality(model, lambda x: compiled_fi(x), img_t, device, 'compiled+BF16',
                  amp_dtype=torch.bfloat16)

    # ============================================================
    print('\n=== 5. torch.compile max-autotune + BF16 ===')
    torch._dynamo.reset()
    compiled_max = torch.compile(forward_image, mode='max-autotune')
    @torch.inference_mode()
    def run_max_bf16():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return compiled_max(img_t)
    print('  Compiling max-autotune (~2-5min)...')
    t0 = time.perf_counter()
    run_max_bf16()
    torch.cuda.synchronize()
    print(f'  First call: {(time.perf_counter()-t0)*1000:.0f}ms')
    ms = benchmark(run_max_bf16, 'forward_image max-autotune+BF16')
    results['5. max-autotune+BF16'] = ms
    check_quality(model, lambda x: compiled_max(x), img_t, device, 'max-autotune+BF16',
                  amp_dtype=torch.bfloat16)

    # ============================================================
    print('\n' + '=' * 70)
    print('SUMMARY (vision encoder only, RTX 4090, 20 runs, 5 warmup)')
    print('=' * 70)
    baseline = results['1. FP32 eager']
    for name, ms in sorted(results.items(), key=lambda x: x[1]):
        speedup = baseline / ms
        print(f'  {name:35s}  {ms:7.1f}ms  ({speedup:.2f}x vs FP32)')


if __name__ == '__main__':
    main()
