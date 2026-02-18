#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile SAM3 vision backbone to TensorRT via torch_tensorrt.

Uses torch.export + torch_tensorrt.dynamo.compile() to avoid ONNX intermediate.
Saves compiled engine as .pt2 (ExportedProgram, PyTorch 2.10+ format).

Usage (inside sam3_pytorch Docker container):
    python3 compile_sam3_trt.py --checkpoint /ws/models/sam3/sam3.pt \
        --output /ws/models/sam3/vision_encoder_trt_fp16.pt2
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def build_model(checkpoint_path, device):
    from sam3.model_builder import build_sam3_image_model
    return build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        load_from_HF=False,
    )


def benchmark(fn, name, n_warmup=3, n_runs=10):
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
    print(f'  {name}: {mean:.1f}ms  std={std:.1f}ms  (min={min(times):.1f}, max={max(times):.1f})')
    return mean


def main():
    parser = argparse.ArgumentParser(description='Compile SAM3 vision backbone to TensorRT')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', default='sam3', choices=['sam3'])
    parser.add_argument('--output', default=None, help='Output path for compiled engine')
    parser.add_argument('--precision', default='fp16', choices=['fp16', 'fp32', 'bf16'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Skip compilation, just benchmark PyTorch baseline')
    parser.add_argument('--load-engine', default=None,
                        help='Load existing .pt2 TRT engine and benchmark vs PyTorch baseline')
    parser.add_argument('--n-warmup', type=int, default=3,
                        help='Number of warmup iterations for benchmark (default: 3)')
    parser.add_argument('--n-runs', type=int, default=10,
                        help='Number of timed iterations for benchmark (default: 10)')
    args = parser.parse_args()

    device = args.device
    image_size = 1008

    # Default output path (.pt2 required by PyTorch 2.10+ torch.export.load)
    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(ckpt_dir, f'vision_encoder_trt_{args.precision}.pt2')

    print(f'Model type: {args.model_type}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output:     {args.output}')
    print(f'Precision:  {args.precision}')
    print()

    # Load model
    print('Loading model...')
    t0 = time.perf_counter()
    model = build_model(args.checkpoint, device)
    model.eval()
    print(f'  Loaded in {time.perf_counter()-t0:.1f}s')

    vision_backbone = model.backbone.vision_backbone
    img_t = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32, device=device)

    # Baseline benchmark (PyTorch FP32)
    print()
    print('=== Baseline (PyTorch FP32) ===')
    with torch.inference_mode():
        baseline_ms = benchmark(
            lambda: vision_backbone(img_t),
            name='vision_backbone FP32',
            n_warmup=args.n_warmup, n_runs=args.n_runs)

    if args.benchmark_only:
        return

    # Load-and-benchmark existing engine (no recompilation)
    if args.load_engine:
        print()
        print(f'=== Load existing TRT engine: {args.load_engine} ===')
        import torch_tensorrt  # noqa: F401
        loaded_ep = torch.export.load(args.load_engine)
        trt_loaded = loaded_ep.module()
        print('  Engine loaded OK')
        print()
        print(f'=== Benchmark loaded TRT engine ===')
        with torch.inference_mode():
            trt_ms = benchmark(
                lambda: trt_loaded(img_t),
                name='loaded TRT engine',
                n_warmup=args.n_warmup, n_runs=args.n_runs)
        print(f'  Speedup vs FP32 baseline: {baseline_ms/trt_ms:.1f}x')
        return

    # Export
    print()
    print('=== torch.export ===')
    t0 = time.perf_counter()
    with torch.inference_mode():
        exported = torch.export.export(vision_backbone, args=(img_t,), strict=False)
    print(f'  Export done in {time.perf_counter()-t0:.1f}s')

    # Compile
    print()
    print(f'=== torch_tensorrt.dynamo.compile ({args.precision}) ===')
    print('  This takes 1-3 minutes...')

    import torch_tensorrt
    precision_map = {
        'fp16': {torch.float16},
        'bf16': {torch.bfloat16},
        'fp32': {torch.float32},
    }
    enabled_precisions = precision_map[args.precision]

    t0 = time.perf_counter()
    trt_model = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[torch_tensorrt.Input(
            shape=(1, 3, image_size, image_size),
            dtype=torch.float32)],
        enabled_precisions=enabled_precisions,
        workspace_size=4 * 1024 ** 3,
        debug=False,
    )
    compile_time = time.perf_counter() - t0
    print(f'  Compilation done in {compile_time:.1f}s')

    # Benchmark compiled model
    print()
    print(f'=== Benchmark TRT {args.precision} ===')
    with torch.inference_mode():
        trt_ms = benchmark(
            lambda: trt_model(img_t),
            name=f'vision_backbone TRT {args.precision}',
            n_warmup=args.n_warmup, n_runs=args.n_runs)
    print(f'  Speedup: {baseline_ms/trt_ms:.1f}x')

    # Save
    print()
    print(f'=== Save to {args.output} ===')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    try:
        torch_tensorrt.save(trt_model, args.output, inputs=[img_t])
        size_mb = os.path.getsize(args.output) / 1e6
        print(f'  Saved: {size_mb:.0f}MB')

        # Test load (PyTorch 2.10+: use .module() on loaded ExportedProgram)
        print('  Testing load...')
        loaded_ep = torch.export.load(args.output)
        loaded = loaded_ep.module()
        with torch.inference_mode():
            loaded_ms = benchmark(
                lambda: loaded(img_t),
                name=f'loaded TRT {args.precision}')
        print(f'  Load test OK: {loaded_ms:.1f}ms')

    except Exception as e:
        print(f'  Save failed: {type(e).__name__}: {e}')
        print()
        print('  Note: torch_tensorrt.save() has known serialization issues in')
        print('  some versions. The compiled model is available in-process.')
        print('  Consider using compile-at-startup caching instead.')

    print()
    print('=== Summary ===')
    print(f'  PyTorch FP32: {baseline_ms:.1f}ms')
    print(f'  TRT {args.precision}:    {trt_ms:.1f}ms')
    print(f'  Speedup:      {baseline_ms/trt_ms:.1f}x')


if __name__ == '__main__':
    main()
