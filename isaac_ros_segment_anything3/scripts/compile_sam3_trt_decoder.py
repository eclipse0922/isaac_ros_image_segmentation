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
Compile SAM3 / EfficientSAM3 decoder to TensorRT via torch_tensorrt.

Architecture note (num_feature_levels=1 for all SAM3 image models):
  - Transformer encoder/decoder uses backbone_fpn[-1] + vision_pos_enc[-1]
  - Segmentation head uses all backbone_fpn levels for fine-grained masks
  - sam2_backbone_out is only for video tracking (None for image grounding)

Input tensors to decoder wrapper:
  fpn_feat_0: (1, 256, 288, 288)  - finest FPN level (for segmentation head)
  fpn_feat_1: (1, 256, 144, 144)  - medium FPN level (for segmentation head)
  fpn_feat_2: (1, 256,  72,  72)  - coarsest FPN level (for transformer + seg head)
  fpn_pos_2:  (1, 256,  72,  72)  - pos encoding for transformer (only last level needed)
  lang_feat:  (seq, 1, 256)        - text features (seq=32)
  lang_mask:  (1, seq)             - text attention mask
  lang_embeds:(1, 1, 256)          - text embeddings before encoder

Usage (inside sam3_pytorch Docker container):
    python3 compile_sam3_trt_decoder.py \\
        --checkpoint /ws/models/sam3/sam3.pt \\
        --model-type sam3 \\
        --output /ws/models/sam3/decoder_trt_fp16.ep

    python3 compile_sam3_trt_decoder.py \\
        --checkpoint /tmp/esam3_models/efficient_sam3.pth \\
        --model-type efficient_sam3 \\
        --output /tmp/esam3_models/decoder_trt_fp16.ep
"""

import argparse
import os
import time

import numpy as np
import torch


def build_model(checkpoint_path, model_type, device):
    if model_type == 'sam3':
        from sam3.model_builder import build_sam3_image_model
        return build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            load_from_HF=False,
        )
    else:
        from sam3.model_builder import build_efficientsam3_image_model
        return build_efficientsam3_image_model(
            checkpoint_path=checkpoint_path,
            backbone_type='tinyvit',
            model_name='11m',
            text_encoder_type='MobileCLIP-S1',
            device=device,
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
    print(f'  {name}: {mean:.1f}ms  std={std:.1f}ms  '
          f'(min={min(times):.1f}, max={max(times):.1f})')
    return mean


class DecoderWrapper(torch.nn.Module):
    """Wraps SAM3 forward_grounding for TRT compilation.

    Accepts backbone_fpn + positional encoding + text features as separate
    tensors, reconstructs backbone_out dict, and calls model.forward_grounding.

    Key architecture facts (num_feature_levels=1 for all SAM3 image models):
      - transformer encoder/decoder only uses backbone_fpn[-1] + vision_pos_enc[-1]
      - segmentation head uses all backbone_fpn levels
      - sam2_backbone_out is only for video tracking (always None here)
    """

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        # Pre-register fixed id tensors as buffers (visible to torch.export)
        self.register_buffer(
            '_img_ids', torch.tensor([0], device=device, dtype=torch.long))
        self.register_buffer(
            '_text_ids', torch.tensor([0], device=device, dtype=torch.long))

    def forward(self,
                fpn_feat_0, fpn_feat_1, fpn_feat_2,
                fpn_pos_2,
                lang_feat, lang_mask, lang_embeds):
        """
        Args:
            fpn_feat_0: (1, 256, 288, 288) - finest FPN level
            fpn_feat_1: (1, 256, 144, 144) - medium FPN level
            fpn_feat_2: (1, 256,  72,  72) - coarsest FPN level = vision_features
            fpn_pos_2:  (1, 256,  72,  72) - pos encoding for transformer
            lang_feat:  (seq, 1, 256)       - language_features (seq_first)
            lang_mask:  (1, seq)            - language_mask
            lang_embeds:(1, 1, 256)         - language_embeds

        Returns:
            (pred_masks, pred_boxes, pred_logits, presence_logit_dec)
        """
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt

        backbone_out = {
            # num_feature_levels=1: transformer only uses backbone_fpn[-1]
            # segmentation head uses all 3 levels
            'vision_features': fpn_feat_2,
            'vision_pos_enc': [fpn_pos_2],  # single-element list (num_feature_levels=1)
            'backbone_fpn': [fpn_feat_0, fpn_feat_1, fpn_feat_2],
            'sam2_backbone_out': None,       # video tracking only
            'language_features': lang_feat,
            'language_mask': lang_mask,
            'language_embeds': lang_embeds,
        }

        find_stage = FindStage(
            img_ids=self._img_ids,
            text_ids=self._text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        # _get_dummy_prompt creates Prompt with zero-size box/point tensors
        geo_prompt = self.model._get_dummy_prompt(num_prompts=1)

        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            geometric_prompt=geo_prompt,
            find_target=None,
        )

        return (
            outputs['pred_masks'],
            outputs['pred_boxes'],
            outputs['pred_logits'],
            outputs['presence_logit_dec'],
        )


def get_backbone_out_shapes(model, device, image_size=1008):
    """Run forward_image + forward_text once to detect actual tensor shapes."""
    img_t = torch.zeros(1, 3, image_size, image_size,
                        dtype=torch.float32, device=device)
    with torch.inference_mode():
        backbone_out_vis = model.backbone.forward_image(img_t)
        text_out = model.backbone.forward_text(['cat'], device=device)
        backbone_out = {**backbone_out_vis, **text_out}

    print('\nbackbone_out structure:')
    fpn_shapes = []
    pos_shapes = []
    for k, v in backbone_out.items():
        if isinstance(v, list):
            print(f'  {k}: list[{len(v)}]')
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    print(f'    [{i}]: {tuple(t.shape)} dtype={t.dtype}')
                    if k == 'backbone_fpn':
                        fpn_shapes.append(tuple(t.shape))
                    elif k == 'vision_pos_enc':
                        pos_shapes.append(tuple(t.shape))
        elif isinstance(v, torch.Tensor):
            print(f'  {k}: {tuple(v.shape)} dtype={v.dtype}')
        elif v is None:
            print(f'  {k}: None')
        else:
            print(f'  {k}: {type(v).__name__}')
    print()

    return backbone_out, fpn_shapes, pos_shapes


def main():
    parser = argparse.ArgumentParser(
        description='Compile SAM3 decoder to TensorRT')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-type', default='sam3',
                        choices=['sam3', 'efficient_sam3'])
    parser.add_argument('--output', default=None,
                        help='Output path for compiled decoder (.ep)')
    parser.add_argument('--precision', default='fp16',
                        choices=['fp16', 'fp32'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Skip compilation, only benchmark PyTorch baseline')
    args = parser.parse_args()

    device = args.device
    image_size = 1008

    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(
            ckpt_dir, f'decoder_trt_{args.precision}.ep')

    print(f'Model type: {args.model_type}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output:     {args.output}')
    print(f'Precision:  {args.precision}')

    # Load model
    print('\nLoading model...')
    t0 = time.perf_counter()
    model = build_model(args.checkpoint, args.model_type, device)
    model.eval()
    print(f'  Loaded in {time.perf_counter()-t0:.1f}s')

    # Detect backbone_out tensor shapes
    backbone_out, fpn_shapes, pos_shapes = get_backbone_out_shapes(
        model, device, image_size)
    print(f'FPN levels: {len(fpn_shapes)}, shapes: {fpn_shapes}')
    print(f'Pos enc levels: {len(pos_shapes)}, shapes: {pos_shapes}')

    # Expect 3 FPN levels: (1,256,288,288), (1,256,144,144), (1,256,72,72)
    if len(fpn_shapes) < 3:
        print(f'WARNING: Expected 3 FPN levels, got {len(fpn_shapes)}. Proceeding anyway.')

    # Build wrapper and run once to verify
    print('\n=== Verify DecoderWrapper ===')
    wrapper = DecoderWrapper(model, device).eval()

    fpn = backbone_out['backbone_fpn']
    pos = backbone_out['vision_pos_enc']
    lang_feat = backbone_out['language_features']   # (seq, 1, 256)
    lang_mask = backbone_out['language_mask']       # (1, seq)
    lang_embeds = backbone_out['language_embeds']   # (1, 1, 256)

    # Use last pos_enc (num_feature_levels=1)
    fpn_pos_2 = pos[-1]

    wrapper_inputs = (fpn[0], fpn[1], fpn[2], fpn_pos_2,
                      lang_feat, lang_mask, lang_embeds)

    with torch.inference_mode():
        out = wrapper(*wrapper_inputs)
    print(f'  pred_masks:        {tuple(out[0].shape)}')
    print(f'  pred_boxes:        {tuple(out[1].shape)}')
    print(f'  pred_logits:       {tuple(out[2].shape)}')
    print(f'  presence_logit_dec:{tuple(out[3].shape)}')

    # Baseline benchmark
    print('\n=== Baseline (PyTorch) ===')
    with torch.inference_mode():
        baseline_ms = benchmark(
            lambda: wrapper(*wrapper_inputs),
            name='decoder PyTorch')

    if args.benchmark_only:
        return

    # Export
    print('\n=== torch.export.export (strict=False) ===')
    print('  This may take 30-60s...')
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            exported = torch.export.export(
                wrapper, args=wrapper_inputs, strict=False)
        print(f'  Export done in {time.perf_counter()-t0:.1f}s')
    except Exception as e:
        print(f'  Export FAILED: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        print()
        print('  torch.export.export failed. The decoder has Python control flow')
        print('  that cannot be traced. Try torch.compile() instead:')
        print('    python3 compile_sam3_trt_decoder.py --benchmark-only')
        print()
        print('  In sam3_node.py, set pytorch_compile_decoder:=True for')
        print('  ~1.5x speedup via torch.compile() (no .ep file needed).')
        return

    # Compile with TRT
    print(f'\n=== torch_tensorrt.dynamo.compile ({args.precision}) ===')
    print('  This takes 2-5 minutes...')

    import torch_tensorrt
    enabled_precisions = (
        {torch.float16} if args.precision == 'fp16' else {torch.float32})

    trt_inputs = [
        torch_tensorrt.Input(shape=s, dtype=torch.float32)
        for s in [fpn_shapes[0], fpn_shapes[1], fpn_shapes[2],
                  pos_shapes[-1] if pos_shapes else fpn_shapes[2]]
    ] + [
        torch_tensorrt.Input(shape=tuple(lang_feat.shape), dtype=torch.float32),
        torch_tensorrt.Input(shape=tuple(lang_mask.shape), dtype=torch.bool),
        torch_tensorrt.Input(shape=tuple(lang_embeds.shape), dtype=torch.float32),
    ]

    t0 = time.perf_counter()
    try:
        trt_model = torch_tensorrt.dynamo.compile(
            exported,
            inputs=trt_inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=4 * 1024 ** 3,
            debug=False,
        )
        compile_time = time.perf_counter() - t0
        print(f'  Compilation done in {compile_time:.1f}s')
    except Exception as e:
        print(f'  TRT compilation FAILED: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return

    # Benchmark TRT
    print(f'\n=== Benchmark TRT {args.precision} ===')
    with torch.inference_mode():
        trt_ms = benchmark(
            lambda: trt_model(*wrapper_inputs),
            name=f'decoder TRT {args.precision}')
    print(f'  Speedup: {baseline_ms/trt_ms:.1f}x  '
          f'({baseline_ms:.1f}ms -> {trt_ms:.1f}ms)')

    # Save
    print(f'\n=== Save to {args.output} ===')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    try:
        torch_tensorrt.save(trt_model, args.output, inputs=list(wrapper_inputs))
        size_mb = os.path.getsize(args.output) / 1e6
        print(f'  Saved: {size_mb:.0f}MB')

        # Test load
        print('  Testing load...')
        loaded_ep = torch.export.load(args.output)
        loaded = loaded_ep.module()
        with torch.inference_mode():
            loaded_ms = benchmark(
                lambda: loaded(*wrapper_inputs),
                name=f'loaded TRT {args.precision}')
        print(f'  Load test OK: {loaded_ms:.1f}ms')

    except Exception as e:
        print(f'  Save/load failed: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()

    print('\n=== Summary ===')
    print(f'  PyTorch:    {baseline_ms:.1f}ms')
    print(f'  TRT {args.precision}: {trt_ms:.1f}ms')
    print(f'  Speedup:    {baseline_ms/trt_ms:.1f}x')
    print(f'  Saved to:   {args.output}')
    print()
    print('  To use in sam3_node.py, the node auto-detects:')
    print(f'    {os.path.dirname(args.checkpoint)}/decoder_trt_fp16.ep')
    print('  Or set: pytorch_trt_decoder_engine:=/path/to/decoder_trt_fp16.ep')


if __name__ == '__main__':
    main()
