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
Side-by-side comparison: upstream Sam3Processor vs our wrapper pipeline.

Runs the SAME image through both paths and compares outputs at each stage
to identify discrepancies. Run inside the sam3_pytorch Docker container.

Usage:
    python3 test_pipeline_comparison.py \
        --checkpoint /tmp/models/efficient_sam3.pth \
        --image /path/to/image.jpg \
        --prompt "person"
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch


def compare_tensors(name, t1, t2, rtol=1e-4, atol=1e-5):
    """Compare two tensors and print differences."""
    if isinstance(t1, torch.Tensor):
        t1 = t1.detach().cpu().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.detach().cpu().numpy()

    if t1.shape != t2.shape:
        print(f'  {name}: SHAPE MISMATCH! {t1.shape} vs {t2.shape}')
        return False

    # Boolean arrays: use exact comparison
    if t1.dtype == bool or t2.dtype == bool:
        match = np.array_equal(t1, t2)
        mismatches = int(np.sum(t1 != t2))
        status = 'OK' if match else 'MISMATCH'
        print(f'  {name}: {status}  '
              f'mismatches={mismatches}/{t1.size}  '
              f'shape={t1.shape}  dtype={t1.dtype}')
        return match

    max_diff = np.max(np.abs(t1 - t2))
    mean_diff = np.mean(np.abs(t1 - t2))
    match = np.allclose(t1, t2, rtol=rtol, atol=atol)

    status = 'OK' if match else 'MISMATCH'
    print(f'  {name}: {status}  '
          f'max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  '
          f'shape={t1.shape}')
    if not match:
        print(f'    t1 range: [{t1.min():.4f}, {t1.max():.4f}]')
        print(f'    t2 range: [{t2.min():.4f}, {t2.max():.4f}]')
    return match


def main():
    parser = argparse.ArgumentParser(
        description='Compare upstream vs wrapper pipeline')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--prompt', default='person')
    parser.add_argument('--backbone-type', default='tinyvit')
    parser.add_argument('--model-name', default='11m')
    parser.add_argument('--text-encoder-type', default='MobileCLIP-S1')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = args.device

    # --- Load model ---
    print('Loading model...')
    from sam3 import build_efficientsam3_image_model
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        device=device,
    )

    # --- Load image ---
    # Sam3Processor.set_image uses image.shape[-2:] which expects PIL or CHW,
    # not HWC numpy. Use PIL for upstream, numpy for our wrapper.
    from PIL import Image as PILImage
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f'ERROR: Cannot read image: {args.image}')
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(img_rgb)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f'Image: {orig_w}x{orig_h}')

    # ====================================================================
    # PATH A: Upstream Sam3Processor
    # ====================================================================
    print('\n' + '=' * 60)
    print('PATH A: Upstream Sam3Processor')
    print('=' * 60)

    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model, resolution=1008, device=device,
                              confidence_threshold=0.0)  # threshold=0 to see all

    state = processor.set_image(img_pil)
    text_outputs_a = model.backbone.forward_text([args.prompt], device=device)
    state['backbone_out'].update(text_outputs_a)
    state['geometric_prompt'] = model._get_dummy_prompt()

    with torch.inference_mode():
        outputs_a = model.forward_grounding(
            backbone_out=state['backbone_out'],
            find_input=processor.find_stage,
            geometric_prompt=state['geometric_prompt'],
            find_target=None,
        )

    pred_logits_a = outputs_a['pred_logits']      # [B, 200, 1]
    pred_boxes_a = outputs_a['pred_boxes']         # [B, 200, 4]
    pred_masks_a = outputs_a['pred_masks']         # [B, 200, 288, 288]
    presence_a = outputs_a['presence_logit_dec']   # [B, 200]

    print(f'  pred_logits: {pred_logits_a.shape}')
    print(f'  pred_boxes: {pred_boxes_a.shape}')
    print(f'  pred_masks: {pred_masks_a.shape}')
    print(f'  presence: {presence_a.shape}')

    det_a = pred_logits_a.sigmoid().squeeze(-1)  # [B, 200]
    pres_a = presence_a.sigmoid()                 # [B, 200]
    scores_a = det_a * pres_a
    print(f'  det_scores max: {det_a.max():.4f}')
    print(f'  presence_scores max: {pres_a.max():.4f}')
    print(f'  final_scores max: {scores_a.max():.4f}')
    print(f'  detections > 0.5: {(scores_a > 0.5).sum().item()}')

    # ====================================================================
    # PATH B: Our wrapper pipeline (from export_efficient_sam3.py)
    # ====================================================================
    print('\n' + '=' * 60)
    print('PATH B: Our wrapper pipeline')
    print('=' * 60)

    # Import our wrappers
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from export_efficient_sam3 import (
        VisionEncoderWrapper, TextEncoderWrapper, DecoderWrapper)

    vision_wrapper = VisionEncoderWrapper(
        model.backbone.vision_backbone,
        scalp=model.backbone.scalp,
    ).to(device).eval()

    text_wrapper = TextEncoderWrapper(
        model.backbone.language_backbone,
    ).to(device).eval()

    decoder_wrapper = DecoderWrapper(model).to(device).eval()

    # --- Preprocess (our way: OpenCV + numpy) ---
    sz = 1008
    resized = cv2.resize(img_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
    img_f = resized.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img_f = (img_f - mean) / std
    img_np = img_f.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    image_tensor_b = torch.from_numpy(img_np).to(device)

    # --- Preprocess (upstream way: torchvision transforms) ---
    from torchvision.transforms import v2
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(sz, sz)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor_a = v2.functional.to_image(img_rgb).to(device)
    image_tensor_a = transform(image_tensor_a).unsqueeze(0)

    print('\n--- Preprocessing comparison ---')
    compare_tensors('image_tensor', image_tensor_a, image_tensor_b,
                    rtol=1e-2, atol=1e-2)

    # --- Vision encoder ---
    print('\n--- Vision encoder ---')
    with torch.inference_mode():
        fpn_b = vision_wrapper(image_tensor_b)

    # Get upstream FPN from backbone_out.
    # forward_image() already applies scalp, so backbone_fpn has 3 features.
    backbone_fpn_a = state['backbone_out']['backbone_fpn']
    vision_pos_a = state['backbone_out']['vision_pos_enc']

    print(f'  Upstream FPN levels: {len(backbone_fpn_a)}, '
          f'Wrapper outputs: {len(fpn_b)}')
    num_fpn = min(len(backbone_fpn_a), len(fpn_b) - 1)  # -1 for pos encoding
    for i in range(num_fpn):
        compare_tensors(f'fpn_feat_{i}', backbone_fpn_a[i], fpn_b[i])

    # --- Text encoder ---
    print('\n--- Text encoder ---')
    from tokenizers import Tokenizer
    tokenizer_path = '/tmp/models/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        # Try to find it
        for p in ['/tmp/models/tokenizer.json',
                  '/workspace/models/tokenizer.json']:
            if os.path.exists(p):
                tokenizer_path = p
                break

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding(length=32, pad_id=0, pad_token='[PAD]')
    tokenizer.enable_truncation(max_length=32)

    enc = tokenizer.encode(args.prompt)
    input_ids = np.zeros((1, 32), dtype=np.int64)
    input_ids[0, :len(enc.ids)] = enc.ids[:32]
    input_ids_tensor = torch.from_numpy(input_ids).to(device)

    with torch.inference_mode():
        text_feats_b, text_mask_b = text_wrapper(input_ids_tensor)

    # Upstream text features
    text_feats_a = text_outputs_a['language_features']   # [32, B, 256]
    text_mask_a = text_outputs_a['language_mask']          # [B, 32]

    # Our wrapper outputs batch-first [B, 32, 256], upstream is [32, B, 256]
    text_feats_b_seqfirst = text_feats_b.transpose(0, 1)

    print(f'  Upstream text_features shape: {text_feats_a.shape}')
    print(f'  Wrapper text_features shape (seq-first): {text_feats_b_seqfirst.shape}')
    compare_tensors('text_features', text_feats_a, text_feats_b_seqfirst)
    compare_tensors('text_mask', text_mask_a, text_mask_b)

    # --- Decoder ---
    print('\n--- Decoder ---')
    with torch.inference_mode():
        # Use upstream pos encoding for fair comparison (last level)
        pos_enc = vision_pos_a[-1] if len(vision_pos_a) > 0 else fpn_b[3]
        decoder_out_b = decoder_wrapper(
            fpn_b[0], fpn_b[1], fpn_b[2],
            pos_enc,
            text_feats_b, text_mask_b)

    masks_b, boxes_b, logits_b, presence_b = decoder_out_b

    print(f'  Upstream pred_logits: {pred_logits_a.shape}')
    print(f'  Wrapper pred_logits: {logits_b.shape}')
    compare_tensors('pred_logits', pred_logits_a, logits_b, rtol=1e-2, atol=1e-2)
    compare_tensors('pred_boxes', pred_boxes_a, boxes_b, rtol=1e-2, atol=1e-2)
    compare_tensors('presence', presence_a, presence_b, rtol=1e-2, atol=1e-2)

    # Score comparison (decoder_wrapper returns tensors, not numpy)
    logits_b_t = logits_b if isinstance(logits_b, torch.Tensor) \
        else torch.from_numpy(logits_b)
    presence_b_t = presence_b if isinstance(presence_b, torch.Tensor) \
        else torch.from_numpy(presence_b)
    det_b = logits_b_t.sigmoid().squeeze(-1)
    pres_b_sig = presence_b_t.sigmoid()
    scores_b = det_b * pres_b_sig
    print(f'\n  Wrapper det_scores max: {det_b.max():.4f}')
    print(f'  Wrapper presence_scores max: {pres_b_sig.max():.4f}')
    print(f'  Wrapper final_scores max: {scores_b.max():.4f}')
    print(f'  Wrapper detections > 0.5: {(scores_b > 0.5).sum().item()}')

    # ====================================================================
    # Summary
    # ====================================================================
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Upstream max final score: {scores_a.max():.4f}')
    print(f'  Wrapper max final score:  {scores_b.max():.4f}')
    print(f'  Upstream detections > 0.5: {(scores_a > 0.5).sum().item()}')
    print(f'  Wrapper detections > 0.5: {(scores_b > 0.5).sum().item()}')

    if scores_a.max() > 0.5 and scores_b.max() < 0.5:
        print('\n  *** UPSTREAM WORKS, WRAPPER FAILS ***')
        print('  The wrapper pipeline has a bug causing low scores.')
    elif scores_a.max() < 0.5 and scores_b.max() < 0.5:
        print('\n  Neither pipeline detects above threshold.')
        print('  This may be a model limitation for this prompt/image.')
    elif abs(float(scores_a.max() - scores_b.max())) < 0.01:
        print('\n  Scores match closely. Pipelines are equivalent.')
    else:
        print('\n  Scores differ. Investigate intermediate stages above.')


if __name__ == '__main__':
    main()
