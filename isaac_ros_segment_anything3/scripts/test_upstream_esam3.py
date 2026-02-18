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
Ground-truth test: run upstream Sam3Processor on a test image.

Validates that the upstream EfficientSAM3 pipeline produces expected scores
and masks. Run inside the sam3_pytorch Docker container.

Usage:
    python3 test_upstream_esam3.py \
        --checkpoint /tmp/models/efficient_sam3.pth \
        --image /path/to/image.jpg \
        --prompt "person"

    # Multiple prompts
    python3 test_upstream_esam3.py \
        --checkpoint /tmp/models/efficient_sam3.pth \
        --image /path/to/image.jpg \
        --prompt "person" --prompt "car"
"""

import argparse
import sys
import time

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Test upstream Sam3Processor on an image')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to EfficientSAM3 checkpoint (.pth)')
    parser.add_argument('--image', required=True,
                        help='Path to test image (jpg/png)')
    parser.add_argument('--prompt', action='append', required=True,
                        help='Text prompt(s), can specify multiple')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output', default=None,
                        help='Path to save output mask (optional)')
    parser.add_argument('--model-type', default='efficient_sam3',
                        choices=['sam3', 'efficient_sam3'],
                        help='Model type (default: efficient_sam3)')
    parser.add_argument('--backbone-type', default='tinyvit')
    parser.add_argument('--model-name', default='11m')
    parser.add_argument('--text-encoder-type', default='MobileCLIP-S1')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Load model
    print(f'Loading model from {args.checkpoint} ...')
    try:
        from sam3.model_builder import (
            build_sam3_image_model, build_efficientsam3_image_model)
    except ImportError:
        print('ERROR: sam3 package not found. Install with:')
        print('  pip install git+https://github.com/SimonZeng7108/'
              'efficientsam3.git@77e830355cb164b6bfe18d1f1f3f35d04ef73e70')
        sys.exit(1)

    if args.model_type == 'sam3':
        model = build_sam3_image_model(
            checkpoint_path=args.checkpoint,
            device=args.device,
            load_from_HF=False,
        )
    else:
        model = build_efficientsam3_image_model(
            checkpoint_path=args.checkpoint,
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            text_encoder_type=args.text_encoder_type,
            device=args.device,
        )

    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(
        model, resolution=1008, device=args.device,
        confidence_threshold=args.threshold)

    # Load image (Sam3Processor.set_image uses image.shape[-2:] which
    # expects PIL or CHW tensor, not HWC numpy. Convert to PIL.)
    from PIL import Image as PILImage
    print(f'Loading image: {args.image}')
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f'ERROR: Cannot read image: {args.image}')
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(img_rgb)
    print(f'  Size: {img_pil.width}x{img_pil.height}')

    # Set image
    t0 = time.perf_counter()
    state = processor.set_image(img_pil)
    t_image = time.perf_counter()
    print(f'  set_image: {(t_image - t0)*1000:.1f}ms')

    # Run each prompt
    for prompt in args.prompt:
        print(f'\n--- Prompt: "{prompt}" ---')
        t1 = time.perf_counter()

        # Reset prompts for each new text prompt
        processor.reset_all_prompts(state)
        state = processor.set_image(img_pil, state)
        state = processor.set_text_prompt(prompt, state)

        t2 = time.perf_counter()
        print(f'  set_text_prompt + forward: {(t2 - t1)*1000:.1f}ms')

        # Results
        if 'scores' in state and len(state['scores']) > 0:
            scores = state['scores'].cpu().numpy()
            boxes = state['boxes'].cpu().numpy()
            masks = state['masks'].cpu().numpy()
            print(f'  Detections: {len(scores)}')
            print(f'  Scores: {scores}')
            print(f'  Boxes (xyxy): {boxes}')
            print(f'  Mask shapes: {masks.shape}')

            # Save overlay if requested
            if args.output and len(scores) > 0:
                overlay = img_rgb.copy()
                for i in range(len(scores)):
                    mask = masks[i, 0] if masks.ndim == 4 else masks[i]
                    color = np.array([255, 50, 50], dtype=np.uint8)
                    overlay[mask > 0] = (
                        0.5 * overlay[mask > 0] + 0.5 * color).astype(np.uint8)
                    # Draw box
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(overlay, f'{prompt}: {scores[i]:.3f}',
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
                out_path = args.output
                cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f'  Saved overlay: {out_path}')
        else:
            print('  No detections above threshold')

        # Also print raw model outputs for debugging
        print('\n  Raw model outputs (before thresholding):')
        # Re-run to get raw outputs
        processor.reset_all_prompts(state)
        state = processor.set_image(img_pil, state)

        # Access model internals
        text_outputs = model.backbone.forward_text([prompt], device=args.device)
        state['backbone_out'].update(text_outputs)
        state['geometric_prompt'] = model._get_dummy_prompt()

        with torch.inference_mode():
            outputs = model.forward_grounding(
                backbone_out=state['backbone_out'],
                find_input=processor.find_stage,
                geometric_prompt=state['geometric_prompt'],
                find_target=None,
            )

        pred_logits = outputs['pred_logits'].cpu().numpy()  # [B, 200, 1]
        pred_boxes = outputs['pred_boxes'].cpu().numpy()     # [B, 200, 4]
        presence = outputs['presence_logit_dec'].cpu().numpy()  # [B, 1]

        det_scores = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        pres_scores = 1.0 / (1.0 + np.exp(-presence.astype(np.float64)))
        final_scores = (det_scores.squeeze(-1) * pres_scores)

        print(f'  pred_logits shape: {pred_logits.shape}')
        print(f'  presence shape: {presence.shape}')
        print(f'  det_scores max: {det_scores.max():.4f}')
        print(f'  presence_scores max: {pres_scores.max():.4f}')
        print(f'  final_scores max: {final_scores.max():.4f}')
        print(f'  top-5 final: {np.sort(final_scores.flatten())[-5:]}')
        print(f'  top-5 presence: {np.sort(pres_scores.flatten())[-5:]}')
        print(f'  top-5 det: {np.sort(det_scores.flatten())[-5:]}')

    print('\nDone.')


if __name__ == '__main__':
    main()
