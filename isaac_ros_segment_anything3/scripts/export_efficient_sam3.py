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
Export EfficientSAM3 PyTorch models to ONNX for Triton Inference Server.

Exports three separate ONNX models matching the SAM3 pipeline interface:
  1. Vision encoder  (image -> FPN features + position encodings)
  2. Text encoder    (input_ids -> text_features + text_mask)
  3. Decoder         (FPN + text -> masks + boxes + logits)

Prerequisites:
  git clone https://github.com/SimonZeng7108/efficientsam3.git
  cd efficientsam3 && pip install -e .

  Download checkpoint from HuggingFace:
    https://huggingface.co/Simon7108528/EfficientSAM3

Usage:
  python3 export_efficient_sam3.py \\
    --checkpoint /path/to/efficient_sam3_tinyvit_11m_mobileclip_s1.pth \\
    --backbone-type tinyvit --model-name 11m \\
    --text-encoder-type MobileCLIP-S1 \\
    --output-dir /tmp/esam3_models

  python3 export_efficient_sam3.py --inspect-only \\
    --checkpoint /path/to/checkpoint.pth \\
    --backbone-type tinyvit --model-name 11m

After export, use download_models.py to set up the Triton repository:
  python3 download_models.py --model-type efficient_sam3 \\
    --local-models /tmp/esam3_models --model-repo /tmp/models
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


# EfficientSAM3 uses 1008x1008 (same as SAM3 teacher)
_IMAGE_SIZE = 1008
_MAX_SEQ_LEN = 32


def load_model(checkpoint_path, backbone_type, model_name,
               text_encoder_type=None):
    """Load an EfficientSAM3 model from checkpoint."""
    try:
        from sam3.model_builder import build_efficientsam3_image_model
    except ImportError:
        print('ERROR: EfficientSAM3 package not found.')
        print('Install with:')
        print('  git clone https://github.com/SimonZeng7108/efficientsam3')
        print('  cd efficientsam3 && pip install -e .')
        sys.exit(1)

    kwargs = {
        'checkpoint_path': checkpoint_path,
        'backbone_type': backbone_type,
        'model_name': model_name,
        'device': 'cpu',
        'compile': False,
    }
    if text_encoder_type:
        kwargs['text_encoder_type'] = text_encoder_type

    print(f'Loading model: backbone={backbone_type}, '
          f'model={model_name}, text_enc={text_encoder_type}')
    model = build_efficientsam3_image_model(**kwargs)
    model.eval()
    return model


def inspect_model(model):
    """Print model structure and submodule names for debugging."""
    print('\n' + '=' * 60)
    print('Model Structure Inspection')
    print('=' * 60)

    total = 0
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        total += param_count
        print(f'  {name:30s} params={param_count:>12,}')
    print(f'  {"TOTAL":30s} params={total:>12,}')

    print(f'\nVision backbone: {type(model.backbone.vision_backbone).__name__}')
    print(f'Language backbone: {type(model.backbone.language_backbone).__name__}')
    print(f'Scalp: {model.backbone.scalp}')

    # Test vision encoder
    print('\nVision encoder forward test:')
    dummy = torch.randn(1, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    with torch.no_grad():
        vout = model.backbone.forward_image(dummy)
    for i, f in enumerate(vout['backbone_fpn']):
        print(f'  fpn[{i}]: {f.shape}')
    for i, p in enumerate(vout['vision_pos_enc']):
        print(f'  pos[{i}]: {p.shape}')

    # Test text encoder
    print('\nText encoder forward test:')
    with torch.no_grad():
        tout = model.backbone.forward_text(['person'], device='cpu')
    print(f'  language_features: {tout["language_features"].shape}')
    print(f'  language_mask: {tout["language_mask"].shape}')


# ---------------------------------------------------------------
# ONNX Wrapper Modules
# ---------------------------------------------------------------

class VisionEncoderWrapper(nn.Module):
    """
    Wraps model.backbone.vision_backbone (Sam3DualViTDetNeck) for ONNX export.

    The neck contains:
      - trunk: student backbone (TinyViT/RepViT/EfficientViT) wrapped with
               ImageStudentEncoder (projects to 1024 channels, 72x72 spatial)
      - convs: SimpleFPN producing 4 scale levels
      - position_encoding: PositionEmbeddingSine

    After scalp=1 (removes lowest-res level), outputs 3 FPN levels + 3 pos.
    """

    def __init__(self, vision_backbone, scalp=1):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.scalp = scalp

    def forward(self, images):
        # Sam3DualViTDetNeck.forward() returns:
        #   (sam3_features, sam3_pos, sam2_features, sam2_pos)
        # sam2 is None for student models (no dual neck)
        sam3_features, sam3_pos, _, _ = self.vision_backbone(images)

        # Apply scalp: remove lowest resolution level
        if self.scalp > 0:
            sam3_features = sam3_features[:-self.scalp]
            sam3_pos = sam3_pos[:-self.scalp]

        # Return 3 FPN features + last position encoding
        # Matches SAM3 Triton interface: fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2
        return (
            sam3_features[0],   # [1, 256, 288, 288]
            sam3_features[1],   # [1, 256, 144, 144]
            sam3_features[2],   # [1, 256, 72, 72]
            sam3_pos[2],        # [1, 256, 72, 72]
        )


class TextEncoderWrapper(nn.Module):
    """
    Wraps TextStudentEncoder for ONNX export, splitting out tokenization.

    The original forward() takes raw strings and tokenizes internally.
    This wrapper takes pre-tokenized input_ids (int64) and runs:
      1. encoder.forward_embedding(input_ids) -> [B, 32, dim]
      2. encoder(embeds, return_all_tokens=True, input_is_embeddings=True)
      3. projector(text_memory) -> [B, 32, 256]

    Output is batch-first [B, 32, 256] to match the Triton interface.
    """

    def __init__(self, language_backbone):
        super().__init__()
        self.encoder = language_backbone.encoder
        self.projector = language_backbone.projector

    def forward(self, input_ids):
        # 1. Embedding lookup
        input_embeds = self.encoder.forward_embedding(input_ids)

        # 2. MobileCLIP transformer
        text_memory = self.encoder(
            input_embeds, return_all_tokens=True, input_is_embeddings=True)

        # 3. Project to output dim (256)
        text_features = self.projector(text_memory)  # [B, 32, 256]

        # 4. Compute mask: True for padding tokens (token_id == 0)
        text_mask = (input_ids == 0)  # [B, 32] bool

        return text_features, text_mask


class DecoderWrapper(nn.Module):
    """
    Wraps the full decoder pipeline for ONNX export.

    Bundles: geometry_encoder + transformer (encoder+decoder) +
             dot_prod_scoring + segmentation_head

    This is a text-only decoder (no box/point prompts).
    For text-only inference, the geometry encoder produces only a CLS token.

    Input: FPN features + text_features [B, 32, 256] + text_mask [B, 32]
    Output: pred_masks, pred_boxes, pred_logits, presence_logits
    """

    def __init__(self, model):
        super().__init__()
        # Extract decoder components from the Sam3Image model
        self.geometry_encoder = model.geometry_encoder
        self.transformer = model.transformer
        self.dot_prod_scoring = model.dot_prod_scoring
        self.segmentation_head = model.segmentation_head
        # Copy config attributes
        self.use_dot_prod_scoring = model.use_dot_prod_scoring
        self.supervise_joint_box_scores = model.supervise_joint_box_scores
        self.detach_presence_in_joint_score = model.detach_presence_in_joint_score

        # Fix device mismatch: decoder pre-caches boxRPB coords on cuda
        # during __init__. Move them to CPU for export.
        dec = self.transformer.decoder
        if dec.compilable_cord_cache is not None:
            dec.compilable_cord_cache = tuple(
                t.cpu() for t in dec.compilable_cord_cache)
        dec.coord_cache = {}  # Clear dict cache

        # Monkey-patch _get_rpb_matrix to avoid data-dependent guards
        # that break torch.export (the `compilable_stored_size == (H, W)`
        # comparison is data-dependent).  We always use the compilable
        # cache with fixed 72x72 feature size.
        import types
        _orig_get_rpb = dec._get_rpb_matrix.__func__

        def _patched_get_rpb(self_dec, reference_boxes, feat_size):
            from sam3.model.box_ops import box_cxcywh_to_xyxy
            import numpy as np
            H, W = feat_size
            boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
            bs, num_queries, _ = boxes_xyxy.shape

            # Always use pre-computed coords (72x72)
            if self_dec.compilable_cord_cache is None:
                self_dec.compilable_cord_cache = self_dec._get_coords(
                    H, W, reference_boxes.device)
            coords_h, coords_w = self_dec.compilable_cord_cache
            # Ensure coords are on the same device as input
            if coords_h.device != reference_boxes.device:
                coords_h = coords_h.to(reference_boxes.device)
                coords_w = coords_w.to(reference_boxes.device)
                self_dec.compilable_cord_cache = (coords_h, coords_w)

            deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(
                -1, 1, 4)[:, :, 1:4:2]
            deltas_y = deltas_y.view(bs, num_queries, -1, 2)
            deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(
                -1, 1, 4)[:, :, 0:3:2]
            deltas_x = deltas_x.view(bs, num_queries, -1, 2)

            if self_dec.boxRPB in ["log", "both"]:
                deltas_x_log = deltas_x * 8
                deltas_x_log = (
                    torch.sign(deltas_x_log)
                    * torch.log2(torch.abs(deltas_x_log) + 1.0)
                    / np.log2(8))
                deltas_y_log = deltas_y * 8
                deltas_y_log = (
                    torch.sign(deltas_y_log)
                    * torch.log2(torch.abs(deltas_y_log) + 1.0)
                    / np.log2(8))
                if self_dec.boxRPB == "log":
                    deltas_x = deltas_x_log
                    deltas_y = deltas_y_log
                else:
                    deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                    deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)

            # MLP: [bs, nq, H or W, n_input] -> [bs, nq, H or W, nheads]
            deltas_x = self_dec.boxRPB_embed_x(deltas_x)
            deltas_y = self_dec.boxRPB_embed_y(deltas_y)

            # Outer sum: [bs, nq, H, nheads] + [bs, nq, W, nheads]
            # -> [bs, nq, H, W, nheads]
            B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
            B = B.flatten(2, 3)   # [bs, nq, H*W, nheads]
            B = B.permute(0, 3, 1, 2)  # [bs, nheads, nq, H*W]
            B = B.contiguous()
            return B

        dec._get_rpb_matrix = types.MethodType(_patched_get_rpb, dec)

    def forward(self, fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2,
                text_features, text_mask):
        """
        Full decoder pipeline for text-only grounding.

        Args:
            fpn_feat_0: [B, 256, 288, 288] - highest res FPN feature
            fpn_feat_1: [B, 256, 144, 144] - mid res FPN feature
            fpn_feat_2: [B, 256, 72, 72]   - lowest res FPN feature (used by encoder)
            fpn_pos_2:  [B, 256, 72, 72]   - position encoding for level 2
            text_features: [B, 32, 256]     - text encoder output (batch-first)
            text_mask: [B, 32]              - True for padding tokens

        Returns:
            pred_masks: [B, 200, 288, 288]
            pred_boxes: [B, 200, 4] (cxcywh normalized)
            pred_logits: [B, 200, 1]
            presence_logits: [B, 200]
        """
        from sam3.model.geometry_encoders import Prompt
        from sam3.model.model_misc import inverse_sigmoid
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        B = text_features.shape[0]
        device = text_features.device

        # Convert text_features to seq-first: [32, B, 256]
        txt_feats = text_features.transpose(0, 1)

        # Prepare image features for encoder (only level 2, flattened)
        # [B, 256, 72, 72] -> [5184, B, 256]
        img_feat = fpn_feat_2.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        img_pos = fpn_pos_2.flatten(2).permute(2, 0, 1)    # [HW, B, C]

        # Geometry encoder: text-only -> just CLS token
        # cls_embed is nn.Embedding(1, 256)
        geo_cls = self.geometry_encoder.cls_embed.weight.unsqueeze(1)  # [1, 1, 256]
        geo_cls = geo_cls.expand(1, B, -1)  # [1, B, 256]
        geo_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

        # Concatenate prompt: text [32, B, 256] + geo CLS [1, B, 256] = [33, B, 256]
        prompt = torch.cat([txt_feats, geo_cls], dim=0)
        prompt_mask = torch.cat([text_mask, geo_mask], dim=1)

        # Run transformer encoder (6 layers, image-prompt cross-attention)
        prompt_pos = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=[img_feat],
            src_key_padding_mask=None,
            src_pos=[img_pos],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=[(72, 72)],
        )

        encoder_hidden = memory['memory']      # [5184, B, 256]
        encoder_pos = memory['pos_embed']      # [5184, B, 256]
        padding_mask = memory.get('padding_mask', None)

        # Run transformer decoder (6 layers, 200 queries)
        query_embed = self.transformer.decoder.query_embed.weight  # [200, 256]
        tgt = query_embed.unsqueeze(1).expand(-1, B, -1)           # [200, B, 256]

        hs, reference_boxes, dec_presence_out, _ = self.transformer.decoder(
            tgt=tgt,
            memory=encoder_hidden,
            memory_key_padding_mask=padding_mask,
            pos=encoder_pos,
            reference_boxes=None,
            level_start_index=memory['level_start_index'],
            spatial_shapes=memory['spatial_shapes'],
            valid_ratios=memory['valid_ratios'],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        hs = hs.transpose(1, 2)                    # [6, B, 200, 256]
        reference_boxes = reference_boxes.transpose(1, 2)  # [6, B, 200, 4]
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)  # [6, B, 200]

        # Score prediction (dot product)
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)

        # Box prediction
        box_head = self.transformer.decoder.bbox_embed
        anchor_offsets = box_head(hs)
        ref_inv = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv + anchor_offsets).sigmoid()

        # Joint scoring with presence logit
        if self.supervise_joint_box_scores and dec_presence_out is not None:
            prob_pres = dec_presence_out.clone().sigmoid()
            if self.detach_presence_in_joint_score:
                prob_pres = prob_pres.detach()
            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_pres.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)

        # Take last decoder layer output
        pred_logits = outputs_class[-1]          # [B, 200, 1]
        pred_boxes = outputs_coord[-1]           # [B, 200, 4]
        presence_logits = dec_presence_out[-1] if dec_presence_out is not None \
            else torch.zeros(B, 200, device=device)

        # Segmentation head: pixel decoder + mask predictor
        backbone_fpn = [fpn_feat_0, fpn_feat_1, fpn_feat_2]
        obj_queries = hs  # [6, B, 200, 256]
        seg_out = self.segmentation_head(
            backbone_feats=backbone_fpn,
            obj_queries=obj_queries,
            image_ids=torch.zeros(B, dtype=torch.long, device=device),
            encoder_hidden_states=encoder_hidden,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        pred_masks = seg_out['pred_masks']  # [B, 200, 288, 288]

        return pred_masks, pred_boxes, pred_logits, presence_logits


# ---------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------

def export_vision_encoder(model, output_path, image_size=_IMAGE_SIZE):
    """Export vision encoder to ONNX."""
    print(f'\nExporting vision encoder to {output_path}')

    wrapper = VisionEncoderWrapper(
        model.backbone.vision_backbone, scalp=model.backbone.scalp)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        outputs = wrapper(dummy_input)
    print(f'  Forward pass OK. Output shapes:')
    names = ['fpn_feat_0', 'fpn_feat_1', 'fpn_feat_2', 'fpn_pos_2']
    for name, out in zip(names, outputs):
        print(f'    {name}: {out.shape}')

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        input_names=['images'],
        output_names=names,
        dynamic_axes={n: {0: 'batch'} for n in ['images'] + names},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f'  Exported: {output_path}')
    return output_path


def export_text_encoder(model, output_path, max_seq_len=_MAX_SEQ_LEN):
    """Export text encoder to ONNX."""
    print(f'\nExporting text encoder to {output_path}')

    wrapper = TextEncoderWrapper(model.backbone.language_backbone)
    wrapper.eval()

    dummy_ids = torch.randint(0, 100, (1, max_seq_len), dtype=torch.long)

    with torch.no_grad():
        text_features, text_mask = wrapper(dummy_ids)
    print(f'  Forward pass OK.')
    print(f'    text_features: {text_features.shape}')
    print(f'    text_mask: {text_mask.shape}')

    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        output_path,
        input_names=['input_ids'],
        output_names=['text_features', 'text_mask'],
        dynamic_axes={
            'input_ids': {0: 'num_prompts'},
            'text_features': {0: 'num_prompts'},
            'text_mask': {0: 'num_prompts'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f'  Exported: {output_path}')
    return output_path


def export_decoder(model, output_path, max_seq_len=_MAX_SEQ_LEN):
    """Export decoder to ONNX."""
    print(f'\nExporting decoder to {output_path}')

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    B = 1
    dummy_fpn_0 = torch.randn(B, 256, 288, 288)
    dummy_fpn_1 = torch.randn(B, 256, 144, 144)
    dummy_fpn_2 = torch.randn(B, 256, 72, 72)
    dummy_pos_2 = torch.randn(B, 256, 72, 72)
    dummy_text = torch.randn(B, max_seq_len, 256)
    dummy_mask = torch.zeros(B, max_seq_len, dtype=torch.bool)

    with torch.no_grad():
        outputs = wrapper(
            dummy_fpn_0, dummy_fpn_1, dummy_fpn_2, dummy_pos_2,
            dummy_text, dummy_mask)
    print(f'  Forward pass OK. Output shapes:')
    out_names = ['pred_masks', 'pred_boxes', 'pred_logits', 'presence_logits']
    for name, out in zip(out_names, outputs):
        print(f'    {name}: {out.shape}')

    # Use legacy (JIT trace) exporter for decoder because the new
    # torch.export-based exporter can't handle some ops in the
    # transformer decoder (non-contiguous views in attention, boxRPB).
    torch.onnx.export(
        wrapper,
        (dummy_fpn_0, dummy_fpn_1, dummy_fpn_2, dummy_pos_2,
         dummy_text, dummy_mask),
        output_path,
        input_names=[
            'fpn_feat_0', 'fpn_feat_1', 'fpn_feat_2', 'fpn_pos_2',
            'prompt_features', 'prompt_mask',
        ],
        output_names=out_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f'  Exported: {output_path}')
    return output_path


def verify_onnx(model_path):
    """Verify exported ONNX model with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f'  [WARN] onnxruntime not installed, skipping verification')
        return True

    print(f'  Verifying: {model_path}')
    try:
        session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        for inp in session.get_inputs():
            print(f'    Input:  {inp.name:25s} shape={inp.shape}  dtype={inp.type}')
        for out in session.get_outputs():
            print(f'    Output: {out.name:25s} shape={out.shape}  dtype={out.type}')
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f'    Size: {size_mb:.1f} MB')
        return True
    except Exception as e:
        print(f'  [FAIL] Verification failed: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Export EfficientSAM3 models to ONNX for Triton')
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to EfficientSAM3 checkpoint (.pth)')
    parser.add_argument(
        '--backbone-type', required=True,
        choices=['repvit', 'tinyvit', 'efficientvit'],
        help='Backbone architecture type')
    parser.add_argument(
        '--model-name', required=True,
        help='Model variant (e.g., m0.9, m1.1, m2.3 for repvit; '
             '5m, 11m, 21m for tinyvit; b0, b1, b2 for efficientvit)')
    parser.add_argument(
        '--text-encoder-type', default=None,
        choices=['MobileCLIP-S0', 'MobileCLIP-S1', 'MobileCLIP2-L'],
        help='Text encoder variant (omit for image-only model)')
    parser.add_argument(
        '--output-dir', default='/tmp/esam3_models',
        help='Output directory for ONNX models')
    parser.add_argument(
        '--image-size', type=int, default=_IMAGE_SIZE,
        help=f'Input image size (default: {_IMAGE_SIZE})')
    parser.add_argument(
        '--inspect-only', action='store_true',
        help='Only inspect model structure, do not export')
    parser.add_argument(
        '--skip-decoder', action='store_true',
        help='Skip decoder export (complex, may fail with some ops)')
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='Skip ONNX verification after export')
    args = parser.parse_args()

    # Load model
    model = load_model(
        args.checkpoint, args.backbone_type, args.model_name,
        args.text_encoder_type)

    if args.inspect_only:
        inspect_model(model)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    inspect_model(model)

    # Export vision encoder
    vision_path = os.path.join(args.output_dir, 'vision-encoder.onnx')
    try:
        export_vision_encoder(model, vision_path, args.image_size)
        if not args.skip_verify:
            verify_onnx(vision_path)
    except Exception as e:
        print(f'\n  [FAIL] Vision encoder export failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Export text encoder
    if args.text_encoder_type:
        text_path = os.path.join(args.output_dir, 'text-encoder.onnx')
        try:
            export_text_encoder(model, text_path)
            if not args.skip_verify:
                verify_onnx(text_path)
        except Exception as e:
            print(f'\n  [FAIL] Text encoder export failed: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('\n  [SKIP] No text encoder (image-only model)')

    # Export decoder
    if not args.skip_decoder:
        decoder_path = os.path.join(args.output_dir, 'decoder.onnx')
        try:
            export_decoder(model, decoder_path)
            if not args.skip_verify:
                verify_onnx(decoder_path)
        except Exception as e:
            print(f'\n  [FAIL] Decoder export failed: {e}')
            import traceback
            traceback.print_exc()
            print('\n  Decoder export is complex. Try --skip-decoder to '
                  'export vision + text only.')
    else:
        print('\n  [SKIP] Decoder export skipped (--skip-decoder)')

    print('\n' + '=' * 60)
    print('Export complete!')
    print(f'Output directory: {args.output_dir}')
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.onnx'):
            size = os.path.getsize(os.path.join(args.output_dir, f))
            print(f'  {f}: {size/1024/1024:.1f} MB')


if __name__ == '__main__':
    main()
