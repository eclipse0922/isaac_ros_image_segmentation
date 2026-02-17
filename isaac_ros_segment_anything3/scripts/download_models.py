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
Download SAM3/EfficientSAM3 models and set up inference backend.

Usage:
    # SAM3 (default, Triton ONNX)
    python3 download_models.py --model-repo /tmp/models

    # EfficientSAM3 Triton (from local export)
    python3 download_models.py --model-type efficient_sam3 \
        --local-models /tmp/esam3_models --model-repo /tmp/models

    # EfficientSAM3 PyTorch (direct CUDA, ~16x faster)
    python3 download_models.py --model-type efficient_sam3 \
        --inference-backend pytorch --model-repo /tmp/models \
        --pytorch-checkpoint /path/to/efficient_sam3.pth

    # Verify only
    python3 download_models.py --model-repo /tmp/models --verify-only
"""

import argparse
import os
import shutil
import sys
import urllib.request

# SAM3 ONNX model URLs from jamjamjon/assets
SAM3_MODELS = {
    'vision_encoder': {
        'url': 'https://github.com/jamjamjon/assets/releases/download/sam3/vision-encoder.onnx',
        'triton_name': 'sam3_vision_encoder',
    },
    'text_encoder': {
        'url': 'https://github.com/jamjamjon/assets/releases/download/sam3/text-encoder.onnx',
        'triton_name': 'sam3_text_encoder',
    },
    'decoder': {
        'url': 'https://github.com/jamjamjon/assets/releases/download/sam3/decoder.onnx',
        'triton_name': 'sam3_decoder',
    },
}

# EfficientSAM3 Triton model names (ONNX files from export script)
EFFICIENT_SAM3_MODELS = {
    'vision_encoder': {
        'filename': 'vision-encoder.onnx',
        'triton_name': 'esam3_vision_encoder',
    },
    'text_encoder': {
        'filename': 'text-encoder.onnx',
        'triton_name': 'esam3_text_encoder',
    },
    'decoder': {
        'filename': 'decoder.onnx',
        'triton_name': 'esam3_decoder',
    },
}

TOKENIZER_URL = \
    'https://github.com/jamjamjon/assets/releases/download/sam3/tokenizer.json'

# EfficientSAM3 PyTorch checkpoint
ESAM3_PYTORCH_CHECKPOINT_NAME = 'efficient_sam3.pth'

# Mapping from numpy dtype to Triton data type string
NUMPY_TO_TRITON_DTYPE = {
    'float16': 'TYPE_FP16',
    'float32': 'TYPE_FP32',
    'float64': 'TYPE_FP64',
    'int8': 'TYPE_INT8',
    'int16': 'TYPE_INT16',
    'int32': 'TYPE_INT32',
    'int64': 'TYPE_INT64',
    'uint8': 'TYPE_UINT8',
    'uint16': 'TYPE_UINT16',
    'bool': 'TYPE_BOOL',
}


def download_file(url, dest):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'  Already exists: {dest} ({size_mb:.1f} MB)')
        return

    print(f'  Downloading: {url}')
    print(f'  Destination: {dest}')

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f'\r  Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f'\n  Saved: {dest} ({size_mb:.1f} MB)')


def verify_onnx_model(model_path, model_name):
    """
    Verify an ONNX model and return its input/output specifications.

    Returns a dict with 'inputs' and 'outputs' lists, each containing
    {'name': str, 'shape': list, 'dtype': str}.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print('  [WARN] onnxruntime not installed. Skipping verification.')
        print('  Install with: pip install onnxruntime-gpu')
        return None

    print(f'\n  Verifying: {model_name} ({model_path})')

    try:
        session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f'  [FAIL] Could not load model: {e}')
        return None

    # Convert ONNX type string to numpy-like dtype
    type_map = {
        'tensor(float)': 'float32',
        'tensor(float16)': 'float16',
        'tensor(double)': 'float64',
        'tensor(int32)': 'int32',
        'tensor(int64)': 'int64',
        'tensor(int8)': 'int8',
        'tensor(uint8)': 'uint8',
        'tensor(bool)': 'bool',
    }

    inputs = []
    for inp in session.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str):
                shape.append(-1)
            else:
                shape.append(dim)
        dtype = inp.type
        numpy_dtype = type_map.get(dtype, dtype)
        inputs.append({
            'name': inp.name,
            'shape': shape,
            'dtype': numpy_dtype,
        })
        print(f'    Input:  {inp.name:30s} shape={shape} dtype={numpy_dtype}')

    outputs = []
    for out in session.get_outputs():
        shape = []
        for dim in out.shape:
            if isinstance(dim, str):
                shape.append(-1)
            else:
                shape.append(dim)
        dtype = out.type
        numpy_dtype = type_map.get(dtype, dtype)
        outputs.append({
            'name': out.name,
            'shape': shape,
            'dtype': numpy_dtype,
        })
        print(f'    Output: {out.name:30s} shape={shape} dtype={numpy_dtype}')

    return {'inputs': inputs, 'outputs': outputs}


def generate_config_pbtxt(triton_name, spec):
    """Generate a Triton config.pbtxt from the ONNX model specification."""
    lines = []
    lines.append(f'name: "{triton_name}"')
    lines.append('platform: "onnxruntime_onnx"')
    lines.append('max_batch_size: 0')

    lines.append('input [')
    for i, inp in enumerate(spec['inputs']):
        triton_dtype = NUMPY_TO_TRITON_DTYPE.get(inp['dtype'], 'TYPE_FP32')
        dims_str = ', '.join(str(d) for d in inp['shape'])
        lines.append('  {')
        lines.append(f'    name: "{inp["name"]}"')
        lines.append(f'    data_type: {triton_dtype}')
        lines.append(f'    dims: [ {dims_str} ]')
        lines.append('  }' + (',' if i < len(spec['inputs']) - 1 else ''))
    lines.append(']')

    lines.append('output [')
    for i, out in enumerate(spec['outputs']):
        triton_dtype = NUMPY_TO_TRITON_DTYPE.get(out['dtype'], 'TYPE_FP32')
        dims_str = ', '.join(str(d) for d in out['shape'])
        lines.append('  {')
        lines.append(f'    name: "{out["name"]}"')
        lines.append(f'    data_type: {triton_dtype}')
        lines.append(f'    dims: [ {dims_str} ]')
        lines.append('  }' + (',' if i < len(spec['outputs']) - 1 else ''))
    lines.append(']')

    lines.append('instance_group [')
    lines.append('  {')
    lines.append('    count: 1')
    lines.append('    kind: KIND_GPU')
    lines.append('  }')
    lines.append(']')

    lines.append('version_policy: {')
    lines.append('  specific {')
    lines.append('    versions: [ 1 ]')
    lines.append('  }')
    lines.append('}')

    return '\n'.join(lines) + '\n'


def _setup_models(model_repo_path, models_dict, model_type_label,
                  source_dir=None, verify=True):
    """
    Set up Triton model repository for a set of models.

    For SAM3: downloads ONNX models from URLs.
    For EfficientSAM3: copies from local export directory.
    """
    for model_key, model_info in models_dict.items():
        triton_name = model_info['triton_name']
        model_dir = os.path.join(model_repo_path, triton_name, '1')
        os.makedirs(model_dir, exist_ok=True)

        print(f'\n[{model_key}]')
        model_path = os.path.join(model_dir, 'model.onnx')

        if 'url' in model_info:
            # Download from URL (SAM3)
            download_file(model_info['url'], model_path)
        elif source_dir and 'filename' in model_info:
            # Copy from local directory (EfficientSAM3)
            src = os.path.join(source_dir, model_info['filename'])
            if not os.path.exists(src):
                print(f'  [WARN] Source not found: {src}')
                continue
            if not os.path.exists(model_path):
                shutil.copy2(src, model_path)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f'  Copied: {src} -> {model_path} ({size_mb:.1f} MB)')
            else:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f'  Already exists: {model_path} ({size_mb:.1f} MB)')

    if verify:
        print('\n' + '=' * 60)
        print(f'Verifying {model_type_label} models and generating configs...')

        for model_key, model_info in models_dict.items():
            triton_name = model_info['triton_name']
            model_path = os.path.join(
                model_repo_path, triton_name, '1', 'model.onnx')
            config_path = os.path.join(
                model_repo_path, triton_name, 'config.pbtxt')

            if not os.path.exists(model_path):
                print(f'\n  [SKIP] {model_key}: model.onnx not found')
                continue

            spec = verify_onnx_model(model_path, model_key)
            if spec is not None:
                config_content = generate_config_pbtxt(triton_name, spec)
                with open(config_path, 'w') as f:
                    f.write(config_content)
                print(f'  Generated: {config_path}')
            else:
                print(f'  [WARN] Skipping config generation for {model_key}')


def setup_pytorch_backend(model_repo_path, checkpoint_src):
    """
    Set up EfficientSAM3 PyTorch backend.

    Copies the .pth checkpoint into the model repository and downloads
    the tokenizer. Only tokenizer + .pth are needed (no Triton/ONNX).

    Args:
        model_repo_path: Destination directory for models.
        checkpoint_src: Path to EfficientSAM3 .pth checkpoint file.
    """
    print('Setting up EfficientSAM3 PyTorch backend')
    print('=' * 60)

    os.makedirs(model_repo_path, exist_ok=True)

    # Copy PyTorch checkpoint
    dest = os.path.join(model_repo_path, ESAM3_PYTORCH_CHECKPOINT_NAME)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'  Already exists: {dest} ({size_mb:.1f} MB)')
    else:
        if not os.path.isfile(checkpoint_src):
            print(f'ERROR: Checkpoint not found: {checkpoint_src}')
            print('Download from HuggingFace:')
            print('  https://huggingface.co/Simon7108528/EfficientSAM3')
            print('Or train with:')
            print('  https://github.com/SimonZeng7108/efficientsam3')
            sys.exit(1)
        shutil.copy2(checkpoint_src, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'  Copied: {checkpoint_src} -> {dest} ({size_mb:.1f} MB)')

    # Download tokenizer (same CLIP BPE tokenizer)
    print('\n[tokenizer]')
    tokenizer_dest = os.path.join(model_repo_path, 'tokenizer.json')
    download_file(TOKENIZER_URL, tokenizer_dest)

    print('\n' + '=' * 60)
    print('PyTorch backend setup complete!')
    print(f'\nCheckpoint: {dest}')
    print(f'Tokenizer: {tokenizer_dest}')
    print(f'\nUsage:')
    print(f'  ros2 launch isaac_ros_segment_anything3 '
          f'isaac_ros_segment_anything3.launch.py \\')
    print(f'    model_type:=efficient_sam3 '
          f'inference_backend:=pytorch \\')
    print(f'    model_repository_path:={model_repo_path}')


def setup_triton_repo(model_repo_path, model_type='sam3',
                      local_models=None, verify=True):
    """
    Set up Triton model repository for SAM3 or EfficientSAM3.

    Directory structure (both model types can coexist):
        model_repo_path/
        ├── sam3_vision_encoder/        # SAM3
        │   ├── config.pbtxt
        │   └── 1/model.onnx
        ├── sam3_text_encoder/
        ├── sam3_decoder/
        ├── esam3_vision_encoder/       # EfficientSAM3
        │   ├── config.pbtxt
        │   └── 1/model.onnx
        ├── esam3_text_encoder/
        ├── esam3_decoder/
        └── tokenizer.json              # Shared (same CLIP BPE)
    """
    print(f'Setting up {model_type} Triton model repository '
          f'at: {model_repo_path}')
    print('=' * 60)

    if model_type == 'sam3':
        _setup_models(model_repo_path, SAM3_MODELS, 'SAM3', verify=verify)
    elif model_type == 'efficient_sam3':
        if not local_models:
            print('ERROR: --local-models required for efficient_sam3.')
            print('Export models first with export_efficient_sam3.py')
            sys.exit(1)
        _setup_models(model_repo_path, EFFICIENT_SAM3_MODELS,
                      'EfficientSAM3', source_dir=local_models, verify=verify)
    else:
        print(f'ERROR: Unknown model type: {model_type}')
        sys.exit(1)

    # Download tokenizer (shared between SAM3 and EfficientSAM3)
    print('\n[tokenizer]')
    tokenizer_dest = os.path.join(model_repo_path, 'tokenizer.json')
    download_file(TOKENIZER_URL, tokenizer_dest)

    print('\n' + '=' * 60)
    print('Setup complete!')
    print(f'\nTo start Triton server:')
    print(f'  tritonserver --model-repository={model_repo_path}')


def verify_only(model_repo_path, model_type='sam3'):
    """Verify existing models and regenerate configs."""
    if model_type == 'sam3':
        models = SAM3_MODELS
    elif model_type == 'efficient_sam3':
        models = EFFICIENT_SAM3_MODELS
    else:
        print(f'ERROR: Unknown model type: {model_type}')
        sys.exit(1)

    print(f'Verifying {model_type} models in: {model_repo_path}')
    print('=' * 60)

    for model_key, model_info in models.items():
        triton_name = model_info['triton_name']
        model_path = os.path.join(
            model_repo_path, triton_name, '1', 'model.onnx')
        config_path = os.path.join(
            model_repo_path, triton_name, 'config.pbtxt')

        if not os.path.exists(model_path):
            print(f'\n  [SKIP] {model_key}: model.onnx not found')
            continue

        spec = verify_onnx_model(model_path, model_key)
        if spec is not None:
            config_content = generate_config_pbtxt(triton_name, spec)
            with open(config_path, 'w') as f:
                f.write(config_content)
            print(f'  Generated: {config_path}')


def download_demo_video(model_repo_path, url_or_default='default'):
    """Download a sample demo video for Foxglove demos."""
    dest = os.path.join(model_repo_path, 'demo.mp4')
    if os.path.isfile(dest):
        print(f'Demo video already exists: {dest}')
        return dest

    if url_or_default == 'default':
        print('No demo video URL provided.')
        print(f'Please download a sample video and save to: {dest}')
        print('Suggested sources (free, no attribution required):')
        print('  - https://pixabay.com/videos/search/people%20walking/')
        print('  - https://www.pexels.com/search/videos/people%20walking/')
        print('')
        print('Or provide a direct URL:')
        print('  python3 download_models.py --demo-video <URL>')
        return dest

    os.makedirs(model_repo_path, exist_ok=True)
    print(f'Downloading demo video from {url_or_default} ...')
    try:
        download_file(url_or_default, dest)
        print(f'Demo video saved: {dest}')
    except Exception as e:
        print(f'Failed to download demo video: {e}')
        print(f'Manually place a video at: {dest}')
    return dest


def main():
    parser = argparse.ArgumentParser(
        description='Download SAM3/EfficientSAM3 models '
                    'and set up inference backend')
    parser.add_argument(
        '--model-repo', default='/tmp/models',
        help='Model repository path (default: /tmp/models)')
    parser.add_argument(
        '--model-type', default='sam3',
        choices=['sam3', 'efficient_sam3'],
        help='Model type to set up (default: sam3)')
    parser.add_argument(
        '--inference-backend', default='triton',
        choices=['triton', 'pytorch'],
        help='Inference backend (default: triton)')
    parser.add_argument(
        '--local-models', default=None,
        help='Path to local ONNX models (required for efficient_sam3 triton)')
    parser.add_argument(
        '--pytorch-checkpoint', default=None,
        help='Path to EfficientSAM3 .pth checkpoint '
             '(required for --inference-backend pytorch)')
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only verify existing ONNX models and regenerate configs')
    parser.add_argument(
        '--demo-video', default=None, nargs='?', const='default',
        help='Download a sample demo video to MODEL_REPO/demo.mp4. '
             'Optionally provide a direct URL to a specific mp4 file.')
    args = parser.parse_args()

    if args.demo_video is not None:
        download_demo_video(args.model_repo, args.demo_video)

    if args.verify_only:
        verify_only(args.model_repo, args.model_type)
    elif args.inference_backend == 'pytorch':
        if args.model_type != 'efficient_sam3':
            print('ERROR: PyTorch backend only supports '
                  '--model-type efficient_sam3')
            sys.exit(1)
        if not args.pytorch_checkpoint:
            print('ERROR: --pytorch-checkpoint is required '
                  'for pytorch backend')
            sys.exit(1)
        setup_pytorch_backend(args.model_repo, args.pytorch_checkpoint)
    else:
        setup_triton_repo(
            args.model_repo, args.model_type, args.local_models)


if __name__ == '__main__':
    main()
