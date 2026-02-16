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
Download SAM3 ONNX models and set up Triton model repository.

Usage:
    python3 download_models.py --model-repo /tmp/models
    python3 download_models.py --model-repo /tmp/models --verify-only
"""

import argparse
import os
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

TOKENIZER_URL = \
    'https://github.com/jamjamjon/assets/releases/download/sam3/tokenizer.json'

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


def setup_triton_repo(model_repo_path, verify=True):
    """
    Download SAM3 models and set up Triton model repository.

    Directory structure:
        model_repo_path/
        ├── sam3_vision_encoder/
        │   ├── config.pbtxt
        │   └── 1/
        │       └── model.onnx
        ├── sam3_text_encoder/
        │   ├── config.pbtxt
        │   └── 1/
        │       └── model.onnx
        ├── sam3_decoder/
        │   ├── config.pbtxt
        │   └── 1/
        │       └── model.onnx
        └── tokenizer.json
    """
    print(f'Setting up SAM3 Triton model repository at: {model_repo_path}')
    print('=' * 60)

    # Download models
    for model_key, model_info in SAM3_MODELS.items():
        triton_name = model_info['triton_name']
        model_dir = os.path.join(model_repo_path, triton_name, '1')
        os.makedirs(model_dir, exist_ok=True)

        print(f'\n[{model_key}]')
        model_path = os.path.join(model_dir, 'model.onnx')
        download_file(model_info['url'], model_path)

    # Download tokenizer
    print('\n[tokenizer]')
    tokenizer_dest = os.path.join(model_repo_path, 'tokenizer.json')
    download_file(TOKENIZER_URL, tokenizer_dest)

    # Verify and generate configs
    if verify:
        print('\n' + '=' * 60)
        print('Verifying models and generating config.pbtxt files...')

        for model_key, model_info in SAM3_MODELS.items():
            triton_name = model_info['triton_name']
            model_path = os.path.join(
                model_repo_path, triton_name, '1', 'model.onnx')
            config_path = os.path.join(
                model_repo_path, triton_name, 'config.pbtxt')

            spec = verify_onnx_model(model_path, model_key)
            if spec is not None:
                config_content = generate_config_pbtxt(triton_name, spec)
                with open(config_path, 'w') as f:
                    f.write(config_content)
                print(f'  Generated: {config_path}')
            else:
                print(f'  [WARN] Skipping config generation for {model_key}')

    print('\n' + '=' * 60)
    print('Setup complete!')
    print(f'\nTo start Triton server:')
    print(f'  tritonserver --model-repository={model_repo_path}')


def verify_only(model_repo_path):
    """Verify existing models and regenerate configs."""
    print(f'Verifying SAM3 models in: {model_repo_path}')
    print('=' * 60)

    for model_key, model_info in SAM3_MODELS.items():
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


def main():
    parser = argparse.ArgumentParser(
        description='Download SAM3 ONNX models and set up Triton repository')
    parser.add_argument(
        '--model-repo', default='/tmp/models',
        help='Triton model repository path (default: /tmp/models)')
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only verify existing models and regenerate configs')
    args = parser.parse_args()

    if args.verify_only:
        verify_only(args.model_repo)
    else:
        setup_triton_repo(args.model_repo)


if __name__ == '__main__':
    main()
