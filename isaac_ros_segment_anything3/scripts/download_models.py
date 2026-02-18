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
Download SAM3 PyTorch checkpoint for direct CUDA inference.

Usage:
    # Download SAM3 checkpoint to /tmp/models/
    python3 download_models.py

    # Download to custom path
    python3 download_models.py --model-repo /path/to/models

    # Verify existing checkpoint
    python3 download_models.py --verify-only
"""

import argparse
import os
import sys
import urllib.request

# SAM3 PyTorch checkpoint (~3.3 GB)
# Download from HuggingFace (preferred) or alternate sources
SAM3_CHECKPOINT_NAME = 'sam3.pt'


def download_file(url, dest):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'  Already exists: {dest} ({size_mb:.0f} MB)')
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
    print(f'\n  Saved: {dest} ({size_mb:.0f} MB)')


def download_sam3_checkpoint(model_repo_path):
    """
    Download SAM3 PyTorch checkpoint from HuggingFace.

    The SAM3 full model checkpoint (~3.3 GB) is downloaded to:
        {model_repo_path}/sam3.pt

    Args:
        model_repo_path: Directory to place the checkpoint.
    """
    print('Setting up SAM3 PyTorch backend')
    print('=' * 60)

    os.makedirs(model_repo_path, exist_ok=True)
    dest = os.path.join(model_repo_path, SAM3_CHECKPOINT_NAME)

    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'✓ Checkpoint already exists: {dest} ({size_mb:.0f} MB)')
        return dest

    # Try HuggingFace download via huggingface_hub
    print('[SAM3 checkpoint]')
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        print('  Downloading via HuggingFace Hub...')
        path = hf_hub_download(
            repo_id='facebook/sam3',
            filename='sam3.pt',
        )
        shutil.copy2(path, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f'  ✓ Saved: {dest} ({size_mb:.0f} MB)')
    except Exception as e:
        print(f'  HuggingFace download failed: {e}')
        print('  Please download sam3.pt manually and place at:')
        print(f'    {dest}')
        print('  Source: https://huggingface.co/facebook/sam3')
        sys.exit(1)

    print('\n' + '=' * 60)
    print('Setup complete!')
    print(f'\nCheckpoint: {dest}')
    print('\nTo start the SAM3 ROS2 node:')
    print('  ros2 run isaac_ros_segment_anything3 sam3_node.py \\')
    print(f'    --ros-args -p pytorch_checkpoint:={dest}')
    return dest


def verify_checkpoint(model_repo_path):
    """Verify the SAM3 checkpoint is present and loadable."""
    dest = os.path.join(model_repo_path, SAM3_CHECKPOINT_NAME)

    if not os.path.isfile(dest):
        print(f'✗ Checkpoint not found: {dest}')
        print('  Run download_models.py to download.')
        sys.exit(1)

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f'✓ Checkpoint found: {dest} ({size_mb:.0f} MB)')

    try:
        import torch
        ckpt = torch.load(dest, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())[:5]
            print(f'  Keys: {keys}...')
        print('  ✓ Checkpoint loaded successfully')
    except ImportError:
        print('  (PyTorch not installed, skipping load verification)')
    except Exception as e:
        print(f'  ✗ Failed to load checkpoint: {e}')
        sys.exit(1)


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
        description='Download SAM3 PyTorch checkpoint for direct CUDA inference')
    parser.add_argument(
        '--model-repo', default='/tmp/models',
        help='Model repository path (default: /tmp/models)')
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Only verify the existing checkpoint (no download)')
    parser.add_argument(
        '--demo-video', default=None, nargs='?', const='default',
        help='Download a sample demo video to MODEL_REPO/demo.mp4. '
             'Optionally provide a direct URL to a specific mp4 file.')
    args = parser.parse_args()

    if args.demo_video is not None:
        download_demo_video(args.model_repo, args.demo_video)
        return

    if args.verify_only:
        verify_checkpoint(args.model_repo)
    else:
        download_sam3_checkpoint(args.model_repo)


if __name__ == '__main__':
    main()
