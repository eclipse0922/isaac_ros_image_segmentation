#!/bin/bash
# Download r2b_robotarm dataset using NGC CLI (more reliable for large files)

set -e

echo "=========================================="
echo "Downloading r2b_robotarm via NGC CLI"
echo "=========================================="
echo ""

# Detect workspace directory
if [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
else
    WORKSPACE_DIR="$(pwd)"
fi

cd "$WORKSPACE_DIR"

# Install NGC CLI if not present
if ! command -v ngc &> /dev/null && [ ! -f "./ngc-cli/ngc" ]; then
    echo "Installing NGC CLI..."
    wget --content-disposition https://ngc.nvidia.com/downloads/ngccli_linux.zip
    unzip -o ngccli_linux.zip
    chmod u+x ngc-cli/ngc
    NGC_CMD="./ngc-cli/ngc"
    echo "✓ NGC CLI installed"
elif command -v ngc &> /dev/null; then
    NGC_CMD="ngc"
    echo "✓ NGC CLI already installed"
else
    NGC_CMD="./ngc-cli/ngc"
    echo "✓ NGC CLI found at ./ngc-cli/ngc"
fi

echo ""

# Download dataset if not already present
NGC_DOWNLOAD_DIR="datasets/r2bdataset2024_v1"

if [ ! -d "$NGC_DOWNLOAD_DIR" ]; then
    echo "Downloading r2bdataset2024 (~3.3GB total, includes r2b_robotarm)..."
    echo "This may take 5-10 minutes depending on connection speed..."
    echo ""

    $NGC_CMD registry resource download-version "nvidia/isaac/r2bdataset2024:1" \
        --dest datasets/
else
    echo "✓ NGC download already exists at $NGC_DOWNLOAD_DIR"
fi

# Debug: show what was downloaded
echo ""
echo "Downloaded file structure:"
find "$NGC_DOWNLOAD_DIR" -type f | head -20

# Create symlink or copy to expected location
echo ""
echo "Setting up dataset paths..."
mkdir -p datasets/r2b_dataset

# Find the actual r2b_robotarm directory
ROBOTARM_DB3=$(find "$NGC_DOWNLOAD_DIR" -name "r2b_robotarm_0.db3" 2>/dev/null | head -1)

if [ -n "$ROBOTARM_DB3" ]; then
    ROBOTARM_DIR=$(dirname "$ROBOTARM_DB3")
    echo "  Found r2b_robotarm at: $ROBOTARM_DIR"

    # Create symlink to expected location
    if [ ! -e "datasets/r2b_dataset/r2b_robotarm" ]; then
        ln -sf "$(realpath "$ROBOTARM_DIR")" "datasets/r2b_dataset/r2b_robotarm"
        echo "  Linked: datasets/r2b_dataset/r2b_robotarm -> $ROBOTARM_DIR"
    else
        echo "  datasets/r2b_dataset/r2b_robotarm already exists"
    fi
else
    echo "  r2b_robotarm_0.db3 not found in download. Listing all files:"
    find "$NGC_DOWNLOAD_DIR" -type f
    echo ""
    echo "✗ Could not find r2b_robotarm_0.db3"
    exit 1
fi

# Verify
echo ""
if [ -f "datasets/r2b_dataset/r2b_robotarm/r2b_robotarm_0.db3" ]; then
    echo "✓ Dataset ready!"
    echo ""
    ls -lh datasets/r2b_dataset/r2b_robotarm/
    echo ""
    echo "Run the benchmark:"
    echo "  ./test_sam3_benchmark.sh"
else
    echo "✗ Verification failed"
    exit 1
fi
