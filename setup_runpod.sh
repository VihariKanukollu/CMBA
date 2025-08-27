#!/bin/bash
# RunPod Setup Script for HRM (Hierarchical Reasoning Model)
# This script sets up the environment on RunPod instances

set -e  # Exit on any error

echo "🚀 Setting up HRM environment on RunPod..."
echo "================================================="

# Update system
echo "📦 Updating system packages..."
apt-get update -qq && apt-get install -y -qq \
    wget \
    git \
    build-essential \
    ninja-build \
    python3-dev \
    python3-pip \
    unzip \
    curl

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA is available
echo "🔧 Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    
    # Detect GPU generation for FlashAttention
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "🔍 GPU: $GPU_NAME"
    
    if [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]]; then
        GPU_GENERATION="hopper"
        echo "🏗️  Hopper GPU detected - will use FlashAttention 3"
    else
        GPU_GENERATION="ampere_or_earlier"
        echo "🏗️  Ampere/earlier GPU detected - will use FlashAttention 2"
    fi
else
    echo "❌ No NVIDIA GPU found. Please use a GPU-enabled RunPod instance."
    exit 1
fi

# Install Python dependencies for building extensions
echo "🐍 Installing Python build dependencies..."
pip3 install --upgrade pip
pip3 install packaging ninja wheel setuptools setuptools-scm

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
# Use the CUDA version that comes with most RunPod instances
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA installation
echo "🧪 Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install FlashAttention based on GPU generation
echo "⚡ Installing FlashAttention..."
if [ "$GPU_GENERATION" = "hopper" ]; then
    # For H100/H200 GPUs - FlashAttention 3
    echo "Installing FlashAttention 3 for Hopper GPUs..."
    if [ ! -d "flash-attention" ]; then
        git clone https://github.com/Dao-AILab/flash-attention.git
    fi
    cd flash-attention/hopper
    python3 setup.py install
    cd ../..
else
    # For A100/V100 and earlier - FlashAttention 2
    echo "Installing FlashAttention 2 for Ampere/earlier GPUs..."
    pip3 install flash-attn --no-build-isolation
fi

# Install project dependencies
echo "📚 Installing project dependencies..."
if [ -f "HRM/requirements.txt" ]; then
    pip3 install -r HRM/requirements.txt
elif [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "⚠️  No requirements.txt found, installing common dependencies..."
    pip3 install torch adam-atan2 einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub
fi

# Set up Weights & Biases
echo "📊 Setting up Weights & Biases..."
echo "Please run 'wandb login' after this script completes to authenticate with W&B"

# Initialize git submodules if they exist
echo "📦 Initializing git submodules..."
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p checkpoints

# Set up environment variables for optimal performance
echo "⚙️  Setting up environment variables..."
cat << EOF >> ~/.bashrc
# HRM Environment Variables
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Optimize for training
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Disable compilation cache issues
export TORCH_COMPILE_DEBUG=0
EOF

echo ""
echo "✅ HRM setup completed successfully!"
echo "================================================="
echo ""
echo "🎯 Next Steps:"
echo "1. Run 'source ~/.bashrc' or restart your terminal"
echo "2. Run 'wandb login' to authenticate with Weights & Biases"
echo "3. Navigate to the HRM directory: cd HRM"
echo "4. Test the installation: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "5. Run a quick demo: python dataset/build_sudoku_dataset.py --help"
echo ""
echo "🚀 Ready to start training! Check the README.md for training commands."
echo "💡 For RunPod: Remember to save your work and models to /workspace/outputs for persistence"
