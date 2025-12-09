#!/bin/bash
# Together Instant Cluster bootstrap for the GRPO Blackjack demo.

set -euo pipefail

echo "🚀 Setting up GRPO BlackJack Training on Together Instant Cluster"
echo "=================================================================="

apt-get update
apt-get install -y git build-essential
rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip

if [ ! -d "/workspace/OpenEnv" ]; then
    echo "📦 Cloning OpenEnv..."
    git clone https://github.com/meta-pytorch/OpenEnv.git /workspace/OpenEnv
else
    echo "✅ OpenEnv already present"
fi

if [ ! -d "/workspace/torchforge" ]; then
    echo "📦 Cloning TorchForge..."
    git clone https://github.com/meta-pytorch/torchforge.git /workspace/torchforge
else
    echo "✅ TorchForge already present"
fi

echo "📦 Installing OpenEnv..."
cd /workspace/OpenEnv
pip install -e .
pip install open_spiel
pip install omegaconf

echo "📦 Installing CUDA 12.8 PyTorch stack..."
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128 torchvision==0.24.0+cu128
pip install numpy==2.2.2

echo "📦 Installing TorchForge (this may take a while)..."
cd /workspace/torchforge
./scripts/install.sh || {
    echo "⚠️  Forge install script failed, falling back to pip editable install"
    pip install -e .
}

echo "📦 Aligning OpenAI client with OpenEnv requirements..."
pip install 'openai>=2.7.2'

cat <<'EOF' >/etc/profile.d/grpo-demo.sh
export OPENENV_PATH="/workspace/OpenEnv/src"
export GRPO_DEMO_PATH="/workspace/grpo-demo"
EOF

echo ""
echo "=================================================================="
echo "✅ Setup complete! (workspace pod now has OpenEnv + TorchForge deps)"
echo "=================================================================="
