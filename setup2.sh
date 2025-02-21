#!/bin/bash
set -e

echo "=== Setting up the Python environment ==="
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
echo "=== Installing requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

# Install custom TRL branch for memory optimization
echo "=== Installing custom TRL branch ==="
git clone https://github.com/huggingface/trl
cd trl
git checkout main
git fetch origin pull/2669/head:grpo-vram-optimization
git checkout -b trl_custom
git cherry-pick ccc95472f6245f2db00986a08ca16da68bf32c14 # tested, stable commit from PR#2669 by @andyl98
#git merge --no-ff grpo-vram-optimization
#git cherry-pick ccc95472f6245f2db00986a08ca16da68bf32c14 # tested, stable commit from PR#2669 by @andyl98 TODO
pip install --force-reinstall .
cd ..

# Ask user for Huggingface API key to set HF_TOKEN env var
echo "=== Setting up Huggingface API key ==="
read -rp "Please enter your Huggingface API key to access gated models such as Llama-3.1-8B-Instruct: " HF_TOKEN
export HF_TOKEN
echo "export HF_TOKEN=$HF_TOKEN" >> venv/bin/activate
echo "HF_TOKEN has been set to: $HF_TOKEN"

# Ask user for Wandb API key to log training runs
echo "=== Setting up Wandb API key ==="
read -rp "Please enter your Wandb API key to log training runs: " WANDB_API_KEY
export WANDB_API_KEY
echo "export WANDB_API_KEY=$WANDB_API_KEY" >> venv/bin/activate
echo "Wandb API key has been set to: $WANDB_API_KEY"

echo "=== Setup complete! ==="