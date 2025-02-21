# Install custom TRL branch for memory optimization
echo "=== Installing custom TRL branch ==="
git clone --branch grpo-vram-optimization https://github.com/andyl98/trl.git trl_custom
#git clone https://github.com/huggingface/trl.git trl_custom
cd trl_custom
# Tested, stable commit
git checkout ccc95472f6245f2db00986a08ca16da68bf32c14
pip install .
cd ..