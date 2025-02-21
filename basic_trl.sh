# Install custom TRL branch for memory optimization
echo "=== Installing custom TRL branch ==="
git clone https://github.com/huggingface/trl.git trl
cd trl
pip install --force-reinstall .
cd ..