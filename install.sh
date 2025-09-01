git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
pyenv local 3.12.11
uv venv venv --python 3.12.11 --seed
source venv/bin/activate
uv pip install -U vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly
uv pip install -U accelerate trl peft
