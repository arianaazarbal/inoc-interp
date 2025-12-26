#!/bin/bash
set -e

echo "=== Repository Setup Script ==="

# ============================================
# SYSTEM SETUP
# ============================================

setup_system() {
    echo "Setting up system packages..."
    
    apt-get update
    apt-get install -y tmux sudo nodejs npm gh
    
    # Install uv
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="/root/.local/bin:$PATH"
    fi
}

# ============================================
# SSH/GIT IDENTITY SETUP
# ============================================

setup_identity() {
    echo "Setting up SSH and git identity..."
    
    # Try network volume first
    if [ -d "/workspace/.ssh" ]; then
        rm -rf ~/.ssh
        ln -sf /workspace/.ssh ~/.ssh
        chmod 700 ~/.ssh
        chmod 600 ~/.ssh/id_ed25519 2>/dev/null || true
        echo "✓ SSH keys linked from network volume"
    # Fall back to environment variable
    elif [ -n "$SSH_PRIVATE_KEY" ]; then
        mkdir -p ~/.ssh
        echo "$SSH_PRIVATE_KEY" > ~/.ssh/id_ed25519
        chmod 600 ~/.ssh/id_ed25519
        ssh-keyscan github.com >> ~/.ssh/known_hosts
        echo "✓ SSH key loaded from environment variable"
    else
        echo "⚠ WARNING: No SSH keys found"
        echo "  Option 1: Create network volume with SSH keys at /workspace/.ssh/"
        echo "  Option 2: Set SSH_PRIVATE_KEY environment variable in RunPod"
    fi
    
    # Git config
    if [ -f "/workspace/.gitconfig" ]; then
        ln -sf /workspace/.gitconfig ~/.gitconfig
        echo "✓ Git config linked from network volume"
    else
        git config --global user.name "Ariana Azarbal"
        git config --global user.email "arianaazarbal@icloud.com"
        git config --global pull.rebase true
        echo "✓ Git config created"
    fi
    
    # Test GitHub connection
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "✓ GitHub SSH working"
    else
        echo "⚠ GitHub SSH not authenticated"
        echo "  You may need to run: gh auth login"
    fi
}

# ============================================
# NODE/NPM SETUP
# ============================================

setup_node() {
    echo "Setting up Node.js environment..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

    
    nvm install --lts

    node -v
    npm -v

    npm install -g @anthropic-ai/claude-code
    
    echo "✓ Node version: $(node -v)"
    echo "✓ npm version: $(npm -v)"
}

# ============================================
# PYTHON ENVIRONMENT SETUP
# ============================================
create_default_venv() {
    echo "No existing venv found. Creating new environment with common ML packages..."
    
    export PATH="$HOME/.local/bin:$PATH"
    
    # Initialize uv project if no pyproject.toml
    if [ ! -f "pyproject.toml" ]; then
        uv init --no-readme --no-workspace
        echo "✓ Created pyproject.toml"
    fi
    
    # Add dependencies (this creates venv and installs)
    echo "Installing common packages (this may take a few minutes)..."
    
    # Add PyTorch with CUDA
    uv add torch
    
    # Add other packages
    uv add \
        transformers \
        transformer-lens \
        sae-lens \
        trl \
        datasets \
        accelerate \
        pandas \
        numpy \
        matplotlib \
        seaborn \
        plotly \
        tqdm \
        scikit-learn \
        jupyter \
        ipython \
        wandb \
        einops \
        jaxtyping
    
    echo "✓ Virtual environment created with ML packages"
}

# ============================================
# PROJECT SETUP
# ============================================

setup_project() {
    echo "Setting up project..."
    
    # Make sure we're in a git repo
    if [ ! -d ".git" ]; then
        echo "⚠ Not in a git repository. Skipping git config."
    else
        # Configure git for this repo
        git config --local pull.rebase true
        git config --local user.email "arianaazarbal@icloud.com"
        git config --local user.name "Ariana Azarbal"
    fi
    
    export PATH="$HOME/.local/bin:$PATH"
    
    # Check if venv exists
    if [ ! -d ".venv" ]; then
        create_default_venv
    else
        echo "✓ Existing venv found"
        
        # If pyproject.toml exists, sync dependencies
        if [ -f "pyproject.toml" ]; then
            echo "Installing project dependencies..."
            uv sync
            
            # Install in editable mode if it's a package
            if grep -q "tool.setuptools" pyproject.toml || grep -q "tool.poetry" pyproject.toml; then
                uv pip install -e .
            fi
        fi
    fi
    
    # Show venv activation command
    if [ -d ".venv" ]; then
        echo ""
        echo "✓ Virtual environment ready"
        echo "  To activate: source .venv/bin/activate"
    fi
}
# ============================================
# CURSOR EXTENSIONS
# ============================================

setup_extensions() {
    echo "Configuring recommended extensions..."
    
    mkdir -p .vscode
    
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "charliermarsh.ruff"
    ]
}
EOF
    
    echo "✓ Created extensions.json"
    echo "  Cursor will prompt to install these when you connect"
}
# ============================================
# MAIN EXECUTION
# ============================================

add_gitignore() {
    if [ ! -f ".gitignore" ]; then
        echo "Creating .gitignore file..."
        cat > .gitignore << 'EOF'
# Python
.venv/
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.python_version

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Model weights and data
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
checkpoints/
wandb/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
EOF
    echo "✓ Created .gitignore"
    fi
}
main() {
    setup_system
    setup_identity
    setup_node
    setup_project
    setup_extensions
    add_gitignore

    echo ""
    echo "========================================="
    echo "Setup complete! 🚀"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Activate venv: source .venv/bin/activate"
    echo "  2. Start tmux: tmux"
    if ! ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "  3. Authenticate GitHub: gh auth login"
    fi
    echo ""
    echo "Installed packages include:"
    echo "  - PyTorch (CUDA 12.1)"
    echo "  - Transformers, TransformerLens, SAE Lens"
    echo "  - TRL, Datasets, Accelerate"
    echo "  - Pandas, NumPy, Matplotlib, Plotly"
    echo "  - Jupyter, Weights & Biases"
}

main "$@"