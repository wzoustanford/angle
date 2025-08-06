#!/bin/bash
# Setup script for new AWS GPU instance
# Run this script on the new instance after launching

set -e  # Exit on any error

echo "=========================================="
echo "Setting up new AWS instance for RL experiments"
echo "=========================================="

# Update system
echo "1. Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential packages
echo "2. Installing essential packages..."
sudo apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    python3-pip \
    python3-dev \
    build-essential

# Install Python packages
echo "3. Installing Python packages..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install gymnasium[atari]
pip3 install ale-py
pip3 install numpy matplotlib scipy
pip3 install opencv-python
pip3 install tensorboard
pip3 install psutil

# Install ROMs for Atari games
echo "4. Installing Atari ROMs..."
pip3 install autorom
python3 -m atari_py.import_roms

# Create directory structure
echo "5. Setting up directory structure..."
mkdir -p ~/rl_experiments
cd ~/rl_experiments

# Clone or create the RL codebase
echo "6. Setting up RL codebase..."
# Option 1: If you have the code in a git repo
# git clone YOUR_REPO_URL .

# Option 2: Create the basic structure (we'll copy files separately)
mkdir -p model config experiments results

echo "7. Setting up GPU monitoring..."
# Install nvidia tools if not present
which nvidia-smi >/dev/null || {
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-470
}

# Create monitoring script
cat > gpu_monitor.sh << 'EOF'
#!/bin/bash
echo "GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "Memory Status:"
free -h
echo "Disk Status:"
df -h /
echo "================="
EOF
chmod +x gpu_monitor.sh

echo "8. Creating tmux session setup..."
cat > start_experiment.sh << 'EOF'
#!/bin/bash
# Start experiment in tmux session

SESSION_NAME="rl_experiment"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new session
tmux new-session -d -s $SESSION_NAME

# Create windows
tmux new-window -t $SESSION_NAME:1 -n 'experiment'
tmux new-window -t $SESSION_NAME:2 -n 'monitor'

# Set up monitoring window
tmux send-keys -t $SESSION_NAME:2 'watch -n 10 ./gpu_monitor.sh' C-m

# Set up experiment window
tmux send-keys -t $SESSION_NAME:1 'cd ~/rl_experiments' C-m

echo "Tmux session '$SESSION_NAME' created!"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Switch windows with: Ctrl+B then window number"
EOF
chmod +x start_experiment.sh

echo "9. Setting up result collection..."
mkdir -p results/experiments
mkdir -p results/logs
mkdir -p results/checkpoints

# Create result collection script
cat > collect_results.sh << 'EOF'
#!/bin/bash
# Collect and package results for download

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_package_$TIMESTAMP"

echo "Collecting results into $RESULTS_DIR..."

mkdir -p $RESULTS_DIR
cp -r results/ $RESULTS_DIR/
cp -r experiments/ $RESULTS_DIR/
cp *.log $RESULTS_DIR/ 2>/dev/null || true

# Create summary
echo "Experiment Results - $TIMESTAMP" > $RESULTS_DIR/README.txt
echo "=======================================" >> $RESULTS_DIR/README.txt
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)" >> $RESULTS_DIR/README.txt
echo "Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)" >> $RESULTS_DIR/README.txt
echo "Collection Date: $(date)" >> $RESULTS_DIR/README.txt
echo "" >> $RESULTS_DIR/README.txt
echo "Contents:" >> $RESULTS_DIR/README.txt
find $RESULTS_DIR -type f -name "*.json" -o -name "*.md" -o -name "*.png" >> $RESULTS_DIR/README.txt

tar -czf $RESULTS_DIR.tar.gz $RESULTS_DIR
echo "Results packaged in: $RESULTS_DIR.tar.gz"
ls -lh $RESULTS_DIR.tar.gz
EOF
chmod +x collect_results.sh

echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo "Next steps:"
echo "1. Copy your RL code to ~/rl_experiments/"
echo "2. Run: ./start_experiment.sh"
echo "3. Attach to tmux: tmux attach -t rl_experiment"
echo "4. Run experiments in the 'experiment' window"
echo "5. Monitor progress in the 'monitor' window"
echo "6. Collect results: ./collect_results.sh"
echo "=========================================="