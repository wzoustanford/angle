#!/bin/bash
# Deploy RL codebase to new AWS instance
# Run this from the current instance to copy files to new instance

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new-instance-ip> [ssh-key-path]"
    echo "Example: $0 18.123.45.67 ~/.ssh/my-key.pem"
    exit 1
fi

NEW_INSTANCE_IP=$1
SSH_KEY=${2:-~/.ssh/id_rsa}

echo "=========================================="
echo "Deploying RL codebase to new instance"
echo "Target: $NEW_INSTANCE_IP"
echo "SSH Key: $SSH_KEY"
echo "=========================================="

# Test connection
echo "1. Testing connection..."
ssh -i $SSH_KEY -o ConnectTimeout=10 ubuntu@$NEW_INSTANCE_IP "echo 'Connection successful'" || {
    echo "ERROR: Cannot connect to $NEW_INSTANCE_IP"
    echo "Make sure:"
    echo "  - Instance is running and accessible"
    echo "  - Security group allows SSH (port 22)"
    echo "  - SSH key is correct"
    exit 1
}

# Copy setup script first
echo "2. Copying setup script..."
scp -i $SSH_KEY setup_new_instance.sh ubuntu@$NEW_INSTANCE_IP:~/

# Run setup script
echo "3. Running setup on new instance..."
ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP "chmod +x setup_new_instance.sh && ./setup_new_instance.sh"

# Copy RL codebase
echo "4. Copying RL codebase..."
cd /home/ubuntu/code/angle/RL

# Create archive of essential files
tar -czf /tmp/rl_codebase.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints' \
    --exclude='results/experiment_*' \
    --exclude='*.log' \
    config/ \
    model/ \
    experiments/alien_icehockey_50_memory_efficient.py \
    experiments/alien_icehockey_experiment.py \
    experiments/quick_full_test.py

# Copy archive to new instance
echo "5. Transferring codebase archive..."
scp -i $SSH_KEY /tmp/rl_codebase.tar.gz ubuntu@$NEW_INSTANCE_IP:~/

# Extract on new instance
echo "6. Extracting codebase on new instance..."
ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP "cd ~/rl_experiments && tar -xzf ~/rl_codebase.tar.gz"

# Copy the optimized 50-episode experiment script
echo "7. Setting up 50-episode experiment..."
scp -i $SSH_KEY alien_icehockey_50_memory_efficient.py ubuntu@$NEW_INSTANCE_IP:~/rl_experiments/experiments/

# Create experiment runner script on new instance
ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP "cat > ~/rl_experiments/run_50_episode_experiment.sh << 'EOF'
#!/bin/bash
# Run 50-episode experiment on new instance

set -e

cd ~/rl_experiments

echo \"Starting 50-Episode 5-Algorithm Experiment\"
echo \"===========================================\"
echo \"Start time: \$(date)\"
echo \"Instance: \$(curl -s http://169.254.169.254/latest/meta-data/instance-id)\"
echo \"Instance type: \$(curl -s http://169.254.169.254/latest/meta-data/instance-type)\"
echo \"GPU info:\"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create log directory
mkdir -p results/logs

# Run experiment with logging
echo \"\\nStarting experiment...\"
python3 experiments/alien_icehockey_50_memory_efficient.py --episodes 50 2>&1 | tee results/logs/experiment_\$(date +%Y%m%d_%H%M%S).log

echo \"\\n===========================================\"
echo \"Experiment completed at: \$(date)\"
echo \"Results location: ~/rl_experiments/experiments/results/\"

# List results
echo \"\\nGenerated files:\"
find experiments/results/ -name \"experiment_50ep_*\" -type d | head -1 | xargs ls -la

echo \"\\nTo collect results, run: ./collect_results.sh\"
EOF"

# Make script executable
ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP "chmod +x ~/rl_experiments/run_50_episode_experiment.sh"

# Copy result collection script
echo "8. Setting up result collection..."
ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP "cp collect_results.sh ~/rl_experiments/"

# Create quick status script
ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP "cat > ~/rl_experiments/check_status.sh << 'EOF'
#!/bin/bash
echo \"=== EXPERIMENT STATUS ===\"
echo \"Time: \$(date)\"
echo \"Uptime: \$(uptime)\"
echo \"\"
echo \"=== PROCESSES ===\"
ps aux | grep python3 | grep -v grep || echo \"No Python experiments running\"
echo \"\"
echo \"=== GPU STATUS ===\"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo \"\"
echo \"=== DISK USAGE ===\"
df -h /
echo \"\"
echo \"=== RECENT LOGS ===\"
if ls results/logs/*.log >/dev/null 2>&1; then
    tail -10 results/logs/*.log | tail -10
else
    echo \"No log files found\"
fi
EOF"

ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP "chmod +x ~/rl_experiments/check_status.sh"

# Clean up temporary files
rm -f /tmp/rl_codebase.tar.gz

echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo ""
echo "TO RUN THE EXPERIMENT ON NEW INSTANCE:"
echo "1. SSH to new instance:"
echo "   ssh -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP"
echo ""
echo "2. Start tmux session:"
echo "   cd ~/rl_experiments"
echo "   ./start_experiment.sh"
echo ""
echo "3. Attach to tmux and run experiment:"
echo "   tmux attach -t rl_experiment"
echo "   ./run_50_episode_experiment.sh"
echo ""
echo "4. Check status anytime:"
echo "   ./check_status.sh"
echo ""
echo "5. Collect results when done:"
echo "   ./collect_results.sh"
echo ""
echo "6. Download results (from your local machine):"
echo "   scp -i $SSH_KEY ubuntu@$NEW_INSTANCE_IP:~/rl_experiments/results_package_*.tar.gz ."
echo "=========================================="