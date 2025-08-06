#!/bin/bash

# Monitor and reschedule script
echo "Monitoring R2D2 experiment and will auto-schedule 50-episode run when complete"

# Get the PID of the current experiment
CURRENT_PID=18986

# Monitor the current experiment
while kill -0 $CURRENT_PID 2>/dev/null; do
    echo "$(date): R2D2 experiment still running (PID: $CURRENT_PID)..."
    sleep 300  # Check every 5 minutes
done

echo "$(date): R2D2 experiment completed!"
echo "Waiting 30 seconds before starting new experiment..."
sleep 30

# Start new 50-episode experiment with corrected priority replay
echo "$(date): Starting new 50-episode experiment with fixed priority replay"
cd /home/ubuntu/code/angle/RL

# Run the new experiment
nohup python -u experiments/single_game_experiment.py \
    --game ALE/Alien-v5 \
    --episodes 50 \
    > alien_50ep_fixed_priority.log 2>&1 &

NEW_PID=$!
echo "$(date): New 50-episode experiment started with PID: $NEW_PID"
echo "Log file: alien_50ep_fixed_priority.log"
echo "Monitor with: tail -f alien_50ep_fixed_priority.log"

# Save experiment info
echo "Experiment started at: $(date)" > alien_50ep_experiment_info.txt
echo "PID: $NEW_PID" >> alien_50ep_experiment_info.txt
echo "Episodes: 50" >> alien_50ep_experiment_info.txt
echo "Game: Alien" >> alien_50ep_experiment_info.txt
echo "Priority Replay Fixed: Yes (α=0.7, β₀=0.5, proper annealing)" >> alien_50ep_experiment_info.txt