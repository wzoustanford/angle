#\!/bin/bash

echo "Autonomous Experiment Status"
echo "============================"

# Check process
PID=$(ps aux | grep "auto_experiment_10ep.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✓ Running (PID: $PID)"
    MEM=$(ps aux | grep "$PID" | grep -v grep | awk '{print $6/1024}' | head -1)
    echo "  Memory: ${MEM}MB"
else
    echo "✗ Not running"
fi

# Check latest directory
LATEST=$(ls -td experiments/results/auto_10ep_* 2>/dev/null | head -1)

if [ -n "$LATEST" ]; then
    echo ""
    echo "Directory: $LATEST"
    
    if [ -f "$LATEST/experiment.log" ]; then
        echo ""
        echo "Progress:"
        tail -5 "$LATEST/experiment.log" | grep -E "✓|Ep |Done|COMPLETED"
    fi
    
    if [ -f "$LATEST/summary.md" ]; then
        echo ""
        echo "✅ EXPERIMENT COMPLETED\!"
        echo ""
        echo "Top 3 Results:"
        grep "^|" "$LATEST/summary.md" | head -5
    fi
fi
