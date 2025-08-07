#\!/bin/bash
echo "Monitoring Comprehensive 20-Episode Experiment"
echo "=============================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "comprehensive_20ep_experiment.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✓ Experiment is running (PID: $PID)"
    MEM=$(ps aux | grep "comprehensive_20ep_experiment.py" | grep -v grep | awk '{print $6/1024}')
    CPU=$(ps aux | grep "comprehensive_20ep_experiment.py" | grep -v grep | awk '{print $3}')
    echo "  Memory: ${MEM}MB"
    echo "  CPU: ${CPU}%"
else
    echo "✗ Experiment not running"
fi

echo ""
echo "Latest Progress:"
echo "----------------"

# Find latest experiment directory
LATEST_DIR=$(ls -td experiments/results/comprehensive_20ep_* 2>/dev/null | head -1)

if [ -n "$LATEST_DIR" ]; then
    echo "Directory: $LATEST_DIR"
    echo ""
    
    # Show last 10 lines of log
    if [ -f "$LATEST_DIR/experiment.log" ]; then
        tail -10 "$LATEST_DIR/experiment.log"
    fi
    
    echo ""
    
    # Check if results exist
    if [ -f "$LATEST_DIR/results.json" ]; then
        echo "✓ Results file exists"
    fi
    
    if [ -f "$LATEST_DIR/summary.md" ]; then
        echo "✓ Summary file exists"
        echo ""
        echo "EXPERIMENT COMPLETED\!"
        echo "View results: cat $LATEST_DIR/summary.md"
    fi
fi
