#!/bin/bash

# Run 5 seeds WITH pretext inhibition
# Expected runtime: ~10 hours (2 hours per seed)
# Usage: sh run_mt5_5seeds_with_grin.sh

echo "Starting MT5 experiments WITH pretext inhibition"
echo "Running 5 seeds sequentially..."
echo "Expected total runtime: ~10 hours"
echo "Start time: $(date)"

N_EXPERTS=4  # 4 experts for MT5
NUM_RECURRENCE = 1

# Run 5 different seeds
for SEED in 42 123 456 789 1011
do
    echo "================================================"
    echo "Starting seed ${SEED} at $(date)"
    echo "================================================"
    
    sh run_metaworld_mt5_quick_test_grin.sh ${N_EXPERTS} ${SEED} ${NUM_RECURRENCE}
    echo "Completed seed ${SEED} at $(date)"
    echo ""
done

echo "================================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "Results saved in: logs/metaworld_mt5/mt5_moore_quick_test_${N_EXPERTS}e_pretexTrue/"
echo "================================================"
