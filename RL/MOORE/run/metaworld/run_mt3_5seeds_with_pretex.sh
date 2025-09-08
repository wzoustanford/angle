#!/bin/bash

# Run 5 seeds WITH pretext inhibition
# Expected runtime: ~10 hours (2 hours per seed)
# Usage: sh run_mt3_5seeds_with_pretex.sh

echo "Starting MT3 experiments WITH pretext inhibition"
echo "Running 5 seeds sequentially..."
echo "Expected total runtime: ~10 hours"
echo "Start time: $(date)"

N_EXPERTS=3  # 3 experts for MT3

# Run 5 different seeds
for SEED in 42 123 456 789 1011
do
    echo "================================================"
    echo "Starting seed ${SEED} at $(date)"
    echo "================================================"
    
    sh run_metaworld_mt3_quick_test.sh ${N_EXPERTS} ${SEED} True
    
    echo "Completed seed ${SEED} at $(date)"
    echo ""
done

echo "================================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "Results saved in: logs/metaworld_mt3/mt3_moore_quick_test_${N_EXPERTS}e_pretexTrue/"
echo "================================================"