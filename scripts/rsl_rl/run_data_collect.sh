#!/bin/bash

# ----------------------------
# CONFIGURATION
# ----------------------------

# List of terrain difficulties
difficulties=(0.0 0.25 0.5 0.75 1.0)

# List of (policy_name, policy_str) pairs
declare -A policies
policies=(
    ["hardware"]="2025-12-03_10-16-29"
    ["fixedDZ"]="2025-12-07_14-30-23"
    ["fixedT"]="2025-12-06_15-29-36"
)

# ----------------------------
# RUN LOOP
# ----------------------------

for difficulty in "${difficulties[@]}"; do
    for policy_name in "${!policies[@]}"; do
        policy_str=${policies[$policy_name]}

        echo "======================================================"
        echo " Running difficulty=$difficulty   policy=$policy_name"
        echo "======================================================"

        python scripts/rsl_rl/data_collect_policy.py \
            --env_type=stepping_stone_finetune \
            --num_envs=4096 \
            --headless
            --difficulty "$difficulty" \
            --policy_name "$policy_name" \
            --policy_str "$policy_str"

    done
done
