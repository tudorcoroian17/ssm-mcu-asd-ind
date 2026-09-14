#!/usr/bin/env bash

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Initialize Conda and navigate to the project directory
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate ssm-mcu-asd || { echo -e "${RED}Failed to activate conda environment.${NC}"; exit 1; }
#
#cd ~/projects/ssm-mcu-asd-ind || { echo -e "${RED}Project directory not found.${NC}"; exit 1; }

# List of commands to execute
scripts=(
    "python -m mcu.export_deploy_matrix --config f4cd557b7e3b.yaml --held-out-case 2 --model-hash 0ac026912cdd"
    "python -m mcu.export_deploy_matrix --config f4cd557b7e3b.yaml --held-out-case 3 --model-hash 2437e6cfa1ab"
    "python -m mcu.export_deploy_matrix --config f4cd557b7e3b.yaml --held-out-case 4 --model-hash 733b93bfdb24"

    "python -m mcu.compute_offline_metrics --config f4cd557b7e3b.yaml --held-out-case 2 --model-hash 0ac026912cdd"
    "python -m mcu.compute_offline_metrics --config f4cd557b7e3b.yaml --held-out-case 3 --model-hash 2437e6cfa1ab"
    "python -m mcu.compute_offline_metrics --config f4cd557b7e3b.yaml --held-out-case 4 --model-hash 733b93bfdb24"

    "python -m mcu.export_deploy_matrix --config b39731b66741.yaml --held-out-case 2 --model-hash a501188e9de1"
    "python -m mcu.export_deploy_matrix --config b39731b66741.yaml --held-out-case 3 --model-hash 314dd8026707"
    "python -m mcu.export_deploy_matrix --config b39731b66741.yaml --held-out-case 4 --model-hash d94988354ca3"

    "python -m mcu.compute_offline_metrics --config b39731b66741.yaml --held-out-case 2 --model-hash a501188e9de1"
    "python -m mcu.compute_offline_metrics --config b39731b66741.yaml --held-out-case 3 --model-hash 314dd8026707"
    "python -m mcu.compute_offline_metrics --config b39731b66741.yaml --held-out-case 4 --model-hash d94988354ca3"

    "python -m mcu.export_deploy_matrix_classic --config f2578cb06991.yaml --held-out-case 2 --model-hash 482eadc31485"
    "python -m mcu.export_deploy_matrix_classic --config f2578cb06991.yaml --held-out-case 3 --model-hash 5295aa5e5f6a"
    "python -m mcu.export_deploy_matrix_classic --config f2578cb06991.yaml --held-out-case 4 --model-hash 8de65745cf2e"

    "python -m mcu.compute_offline_metrics_classic --config f2578cb06991.yaml --held-out-case 2 --model-hash 482eadc31485"
    "python -m mcu.compute_offline_metrics_classic --config f2578cb06991.yaml --held-out-case 3 --model-hash 5295aa5e5f6a"
    "python -m mcu.compute_offline_metrics_classic --config f2578cb06991.yaml --held-out-case 4 --model-hash 8de65745cf2e"

    "python -m mcu.export_deploy_matrix_classic --config 352f70960ed3.yaml --held-out-case 2 --model-hash d610f3c01dd7"
    "python -m mcu.export_deploy_matrix_classic --config 352f70960ed3.yaml --held-out-case 3 --model-hash 03828168f4c2"
    "python -m mcu.export_deploy_matrix_classic --config 352f70960ed3.yaml --held-out-case 4 --model-hash 3660d1d10006"

    "python -m mcu.compute_offline_metrics_classic --config 352f70960ed3.yaml --held-out-case 2 --model-hash d610f3c01dd7"
    "python -m mcu.compute_offline_metrics_classic --config 352f70960ed3.yaml --held-out-case 3 --model-hash 03828168f4c2"
    "python -m mcu.compute_offline_metrics_classic --config 352f70960ed3.yaml --held-out-case 4 --model-hash 3660d1d10006"
)

# Associative array to store output statuses
declare -A results

for cmd in "${scripts[@]}"; do
    echo -e "\n${CYAN}========================================================${NC}"
    echo -e "${CYAN}Starting: ${cmd}${NC}"
    echo -e "${CYAN}========================================================${NC}"

    # Execute the command
    eval "$cmd"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}[SUCCESS] ${cmd} completed successfully.${NC}"
        results["$cmd"]="Success"
    else
        echo -e "\n${RED}[FAILED] ${cmd} exited with code ${exit_code}. Continuing to next...${NC}"
        results["$cmd"]="Failed (Code ${exit_code})"
    fi
done

# Execution Summary
echo -e "\n${YELLOW}====================== SUMMARY ======================${NC}"
for cmd in "${scripts[@]}"; do
    status="${results["$cmd"]}"
    if [[ "$status" == "Success" ]]; then
        echo -e "${GREEN}${cmd} : ${status}${NC}"
    else
        echo -e "${RED}${cmd} : ${status}${NC}"
    fi
done
echo -e "${YELLOW}=====================================================${NC}"