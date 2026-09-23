#!/usr/bin/env bash

# Offline metrics with the q15 features, for every case and model.
#
# Two modes, each with its own results folder under every setup:
#   dropin : only the test split reads cache_features_q15. The head and the
#            thresholds stay as deployed.            -> offline_q15_dropin/
#   recal  : the test and calibration splits read cache_features_q15. The
#            thresholds are recalibrated. The centroid or clusters stay as
#            deployed.                                -> offline_q15_recal/
# The existing offline/ folders are not touched.
#
# Needs the --test-feature-dir, --feature-splits, and --offline-subdir
# patch in mcu/compute_offline_metrics.py and
# mcu/compute_offline_metrics_classic.py.
#
# Settings (environment variables, all optional):
#   FEATURE_DIR   feature folder                    (default: cache_features_q15)
#   MODES         "dropin", "recal", or both        (default: "dropin recal")
#   CASES         held-out cases to run             (default: "1 2 3 4")
#   FAMILIES      "selective", "classic", or both   (default: "selective classic")
#   ONLY_SETUPS   comma-separated setup identifiers, passed as --only.
#                 The identifiers differ between the two families, so set
#                 FAMILIES to one family when you use this.
#   LOG_DIR       one log file per run              (default: logs/offline_q15)
#   DRY_RUN       1 = print the commands and stop   (default: 0)
#   HEARTBEAT_S   seconds between "still running" lines (default: 30)
#
# Examples:
#   ./compute_offline_metrics_q15.sh
#   MODES="dropin" ./compute_offline_metrics_q15.sh
#   CASES="1" FAMILIES="selective" ONLY_SETUPS="full_fp32_euclidean_fp32" ./compute_offline_metrics_q15.sh
#   DRY_RUN=1 ./compute_offline_metrics_q15.sh

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Same convention as export_full_deployment_matrix.sh: activate the conda
# environment before you start the script. The script only moves to the
# folder that it is in, so the relative paths work.
cd "$(dirname "$(readlink -f "$0")")" || { echo -e "${RED}Cannot change to the project directory.${NC}"; exit 1; }

FEATURE_DIR="${FEATURE_DIR:-cache_features_q15}"
MODES="${MODES:-dropin recal}"
CASES="${CASES:-1 2 3 4}"
FAMILIES="${FAMILIES:-selective classic}"
ONLY_SETUPS="${ONLY_SETUPS:-}"
LOG_DIR="${LOG_DIR:-logs/offline_q15}"
DRY_RUN="${DRY_RUN:-0}"

HEARTBEAT_S="${HEARTBEAT_S:-30}"

# Python buffers its output when it writes into a pipe (here: into tee), so
# the script looks stuck for minutes. This variable turns the buffering off.
export PYTHONUNBUFFERED=1

format_time() {
    local s=$1
    printf '%dh%02dm%02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

# Prints one line every HEARTBEAT_S seconds while a run is active.
heartbeat_pid=""
heartbeat() {
    local log=$1 start=$2 label=$3 last
    while true; do
        sleep "$HEARTBEAT_S"
        last=$(tail -n 1 "$log" 2>/dev/null | tr -d '\r' | cut -c1-110)
        echo -e "${YELLOW}[still running] ${label} | $(format_time $((SECONDS - start))) | last line: ${last}${NC}"
    done
}

stop_heartbeat() {
    if [ -n "$heartbeat_pid" ]; then
        kill "$heartbeat_pid" 2>/dev/null
        wait "$heartbeat_pid" 2>/dev/null
        heartbeat_pid=""
    fi
}
trap 'stop_heartbeat' EXIT
trap 'stop_heartbeat; echo -e "\n${RED}Stopped by the user.${NC}"; exit 130' INT

# family  config  held-out case  model hash
models=(
#    "selective f4cd557b7e3b.yaml 1 16662b29beb3"
#    "selective f4cd557b7e3b.yaml 2 0ac026912cdd"
#    "selective f4cd557b7e3b.yaml 3 2437e6cfa1ab"
#    "selective f4cd557b7e3b.yaml 4 733b93bfdb24"
#
#    "selective b39731b66741.yaml 1 086acf0275b8"
#    "selective b39731b66741.yaml 2 a501188e9de1"
#    "selective b39731b66741.yaml 3 314dd8026707"
#    "selective b39731b66741.yaml 4 d94988354ca3"

    "classic f2578cb06991.yaml 1 238a49a973a3"
    "classic f2578cb06991.yaml 2 482eadc31485"
    "classic f2578cb06991.yaml 3 5295aa5e5f6a"
    "classic f2578cb06991.yaml 4 8de65745cf2e"

    "classic 352f70960ed3.yaml 1 b77482e85dc3"
    "classic 352f70960ed3.yaml 2 d610f3c01dd7"
    "classic 352f70960ed3.yaml 3 03828168f4c2"
    "classic 352f70960ed3.yaml 4 3660d1d10006"
)

# ---------------------------------------------------------------------------
# Checks before the run
# ---------------------------------------------------------------------------
preflight_fail() {
    if [ "$DRY_RUN" = "1" ]; then
        echo -e "${YELLOW}[WARNING] $1 (ignored, because DRY_RUN=1)${NC}"
    else
        echo -e "${RED}[ERROR] $1${NC}"
        exit 1
    fi
}

for mode in $MODES; do
    case "$mode" in
        dropin|recal) ;;
        *) echo -e "${RED}[ERROR] Unknown mode '${mode}'. Use dropin or recal.${NC}"; exit 1 ;;
    esac
done

if [ ! -f "$FEATURE_DIR/metadata.json" ]; then
    preflight_fail "No $FEATURE_DIR/metadata.json. Build the cache first."
elif ! grep -q '"complete": true' "$FEATURE_DIR/metadata.json"; then
    preflight_fail "$FEATURE_DIR/metadata.json does not say complete: true. Finish the cache first."
fi

# ---------------------------------------------------------------------------
# Build the list of commands
# ---------------------------------------------------------------------------
build_cmd() {
    local family=$1 config=$2 case_id=$3 hash=$4 mode=$5
    local module="mcu.compute_offline_metrics"
    if [ "$family" = "classic" ]; then
        module="mcu.compute_offline_metrics_classic"
    fi

    local splits subdir
    if [ "$mode" = "dropin" ]; then
        splits="test"
        subdir="offline_q15_dropin"
    else
        splits="test calib_normal"
        subdir="offline_q15_recal"
    fi

    local cmd="python -m ${module} --config ${config} --held-out-case ${case_id} --model-hash ${hash}"
    cmd="${cmd} --test-feature-dir ${FEATURE_DIR} --feature-splits ${splits} --offline-subdir ${subdir}"
    if [ -n "$ONLY_SETUPS" ]; then
        cmd="${cmd} --only ${ONLY_SETUPS}"
    fi
    echo "$cmd"
}

commands=()
labels=()
for entry in "${models[@]}"; do
    read -r family config case_id hash <<< "$entry"
    [[ " $CASES " == *" $case_id "* ]] || continue
    [[ " $FAMILIES " == *" $family "* ]] || continue
    for mode in $MODES; do
        commands+=("$(build_cmd "$family" "$config" "$case_id" "$hash" "$mode")")
        labels+=("case${case_id}_${family}_${hash}_${mode}")
    done
done

echo -e "${CYAN}${#commands[@]} runs planned. Feature folder: ${FEATURE_DIR}. Modes: ${MODES}.${NC}"

if [ "$DRY_RUN" = "1" ]; then
    for cmd in "${commands[@]}"; do
        echo "$cmd"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

# Associative arrays to store output statuses and durations
declare -A results
declare -A durations

total=${#commands[@]}
run_start=$SECONDS

for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    label="${labels[$i]}"
    log="${LOG_DIR}/${label}.log"

    echo -e "\n${CYAN}========================================================${NC}"
    echo -e "${CYAN}[$((i + 1))/${total}] Starting: ${cmd}${NC}"
    echo -e "${CYAN}Log: ${log}${NC}"
    echo -e "${CYAN}========================================================${NC}"

    start=$SECONDS
    : > "$log"    # create the log now, so the heartbeat can read it at once
    heartbeat "$log" "$start" "$label" &
    heartbeat_pid=$!

    eval "$cmd" 2>&1 | tee "$log"
    exit_code=${PIPESTATUS[0]}

    stop_heartbeat
    durations["$label"]=$((SECONDS - start))

    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}[SUCCESS] ${label} completed in $(format_time ${durations[$label]}).${NC}"
        results["$label"]="Success"
    else
        echo -e "\n${RED}[FAILED] ${label} exited with code ${exit_code}. Continuing to next...${NC}"
        results["$label"]="Failed (Code ${exit_code})"
    fi

    done_count=$((i + 1))
    elapsed=$((SECONDS - run_start))
    average=$((elapsed / done_count))
    remaining=$((average * (total - done_count)))
    echo -e "${CYAN}Progress: ${done_count}/${total} runs | elapsed $(format_time $elapsed) | rough ETA $(format_time $remaining)${NC}"
done

# Execution Summary
echo -e "\n${YELLOW}====================== SUMMARY ======================${NC}"
for label in "${labels[@]}"; do
    status="${results["$label"]}"
    if [[ "$status" == "Success" ]]; then
        echo -e "${GREEN}${label} : ${status} ($(format_time ${durations[$label]}))${NC}"
    else
        echo -e "${RED}${label} : ${status} ($(format_time ${durations[$label]}))${NC}"
    fi
done
echo -e "${YELLOW}=====================================================${NC}"