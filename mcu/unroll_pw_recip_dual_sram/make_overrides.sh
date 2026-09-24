#!/usr/bin/env bash
#
# Builds the unroll_pw_recip_dual_sram/ override folder for each setup:
#
#   deploy/<case>/<model>/setups/<setup>/unroll_pw_recip_dual_sram/
#       ssm_backbone.c   copied from the canonical folder for this setup's family
#       ssm_backbone.h   copied from the same canonical folder
#       ssm_weights.c    this setup's own weights, SRAM-tagged (see FLASH_KEEP below)
#       ssm_weights.h    this setup's own header, unchanged
#       provenance.txt   which canonical file was used, and checksums
#
# The setup's own files are never modified. The override folder is deleted
# and rebuilt on every run, so the script is safe to run again.
#
# Usage (from anywhere):
#   bash mcu/unroll_pw_recip_dual_sram/make_overrides.sh             # all cases
#   bash mcu/unroll_pw_recip_dual_sram/make_overrides.sh case1 case3 # some cases
#   bash mcu/unroll_pw_recip_dual_sram/make_overrides.sh --dry-run   # routing only

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CANON_DIR="$SCRIPT_DIR"
DEPLOY_DIR="$MCU_DIR/deploy"
OVERRIDE_NAME="unroll_pw_recip_dual_sram"
SECTION=".time_critical.ssm_weights"
TAG="__attribute__((section(\"${SECTION}\")))"

# Every top-level definition shape found in the generated ssm_weights.c
# files (checked across case1):
#   static const int8_t NAME[..] = ...     static const float NAME[..] = ...
#   const float NAME[..] = ...             const float NAME = ...
#   const ssm_true_int8_layer_t NAME[..]   const ssm_block_weights_t NAME[..]
#   const ssm_norm_weights_t NAME = ...
# The anchor ^ skips extern declarations and initializer continuation lines.
TAG_RE='^((static[[:space:]]+)?const[[:space:]]+(int8_t|int16_t|int32_t|float|ssm_true_int8_layer_t|ssm_block_weights_t|ssm_norm_weights_t)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*(\[|=))'

DRY_RUN=0
CASES=()
for arg in "$@"; do
	case "$arg" in
		--dry-run) DRY_RUN=1 ;;
		-h|--help) sed -n '2,19p' "$0"; exit 0 ;;
		*) CASES+=("$arg") ;;
	esac
done
if [ ${#CASES[@]} -eq 0 ]; then
	for d in "$DEPLOY_DIR"/case*/; do CASES+=("$(basename "$d")"); done
fi

# Setup name -> canonical override folder. Order matters: the classic
# patterns (_qab, which also matches _qabar) must come before the
# selective ones, because w_all_* also matches the classic w_all setups.
# Combinations that no export produces fail loudly instead of guessing.
pick_canonical() {
	case "$1" in
		true_int8_*_qab*)  echo "ssm_true_int8_classic_src" ;;
		true_int8_*)       echo "ssm_true_int8_src" ;;
		full_fp32_*_qab*)  return 1 ;;
		full_fp32_*)       echo "ssm_backbone_fullfp32_src" ;;
		w_proj_*_qab*)     return 1 ;;
		w_all_*_qab*)      echo "ssm_backbone_classic_quant_src" ;;
		w_proj_*|w_all_*)  echo "ssm_backbone_quant_src" ;;
		*)                 return 1 ;;
	esac
}

# Canonical override folder -> the ORIGINAL canonical source that the
# exporter copied into the setup. Used to check that the setup's own
# ssm_backbone.c really belongs to the family the name says.
original_src() {
	case "$1" in
		ssm_true_int8_src)              echo "$MCU_DIR/ssm_true_int8_src" ;;
		ssm_true_int8_classic_src)      echo "$MCU_DIR/ssm_true_int8_classic_src" ;;
		ssm_backbone_fullfp32_src)      echo "$MCU_DIR/ssm_backbone_src" ;;
		ssm_backbone_quant_src)         echo "$MCU_DIR/ssm_backbone_src" ;;
		ssm_backbone_classic_quant_src) echo "$MCU_DIR/ssm_backbone_classic_src" ;;
	esac
}

# Weight definitions that stay in FLASH, per canonical folder (an ERE
# matched against the definition line). Empty means "tag everything".
#
# selective full_fp32: every weight is fp32, about 236 KB for the
# d_model=64 / d_inner=128 / 2-layer models -- more than the 256 KB main
# SRAM can hold next to the rest of the sketch (the link failed with
# "cannot move location counter backwards", about 11 KB over). out_proj_w
# (64x128 fp32 per layer, 64 KB total) stays in flash; everything else
# (about 172 KB) goes to SRAM. out_proj, not in_proj, so that the larger
# matrix is the one read from SRAM.
flash_keep_re() {
	case "$1" in
		ssm_backbone_fullfp32_src) echo '[A-Za-z0-9_]*out_proj_w[[:space:]]*\[' ;;
		*)                         echo '' ;;
	esac
}

# All five canonical folders must be complete before anything is written.
for canon in ssm_true_int8_src ssm_true_int8_classic_src ssm_backbone_fullfp32_src \
		ssm_backbone_quant_src ssm_backbone_classic_quant_src; do
	for f in ssm_backbone.c ssm_backbone.h; do
		if [ ! -f "$CANON_DIR/$canon/$f" ]; then
			echo "ERROR: missing $CANON_DIR/$canon/$f" >&2
			exit 1
		fi
	done
done

declare -A COUNT=()
FAILED=()
TOTAL=0

for case_name in "${CASES[@]}"; do
	case_dir="$DEPLOY_DIR/$case_name"
	if [ ! -d "$case_dir" ]; then
		echo "ERROR: no such case folder: $case_dir" >&2
		exit 1
	fi

	for setup_dir in "$case_dir"/*/setups/*/; do
		setup_dir="${setup_dir%/}"
		setup="$(basename "$setup_dir")"
		model="$(basename "$(dirname "$(dirname "$setup_dir")")")"
		label="$case_name/$model/$setup"

		if ! canon="$(pick_canonical "$setup")"; then
			echo "FAIL  $label: no canonical backbone for this setup name"
			FAILED+=("$label")
			continue
		fi

		# full_fp32 setup names are the same on selective and classic models.
		# The exported backbone decides (read-only cmp, so this is safe in --dry-run).
		if [ "$canon" = "ssm_backbone_fullfp32_src" ] && \
				cmp -s "$setup_dir/ssm_backbone.c" "$MCU_DIR/ssm_backbone_classic_src/ssm_backbone.c"; then
			canon="ssm_backbone_classic_quant_src"
		fi

		if [ "$DRY_RUN" -eq 1 ]; then
			printf '%-70s -> %s\n' "$label" "$canon"
			COUNT[$canon]=$(( ${COUNT[$canon]:-0} + 1 ))
			TOTAL=$(( TOTAL + 1 ))
			continue
		fi

		# Check 1: the setup's exported backbone is the family we think it is.
		orig="$(original_src "$canon")"
		if ! cmp -s "$setup_dir/ssm_backbone.c" "$orig/ssm_backbone.c"; then
			echo "FAIL  $label: ssm_backbone.c differs from $orig/ssm_backbone.c" \
				"(wrong family, or the export is older than the canonical source)"
			FAILED+=("$label")
			continue
		fi

		# Check 2: the setup's weights are clean (not already SRAM-tagged).
		src_w="$setup_dir/ssm_weights.c"
		if grep -q "time_critical" "$src_w"; then
			echo "FAIL  $label: ssm_weights.c is already SRAM-tagged -- expected a clean export"
			FAILED+=("$label")
			continue
		fi
		n_defs="$(grep -cE "$TAG_RE" "$src_w" || true)"
		if [ "$n_defs" -eq 0 ]; then
			echo "FAIL  $label: no weight definitions matched in ssm_weights.c"
			FAILED+=("$label")
			continue
		fi

		keep_re="$(flash_keep_re "$canon")"
		n_keep=0
		if [ -n "$keep_re" ]; then
			n_keep="$(grep -E "$TAG_RE" "$src_w" | grep -cE "$keep_re" || true)"
			# The keep pattern names arrays that must exist in this family.
			# Zero matches means the generated names changed -- fail instead
			# of silently tagging everything (which would not link).
			if [ "$n_keep" -eq 0 ]; then
				echo "FAIL  $label: flash-keep pattern matched nothing in ssm_weights.c: $keep_re"
				FAILED+=("$label")
				continue
			fi
		fi
		n_expected=$(( n_defs - n_keep ))

		out="$setup_dir/$OVERRIDE_NAME"
		rm -rf -- "$out"
		mkdir -p -- "$out"

		cp -- "$CANON_DIR/$canon/ssm_backbone.c" "$CANON_DIR/$canon/ssm_backbone.h" "$out/"
		cp -- "$setup_dir/ssm_weights.h" "$out/"
		if [ -n "$keep_re" ]; then
			sed -E "/${keep_re}/! s/${TAG_RE}/${TAG} \\1/" "$src_w" > "$out/ssm_weights.c"
		else
			sed -E "s/${TAG_RE}/${TAG} \\1/" "$src_w" > "$out/ssm_weights.c"
		fi

		# Check 3: every matched definition got exactly one tag.
		n_tagged="$(grep -cF "$TAG" "$out/ssm_weights.c" || true)"
		if [ "$n_tagged" -ne "$n_expected" ]; then
			echo "FAIL  $label: tagged $n_tagged definitions, expected $n_expected"
			rm -rf -- "$out"
			FAILED+=("$label")
			continue
		fi

		{
			echo "canonical: $canon"
			echo "weights_tagged: $n_tagged (section $SECTION)"
			echo "weights_in_flash: $n_keep${keep_re:+ (pattern $keep_re)}"
			echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
			(cd "$out" && sha256sum ssm_backbone.c ssm_backbone.h ssm_weights.c ssm_weights.h)
		} > "$out/provenance.txt"

		printf 'OK    %-70s %-32s %4d defs tagged, %d in flash\n' "$label" "$canon" "$n_tagged" "$n_keep"
		COUNT[$canon]=$(( ${COUNT[$canon]:-0} + 1 ))
		TOTAL=$(( TOTAL + 1 ))
	done
done

echo
echo "Summary: $TOTAL setup(s) done, ${#FAILED[@]} failed"
for canon in "${!COUNT[@]}"; do
	printf '  %-32s %d\n' "$canon" "${COUNT[$canon]}"
done
if [ ${#FAILED[@]} -gt 0 ]; then
	echo "Failed:"
	printf '  %s\n' "${FAILED[@]}"
	exit 1
fi