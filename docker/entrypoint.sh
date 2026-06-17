#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/repo"
WORK_ROOT="${WORK_ROOT:-/work}"
REPORT_DIR="${REPORT_DIR:-${WORK_ROOT}/reports}"
SRC_DIR="${SRC_DIR:-${WORK_ROOT}/optimum-intel}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
OUT_DIR="${OUT_DIR:-${WORK_ROOT}/qwen35_ov_int4}"
EXPORT_OUTPUT_DIR="${EXPORT_OUTPUT_DIR:-/work/exported_models/qwen35_08b_int4}"
OPTIMUM_INTEL_REPO="${OPTIMUM_INTEL_REPO:-https://github.com/huggingface/optimum-intel.git}"
OPTIMUM_INTEL_REF="${OPTIMUM_INTEL_REF:-v2.0.0-release}"
OPENVINO_VERSION="${OPENVINO_VERSION:-2026.2.0}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-v5.11.0}"
RUN_FULL_EXPORT="${RUN_FULL_EXPORT:-0}"
RUN_BASELINE_EXPORT="${RUN_BASELINE_EXPORT:-0}"


mkdir -p "${WORK_ROOT}" "${REPORT_DIR}"
exec > >(tee "${REPORT_DIR}/run.log") 2>&1
mkdir -p "$EXPORT_OUTPUT_DIR"

install_transformers() {
  if [[ -d "/opt/transformers-src" ]]; then
    echo "[setup] installing Transformers from local /opt/transformers-src ref=${TRANSFORMERS_REF}"
    pip install -q --upgrade --force-reinstall --no-deps /opt/transformers-src
  else
    echo "[setup] local Transformers source missing; installing release archive ref=${TRANSFORMERS_REF}"
    pip install -q --upgrade --force-reinstall --no-deps \
      "https://github.com/huggingface/transformers/archive/refs/tags/${TRANSFORMERS_REF}.tar.gz"
  fi
}

echo "[repro] started_at=$(date -Iseconds)"
echo "[repro] model_id=${MODEL_ID}"
echo "[repro] optimum_intel_ref=${OPTIMUM_INTEL_REF}"
echo "[repro] openvino_version=${OPENVINO_VERSION}"
echo "[repro] transformers_ref=${TRANSFORMERS_REF}"
echo "[repro] run_full_export=${RUN_FULL_EXPORT}"

python --version
python - <<'PY' || true
import importlib.metadata as m
for pkg in ["openvino","nncf","transformers","torch","torchvision","numpy","pytest"]:
    try: print(f"{pkg}={m.version(pkg)}")
    except Exception as e: print(f"{pkg}=NOT_INSTALLED ({e!r})")
PY

install_transformers

echo "[setup] cloning optimum-intel"
rm -rf "${SRC_DIR}"
git clone --depth 1 --branch "${OPTIMUM_INTEL_REF}" "${OPTIMUM_INTEL_REPO}" "${SRC_DIR}" || {
  rm -rf "${SRC_DIR}"
  git clone "${OPTIMUM_INTEL_REPO}" "${SRC_DIR}"
  cd "${SRC_DIR}"
  git checkout "${OPTIMUM_INTEL_REF}"
}
cd "${SRC_DIR}"
git rev-parse HEAD | tee "${REPORT_DIR}/optimum_intel_commit.txt"

echo "[patch] applying and testing exact source patch"
UNIT_STATUS=0
set +e
REPO_DIR="${SRC_DIR}" SOURCE_PATCHER="${REPO_ROOT}/scripts/apply_source_patch.py" PYTHON_BIN=python TRANSFORMERS_REF="${TRANSFORMERS_REF}" KEEP_TESTS=1 \
  bash "${REPO_ROOT}/scripts/apply_patch_rebuild_test.sh" 2>&1 | tee "${REPORT_DIR}/patch_unit.log"
UNIT_STATUS="${PIPESTATUS[0]}"
set -e

echo "[patch] unit_status=${UNIT_STATUS}"
git diff --stat | tee "${REPORT_DIR}/patched_diff_stat.txt" || true
git diff > "${REPORT_DIR}/patched.diff" || true

FULL_EXPORT_STATUS="not_run"
if [[ "${UNIT_STATUS}" != "0" ]]; then
  echo "[patch][error] patch/unit stage failed; skipping full export"
elif [[ "${RUN_FULL_EXPORT}" == "1" ]]; then
  echo "[full] running patched full export"
  rm -rf "${OUT_DIR}"
  set +e
  python -m optimum.commands.optimum_cli export openvino \
    --model "${MODEL_ID}" \
    --task image-text-to-text \
    --weight-format int4 \
    --group-size 128 \
    --trust-remote-code \
    "$EXPORT_OUTPUT_DIR"
    # "${OUT_DIR}" 2>&1 | tee "${REPORT_DIR}/patched_export.log"
  FULL_EXPORT_STATUS="${PIPESTATUS[0]}"
  set -e
  echo "[full] patched_export_status=${FULL_EXPORT_STATUS}"
  if [[ -d "${OUT_DIR}" ]]; then
    python "${REPO_ROOT}/scripts/inspect_ir.py" "${OUT_DIR}" 2>&1 | tee "${REPORT_DIR}/patched_ir.log" || true
  fi
fi

python "${REPO_ROOT}/scripts/generate_report.py" \
  --report-dir "${REPORT_DIR}" \
  --model-id "${MODEL_ID}" \
  --mode "patched" \
  --optimum-intel-ref "${OPTIMUM_INTEL_REF}" \
  --transformers-ref "${TRANSFORMERS_REF}" \
  --unit-status "${UNIT_STATUS}" \
  --patch-status "$(if [[ "${UNIT_STATUS}" == "0" ]]; then echo applied; else echo failed; fi)" \
  --full-export-status "${FULL_EXPORT_STATUS}" \
  --baseline-export-status "not_run" \
  --output-md "${REPORT_DIR}/report.md" \
  --output-json "${REPORT_DIR}/report.json"

sed -n '1,240p' "${REPORT_DIR}/report.md"

if [[ "${UNIT_STATUS}" != "0" ]]; then exit "${UNIT_STATUS}"; fi
if [[ "${RUN_FULL_EXPORT}" == "1" && "${FULL_EXPORT_STATUS}" != "0" ]]; then exit "${FULL_EXPORT_STATUS}"; fi
