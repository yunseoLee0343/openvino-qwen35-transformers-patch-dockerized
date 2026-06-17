#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
SOURCE_PATCHER="${SOURCE_PATCHER:-$(dirname "$0")/apply_source_patch.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-v5.11.0}"
TEST_DIR="${TEST_DIR:-$REPO_DIR/.tmp_qwen35_vlm_use_cache_tests}"
KEEP_TESTS="${KEEP_TESTS:-0}"

cd "$REPO_DIR"

echo "[1/7] repository: $REPO_DIR"
test -d ".git"
test -f "optimum/exporters/openvino/__main__.py"
test -f "optimum/intel/openvino/modeling_visual_language.py"
test -f "optimum/exporters/openvino/model_patcher.py"

echo "[2/7] applying exact source patcher: $SOURCE_PATCHER"
"$PYTHON_BIN" "$SOURCE_PATCHER" --repo "$REPO_DIR"

echo "[3/7] installing package in editable mode"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e ".[openvino,nncf]" pytest openvino nncf || "$PYTHON_BIN" -m pip install -e . pytest openvino nncf

echo "[3/7] re-installing requested Transformers after editable install"
if [ -d "/opt/transformers-src" ]; then
  "$PYTHON_BIN" -m pip install -q --upgrade --force-reinstall --no-deps /opt/transformers-src
else
  "$PYTHON_BIN" -m pip install -q --upgrade --force-reinstall --no-deps \
    "https://github.com/huggingface/transformers/archive/refs/tags/${TRANSFORMERS_REF}.tar.gz"
fi

echo "[4/7] bytecode compile check"
"$PYTHON_BIN" -m compileall -q \
  optimum/exporters/openvino/__main__.py \
  optimum/intel/openvino/modeling_visual_language.py \
  optimum/exporters/openvino/model_patcher.py

echo "[5/7] OpenVINO opset import compatibility check"
"$PYTHON_BIN" - <<'PY'
import openvino as ov
try:
    from openvino import opset13 as ops
    print("[openvino] using openvino.opset13")
except ImportError:
    from openvino.runtime import opset13 as ops
    print("[openvino] using openvino.runtime.opset13")
print("[openvino] version", getattr(ov, "__version__", "unknown"))
PY

echo "[5/7] writing focused unit tests"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

cat > "$TEST_DIR/test_qwen35_vlm_use_cache_reload.py" <<'PY'
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openvino as ov
try:
    from openvino import opset13 as ops
except ImportError:
    from openvino.runtime import opset13 as ops

from optimum.exporters.openvino import __main__ as ov_export_main


def _make_model(input_name: str, xml_path: Path) -> Path:
    param = ops.parameter([1], dtype=np.float32, name=input_name)
    param.output(0).set_names({input_name})
    result = ops.result(param)
    model = ov.Model([result], [param], "test_model")
    ov.save_model(model, xml_path)
    return xml_path


def test_ov_model_has_cache_input_returns_true():
    param = ops.parameter([1], dtype=np.float32, name="past_key_values.0.key")
    param.output(0).set_names({"past_key_values.0.key"})
    result = ops.result(param)
    model = ov.Model([result], [param], "cache_input_model")
    assert ov_export_main._ov_model_has_cache_or_state(model) is True


def test_ov_model_without_cache_or_state_returns_false():
    param = ops.parameter([1], dtype=np.float32, name="inputs_embeds")
    param.output(0).set_names({"inputs_embeds"})
    result = ops.result(param)
    model = ov.Model([result], [param], "plain_model")
    assert ov_export_main._ov_model_has_cache_or_state(model) is False


def test_exported_visual_model_requires_cache_reads_resolved_lm_ir(tmp_path, monkeypatch):
    lm_xml = _make_model("cache_params.0", tmp_path / "openvino_language_model.xml")
    monkeypatch.setattr(ov_export_main, "_resolve_visual_lm_ir_path", lambda output, model_cls, trust_remote_code=False: lm_xml)
    assert ov_export_main._exported_visual_model_requires_cache(tmp_path, object, True) is True


def test_resolve_visual_lm_ir_path_uses_vlm_mapping(tmp_path, monkeypatch):
    lm_xml = _make_model("inputs_embeds", tmp_path / "openvino_language_model.xml")
    monkeypatch.setattr(ov_export_main.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="qwen3_5"))

    class FakeVisualModel:
        _all_ov_model_paths = {"lm_model": lm_xml.name}

    import optimum.intel.openvino.modeling_visual_language as vlm_mod
    monkeypatch.setitem(vlm_mod.MODEL_TYPE_TO_CLS_MAPPING, "qwen3_5", FakeVisualModel)
    assert ov_export_main._resolve_visual_lm_ir_path(tmp_path, object, True) == lm_xml
PY

echo "[6/7] running focused unit tests"
"$PYTHON_BIN" -m pytest -q "$TEST_DIR"

echo "[7/7] static assertions"
grep -n "use_cache=requested_use_cache" optimum/exporters/openvino/__main__.py
grep -n "use_cache=self.use_cache" optimum/intel/openvino/modeling_visual_language.py
grep -n "DynamicCache as Qwen3_5DynamicCache" optimum/exporters/openvino/model_patcher.py
grep -n "self.layer_types = getattr(self, \"layer_types\"" optimum/exporters/openvino/model_patcher.py
grep -n "\*\*kwargs" optimum/exporters/openvino/model_patcher.py | head
grep -n "def _repair_cached_attention_mask" optimum/exporters/openvino/model_patcher.py
grep -n "attention_mask = _repair_cached_attention_mask" optimum/exporters/openvino/model_patcher.py
python - <<'PY'
from pathlib import Path
mp = Path("optimum/exporters/openvino/model_patcher.py").read_text()
q35 = mp.find("class Qwen3_5ModelPatcher")
qn = mp.find("class Qwen3NextModelPatcher")
assert q35 != -1
q35_text = mp[q35:]
qnext_text = mp[qn:q35] if qn != -1 and qn < q35 else ""
assert 'self.layer_types = getattr(self, "layer_types", getattr(config, "layer_types"))' in q35_text
assert 'self.layer_types = getattr(self, "layer_types", getattr(config, "layer_types"))' not in qnext_text
print("[static] Qwen3_5 metadata reconstruction present; Qwen3Next untouched")
q35_has = mp.find("            def has_previous_state(self):", q35)
prop_before = mp.rfind("            @property\n", q35, q35_has)
assert not (prop_before != -1 and mp[prop_before + len("            @property\n"):q35_has].strip() == "")
print("[static] Qwen3_5 has_previous_state is method, not property")
PY

if [ "$KEEP_TESTS" != "1" ]; then
  rm -rf "$TEST_DIR"
fi

echo "[done] exact source patch applied, package reinstalled, compile check passed, focused unit tests passed"
