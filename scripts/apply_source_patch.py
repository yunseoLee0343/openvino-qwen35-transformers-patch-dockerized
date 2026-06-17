#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path


HELPER_BLOCK = "\n".join([
"def _ov_model_has_cache_or_state(model: ov.Model) -> bool:",
"    # Return True when an exported OpenVINO model carries a cache/state contract.",
"    for inp in model.inputs:",
"        name = inp.get_any_name()",
"        if \"past_key_values\" in name or \"cache_params\" in name:",
"            return True",
"",
"    for op in model.get_ops():",
"        if op.get_type_name() in {\"ReadValue\", \"Assign\"}:",
"            return True",
"",
"    return False",
"",
"",
"def _resolve_visual_lm_ir_path(",
"    output: Path,",
"    model_cls,",
"    trust_remote_code: bool = False,",
") -> Optional[Path]:",
"    config = AutoConfig.from_pretrained(",
"        output,",
"        trust_remote_code=trust_remote_code,",
"    )",
"",
"    from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING",
"",
"    concrete_model_cls = MODEL_TYPE_TO_CLS_MAPPING.get(config.model_type)",
"    if concrete_model_cls is None:",
"        return None",
"",
"    model_file_names = concrete_model_cls._all_ov_model_paths.copy()",
"    lm_file_name = model_file_names.get(\"lm_model\")",
"    if lm_file_name is None:",
"        return None",
"",
"    lm_path = output / lm_file_name",
"    if not lm_path.exists():",
"        return None",
"",
"    return lm_path",
"",
"",
"def _exported_visual_model_requires_cache(",
"    output: Path,",
"    model_cls,",
"    trust_remote_code: bool = False,",
") -> bool:",
"    lm_path = _resolve_visual_lm_ir_path(",
"        output=output,",
"        model_cls=model_cls,",
"        trust_remote_code=trust_remote_code,",
"    )",
"    if lm_path is None:",
"        return False",
"",
"    model = ov.Core().read_model(lm_path)",
"    return _ov_model_has_cache_or_state(model)",
"",
]) + "\n"

REQUESTED_USE_CACHE_BLOCK = "\n".join([
"    requested_use_cache = task.endswith(\"with-past\")",
"    if not requested_use_cache and task == \"image-text-to-text\":",
"        requested_use_cache = _exported_visual_model_requires_cache(",
"            output=output,",
"            model_cls=model_cls,",
"            trust_remote_code=trust_remote_code,",
"        )",
"",
"    logger.info(",
"        \"OpenVINO quantization reload: task=%s task_with_past=%s resolved_use_cache=%s\",",
"        task,",
"        task.endswith(\"with-past\"),",
"        requested_use_cache,",
"    )",
"",
]) + "\n"


def find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    in_str = None
    escaped = False
    i = open_idx
    while i < len(text):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ("'", '"'):
            if text.startswith(ch * 3, i):
                end = text.find(ch * 3, i + 3)
                if end == -1:
                    raise RuntimeError("unterminated triple-quoted string while scanning")
                i = end + 3
                continue
            in_str = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise RuntimeError("no matching parenthesis found")


def patch_main(repo: Path) -> list[str]:
    path = repo / "optimum/exporters/openvino/__main__.py"
    text = path.read_text(encoding="utf-8")
    notes = []

    if "import openvino as ov" not in text:
        marker = "from .utils import ("
        idx = text.find(marker)
        close = text.find("\n)", idx)
        if idx == -1 or close == -1:
            raise RuntimeError("could not locate .utils import block")
        text = text[: close + 2] + "\n\nimport openvino as ov" + text[close + 2 :]
        notes.append("added import openvino as ov")

    if "def _ov_model_has_cache_or_state(" not in text:
        m = re.search(r"^def _main_quantize\(", text, flags=re.M)
        if not m:
            raise RuntimeError("could not locate def _main_quantize")
        text = text[:m.start()] + HELPER_BLOCK + text[m.start():]
        notes.append("inserted exported IR cache helpers")

    m = re.search(r"^def _main_quantize\(", text, flags=re.M)
    if not m:
        raise RuntimeError("could not locate def _main_quantize")
    next_m = re.search(r"^def\s+\w+\(", text[m.end():], flags=re.M)
    start = m.start()
    end = m.end() + next_m.start() if next_m else len(text)
    body = text[start:end]

    if 'requested_use_cache = task.endswith("with-past")' not in body:
        marker = "    # Step 2. Load the exported model"
        rel = body.find(marker)
        if rel == -1:
            raise RuntimeError("could not locate Step 2 marker")
        body = body[:rel] + REQUESTED_USE_CACHE_BLOCK + body[rel:]
        notes.append("inserted requested_use_cache calculation")

    call_m = re.search(r"model\s*=\s*model_cls\.from_pretrained\s*\(", body)
    if not call_m:
        raise RuntimeError("could not locate model_cls.from_pretrained")
    close = find_matching_paren(body, call_m.end() - 1)
    call = body[call_m.start(): close + 1]
    if "use_cache=requested_use_cache" not in call:
        call2 = re.sub(
            r"use_cache\s*=\s*task\.endswith\([\"']with-past[\"']\)",
            "use_cache=requested_use_cache",
            call,
            count=1,
        )
        if call2 == call:
            raise RuntimeError("could not replace task.endswith use_cache")
        body = body[:call_m.start()] + call2 + body[close + 1:]
        notes.append("replaced reload use_cache heuristic")

    text = text[:start] + body + text[end:]
    path.write_text(text, encoding="utf-8", newline="\n")
    return notes

def patch_convert(repo: Path) -> list[str]:
    path = repo / "optimum/exporters/openvino/convert.py"
    text = path.read_text(encoding="utf-8")
    notes = []

    marker = "Source patch: allow Transformers versions newer than MAX_TRANSFORMERS_VERSION"

    if marker not in text:
        old_block = '''    if max_version is not None:
        if isinstance(max_version, Version):
            max_version = max_version.base_version
        if is_transformers_version(">", max_version):
            raise ValueError(
                f"The current version of Transformers does not allow for the export of the model. Maximum required is "
                f"{config.MAX_TRANSFORMERS_VERSION.replace('99', '*')}, got: {_transformers_version}"
            )
'''

        new_block = '''    if max_version is not None:
        if isinstance(max_version, Version):
            max_version = max_version.base_version
        if is_transformers_version(">", max_version):
            # Source patch: allow Transformers versions newer than MAX_TRANSFORMERS_VERSION.
            # Required for validating the patched Qwen3.5 Transformers v5.11 cache contract.
            logger.warning(
                "Exporting with unsupported Transformers version %s; configured maximum is %s. "
                "Compatibility is provided by the local Qwen3.5 source patches.",
                _transformers_version,
                config.MAX_TRANSFORMERS_VERSION.replace("99", "*"),
            )
'''

        if old_block not in text:
            raise RuntimeError(
                "could not locate maximum Transformers version check in convert.py"
            )

        text = text.replace(old_block, new_block, 1)
        notes.append("allowed Transformers versions newer than export config maximum")

    path.write_text(text, encoding="utf-8", newline="\n")
    return notes


def patch_vlm(repo: Path) -> list[str]:
    path = repo / "optimum/intel/openvino/modeling_visual_language.py"
    text = path.read_text(encoding="utf-8")
    notes = []
    if "use_cache=self.use_cache" not in text:
        needle = "            compile_only=self._compile_only,\n"
        if needle not in text:
            raise RuntimeError("could not locate compile_only anchor in modeling_visual_language.py")
        text = text.replace(needle, needle + "            use_cache=self.use_cache,\n", 1)
        notes.append("propagated use_cache into OVModelWithEmbedForCausalLM")
        path.write_text(text, encoding="utf-8", newline="\n")
    return notes


def patch_model_patcher(repo: Path) -> list[str]:
    path = repo / "optimum/exporters/openvino/model_patcher.py"
    text = path.read_text(encoding="utf-8")
    notes = []

    repair_helper = (
        "import logging\n"
        "logger = logging.getLogger('InferenceDebugger')\n"
        "\n"
        "def _repair_cached_attention_mask(\n"
        "    attention_mask,\n"
        "    inputs_embeds,\n"
        "    wrapped_cache_params,\n"
        "):\n"
        "    if (\n"
        "        wrapped_cache_params is None\n"
        "        or attention_mask is None\n"
        "        or inputs_embeds is None\n"
        "    ):\n"
        "        return attention_mask\n"
        "\n"
        "    past_len = wrapped_cache_params.get_seq_length()\n"
        "    current_len = inputs_embeds.shape[1]\n"
        "    mask_len = attention_mask.shape[-1]\n"
        "\n"
        "    logger.warning(f'[MaskRepair] Entry - past_len: {past_len}, current_len: {current_len}, mask_len: {mask_len}, mask_shape: {attention_mask.shape}')\n"
        "\n"
        "    if past_len > 0 and mask_len == current_len:\n"
        "        logger.warning(f'[MaskRepair] Condition Met: Concat triggered.')\n"
        "        past_mask = torch.ones(\n"
        "            attention_mask.shape[0],\n"
        "            past_len,\n"
        "            dtype=attention_mask.dtype,\n"
        "            device=attention_mask.device,\n"
        "        )\n"
        "        attention_mask = torch.cat([past_mask, attention_mask], dim=-1)\n"
        "        logger.warning(f'[MaskRepair] Final mask shape: {attention_mask.shape}')\n"
        "    else:\n"
        "        logger.warning(f'[MaskRepair] Condition Failed: Skipping Concat.')\n"
        "        \n"
        "    return attention_mask\n"
        "\n"
        "\n"
    )
    
    repair_call = (
        "            attention_mask = _repair_cached_attention_mask(\n"
        "                attention_mask=attention_mask,\n"
        "                inputs_embeds=inputs_embeds,\n"
        "                wrapped_cache_params=wrapped_cache_params,\n"
        "            )\n"
        "\n"
    )

    # Qwen3_5 import fallback only.
    old_import = "        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache\n"
    new_import = (
        "        try:\n"
        "            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache\n"
        "        except ImportError:\n"
        "            from transformers.cache_utils import DynamicCache as Qwen3_5DynamicCache\n"
    )
    if old_import in text:
        text = text.replace(old_import, new_import, 1)
        notes.append("added Qwen3_5DynamicCache fallback import")
    elif "DynamicCache as Qwen3_5DynamicCache" not in text:
        raise RuntimeError("could not locate Qwen3_5DynamicCache import")

    # Qwen3_5 gated delta net forward **kwargs only.
    fwd_start = text.find("def qwen3_5_gated_delta_net_forward(")
    fwd_end = text.find("):", fwd_start)
    if fwd_start == -1 or fwd_end == -1:
        raise RuntimeError("could not locate qwen3_5_gated_delta_net_forward signature")
    fwd_sig = text[fwd_start:fwd_end + 3]
    if "**kwargs" not in fwd_sig:
        if "    attention_mask: Optional[torch.Tensor] = None,\n" not in fwd_sig:
            raise RuntimeError("could not locate attention_mask line in qwen3_5_gated_delta_net_forward signature")
        fwd_sig2 = fwd_sig.replace(
            "    attention_mask: Optional[torch.Tensor] = None,\n",
            "    attention_mask: Optional[torch.Tensor] = None,\n"
            "    **kwargs,\n",
            1,
        )
        text = text[:fwd_start] + fwd_sig2 + text[fwd_end + 3:]
        notes.append("added **kwargs to qwen3_5_gated_delta_net_forward")

        # Synchronize flattened OpenVINO KV inputs with Transformers DynamicCache layers.
    qwen35_cls = text.find("class Qwen3_5ModelPatcher")
    qwen35_end = text.find("\nclass ", qwen35_cls + 1)
    if qwen35_end == -1:
        qwen35_end = len(text)

    qwen35_text = text[qwen35_cls:qwen35_end]

    if "def _seed_full_attention_layers_from_flat_cache(" not in qwen35_text:
        update_start_rel = qwen35_text.find(
            "            def update(\n"
        )
        has_previous_rel = qwen35_text.find(
            "            def has_previous_state(self):"
        )

        if update_start_rel == -1 or has_previous_rel == -1:
            raise RuntimeError(
                "could not locate Qwen3_5DynamicCacheWrap cache methods"
            )

        update_start = qwen35_cls + update_start_rel
        has_previous_start = qwen35_cls + has_previous_rel

        synchronized_cache_methods = (
            "            def _seed_full_attention_layers_from_flat_cache(self):\n"
            "                for model_layer_idx, flat_cache_idx in self.full_attn_mapping.items():\n"
            "                    if flat_cache_idx >= len(self.key_cache):\n"
            "                        continue\n"
            "\n"
            "                    key = self.key_cache[flat_cache_idx]\n"
            "                    value = self.value_cache[flat_cache_idx]\n"
            "                    if key is None or value is None:\n"
            "                        continue\n"
            "\n"
            "                    layer = self.layers[model_layer_idx]\n"
            "                    layer.keys = key\n"
            "                    layer.values = value\n"
            "                    layer.dtype = key.dtype\n"
            "                    layer.device = key.device\n"
            "                    layer.is_initialized = True\n"
            "\n"
            "            def update(\n"
            "                self,\n"
            "                key_states: torch.Tensor,\n"
            "                value_states: torch.Tensor,\n"
            "                layer_idx: int,\n"
            "                cache_kwargs: Optional[dict[str, Any]] = None,\n"
            "            ) -> tuple[torch.Tensor, torch.Tensor]:\n"
            "                flat_cache_idx = self.full_attn_mapping[layer_idx]\n"
            "\n"
            "                key_states, value_states = self.layers[layer_idx].update(\n"
            "                    key_states,\n"
            "                    value_states,\n"
            "                )\n"
            "\n"
            "                self.key_cache[flat_cache_idx] = key_states\n"
            "                self.value_cache[flat_cache_idx] = value_states\n"
            "                return key_states, value_states\n"
            "\n"
            "            def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:\n"
            "                if layer_idx not in self.transformer_layers:\n"
            "                    layer_idx = self.transformer_layers[0]\n"
            "                return self.layers[layer_idx].get_seq_length()\n"
            "\n"
            "            def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:\n"
            "                if layer_idx not in self.transformer_layers:\n"
            "                    layer_idx = self.transformer_layers[0]\n"
            "                return self.layers[layer_idx].get_mask_sizes(query_length)\n"
            "\n"
        )

        text = (
            text[:update_start]
            + synchronized_cache_methods
            + text[has_previous_start:]
        )

        notes.append(
            "synchronized Qwen3_5 flattened KV cache with parent DynamicCache layers"
        )

        qwen35_cls = text.find("class Qwen3_5ModelPatcher")
    seed_method = text.find(
        "            def _seed_full_attention_layers_from_flat_cache(self):",
        qwen35_cls,
    )

    if seed_method == -1:
        raise RuntimeError("could not locate inserted DynamicCache seed method")

    seed_call = (
        "                self._seed_full_attention_layers_from_flat_cache()\n"
        "\n"
    )

    init_region = text[qwen35_cls:seed_method]
    if "self._seed_full_attention_layers_from_flat_cache()" not in init_region:
        text = text[:seed_method] + seed_call + text[seed_method:]
        notes.append("seeded parent DynamicCache layers from flattened KV inputs")

    qwen35_cls = text.find("class Qwen3_5ModelPatcher")
    if qwen35_cls == -1:
        raise RuntimeError("could not locate class Qwen3_5ModelPatcher")

    # Helper for cached export path where attention_mask covers only current_len,
    # while wrapped cache already carries past_len.
    if "def _repair_cached_attention_mask(" not in text:
        text = text[:qwen35_cls] + repair_helper + text[qwen35_cls:]
        notes.append("added Qwen3_5 cached attention_mask repair helper")
        qwen35_cls = text.find("class Qwen3_5ModelPatcher")
        if qwen35_cls == -1:
            raise RuntimeError("could not relocate class Qwen3_5ModelPatcher after helper insertion")

    # Qwen3_5DynamicCacheWrap metadata reconstruction only.
    dyn_anchor = (
        "                # Call parent constructor with all required arguments\n"
        "                super().__init__(config=config)\n"
        "\n"
        "                self.conv_states = conv_states\n"
    )
    dyn_repl = (
        "                # Call parent constructor with all required arguments\n"
        "                super().__init__(config=config)\n"
        "\n"
        "                self.layer_types = getattr(self, \"layer_types\", getattr(config, \"layer_types\"))\n"
        "                self.transformer_layers = getattr(\n"
        "                    self,\n"
        "                    \"transformer_layers\",\n"
        "                    [i for i, layer_type in enumerate(self.layer_types) if layer_type == \"full_attention\"],\n"
        "                )\n"
        "                self.linear_attention_layers = getattr(\n"
        "                    self,\n"
        "                    \"linear_attention_layers\",\n"
        "                    [i for i, layer_type in enumerate(self.layer_types) if layer_type == \"linear_attention\"],\n"
        "                )\n"
        "                self.last_linear_layer = getattr(\n"
        "                    self,\n"
        "                    \"last_linear_layer\",\n"
        "                    self.linear_attention_layers[-1] if len(self.linear_attention_layers) > 0 else None,\n"
        "                )\n"
        "\n"
        "                self.conv_states = conv_states\n"
    )
    if "self.layer_types = getattr(self, \"layer_types\", getattr(config, \"layer_types\"))" not in text[qwen35_cls:]:
        rel = text.find(dyn_anchor, qwen35_cls)
        if rel == -1:
            raise RuntimeError("could not locate Qwen3_5DynamicCacheWrap __init__ metadata anchor")
        text = text[:rel] + dyn_repl + text[rel + len(dyn_anchor):]
        notes.append("reconstructed Qwen3_5DynamicCacheWrap layer metadata from config.layer_types")

    # Insert attention-mask repair immediately after wrapped_cache_params is constructed.
    qwen35_cls = text.find("class Qwen3_5ModelPatcher")
    qwen35_end = text.find("\nclass ", qwen35_cls + 1)
    if qwen35_end == -1:
        qwen35_end = len(text)
    qwen35_text = text[qwen35_cls:qwen35_end]

    if "attention_mask = _repair_cached_attention_mask(" not in qwen35_text:
        wrap_assign_rel = qwen35_text.find("wrapped_cache_params = Qwen3_5DynamicCacheWrap(")
        if wrap_assign_rel == -1:
            raise RuntimeError("could not locate wrapped_cache_params = Qwen3_5DynamicCacheWrap(...) assignment")

        wrap_abs = qwen35_cls + wrap_assign_rel
        open_paren = text.find("(", wrap_abs)
        if open_paren == -1:
            raise RuntimeError("could not locate Qwen3_5DynamicCacheWrap call open parenthesis")

        close_paren = find_matching_paren(text, open_paren)
        insert_at = close_paren + 1
        if insert_at < len(text) and text[insert_at] == "\n":
            insert_at += 1

        text = text[:insert_at] + repair_call + text[insert_at:]
        notes.append("repaired cached Qwen3_5 attention_mask length after cache wrapping")

    # Qwen3_5 has_previous_state must be a method for Transformers v5.11.0,
    # which calls past_key_values.has_previous_state(). Do not touch Qwen3Next.
    qwen35_cls = text.find("class Qwen3_5ModelPatcher")
    if qwen35_cls == -1:
        raise RuntimeError("could not relocate class Qwen3_5ModelPatcher")

    qwen35_has = text.find("            def has_previous_state(self):", qwen35_cls)
    if qwen35_has == -1:
        raise RuntimeError("could not locate Qwen3_5 has_previous_state")

    prop_before = text.rfind("            @property\n", qwen35_cls, qwen35_has)
    if prop_before != -1:
        between = text[prop_before + len("            @property\n"):qwen35_has]
        if between.strip() == "":
            text = text[:prop_before] + text[qwen35_has:]
            qwen35_has = prop_before
            notes.append("removed @property from Qwen3_5 has_previous_state")

    guard_window = text[qwen35_has:qwen35_has + 240]
    if "if self.last_linear_layer is None:" not in guard_window:
        insert_at = text.find("\n", qwen35_has) + 1
        guard = (
            "                if self.last_linear_layer is None:\n"
            "                    return False\n"
            "\n"
        )
        text = text[:insert_at] + guard + text[insert_at:]
        notes.append("added Qwen3_5 has_previous_state last_linear_layer None guard")

    path.write_text(text, encoding="utf-8", newline="\n")
    return notes


def validate(repo: Path) -> None:
    main = (repo / "optimum/exporters/openvino/__main__.py").read_text(encoding="utf-8")
    vlm = (repo / "optimum/intel/openvino/modeling_visual_language.py").read_text(encoding="utf-8")
    mp = (repo / "optimum/exporters/openvino/model_patcher.py").read_text(encoding="utf-8")
    convert = (repo / "optimum/exporters/openvino/convert.py").read_text(
        encoding="utf-8"
    )
    for marker in [
        "def _ov_model_has_cache_or_state(",
        "def _resolve_visual_lm_ir_path(",
        "def _exported_visual_model_requires_cache(",
        'requested_use_cache = task.endswith("with-past")',
        "use_cache=requested_use_cache",
    ]:
        if marker not in main:
            raise RuntimeError(f"missing marker in __main__.py: {marker}")
        
    for marker in [
        "def _seed_full_attention_layers_from_flat_cache(self):",
        "self._seed_full_attention_layers_from_flat_cache()",
        "key_states, value_states = self.layers[layer_idx].update(",
        "return self.layers[layer_idx].get_seq_length()",
        "return self.layers[layer_idx].get_mask_sizes(query_length)",
    ]:
        if marker not in mp:
            raise RuntimeError(f"missing marker in model_patcher.py: {marker}")

    if 'use_cache=task.endswith("with-past")' in main:
        raise RuntimeError("stale task.endswith use_cache remains")
    if "use_cache=self.use_cache" not in vlm:
        raise RuntimeError("missing use_cache propagation in modeling_visual_language.py")
    if "DynamicCache as Qwen3_5DynamicCache" not in mp:
        raise RuntimeError("missing Qwen3_5DynamicCache fallback")
    qwen35_cls = mp.find("class Qwen3_5ModelPatcher")
    qwen_next_cls = mp.find("class Qwen3NextModelPatcher")
    if qwen35_cls == -1:
        raise RuntimeError("missing class Qwen3_5ModelPatcher")
    qwen35_text = mp[qwen35_cls:]
    if "def _repair_cached_attention_mask(" not in mp:
        raise RuntimeError("missing Qwen3_5 cached attention_mask repair helper")
    qwennext_text = mp[qwen_next_cls:qwen35_cls] if qwen_next_cls != -1 and qwen_next_cls < qwen35_cls else ""
    if "self.layer_types = getattr(self, \"layer_types\", getattr(config, \"layer_types\"))" not in qwen35_text:
        raise RuntimeError("missing Qwen3_5DynamicCacheWrap layer metadata reconstruction")
    if "self.layer_types = getattr(self, \"layer_types\", getattr(config, \"layer_types\"))" in qwennext_text:
        raise RuntimeError("unexpected Qwen3Next metadata reconstruction")
    fwd_start = mp.find("def qwen3_5_gated_delta_net_forward(")
    fwd_end = mp.find("):", fwd_start)
    if fwd_start == -1 or fwd_end == -1 or "**kwargs" not in mp[fwd_start:fwd_end]:
        raise RuntimeError("missing qwen3_5_gated_delta_net_forward **kwargs compatibility")
    qwen35_has = mp.find("            def has_previous_state(self):", qwen35_cls)
    if qwen35_has == -1 or "if self.last_linear_layer is None:" not in mp[qwen35_has:qwen35_has + 240]:
        raise RuntimeError("missing Qwen3_5 has_previous_state guard")
    prop_before = mp.rfind("            @property\n", qwen35_cls, qwen35_has)
    if prop_before != -1 and mp[prop_before + len("            @property\n"):qwen35_has].strip() == "":
        raise RuntimeError("Qwen3_5 has_previous_state must be a method, not @property")
    if "Source patch: allow Transformers versions newer than MAX_TRANSFORMERS_VERSION" not in convert:
        raise RuntimeError("missing Transformers maximum-version compatibility bypass")
    if "if is_transformers_version(\"<\", min_version):" not in convert:
        raise RuntimeError("minimum Transformers version check must remain enabled")
    if (
        "if is_transformers_version(\">\", max_version):" in convert
        and "Maximum required is " in convert
        and "raise ValueError(" in convert[
            convert.find('if is_transformers_version(">", max_version):'):
            convert.find('if stateful:')
        ]
    ):
        raise RuntimeError("stale maximum Transformers version ValueError remains")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    notes = []
    notes += patch_main(repo)
    notes += patch_convert(repo)
    notes += patch_vlm(repo)
    notes += patch_model_patcher(repo)
    validate(repo)
    print("[source-patcher] applied:")
    for note in notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
