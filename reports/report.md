# Qwen3.5 VLM OpenVINO `use_cache` Reload Reproduction Report

## Summary

- Generated at: `2026-06-15T20:23:43.289283+00:00`
- Model: `Qwen/Qwen3.5-0.8B`
- Mode: `patched`
- Optimum Intel ref: `v2.0.0-release`
- Transformers ref: `v5.11.0`
- Patch status: `applied`
- Focused unit-test status: `0`
- Patched full export status: `0`
- Baseline export status: `not_run`

## Core claim

The public VLM task string can be `image-text-to-text`, while the exported LM IR can still carry a stateful cache contract. The quantization reload path should therefore infer `use_cache` from the exported LM IR rather than only from `task.endswith("with-past")`.

## Detected signals

- Patched export log exists: `True`
- `resolved_use_cache=True` observed: `False`
- Weight compression observed: `True`
- Patched IR has cache/state signal: `True`

## Key patched-export lines

```text
INFO:nncf:Statistics of the bitwidth distribution:
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:13 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:01 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:00 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:02 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:00 • 0:00:00
```

## Key patched-IR lines

```text
[IR] file=openvino_language_model.xml
  inputs= ['attention_mask', 'inputs_embeds', 'position_ids', 'beam_idx']
  has_cache_or_state= True
  ReadValue= 48
  Assign= 48
[IR] file=openvino_text_embeddings_model.xml
  inputs= ['input']
  has_cache_or_state= False
  ReadValue= 0
  Assign= 0
[IR] file=openvino_vision_embeddings_merger_model.xml
  inputs= ['hidden_states', 'attention_mask', 'rotary_pos_emb']
  has_cache_or_state= False
  ReadValue= 0
  Assign= 0
[IR] file=openvino_vision_embeddings_model.xml
  inputs= ['hidden_states']
  has_cache_or_state= False
  ReadValue= 0
  Assign= 0
[IR] file=openvino_vision_embeddings_pos_model.xml
  inputs= ['input']
  has_cache_or_state= False
  ReadValue= 0
  Assign= 0
```

## Unit-test lines

```text
Requirement already satisfied: pytest in /usr/local/lib/python3.11/site-packages (9.1.0)
Requirement already satisfied: iniconfig>=1.0.1 in /usr/local/lib/python3.11/site-packages (from pytest) (2.3.0)
Requirement already satisfied: pluggy<2,>=1.5 in /usr/local/lib/python3.11/site-packages (from pytest) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in /usr/local/lib/python3.11/site-packages (from pytest) (2.20.0)
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
4 passed, 3 warnings in 6.98s
442:            use_cache=self.use_cache,
[done] exact source patch applied, package reinstalled, compile check passed, focused unit tests passed
```

## Baseline lines

```text
Traceback (most recent call last):
ImportError: cannot import name 'Qwen3_5DynamicCache' from 'transformers.models.qwen3_5.modeling_qwen3_5' (/usr/local/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py)
```

## Report artifacts

- `.gitkeep`
- `baseline_export.log`
- `issue_report.json`
- `issue_report.md`
- `issue_repro.log`
- `optimum_intel_commit.txt`
- `package_versions.txt`
- `patch_unit.log`
- `patched.diff`
- `patched_diff_stat.txt`
- `patched_export.log`
- `patched_ir.log`
- `report.json`
- `report.md`
- `run.log`
- `upstream_commit.txt`

## Maintainer interpretation

A passing focused unit test validates the patch mechanics without downloading Qwen3.5. A full export run with `RUN_FULL_EXPORT=1` validates the end-to-end export and NNCF reload path. The important line is `resolved_use_cache=True` for `task=image-text-to-text task_with_past=False`.
