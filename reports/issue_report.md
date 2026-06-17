# Issue Reproduction Report: Unpatched Qwen3.5 VLM OpenVINO INT4 Export

## Purpose

This report is generated from an **unpatched upstream** `optimum-intel` checkout. It is intended for issue filing, not for validating the proposed fix.

## Configuration

- Generated at: `2026-06-13T04:01:28.936841+00:00`
- Model: `Qwen/Qwen3.5-0.8B`
- Optimum Intel ref: `v2.0.0`
- Transformers ref: `v5.11.0`
- Baseline export status: `1`

## Expected issue signal

The public task is `image-text-to-text`, so the unpatched reload path derives `use_cache` from `task.endswith("with-past")`, which is false. If the exported LM IR is stateful, that task suffix is not enough to preserve the actual IR cache/state contract.

## Observed signals

- Export log exists: `True`
- Weight compression observed: `False`
- Traceback / exception observed: `True`
- IR log exists: `False`
- IR has cache/state signal: `False`
- ReadValue signal observed: `False`
- Assign signal observed: `False`

## Key export lines

```text
Traceback (most recent call last):
ImportError: cannot import name 'Qwen3_5DynamicCache' from 'transformers.models.qwen3_5.modeling_qwen3_5' (/usr/local/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py)
```

## Key IR lines

```text
```

## Package versions

```text
openvino=2026.2.0
nncf=3.2.0
optimum=2.2.0
optimum-intel=2.0.0
transformers=5.11.0
torch=2.12.0+cpu
torchvision=0.27.0+cpu
numpy=2.4.6
```

## Artifacts

- `.gitkeep`
- `baseline_export.log`
- `issue_repro.log`
- `optimum_intel_commit.txt`
- `package_versions.txt`
- `patch_unit.log`
- `patched.diff`
- `patched_diff_stat.txt`
- `patched_export.log`
- `report.json`
- `report.md`
- `run.log`
- `upstream_commit.txt`

## Maintainer note

This repository has a separate patched reproducer. The issue-facing baseline report should be attached first. Then the patched report can be used to show that resolving `use_cache` from the exported LM IR changes the reload decision to `resolved_use_cache=True` and allows the export path to complete.
