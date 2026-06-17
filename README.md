# Qwen3.5 OpenVINO Export Patcher / Reproducer

This repository provides a Docker-based reproducer for the Qwen3.5 OpenVINO export path in `huggingface/optimum-intel`.

The prebuilt image is available on Docker Hub:

```text
yunseolee0343/qwen35_ov_use_cache_repro_repo:latest
```

The image is intended to help reproduce and validate a narrow Optimum Intel patch around OpenVINO VLM quantization reload behavior:

1. Detect whether the exported OpenVINO language-model IR carries a cache/state contract.
2. Preserve `use_cache=True` during the second-stage quantization reload when the public task is `image-text-to-text` but the exported LM IR is stateful.
3. Apply the minimum Qwen3.5 compatibility shims needed to reach and validate that reload path with the selected Transformers version.

## References

Tested reference versions:

```text
optimum-intel ref: v2.0.0-release
openvino: 2026.2.0
transformers ref: v5.11.0
model: Qwen/Qwen3.5-0.8B
task: image-text-to-text
weight format: int4
```

Relevant Optimum Intel source areas:

```text
optimum/exporters/openvino/__main__.py
optimum/intel/openvino/modeling_visual_language.py
optimum/exporters/openvino/model_patcher.py
```

## Quick Start

Pull the image:

```bash
docker pull yunseolee0343/qwen35_ov_use_cache_repro_repo:latest
```

Run the full export/reload reproducer:

```bash
mkdir -p reports exported_models

docker run --rm \
  -e MODEL_ID="Qwen/Qwen3.5-0.8B" \
  -e OPTIMUM_INTEL_REF="v2.0.0-release" \
  -e TRANSFORMERS_REF="v5.11.0" \
  -e RUN_FULL_EXPORT=1 \
  -v "$PWD/reports:/work/reports" \
  -v "$PWD/exported_models:/work/patched_export" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  yunseolee0343/qwen35_ov_use_cache_repro_repo:latest
```

PowerShell:

```powershell
mkdir reports -Force
mkdir exported_models -Force

docker run --rm `
  -e MODEL_ID="Qwen/Qwen3.5-0.8B" `
  -e OPTIMUM_INTEL_REF="v2.0.0-release" `
  -e TRANSFORMERS_REF="v5.11.0" `
  -e RUN_FULL_EXPORT=1 `
  -v "${PWD}\reports:/work/reports" `
  -v "${PWD}\exported_models:/work/patched_export" `
  -v "${HOME}\.cache\huggingface:/root/.cache/huggingface" `
  yunseolee0343/qwen35_ov_use_cache_repro_repo:latest
```

If the model download is rate-limited, pass a Hugging Face token:

```bash
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_ID="Qwen/Qwen3.5-0.8B" \
  -e OPTIMUM_INTEL_REF="v2.0.0-release" \
  -e TRANSFORMERS_REF="v5.11.0" \
  -e RUN_FULL_EXPORT=1 \
  -v "$PWD/reports:/work/reports" \
  -v "$PWD/exported_models:/work/patched_export" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  yunseolee0343/qwen35_ov_use_cache_repro_repo:latest
```

## Expected Artifacts

The container writes logs and reports under the mounted `reports` directory. It writes the patched exported OpenVINO model under the mounted `exported_models` directory.

Useful report files include:

```text
baseline_export.log
patched_export.log
patched_ir.log
patch_unit.log
patched.diff
report.md
report.json
package_versions.txt
optimum_intel_commit.txt
upstream_commit.txt
```

The most important inspection result is the OpenVINO language-model IR state signal:

```text
[IR] file=openvino_language_model.xml
  inputs= [...]
  has_cache_or_state= True
  ReadValue= 48
  Assign= 48
```

This means the exported LM IR is stateful even though the public task string is `image-text-to-text`.

## Patch Scope

The patcher is intentionally broader than the final upstream PR should be, because it must get past compatibility issues before it can validate the quantization reload path.

For an upstream PR, keep the first patch narrow:

```text
1. __main__.py
   - detect cache/state in the exported OpenVINO LM IR
   - resolve requested_use_cache=True for stateful image-text-to-text VLM exports
   - use requested_use_cache in model_cls.from_pretrained(...)

2. modeling_visual_language.py
   - pass use_cache=self.use_cache into OVModelWithEmbedForCausalLM(...)
```

Do not include these local validation edits in the narrow PR:

```text
- convert.py Transformers maximum-version bypass
- _repair_cached_attention_mask(...)
- InferenceDebugger / MaskRepair logging
- broad Qwen3.5 cache rewrites unless submitted as a separate compatibility PR
```

## Why This Reproducer Exists

The current reload path in `_main_quantize()` derives `use_cache` from the task suffix:

```python
use_cache=task.endswith("with-past")
```

For VLM exports, this is not sufficient. A model exported under `image-text-to-text` can still have a stateful language-model IR with `ReadValue` and `Assign` operations.

The reload decision should therefore preserve the exported IR contract, not only the public task string.

## Local Image Tag Used to Publish

The Docker Hub image was published from the local image with:

```bash
docker tag qwen35-ov-use-cache-repro:local yunseolee0343/qwen35_ov_use_cache_repro_repo:latest
docker push yunseolee0343/qwen35_ov_use_cache_repro_repo:latest
```

Users do not need to build the image locally unless they want to modify the reproducer.

## Optional Local Build

If you want to rebuild the image locally:

```bash
docker build \
  --build-arg OPENVINO_VERSION=2026.2.0 \
  --build-arg TRANSFORMERS_REF=v5.11.0 \
  -t qwen35_ov_use_cache_repro_repo:local .
```

Then run it with:

```bash
docker run --rm \
  -e MODEL_ID="Qwen/Qwen3.5-0.8B" \
  -e OPTIMUM_INTEL_REF="v2.0.0-release" \
  -e TRANSFORMERS_REF="v5.11.0" \
  -e RUN_FULL_EXPORT=1 \
  -v "$PWD/reports:/work/reports" \
  -v "$PWD/exported_models:/work/patched_export" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  qwen35_ov_use_cache_repro_repo:local
```

## Notes on Transformers Source

The image caches the selected Transformers source during build under:

```text
/opt/transformers-src
```

After `pip install -e .[openvino,nncf]` installs `optimum-intel`, dependency resolution may change the installed Transformers package. The entrypoint reinstalls Transformers from the cached source instead of cloning it again at runtime.

This avoids transient runtime failures such as:

```text
RPC failed; curl 56 GnuTLS recv error
fatal: early EOF
```

## Maintainer-Facing Interpretation

This reproducer should be read as evidence for a small upstream fix:

```text
image-text-to-text task string != no cache/state contract
```

The exported OpenVINO LM IR should be the source of truth for whether the quantization reload path needs `use_cache=True`.
