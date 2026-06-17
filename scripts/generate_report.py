#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path


def read_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def contains(path: Path, needle: str) -> bool:
    return needle in read_text(path, max_chars=2_000_000)


def extract_lines(path: Path, patterns: list[str], limit: int = 40) -> list[str]:
    text = read_text(path, max_chars=2_000_000)
    out: list[str] = []
    for line in text.splitlines():
        if any(p in line for p in patterns):
            out.append(line)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--optimum-intel-ref", required=True)
    ap.add_argument("--transformers-ref", required=True)
    ap.add_argument("--unit-status", required=True)
    ap.add_argument("--patch-status", required=True)
    ap.add_argument("--full-export-status", required=True)
    ap.add_argument("--baseline-export-status", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    patched_export_log = report_dir / "patched_export.log"
    baseline_export_log = report_dir / "baseline_export.log"
    patched_ir_log = report_dir / "patched_ir.log"
    patch_unit_log = report_dir / "patch_unit.log"

    resolved_use_cache_true = contains(patched_export_log, "resolved_use_cache=True") or contains(
        patched_export_log, "resolved_use_cache true"
    )
    weight_compression_done = contains(patched_export_log, "Applying Weight Compression")
    ir_has_state = contains(patched_ir_log, "has_cache_or_state= True") or contains(
        patched_ir_log, "has_cache_or_state=True"
    )

    key_lines = {
        "patched_export": extract_lines(
            patched_export_log,
            ["resolved_use_cache", "Applying Weight Compression", "Statistics of the bitwidth"],
        ),
        "patched_ir": extract_lines(
            patched_ir_log,
            ["[IR]", "has_cache_or_state", "ReadValue", "Assign", "inputs="],
        ),
        "unit": extract_lines(
            patch_unit_log,
            ["passed", "failed", "[done]", "pytest", "use_cache=self.use_cache"],
        ),
        "baseline": extract_lines(
            baseline_export_log,
            ["error", "Error", "Traceback", "resolved_use_cache", "Applying Weight Compression"],
        ),
    }

    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "model_id": args.model_id,
        "mode": args.mode,
        "optimum_intel_ref": args.optimum_intel_ref,
        "transformers_ref": args.transformers_ref,
        "patch_status": args.patch_status,
        "unit_status": args.unit_status,
        "full_export_status": args.full_export_status,
        "baseline_export_status": args.baseline_export_status,
        "signals": {
            "patched_export_log_exists": patched_export_log.exists(),
            "resolved_use_cache_true": resolved_use_cache_true,
            "weight_compression_seen": weight_compression_done,
            "patched_ir_has_cache_or_state": ir_has_state,
        },
        "artifacts": sorted(p.name for p in report_dir.glob("*")),
        "key_lines": key_lines,
    }

    md_lines = [
        "# Qwen3.5 VLM OpenVINO `use_cache` Reload Reproduction Report",
        "",
        "## Summary",
        "",
        f"- Generated at: `{status['generated_at']}`",
        f"- Model: `{args.model_id}`",
        f"- Mode: `{args.mode}`",
        f"- Optimum Intel ref: `{args.optimum_intel_ref}`",
        f"- Transformers ref: `{args.transformers_ref}`",
        f"- Patch status: `{args.patch_status}`",
        f"- Focused unit-test status: `{args.unit_status}`",
        f"- Patched full export status: `{args.full_export_status}`",
        f"- Baseline export status: `{args.baseline_export_status}`",
        "",
        "## Core claim",
        "",
        "The public VLM task string can be `image-text-to-text`, while the exported LM IR can still carry a stateful cache contract. The quantization reload path should therefore infer `use_cache` from the exported LM IR rather than only from `task.endswith(\"with-past\")`.",
        "",
        "## Detected signals",
        "",
        f"- Patched export log exists: `{patched_export_log.exists()}`",
        f"- `resolved_use_cache=True` observed: `{resolved_use_cache_true}`",
        f"- Weight compression observed: `{weight_compression_done}`",
        f"- Patched IR has cache/state signal: `{ir_has_state}`",
        "",
        "## Key patched-export lines",
        "",
        "```text",
        *key_lines["patched_export"],
        "```",
        "",
        "## Key patched-IR lines",
        "",
        "```text",
        *key_lines["patched_ir"],
        "```",
        "",
        "## Unit-test lines",
        "",
        "```text",
        *key_lines["unit"],
        "```",
        "",
        "## Baseline lines",
        "",
        "```text",
        *key_lines["baseline"],
        "```",
        "",
        "## Report artifacts",
        "",
        *[f"- `{name}`" for name in status["artifacts"]],
        "",
        "## Maintainer interpretation",
        "",
        "A passing focused unit test validates the patch mechanics without downloading Qwen3.5. A full export run with `RUN_FULL_EXPORT=1` validates the end-to-end export and NNCF reload path. The important line is `resolved_use_cache=True` for `task=image-text-to-text task_with_past=False`.",
        "",
    ]

    Path(args.output_md).write_text("\n".join(md_lines), encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
