import sys
from pathlib import Path
import openvino as ov

def has_cache_or_state(model: ov.Model) -> bool:
    for inp in model.inputs:
        name = inp.get_any_name()
        if "past_key_values" in name or "cache_params" in name:
            return True
    for op in model.get_ops():
        if op.get_type_name() in {"ReadValue", "Assign"}:
            return True
    return False

def main():
    out = Path(sys.argv[1])
    core = ov.Core()

    SKIP_IR_NAMES = {
        "openvino_tokenizer.xml",
        "openvino_detokenizer.xml",
    }

    for xml in sorted(out.glob("*.xml")):
        if xml.name in SKIP_IR_NAMES:
            print(f"[IR signature] skipped extension model: {xml.name}")
            continue

        model = core.read_model(xml)
        op_counts = {}
        for op in model.get_ops():
            op_counts[op.get_type_name()] = op_counts.get(op.get_type_name(), 0) + 1
        print(f"[IR] file={xml.name}")
        print("  inputs=", [i.get_any_name() for i in model.inputs])
        print("  outputs=", [o.get_any_name() for o in model.outputs])
        print("  has_cache_or_state=", has_cache_or_state(model))
        print("  ReadValue=", op_counts.get("ReadValue", 0))
        print("  Assign=", op_counts.get("Assign", 0))
        print("  Parameter=", op_counts.get("Parameter", 0))
        print("  Result=", op_counts.get("Result", 0))

if __name__ == "__main__":
    main()
