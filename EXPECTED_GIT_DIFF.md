```diff
diff --git a/optimum/exporters/openvino/__main__.py b/optimum/exporters/openvino/__main__.py
index d2bfda8..d215fb9 100644
--- a/optimum/exporters/openvino/__main__.py
+++ b/optimum/exporters/openvino/__main__.py
@@ -52,6 +52,8 @@ from .utils import (
     patch_qwenvl_configs,
 )

+import openvino as ov
+

 if is_transformers_version(">=", "4.55"):
     from transformers import Mxfp4Config
@@ -615,6 +617,64 @@ def main_export(
                 AutoBitLinear.load_hook = orig_load_hook


+def _ov_model_has_cache_or_state(model: ov.Model) -> bool:
+    # Return True when an exported OpenVINO model carries a cache/state contract.
+    for inp in model.inputs:
+        name = inp.get_any_name()
+        if "past_key_values" in name or "cache_params" in name:
+            return True
+
+    for op in model.get_ops():
+        if op.get_type_name() in {"ReadValue", "Assign"}:
+            return True
+
+    return False
+
+
+def _resolve_visual_lm_ir_path(
+    output: Path,
+    model_cls,
+    trust_remote_code: bool = False,
+) -> Optional[Path]:
+    config = AutoConfig.from_pretrained(
+        output,
+        trust_remote_code=trust_remote_code,
+    )
+
+    from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING
+
+    concrete_model_cls = MODEL_TYPE_TO_CLS_MAPPING.get(config.model_type)
+    if concrete_model_cls is None:
+        return None
+
+    model_file_names = concrete_model_cls._all_ov_model_paths.copy()
+    lm_file_name = model_file_names.get("lm_model")
+    if lm_file_name is None:
+        return None
+
+    lm_path = output / lm_file_name
+    if not lm_path.exists():
+        return None
+
+    return lm_path
+
+
+def _exported_visual_model_requires_cache(
+    output: Path,
+    model_cls,
+    trust_remote_code: bool = False,
+) -> bool:
+    lm_path = _resolve_visual_lm_ir_path(
+        output=output,
+        model_cls=model_cls,
+        trust_remote_code=trust_remote_code,
+    )
+    if lm_path is None:
+        return False
+
+    model = ov.Core().read_model(lm_path)
+    return _ov_model_has_cache_or_state(model)
+
 def _main_quantize(
     model_name_or_path: str,
     task: str,
@@ -727,13 +787,28 @@ def _main_quantize(
         except (AttributeError, ImportError, KeyError) as e:
             raise RuntimeError(f"Wasn't able to locate OpenVINO class for task {original_task} ({task}).") from e

+    requested_use_cache = task.endswith("with-past")
+    if not requested_use_cache and task == "image-text-to-text":
+        requested_use_cache = _exported_visual_model_requires_cache(
+            output=output,
+            model_cls=model_cls,
+            trust_remote_code=trust_remote_code,
+        )
+
+    logger.info(
+        "OpenVINO quantization reload: task=%s task_with_past=%s resolved_use_cache=%s",
+        task,
+        task.endswith("with-past"),
+        requested_use_cache,
+    )
+
     # Step 2. Load the exported model
     model = model_cls.from_pretrained(
         output,
         compile=False,
         trust_remote_code=trust_remote_code,
         cache_dir=cache_dir,
-        use_cache=task.endswith("with-past"),
+        use_cache=requested_use_cache,
         **(model_kwargs or {}),
     )

diff --git a/optimum/exporters/openvino/model_patcher.py b/optimum/exporters/openvino/model_patcher.py
index 2ad7124..qwen35 100644
--- a/optimum/exporters/openvino/model_patcher.py
+++ b/optimum/exporters/openvino/model_patcher.py
@@ -9350,6 +9350,7 @@ def qwen3_5_gated_delta_net_forward(
     cache_params=None,
     cache_position: Optional[torch.LongTensor] = None,
     attention_mask: Optional[torch.Tensor] = None,
+    **kwargs,
 ):
     def apply_mask_to_padding_states(hidden_states, attention_mask):
         """
@@ -9445,7 +9446,10 @@ class Qwen3_5ModelPatcher(OVDecoderModelPatcher):
         model: "PreTrainedModel",
         model_kwargs: Optional[Dict[str, Any]] = None,
     ):
-        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
+        try:
+            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
+        except ImportError:
+            from transformers.cache_utils import DynamicCache as Qwen3_5DynamicCache

         from openvino.frontend.pytorch import ConversionExtension, ModuleExtension

@@ -9464,6 +9468,23 @@ class Qwen3_5ModelPatcher(OVDecoderModelPatcher):
                 # Call parent constructor with all required arguments
                 super().__init__(config=config)

+                self.layer_types = getattr(self, "layer_types", getattr(config, "layer_types"))
+                self.transformer_layers = getattr(
+                    self,
+                    "transformer_layers",
+                    [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"],
+                )
+                self.linear_attention_layers = getattr(
+                    self,
+                    "linear_attention_layers",
+                    [i for i, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"],
+                )
+                self.last_linear_layer = getattr(
+                    self,
+                    "last_linear_layer",
+                    self.linear_attention_layers[-1] if len(self.linear_attention_layers) > 0 else None,
+                )
+
                 self.conv_states = conv_states
                 self.recurrent_states = recurrent_states
                 self.key_cache = key_cache
@@ -9511,7 +9532,9 @@ class Qwen3_5ModelPatcher(OVDecoderModelPatcher):
                 if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
                     return 0
                 return self.key_cache[layer_idx].shape[-2]

-            @property
             def has_previous_state(self):
+                if self.last_linear_layer is None:
+                    return False
+
                 """We have a previous state if the last linear (conv) layer was already updated."""
                 layer_idx = self.linear_attn_mapping[self.last_linear_layer]
                 return self.conv_states[layer_idx] is not None
diff --git a/optimum/intel/openvino/modeling_visual_language.py b/optimum/intel/openvino/modeling_visual_language.py
index 076e3ff..15467b4 100644
--- a/optimum/intel/openvino/modeling_visual_language.py
+++ b/optimum/intel/openvino/modeling_visual_language.py
@@ -439,6 +439,7 @@ class OVModelForVisualCausalLM(OVBaseModel, GenerationMixin):
             quantization_config=quantization_config,
             compile=self._compile_only or enable_compilation,
             compile_only=self._compile_only,
+            use_cache=self.use_cache,
         )
         self.vision_embeddings = OVVisionEmbedding(vision_embeddings, self)
         for part in self.additional_parts:

```
