"""Pinned SmolVLA checkpoint and dependency constants."""

DEFAULT_SMOLVLA_REVISION = "3326b100334ffc0a0bd1ec27e3afb1cfa2a6000c"
EXPECTED_LEROBOT_VERSION = "0.3.3"
WEIGHTS_FILENAME = "model.safetensors"

VERIFICATION_TENSORS = {
    "model._orig_mod.action_in_proj.weight": "model.action_in_proj.weight",
    (
        "model._orig_mod.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
    ): "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight",
    ("model._orig_mod.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"): (
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"
    ),
}
