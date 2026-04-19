"""
Model preset registry.

Declares supported vision-language models and their per-model quirks in one
place. The model loader, settings UI, and processing pipeline all consult
this registry to decide how to load a model, what content-block shape to feed
it, and which knobs to expose.
"""

from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Preset entries
# ---------------------------------------------------------------------------
#
# Fields:
#   model_id                  HuggingFace repo id
#   label                     Human-readable label for the UI dropdown
#   description               One-line help text
#   loader                    "image_text_to_text" | "gemma4"
#   content_type              "image_list" (one image block per frame)
#                             | "video_block" (single video block w/ frame list)
#   supports_sage_attention   Whether SageAttention is safe to use
#   supports_torch_compile    Whether torch.compile is safe to use
#   supports_multi_gpu_shard  If True, loads with device_map="auto" on one
#                             worker (forces batch_size=1)
#   quantization              None | "int4_torchao"
#   default_max_frames        Sensible default for the UI / processing
#   default_frame_size        Sensible default (must be 48-divisible for Gemma 4)
#   approx_vram_gb            Rough load-time VRAM estimate
#   frame_size_divisor        If set, frame sizes are rounded to this multiple
#                             (Gemma 4 requires 48)
#   gen_defaults              Optional generation param overrides
#   vision_token_budget       Gemma 4 soft-token budget (70/140/280/560/1120)
#   enable_thinking           Gemma 4 thinking mode default

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3-vl-8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "label": "Qwen3-VL 8B (default)",
        "description": "Alibaba's 8B video-language model. Fast, proven, needs ~16GB VRAM.",
        "loader": "image_text_to_text",
        "content_type": "image_list",
        "supports_sage_attention": False,
        "supports_torch_compile": True,
        "supports_multi_gpu_shard": False,
        "quantization": None,
        "default_max_frames": 16,
        "default_frame_size": 336,
        "approx_vram_gb": 16,
        "frame_size_divisor": None,
    },
    "gemma-4-26b-a4b": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B-A4B (video-native, bf16)",
        "description": "Google's MoE (4B active / 26B total). Video-native. Needs ~52GB VRAM (multi-GPU shard).",
        "loader": "gemma4",
        "content_type": "video_block",
        "supports_sage_attention": False,
        "supports_torch_compile": False,
        "supports_multi_gpu_shard": True,
        "quantization": None,
        "default_max_frames": 32,
        "default_frame_size": 336,  # divisible by 48
        "approx_vram_gb": 52,
        "frame_size_divisor": 48,
        "vision_token_budget": 280,
        "enable_thinking": False,
        "gen_defaults": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    },
    "gemma-4-26b-a4b-int4": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B-A4B (video-native, int4)",
        "description": "Same as above but int4-quantized via TorchAo. Needs ~15GB VRAM.",
        "loader": "gemma4",
        "content_type": "video_block",
        "supports_sage_attention": False,
        "supports_torch_compile": False,
        "supports_multi_gpu_shard": False,  # int4 fits on one GPU
        "quantization": "int4_torchao",
        "default_max_frames": 32,
        "default_frame_size": 336,
        "approx_vram_gb": 15,
        "frame_size_divisor": 48,
        "vision_token_budget": 280,
        "enable_thinking": False,
        "gen_defaults": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    },
}

DEFAULT_PRESET = "qwen3-vl-8b"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    """Return a preset dict by id, or None if not found."""
    return MODEL_PRESETS.get(preset_id)


def resolve_preset(preset_id: Optional[str], model_id: Optional[str]) -> Dict[str, Any]:
    """
    Resolve the active preset.

    Resolution order:
      1. If preset_id matches a known preset, use it.
      2. If model_id matches a preset's model_id, use that preset.
      3. Otherwise fall back to the default preset but override its model_id
         with the provided custom one (keeps the free-text escape hatch alive).
    """
    if preset_id and preset_id in MODEL_PRESETS:
        return MODEL_PRESETS[preset_id]

    if model_id:
        for preset in MODEL_PRESETS.values():
            if preset["model_id"] == model_id:
                return preset
        # Custom model id — return default preset shape with the custom id substituted.
        fallback = dict(MODEL_PRESETS[DEFAULT_PRESET])
        fallback["model_id"] = model_id
        fallback["label"] = f"Custom: {model_id}"
        fallback["description"] = "Custom model id (not a known preset)"
        return fallback

    return MODEL_PRESETS[DEFAULT_PRESET]


def list_presets_public() -> list:
    """Return preset metadata for the frontend dropdown (stable field set)."""
    out = []
    for preset_id, p in MODEL_PRESETS.items():
        out.append({
            "id": preset_id,
            "model_id": p["model_id"],
            "label": p["label"],
            "description": p["description"],
            "approx_vram_gb": p["approx_vram_gb"],
            "default_max_frames": p["default_max_frames"],
            "default_frame_size": p["default_frame_size"],
            "supports_multi_gpu_shard": p["supports_multi_gpu_shard"],
            "quantization": p["quantization"],
            "supports_sage_attention": p["supports_sage_attention"],
            "supports_torch_compile": p["supports_torch_compile"],
            "is_video_native": p["content_type"] == "video_block",
        })
    return out


def frame_size_for_preset(preset: Dict[str, Any], requested: int) -> int:
    """Round a requested frame size up to the preset's divisor (if any)."""
    divisor = preset.get("frame_size_divisor")
    if not divisor:
        return requested
    return ((requested + divisor - 1) // divisor) * divisor
