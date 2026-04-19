"""
Model loader with per-preset strategies.

Dispatches to a strategy selected from `backend.model_presets`:

    image_text_to_text  -> Qwen-VL family and similar image-text models
    gemma4              -> Google's Gemma 4 family (video-native, optional int4)

The public entry points (`load_model`, `generate_caption`, `clear_cache`) keep
the same signatures the rest of the codebase already uses, so
`processing.py` does not need to know which strategy is active.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from huggingface_hub import snapshot_download

from backend import config
from backend.model_presets import (
    MODEL_PRESETS,
    DEFAULT_PRESET,
    get_preset,
    resolve_preset,
    frame_size_for_preset,
)

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}
_SAGE_ATTENTION_ENABLED = False
_SAGE_ERROR_LOGGED = False


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.float16)


def enable_sage_attention() -> bool:
    """Monkey-patch F.scaled_dot_product_attention with SageAttention. Idempotent."""
    global _SAGE_ATTENTION_ENABLED

    if _SAGE_ATTENTION_ENABLED:
        print("[Model Loader] SageAttention already enabled")
        return True

    print(f"[Model Loader] Python: {sys.executable}")
    print(f"[Model Loader] Attempting to enable SageAttention...")

    try:
        from sageattention import sageattn
        print("[Model Loader] SageAttention module imported successfully!")

        if not hasattr(F, '_original_scaled_dot_product_attention'):
            F._original_scaled_dot_product_attention = F.scaled_dot_product_attention

        def sdpa_sageattn_wrapper(query, key, value, attn_mask=None, dropout_p=0.0,
                                   is_causal=False, scale=None, **kwargs):
            if attn_mask is not None:
                return F._original_scaled_dot_product_attention(
                    query, key, value, attn_mask=attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, scale=scale
                )

            try:
                return sageattn(
                    query, key, value,
                    tensor_layout="HND",
                    is_causal=is_causal,
                    sm_scale=scale,
                )
            except Exception as e:
                global _SAGE_ERROR_LOGGED
                if not _SAGE_ERROR_LOGGED and "headdim" in str(e):
                    print(f"[Model Loader] SageAttention incompatible with model head dimension, using SDPA")
                    _SAGE_ERROR_LOGGED = True
                elif not _SAGE_ERROR_LOGGED:
                    print(f"[Model Loader] SageAttention error, falling back to SDPA: {e}")
                    _SAGE_ERROR_LOGGED = True
                return F._original_scaled_dot_product_attention(
                    query, key, value, attn_mask=attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, scale=scale
                )

        F.scaled_dot_product_attention = sdpa_sageattn_wrapper
        _SAGE_ATTENTION_ENABLED = True

        print("[Model Loader] SageAttention ENABLED")
        return True

    except ImportError as e:
        print(f"[Model Loader] SageAttention not available: {e}")
        return False
    except Exception as e:
        print(f"[Model Loader] Failed to enable SageAttention: {e}")
        return False


def download_model(model_id: str, local_dir: Path) -> Path:
    """Download model weights if not already cached. Returns local path."""
    model_name = model_id.split("/")[-1]
    model_path = local_dir / model_name

    if (model_path / "config.json").exists():
        print(f"[Model Loader] Model already downloaded: {model_path}")
        return model_path

    print(f"[Model Loader] Downloading model: {model_id}")
    start = time.time()
    downloaded_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_path),
    )
    print(f"[Model Loader] Download complete in {time.time() - start:.1f}s")
    return Path(downloaded_path)


# ---------------------------------------------------------------------------
# Strategy: generic image-text-to-text (Qwen-VL family)
# ---------------------------------------------------------------------------

def _load_image_text_to_text(
    preset: Dict[str, Any],
    model_path: Path,
    torch_dtype: torch.dtype,
    torch_device: torch.device,
    sage_enabled: bool,
    use_torch_compile: bool,
) -> Dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(torch_device)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    compiled = False
    if use_torch_compile and preset["supports_torch_compile"]:
        try:
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
            compiled = True
            print("[Model Loader] torch.compile applied")
        except Exception as e:
            print(f"[Model Loader] torch.compile failed: {e}")

    return {
        "model": model,
        "processor": processor,
        "sage_attention": sage_enabled,
        "torch_compiled": compiled,
        "device_map": None,
    }


def _generate_image_text_to_text(
    model_info: Dict[str, Any],
    images: List,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, Any]]:
    model = model_info["model"]
    processor = model_info["processor"]
    device = model_info["device"]

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    encode_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    encode_time = time.time() - encode_start
    input_tokens = inputs["input_ids"].shape[1]

    use_sampling = temperature > 0.1
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }
    if use_sampling:
        gen_kwargs.update({
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 40,
        })
    else:
        gen_kwargs["do_sample"] = False

    generate_start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**gen_kwargs)
    generate_time = time.time() - generate_start

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output_tokens = trimmed[0].shape[0]

    return output_text, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "encode_time": encode_time,
        "generate_time": generate_time,
        "total_time": encode_time + generate_time,
        "tokens_per_sec": output_tokens / generate_time if generate_time > 0 else 0,
        "num_frames": len(images),
    }


# ---------------------------------------------------------------------------
# Strategy: Gemma 4 (video-native, optional int4)
# ---------------------------------------------------------------------------

def _load_gemma4(
    preset: Dict[str, Any],
    model_path: Path,
    torch_dtype: torch.dtype,
    torch_device: torch.device,
) -> Dict[str, Any]:
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor

    quantization_config = None
    if preset.get("quantization") == "int4_torchao":
        from transformers import TorchAoConfig
        from torchao.quantization import Int4WeightOnlyConfig
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        # TILE_PACKED_TO_4D runs on CUDA tinygemm kernels; the default PLAIN
        # format requires the mslk backend which is unavailable on Windows.
        quantization_config = TorchAoConfig(
            Int4WeightOnlyConfig(
                group_size=128,
                int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
            )
        )
        print("[Model Loader] Using int4 TorchAo quantization (tile_packed_to_4d, group_size=128)")

    # Gemma 4 supports device_map="auto" for sharding across multiple GPUs.
    # For int4 (fits on one GPU) or single-GPU setups, pin to the requested device.
    shard = preset.get("supports_multi_gpu_shard", False) and torch.cuda.device_count() > 1 and quantization_config is None

    load_kwargs = {
        "dtype": torch_dtype,
        "attn_implementation": "sdpa",
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    if shard:
        load_kwargs["device_map"] = "auto"
        print(f"[Model Loader] Sharding Gemma 4 across {torch.cuda.device_count()} GPUs (device_map='auto')")
    else:
        load_kwargs["device_map"] = {"": torch_device}

    model = Gemma4ForConditionalGeneration.from_pretrained(str(model_path), **load_kwargs)
    processor = AutoProcessor.from_pretrained(str(model_path))

    return {
        "model": model,
        "processor": processor,
        "sage_attention": False,
        "torch_compiled": False,
        "device_map": "auto" if shard else "single",
    }


def _generate_gemma4(
    model_info: Dict[str, Any],
    images: List,
    prompt: str,
    max_tokens: int,
    temperature: float,
    preset: Dict[str, Any],
    video_fps: float = None,
) -> Tuple[str, Dict[str, Any]]:
    model = model_info["model"]
    processor = model_info["processor"]
    device = model_info["device"]

    # Gemma 4 wants a single video content block carrying the frame list. The
    # chat template expands {"type": "video", "video": ...} into the right
    # tokens and hands the frames to Gemma4VideoProcessor.
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": images},
            {"type": "text", "text": prompt},
        ],
    }]

    encode_start = time.time()
    # Tell the Gemma4VideoProcessor not to re-sample our pre-extracted frames,
    # and feed it a full VideoMetadata — fps AND frames_indices are both
    # required for Gemma 4 to compute per-frame timestamps for the prompt.
    videos_kwargs = {"do_sample_frames": False}
    effective_fps = float(video_fps) if (video_fps and video_fps > 0) else 24.0
    videos_kwargs["video_metadata"] = [{
        "fps": effective_fps,
        "duration": len(images) / effective_fps,
        "total_num_frames": len(images),
        "frames_indices": list(range(len(images))),
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"videos_kwargs": videos_kwargs},
    )
    inputs.pop("token_type_ids", None)
    # For sharded models, accelerate routes inputs; for single-device, move manually.
    if model_info.get("device_map") != "auto":
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    encode_time = time.time() - encode_start
    input_tokens = inputs["input_ids"].shape[1]

    # Gemma 4 recommended defaults (overridden by user temperature if set low)
    defaults = preset.get("gen_defaults") or {}
    use_sampling = temperature > 0.1
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
    }
    if use_sampling:
        gen_kwargs.update({
            "temperature": temperature,
            "do_sample": True,
            "top_p": defaults.get("top_p", 0.95),
            "top_k": defaults.get("top_k", 64),
        })
    else:
        gen_kwargs["do_sample"] = False

    generate_start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**gen_kwargs)
    generate_time = time.time() - generate_start

    trimmed_ids = generated_ids[0][input_tokens:]
    output_text = processor.decode(trimmed_ids, skip_special_tokens=True)
    # Gemma 4 supports a structured `parse_response` to strip thinking/channel tokens.
    try:
        parsed = processor.parse_response(output_text)
        if isinstance(parsed, str) and parsed:
            output_text = parsed
        elif isinstance(parsed, dict) and parsed.get("response"):
            output_text = parsed["response"]
    except Exception:
        pass

    output_tokens = trimmed_ids.shape[0]
    return output_text.strip(), {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "encode_time": encode_time,
        "generate_time": generate_time,
        "total_time": encode_time + generate_time,
        "tokens_per_sec": output_tokens / generate_time if generate_time > 0 else 0,
        "num_frames": len(images),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = None,
    device: str = None,
    dtype: str = None,
    use_sage_attention: bool = None,
    use_torch_compile: bool = None,
    force_reload: bool = False,
    preset_id: str = None,
) -> Dict[str, Any]:
    """
    Load a model according to its preset.

    `preset_id` takes precedence over `model_id`. If neither is given, the
    default preset is used. If `model_id` is provided without a matching
    preset, a custom-model fallback is used (default preset shape, custom id).
    """
    preset = resolve_preset(preset_id, model_id)
    effective_model_id = preset["model_id"]

    device = device or config.DEVICE
    dtype = dtype or config.DTYPE
    use_sage_attention = use_sage_attention if use_sage_attention is not None else config.USE_SAGE_ATTENTION
    use_torch_compile = use_torch_compile if use_torch_compile is not None else config.USE_TORCH_COMPILE

    # Honour preset capabilities — never force-enable something the preset
    # has declared unsupported.
    if not preset["supports_sage_attention"]:
        use_sage_attention = False
    if not preset["supports_torch_compile"]:
        use_torch_compile = False

    cache_key = f"{effective_model_id}_{device}_{dtype}_{preset.get('quantization') or 'none'}"
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"[Model Loader] Using cached model")
        return _MODEL_CACHE[cache_key]

    print("\n" + "=" * 60)
    print("[Model Loader] LOADING MODEL")
    print("=" * 60)
    print(f"[Model Loader] Preset: {preset['label']}")
    print(f"[Model Loader] Model: {effective_model_id}")
    print(f"[Model Loader] Loader strategy: {preset['loader']}")
    print(f"[Model Loader] Device: {device}")
    print(f"[Model Loader] Dtype: {dtype}")
    print(f"[Model Loader] Quantization: {preset.get('quantization') or 'none'}")
    print(f"[Model Loader] SageAttention: {use_sage_attention}")
    print(f"[Model Loader] torch.compile: {use_torch_compile}")
    print("=" * 60 + "\n")

    total_start = time.time()

    sage_enabled = enable_sage_attention() if use_sage_attention else False

    model_path = download_model(effective_model_id, config.MODELS_DIR)
    torch_dtype = get_dtype(dtype)
    torch_device = torch.device(device)

    if preset["loader"] == "image_text_to_text":
        strategy_info = _load_image_text_to_text(
            preset, model_path, torch_dtype, torch_device,
            sage_enabled=sage_enabled, use_torch_compile=use_torch_compile,
        )
    elif preset["loader"] == "gemma4":
        strategy_info = _load_gemma4(preset, model_path, torch_dtype, torch_device)
    else:
        raise ValueError(f"Unknown loader strategy: {preset['loader']}")

    model_info = {
        **strategy_info,
        "model_id": effective_model_id,
        "model_path": str(model_path),
        "device": torch_device,
        "dtype": torch_dtype,
        "preset": preset,
        "preset_id": _find_preset_id(preset),
    }
    _MODEL_CACHE[cache_key] = model_info

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("[Model Loader] MODEL LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"[Model Loader] Total time: {total_time:.1f}s")
    if torch.cuda.is_available():
        vram_gb = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3
        print(f"[Model Loader] VRAM used (all GPUs): {vram_gb:.2f} GB")
    print("=" * 60 + "\n")

    return model_info


def _find_preset_id(preset: Dict[str, Any]) -> Optional[str]:
    """Reverse-lookup a preset id from its dict (for custom models returns None)."""
    for pid, p in MODEL_PRESETS.items():
        if p is preset:
            return pid
    # custom fallback (resolve_preset clones the default dict) — match by model_id + label
    for pid, p in MODEL_PRESETS.items():
        if p["model_id"] == preset["model_id"] and p["label"] == preset["label"]:
            return pid
    return None


def generate_caption(
    model_info: Dict[str, Any],
    images: list,
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    video_fps: float = None,
) -> Tuple[str, Dict[str, Any]]:
    """Generate a caption via the strategy selected at load time."""
    max_tokens = max_tokens or config.MAX_TOKENS
    temperature = temperature if temperature is not None else config.TEMPERATURE
    preset = model_info["preset"]

    if preset["loader"] == "image_text_to_text":
        return _generate_image_text_to_text(model_info, images, prompt, max_tokens, temperature)
    elif preset["loader"] == "gemma4":
        return _generate_gemma4(model_info, images, prompt, max_tokens, temperature, preset, video_fps=video_fps)
    else:
        raise ValueError(f"Unknown loader strategy: {preset['loader']}")


def clear_cache():
    """Clear model cache and free GPU memory."""
    global _MODEL_CACHE

    for cache_key in list(_MODEL_CACHE.keys()):
        info = _MODEL_CACHE.get(cache_key)
        if info:
            for key in ("model", "processor"):
                if key in info:
                    del info[key]
    _MODEL_CACHE.clear()

    import gc
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    print("[Model Loader] Cache cleared and GPU memory freed")


if __name__ == "__main__":
    print("Testing model loader with default preset...")
    info = load_model()
    print(f"\nLoaded preset: {info.get('preset_id')} ({info['model_id']})")
