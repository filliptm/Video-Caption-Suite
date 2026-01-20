"""
Optimized Model Loader for Qwen3-VL-8B
Includes SageAttention monkey-patching and torch.compile support
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from huggingface_hub import snapshot_download

import config

# Global state
_MODEL_CACHE: Dict[str, Any] = {}
_SAGE_ATTENTION_ENABLED = False
_SAGE_ERROR_LOGGED = False  # Track if we've already logged the headdim error


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype"""
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
    """
    Enable SageAttention by monkey-patching PyTorch's scaled_dot_product_attention.
    Provides 2-5x speedup over standard attention.

    Returns:
        True if SageAttention was enabled successfully
    """
    global _SAGE_ATTENTION_ENABLED

    if _SAGE_ATTENTION_ENABLED:
        print("[Model Loader] SageAttention already enabled")
        return True

    print(f"[Model Loader] Python: {sys.executable}")
    print(f"[Model Loader] Attempting to enable SageAttention...")

    try:
        from sageattention import sageattn
        print("[Model Loader] SageAttention module imported successfully!")

        # Store original function
        if not hasattr(F, '_original_scaled_dot_product_attention'):
            F._original_scaled_dot_product_attention = F.scaled_dot_product_attention

        def sdpa_sageattn_wrapper(query, key, value, attn_mask=None, dropout_p=0.0,
                                   is_causal=False, scale=None, **kwargs):
            """
            Wrapper to make sageattn compatible with F.scaled_dot_product_attention signature.
            Falls back to original SDPA when attention mask is needed.
            """
            # SageAttention doesn't support attention masks - fall back
            if attn_mask is not None:
                return F._original_scaled_dot_product_attention(
                    query, key, value, attn_mask=attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, scale=scale
                )

            try:
                # SageAttention call with HND tensor layout
                # HND = (batch, heads, seq_len, head_dim)
                return sageattn(
                    query, key, value,
                    tensor_layout="HND",
                    is_causal=is_causal,
                    sm_scale=scale,
                )
            except Exception as e:
                # Graceful fallback on any error - only log once to avoid spam
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

        # Apply monkey-patch
        F.scaled_dot_product_attention = sdpa_sageattn_wrapper
        _SAGE_ATTENTION_ENABLED = True

        print("[Model Loader] SageAttention ENABLED")
        print("[Model Loader] Expected speedup: 2-5x over standard attention")
        return True

    except ImportError as e:
        print(f"[Model Loader] SageAttention not available: {e}")
        if "triton" in str(e).lower():
            print("[Model Loader] SageAttention requires Triton. Install with:")
            print("[Model Loader]   pip install triton-windows  (for Windows)")
            print("[Model Loader]   pip install triton          (for Linux)")
        else:
            print("[Model Loader] Install with: pip install sageattention triton-windows")
        return False
    except Exception as e:
        print(f"[Model Loader] Failed to enable SageAttention: {e}")
        return False


def download_model(model_id: str, local_dir: Path) -> Path:
    """
    Download model from HuggingFace Hub if not already present.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-VL-8B-Instruct")
        local_dir: Local directory to store model

    Returns:
        Path to downloaded model
    """
    # Check if model already exists
    model_name = model_id.split("/")[-1]
    model_path = local_dir / model_name

    # Check for key model files
    if (model_path / "config.json").exists():
        print(f"[Model Loader] Model already downloaded: {model_path}")
        return model_path

    print(f"[Model Loader] Downloading model: {model_id}")
    print(f"[Model Loader] This may take a while (~16GB for Qwen3-VL-8B)...")

    start_time = time.time()

    # Download using huggingface_hub
    downloaded_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
    )

    elapsed = time.time() - start_time
    print(f"[Model Loader] Download complete in {elapsed:.1f}s")
    print(f"[Model Loader] Model path: {downloaded_path}")

    return Path(downloaded_path)


def load_model(
    model_id: str = None,
    device: str = None,
    dtype: str = None,
    use_sage_attention: bool = None,
    use_torch_compile: bool = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Load Qwen3-VL model with all optimizations.

    Args:
        model_id: HuggingFace model ID (default: from config)
        device: Target device (default: from config)
        dtype: Model precision (default: from config)
        use_sage_attention: Enable SageAttention (default: from config)
        use_torch_compile: Enable torch.compile (default: from config)
        force_reload: Force reload even if cached

    Returns:
        Dict with model, processor, and metadata
    """
    # Use config defaults if not specified
    model_id = model_id or config.MODEL_ID
    device = device or config.DEVICE
    dtype = dtype or config.DTYPE
    use_sage_attention = use_sage_attention if use_sage_attention is not None else config.USE_SAGE_ATTENTION
    use_torch_compile = use_torch_compile if use_torch_compile is not None else config.USE_TORCH_COMPILE

    # Check cache
    cache_key = f"{model_id}_{device}_{dtype}"
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"[Model Loader] Using cached model")
        return _MODEL_CACHE[cache_key]

    print("\n" + "=" * 60)
    print("[Model Loader] LOADING MODEL")
    print("=" * 60)
    print(f"[Model Loader] Model: {model_id}")
    print(f"[Model Loader] Device: {device}")
    print(f"[Model Loader] Dtype: {dtype}")
    print(f"[Model Loader] SageAttention: {use_sage_attention}")
    print(f"[Model Loader] torch.compile: {use_torch_compile}")
    print("=" * 60 + "\n")

    total_start = time.time()

    # Step 1: Enable SageAttention if requested
    sage_enabled = False
    if use_sage_attention:
        print("[Model Loader] Step 1/4: Enabling SageAttention...")
        sage_enabled = enable_sage_attention()
    else:
        print("[Model Loader] Step 1/4: SageAttention disabled by config")

    # Step 2: Download model if needed
    print("\n[Model Loader] Step 2/4: Checking model files...")
    model_path = download_model(model_id, config.MODELS_DIR)

    # Step 3: Load model
    print("\n[Model Loader] Step 3/4: Loading model weights...")
    print("[Model Loader] This may take 30-60 seconds...")

    from transformers import AutoModelForVision2Seq, AutoProcessor

    load_start = time.time()
    torch_dtype = get_dtype(dtype)
    torch_device = torch.device(device)

    # Determine attention implementation
    # If SageAttention is enabled, we use SDPA (which is now monkey-patched)
    attn_impl = "sdpa" if sage_enabled else "sdpa"

    model = AutoModelForVision2Seq.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(torch_device)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    load_time = time.time() - load_start
    print(f"[Model Loader] Model loaded in {load_time:.1f}s")

    # Step 4: Apply torch.compile if requested
    compiled = False
    if use_torch_compile:
        print("\n[Model Loader] Step 4/4: Applying torch.compile()...")
        print("[Model Loader] First inference will be slower due to JIT compilation")

        compile_start = time.time()
        try:
            model = torch.compile(
                model,
                mode="default",      # Better for dynamic shapes
                fullgraph=False,     # Allow graph breaks
                dynamic=True,        # Support variable sequence lengths
            )
            compiled = True
            compile_time = time.time() - compile_start
            print(f"[Model Loader] torch.compile applied in {compile_time:.1f}s")
        except Exception as e:
            print(f"[Model Loader] torch.compile failed: {e}")
            print("[Model Loader] Continuing with uncompiled model")
    else:
        print("\n[Model Loader] Step 4/4: torch.compile disabled by config")

    # Build model info dict
    model_info = {
        "model": model,
        "processor": processor,
        "model_id": model_id,
        "model_path": str(model_path),
        "device": torch_device,
        "dtype": torch_dtype,
        "sage_attention": sage_enabled,
        "torch_compiled": compiled,
    }

    # Cache the model
    _MODEL_CACHE[cache_key] = model_info

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("[Model Loader] MODEL LOADED SUCCESSFULLY!")
    print("=" * 60)
    print(f"[Model Loader] Total time: {total_time:.1f}s")
    print(f"[Model Loader] SageAttention: {'Enabled' if sage_enabled else 'Disabled'}")
    print(f"[Model Loader] torch.compile: {'Enabled' if compiled else 'Disabled'}")
    print(f"[Model Loader] VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("=" * 60 + "\n")

    return model_info


def generate_caption(
    model_info: Dict[str, Any],
    images: list,
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a caption for a list of video frames.

    Args:
        model_info: Dict from load_model()
        images: List of PIL Images (video frames)
        prompt: Text prompt for captioning
        max_tokens: Maximum tokens to generate (default: from config)
        temperature: Sampling temperature (default: from config)

    Returns:
        Tuple of (caption_text, metadata_dict)
    """
    max_tokens = max_tokens or config.MAX_TOKENS
    temperature = temperature or config.TEMPERATURE

    model = model_info["model"]
    processor = model_info["processor"]
    device = model_info["device"]

    # Build message with images
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # Process inputs
    encode_start = time.time()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Remove token_type_ids if present (not needed and can cause issues)
    inputs.pop("token_type_ids", None)

    # Move to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    encode_time = time.time() - encode_start
    input_tokens = inputs["input_ids"].shape[1]

    # Generate
    generate_start = time.time()

    with torch.inference_mode():
        # Use greedy decoding for stability, sampling for creativity
        use_sampling = temperature > 0.1

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "repetition_penalty": 1.15,      # Moderate repetition penalty
            "no_repeat_ngram_size": 3,       # Prevent 3-gram repetition
        }

        if use_sampling:
            generate_kwargs.update({
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.95,               # Nucleus sampling
                "top_k": 40,                 # Limit vocabulary
            })
        else:
            # Greedy decoding - most stable
            generate_kwargs["do_sample"] = False

        generated_ids = model.generate(**generate_kwargs)

    generate_time = time.time() - generate_start

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    output_tokens = generated_ids_trimmed[0].shape[0]
    tokens_per_sec = output_tokens / generate_time if generate_time > 0 else 0

    metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "encode_time": encode_time,
        "generate_time": generate_time,
        "total_time": encode_time + generate_time,
        "tokens_per_sec": tokens_per_sec,
        "num_frames": len(images),
    }

    return output_text, metadata


def clear_cache():
    """Clear model cache and free GPU memory"""
    global _MODEL_CACHE

    # Explicitly delete model and processor objects to free GPU memory
    for cache_key in list(_MODEL_CACHE.keys()):
        model_info = _MODEL_CACHE.get(cache_key)
        if model_info:
            # Delete model first (largest GPU memory consumer)
            if "model" in model_info:
                del model_info["model"]
            # Delete processor
            if "processor" in model_info:
                del model_info["processor"]

    _MODEL_CACHE.clear()

    # Force garbage collection before clearing CUDA cache
    import gc
    gc.collect()

    if torch.cuda.is_available():
        # Synchronize all CUDA devices before clearing cache
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Additional aggressive memory cleanup
        torch.cuda.reset_peak_memory_stats()

    # Run GC again after CUDA cleanup
    gc.collect()

    print("[Model Loader] Cache cleared and GPU memory freed")


if __name__ == "__main__":
    # Test model loading
    print("Testing model loader...")
    model_info = load_model()
    print("\nModel loaded successfully!")
    print(f"Device: {model_info['device']}")
    print(f"SageAttention: {model_info['sage_attention']}")
    print(f"Compiled: {model_info['torch_compiled']}")
