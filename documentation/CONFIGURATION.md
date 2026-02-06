# Configuration Reference

Complete reference for all configuration options in Video Caption Suite.

## Configuration Files

| File | Purpose |
|------|---------|
| `backend/config.py` | Backend Python settings |
| `user_config.json` | User working directory and preferences (auto-generated, persisted) |
| `settings.json` | Persisted user settings (auto-generated) |
| `prompts.json` | Saved prompt library (auto-generated) |
| `frontend/vite.config.ts` | Frontend build configuration |
| `frontend/tailwind.config.js` | CSS framework configuration |

---

## Backend Configuration (backend/config.py)

**File:** `backend/config.py`

### Model Settings

```python
# HuggingFace model identifier
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Compute device
DEVICE = "cuda"  # or "cpu"

# Model precision
DTYPE = "bfloat16"  # "float16", "bfloat16", or "float32"
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `MODEL_ID` | str | `"Qwen/Qwen3-VL-8B-Instruct"` | HuggingFace model ID |
| `DEVICE` | str | `"cuda"` | `cuda` for GPU, `cpu` for CPU-only |
| `DTYPE` | str | `"bfloat16"` | Model precision (affects VRAM usage) |

### Inference Settings

```python
# Maximum frames to extract per video
MAX_FRAMES_PER_VIDEO = 128

# Frame resize target (pixels, preserves aspect ratio)
FRAME_SIZE = 336

# Maximum tokens to generate
MAX_TOKENS = 512

# Sampling temperature (0 = greedy, higher = more creative)
TEMPERATURE = 0.3
```

| Setting | Type | Default | Range | Description |
|---------|------|---------|-------|-------------|
| `MAX_FRAMES_PER_VIDEO` | int | `128` | 1-128 | More frames = better temporal understanding, more VRAM |
| `FRAME_SIZE` | int | `336` | 224-672 | Larger = more detail, more VRAM |
| `MAX_TOKENS` | int | `512` | 64-2048 | Maximum caption length |
| `TEMPERATURE` | float | `0.3` | 0.0-2.0 | 0 = deterministic, >1 = very creative |

### Optimization Settings

```python
# SageAttention (2-5x attention speedup)
USE_SAGE_ATTENTION = False

# torch.compile JIT optimization
USE_TORCH_COMPILE = True
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `USE_SAGE_ATTENTION` | bool | `False` | Enable SageAttention (requires Triton, incompatible with Qwen3-VL head dims) |
| `USE_TORCH_COMPILE` | bool | `True` | Enable torch.compile (slower first run, faster subsequent) |

### Directory Settings

```python
# Default working directory (persisted to user_config.json)
WORKING_DIRECTORY = None  # Set via API

# Search subfolders for videos (persisted to user_config.json)
TRAVERSE_SUBFOLDERS = False

# Model download/cache location
MODELS_DIR = Path("./models")

# Output file extension
OUTPUT_EXTENSION = ".txt"
```

| Setting | Type | Default | Persisted | Description |
|---------|------|---------|-----------|-------------|
| `WORKING_DIRECTORY` | str/None | `None` | Yes | Video source folder |
| `TRAVERSE_SUBFOLDERS` | bool | `False` | Yes | Recursively search subfolders |
| `MODELS_DIR` | Path | `./models` | No | Model download location |
| `OUTPUT_EXTENSION` | str | `.txt` | No | Caption file extension |

**Note:** Working directory and traverse subfolders settings are automatically saved to `user_config.json` when changed and restored on server restart.

### Server Settings

```python
# API server port
API_PORT = 8000

# API server host
API_HOST = "0.0.0.0"
```

---

## User Config (user_config.json)

**Location:** Project root (auto-generated)

Stores the user's working directory, folder preferences, and media type filters. Automatically loaded on server startup and saved when settings change.

```json
{
  "working_directory": "C:/Videos/MyProject",
  "traverse_subfolders": false,
  "include_videos": true,
  "include_images": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `working_directory` | string/null | `null` | Path to media source folder |
| `traverse_subfolders` | bool | `false` | Whether to search subfolders |
| `include_videos` | bool | `true` | Include video files in media list |
| `include_images` | bool | `false` | Include image files in media list |

**Supported Extensions:**
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`
- Images: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`

**Behavior:**
- If `working_directory` path no longer exists, it is ignored and defaults to user's home directory
- Settings are saved immediately when changed via the Directory settings in the UI
- This file enables project resumption after server restarts
- Media type filters determine which file types appear in the media grid

---

## User Settings (settings.json)

**Location:** Project root (auto-generated)

Settings are persisted when changed via the API. All fields from `backend/schemas.py:Settings`.

```json
{
  "model_id": "Qwen/Qwen3-VL-8B-Instruct",
  "device": "cuda",
  "dtype": "bfloat16",
  "max_frames": 32,
  "frame_size": 336,
  "max_tokens": 512,
  "temperature": 0.3,
  "prompt": "Describe this video in detail...",
  "include_metadata": false,
  "use_sage_attention": false,
  "use_torch_compile": true,
  "batch_size": 1
}
```

### Settings Schema

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `model_id` | string | `"Qwen/Qwen3-VL-8B-Instruct"` | - | Model identifier |
| `device` | enum | `"cuda"` | `cuda`, `cpu` | Compute device |
| `dtype` | enum | `"bfloat16"` | `float16`, `bfloat16`, `float32` | Precision |
| `max_frames` | int | `32` | 1-128 | Frames per video |
| `frame_size` | int | `336` | 224-672 | Frame dimensions |
| `max_tokens` | int | `512` | 64-2048 | Output length |
| `temperature` | float | `0.3` | 0.0-2.0 | Creativity |
| `prompt` | string | See below | - | Captioning prompt |
| `include_metadata` | bool | `false` | - | Add metadata to output |
| `use_sage_attention` | bool | `false` | - | SageAttention |
| `use_torch_compile` | bool | `true` | - | JIT compilation |
| `batch_size` | int | `1` | 1-8 | GPUs to use |

### Default Prompt

```
Describe this video in detail, including:
- The main subjects and their actions
- The setting and environment
- Any notable visual elements or changes
- The overall mood or atmosphere

Be specific and descriptive, focusing on what is actually visible in the video.
```

---

## Prompt Library (prompts.json)

**Location:** Project root (auto-generated)

```json
{
  "prompts": [
    {
      "id": "abc123-def456",
      "name": "Detailed Description",
      "prompt": "Describe this video...",
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-15T10:30:00Z"
    }
  ]
}
```

---

## Frontend Configuration

### Vite Configuration

**File:** `frontend/vite.config.ts`

```typescript
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
})
```

| Setting | Value | Description |
|---------|-------|-------------|
| `server.port` | `5173` | Development server port |
| `proxy./api` | `http://localhost:8000` | API proxy target |
| `proxy./ws` | `ws://localhost:8000` | WebSocket proxy |

### TypeScript Configuration

**File:** `frontend/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "preserve",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

### Tailwind Configuration

**File:** `frontend/tailwind.config.js`

```javascript
module.exports = {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#faf5ff',
          100: '#f3e8ff',
          200: '#e9d5ff',
          300: '#d8b4fe',
          400: '#c084fc',
          500: '#a855f7',
          600: '#9333ea',
          700: '#7c3aed',
          800: '#6b21a8',
          900: '#581c87',
          950: '#2e1065'
        },
        dark: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          850: '#1a1f2e',
          900: '#0f172a',
          950: '#0f1219'
        }
      }
    }
  },
  plugins: []
}
```

---

## Environment Variables

Currently, no environment variables are used. All configuration is file-based.

For production deployment, consider adding:

```bash
# .env (proposed)
API_HOST=0.0.0.0
API_PORT=8000
MODELS_DIR=/data/models
CUDA_VISIBLE_DEVICES=0,1  # Limit GPU access
```

---

## VRAM Requirements by Configuration

| Configuration | Estimated VRAM |
|---------------|----------------|
| Default (8B, bf16, 32 frames) | ~16-18 GB |
| High quality (8B, bf16, 128 frames) | ~20-24 GB |
| Low VRAM (8B, bf16, 16 frames, 224px) | ~14-16 GB |
| Float32 (8B, fp32, 32 frames) | ~32+ GB |
| Multi-GPU 2x | ~16 GB per GPU |

### Memory Optimization Tips

1. **Reduce `max_frames`**: 16-32 for short videos, 64+ for long videos
2. **Reduce `frame_size`**: 224 minimum, 336 balanced, 672 maximum quality
3. **Use `bfloat16`**: Best balance of speed and precision
4. **Disable `torch_compile`**: Saves memory during compilation phase
5. **Single GPU**: `batch_size=1` uses less total VRAM

---

## Performance Tuning

### For Speed

```json
{
  "max_frames": 16,
  "frame_size": 336,
  "max_tokens": 256,
  "temperature": 0.0,
  "use_torch_compile": true,
  "batch_size": 2
}
```

### For Quality

```json
{
  "max_frames": 64,
  "frame_size": 672,
  "max_tokens": 1024,
  "temperature": 0.3,
  "use_torch_compile": true,
  "batch_size": 1
}
```

### For Low VRAM (~12GB)

```json
{
  "max_frames": 8,
  "frame_size": 224,
  "max_tokens": 256,
  "dtype": "float16",
  "use_torch_compile": false,
  "batch_size": 1
}
```
