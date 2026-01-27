# Architecture Overview

This document describes the system architecture, data flow, and component relationships of Video Caption Suite.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Vue 3)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  App.vue    │  │   Stores    │  │ Composables │  │     Components      │ │
│  │  (Root)     │  │  - video    │  │ - useApi    │  │  - Settings panels  │ │
│  │             │  │  - progress │  │ - useWS     │  │  - Video grid       │ │
│  │             │  │  - settings │  │ - useResize │  │  - Progress display │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                    HTTP REST   │   WebSocket
                    (port 5173) │   (ws://localhost:8000)
                                │
┌───────────────────────────────┴─────────────────────────────────────────────┐
│                           BACKEND (FastAPI)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         api.py (FastAPI Server)                         ││
│  │  - REST endpoints (settings, videos, captions, processing)              ││
│  │  - WebSocket endpoint (/ws/progress)                                    ││
│  │  - SSE streaming for video lists                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│  ┌──────────────────┐  ┌──────────┴───────────┐  ┌───────────────────────┐  │
│  │   schemas.py     │  │   processing.py      │  │    gpu_utils.py       │  │
│  │   (Pydantic)     │  │   (ProcessingMgr)    │  │    (GPU detection)    │  │
│  └──────────────────┘  └──────────┬───────────┘  └───────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────────┐
│                          CORE MODULES                                        │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐   │
│  │    model_loader.py      │  │         video_processor.py              │   │
│  │  - Model download       │  │  - Frame extraction (OpenCV)            │   │
│  │  - SageAttention        │  │  - Video metadata                       │   │
│  │  - torch.compile        │  │  - Resize/sampling                      │   │
│  │  - Caption generation   │  │  - Directory scanning                   │   │
│  │  - Memory management    │  │                                         │   │
│  └────────────┬────────────┘  └─────────────────────────────────────────┘   │
└───────────────┼─────────────────────────────────────────────────────────────┘
                │
┌───────────────┴─────────────────────────────────────────────────────────────┐
│                           GPU / MODEL LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    PyTorch + Transformers                               ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    ││
│  │  │   cuda:0    │  │   cuda:1    │  │   cuda:2    │  │   cuda:N    │    ││
│  │  │  Qwen3-VL   │  │  Qwen3-VL   │  │  Qwen3-VL   │  │  Qwen3-VL   │    ││
│  │  │  (~16GB)    │  │  (~16GB)    │  │  (~16GB)    │  │  (~16GB)    │    ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Application Initialization

```
Browser loads App.vue
        │
        ├──► settingsStore.fetchSettings()
        │           │
        │           └──► GET /api/settings ──► Returns Settings JSON
        │           └──► GET /api/system/gpu ──► Returns GPU info
        │
        ├──► videoStore.fetchVideos()
        │           │
        │           └──► GET /api/videos/stream (SSE)
        │                       │
        │                       └──► Streams video info progressively
        │                           (handles large libraries efficiently)
        │
        └──► useWebSocket.connect()
                    │
                    └──► WS /ws/progress ──► Persistent connection
                                            for real-time updates
```

### 2. Video Processing Flow

```
User clicks "Process Selected Videos"
        │
        ▼
POST /api/process/start
  { video_names: [...], settings: {...} }
        │
        ▼
ProcessingManager.process_videos()
        │
        ├──► Check model loaded?
        │           │
        │    No ◄───┴───► Yes
        │    │              │
        │    ▼              │
        │  load_model()     │
        │    │              │
        │    ├─► Download from HuggingFace (if needed)
        │    ├─► Apply SageAttention (if enabled)
        │    ├─► Apply torch.compile (if enabled)
        │    └─► Move to GPU(s)
        │              │
        └──────────────┘
                │
                ▼
        For each video (parallel if multi-GPU):
        ┌───────────────────────────────────────┐
        │  1. video_processor.process_video()   │
        │     └─► Extract frames (OpenCV)       │
        │     └─► Resize to frame_size          │
        │                                       │
        │  2. model_loader.generate_caption()   │
        │     └─► Encode frames + prompt        │
        │     └─► Model inference               │
        │     └─► Decode output tokens          │
        │                                       │
        │  3. Save caption to .txt file         │
        │                                       │
        │  4. Emit progress via WebSocket       │
        │     └─► progressStore updates         │
        │     └─► UI re-renders                 │
        └───────────────────────────────────────┘
                │
                ▼
        Processing complete
        Stage = "complete"
```

### 3. Multi-GPU Processing

```
batch_size > 1 detected
        │
        ▼
load_models_parallel()
        │
        ├──► Load model on cuda:0
        ├──► Load model on cuda:1
        └──► Load model on cuda:N
             (Sequential to avoid OOM)
        │
        ▼
_process_videos_parallel()
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    Video Queue                          │
│  [video1, video2, video3, video4, video5, ...]         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┬───────────────┬───────────────┐
│   Worker 0    │   Worker 1    │   Worker N    │
│   (cuda:0)    │   (cuda:1)    │   (cuda:N)    │
├───────────────┼───────────────┼───────────────┤
│ Pull video1   │ Pull video2   │ Pull video3   │
│ Process...    │ Process...    │ Process...    │
│ Complete      │ Complete      │ Complete      │
│ Pull video4   │ Pull video5   │ Pull video6   │
│ ...           │ ...           │ ...           │
└───────────────┴───────────────┴───────────────┘
        │
        ▼
All workers report progress independently
Combined in progressStore for UI display
```

## Component Relationships

### Backend Module Dependencies

```
api.py
  ├── schemas.py (Pydantic models)
  ├── processing.py (ProcessingManager)
  ├── gpu_utils.py (GPU detection)
  └── config.py (settings)

processing.py
  ├── model_loader.py (load_model, generate_caption, clear_cache)
  ├── video_processor.py (process_video)
  ├── schemas.py (Settings, ProgressUpdate)
  └── config.py

model_loader.py
  ├── config.py
  ├── torch, transformers (external)
  └── sageattention (optional)

video_processor.py
  ├── config.py
  ├── os (scandir/walk for file discovery)
  └── cv2, PIL (external)
```

### Frontend Component Hierarchy

```
App.vue
├── LayoutSidebar
│   └── SettingsPanel
│       ├── DirectorySettings
│       ├── ModelSettings
│       ├── InferenceSettings
│       ├── PromptSettings
│       └── PromptLibrary
│
├── VideoGrid (main content area)
│   ├── VideoGridToolbar
│   └── VideoTile (repeated)
│       ├── Thumbnail
│       ├── Selection checkbox
│       └── Caption preview
│
├── CaptionPanel (right sidebar)
│   └── CaptionViewer
│
└── StatusPanel / ProgressBar (header)
    ├── StageProgress
    ├── TokenCounter
    └── ProgressRing
```

### Store Dependencies

```
App.vue
  ├── useVideoStore()
  │     ├── videos, captions, selectedVideos
  │     └── fetchVideos(), toggleSelection()
  │
  ├── useProgressStore()
  │     ├── stage, progress, workers
  │     └── updateFromWebSocket()
  │
  └── useSettingsStore()
        ├── settings, gpuInfo
        └── fetchSettings(), updateSettings()

Composables:
  useWebSocket() ──► progressStore.updateFromWebSocket()
  useApi() ──► HTTP requests to backend
```

## Key Design Decisions

### 1. WebSocket for Progress Updates
- Real-time updates without polling
- Automatic reconnection with exponential backoff
- Server broadcasts to all connected clients

### 2. SSE for Video Listing
- Handles large video libraries (1000+ files)
- Progressive loading improves perceived performance
- Reduces memory usage vs. single large response

### 3. Model Caching
- Models stay loaded between processing runs
- Manual unload button for VRAM management
- Cache key includes model_id + device + dtype

### 4. Multi-GPU Strategy
- Each GPU gets independent model copy
- Dynamic work distribution (no pre-assignment)
- Sequential model loading to avoid OOM

### 5. Memory Management
- Explicit `del` on model objects before `gc.collect()`
- CUDA synchronization before `empty_cache()`
- Clear all references before cache cleanup

## File Locations and Line References

| Component | File | Key Lines |
|-----------|------|-----------|
| FastAPI app creation | `backend/api.py` | 1-50 |
| WebSocket handler | `backend/api.py` | 120-180 |
| Video endpoints | `backend/api.py` | 400-600 |
| Processing endpoints | `backend/api.py` | 800-900 |
| ProcessingManager | `backend/processing.py` | 85-250 |
| Parallel processing | `backend/processing.py` | 264-464 |
| Model loading | `model_loader.py` | 158-295 |
| Caption generation | `model_loader.py` | 298-407 |
| Memory cleanup | `model_loader.py` | 410-444 |
| Frame extraction | `video_processor.py` | 80-150 |
| Vue root component | `frontend/src/App.vue` | 1-464 |
| Video store | `frontend/src/stores/videoStore.ts` | Full file |
| Progress store | `frontend/src/stores/progressStore.ts` | Full file |
| WebSocket composable | `frontend/src/composables/useWebSocket.ts` | Full file |

## Security Considerations

1. **Path Traversal Prevention**: All file paths validated against `..` sequences
2. **Input Validation**: Pydantic models enforce type constraints
3. **CORS**: Configured for development; tighten for production
4. **No Authentication**: Currently designed for local use only

## Performance Optimizations

### Model Inference
1. **SageAttention**: 2-5x attention speedup (when compatible)
2. **torch.compile**: JIT compilation for optimized inference
3. **Batch Processing**: Parallel GPU utilization

### Media Loading
4. **Single-Pass File Discovery**: `find_all_media()` in `video_processor.py` uses `os.scandir()` (flat) or `os.walk()` (recursive) with pre-computed extension sets, replacing 16-28 per-extension `glob()` calls with a single directory traversal
5. **PIL Image Thumbnails**: Image files use `Pillow` for thumbnail generation (fast), while video files use `ffmpeg`. Routed automatically by file extension via `_generate_any_thumbnail()`
6. **Background Thumbnail Pre-generation**: After SSE streaming completes, `asyncio.create_task()` kicks off a `ThreadPoolExecutor(max_workers=4)` to pre-generate all uncached thumbnails in batches of 50
7. **Non-blocking Thumbnail Endpoint**: On-demand thumbnail requests use `asyncio.to_thread()` so generation doesn't block the FastAPI event loop
8. **Caption Preview Optimization**: Only reads first 200 bytes of caption files for preview text, instead of loading entire files

### Frontend
9. **Virtual Scrolling**: Efficient rendering of large video grids (only visible + buffer rows)
10. **SSE Batch Throttling**: Incoming video batches accumulate in a plain (non-reactive) array and flush to Vue reactive state at most every 150ms, reducing reactivity cascades from ~50 to ~10 for large libraries
11. **Video Preview `preload="none"`**: Hover preview video elements use `preload="none"` to prevent browsers from downloading video data until playback starts
12. **Thumbnail Caching**: MD5-based cache avoids regeneration on subsequent loads
