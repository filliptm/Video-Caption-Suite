# CLAUDE.md - AI Assistant Guidelines

This file provides guidelines for AI assistants (Claude, GPT, etc.) working on the Video Caption Suite project. **Any changes to the codebase must be reflected in the documentation.**

## Project Overview

Video Caption Suite is a video captioning application using the Qwen3-VL-8B vision-language model. It consists of:

- **Backend**: Python/FastAPI server (`backend/`)
- **Frontend**: Vue 3/TypeScript application (`frontend/src/`)
- **Documentation**: Comprehensive docs (`documentation/`)

## Documentation Sync Requirements

**CRITICAL: When making any code changes, you MUST update the corresponding documentation.**

### Documentation Files to Update

| Change Type | Files to Update |
|-------------|-----------------|
| New API endpoint | `documentation/API.md` |
| API endpoint changes | `documentation/API.md` |
| New Vue component | `documentation/FRONTEND.md` |
| Component prop/event changes | `documentation/FRONTEND.md` |
| New/changed settings | `documentation/CONFIGURATION.md` |
| Architecture changes | `documentation/ARCHITECTURE.md` |
| Setup/build changes | `documentation/DEVELOPMENT.md` |
| Major features | `documentation/README.md` |

### Cross-Reference Checklist

Before completing any task, verify:

1. [ ] Code changes compile/run without errors
2. [ ] TypeScript types match backend schemas
3. [ ] API documentation matches actual endpoints
4. [ ] Frontend documentation matches actual components
5. [ ] Configuration documentation matches actual options
6. [ ] Line number references in docs are still accurate

## Key File Locations

### Backend

| Purpose | File | Key Sections |
|---------|------|--------------|
| API Server | `backend/api.py` | Endpoints, WebSocket, CORS |
| Data Models | `backend/schemas.py` | Settings, Progress, Video/Caption, Analytics |
| Processing | `backend/processing.py` | ProcessingManager, multi-GPU |
| Analytics | `backend/analytics.py` | Word frequency, n-grams, correlations |
| Resource Monitor | `backend/resource_monitor.py` | ResourceMonitor, CPU/RAM/GPU metrics |
| Model Loading | `backend/model_loader.py` | load_model, generate_caption, clear_cache |
| Video Processing | `backend/video_processor.py` | extract_frames, process_video |
| Configuration | `backend/config.py` | All defaults |

### Frontend

| Purpose | File |
|---------|------|
| Root Component | `frontend/src/App.vue` |
| Video State | `frontend/src/stores/videoStore.ts` |
| Progress State | `frontend/src/stores/progressStore.ts` |
| Settings State | `frontend/src/stores/settingsStore.ts` |
| Analytics State | `frontend/src/stores/analyticsStore.ts` |
| Resource State | `frontend/src/stores/resourceStore.ts` |
| API Calls | `frontend/src/composables/useApi.ts` |
| WebSocket | `frontend/src/composables/useWebSocket.ts` |
| Resource WebSocket | `frontend/src/composables/useResourceWebSocket.ts` |
| Types | `frontend/src/types/*.ts` |
| Analytics Components | `frontend/src/components/analytics/*.vue` |
| Resource Monitor | `frontend/src/components/layout/ResourceMonitor.vue` |

## Code Patterns

### Backend Patterns

**Adding an API endpoint:**
```python
# backend/api.py
@app.post("/api/my-endpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    """Docstring describing the endpoint"""
    # Implementation
    return MyResponse(...)
```

**Updating schemas:**
```python
# backend/schemas.py
class MyModel(BaseModel):
    field: str = Field(default="value", description="Description")
```

### Frontend Patterns

**Vue component structure:**
```vue
<script setup lang="ts">
// Props and emits
interface Props { ... }
const props = defineProps<Props>()
const emit = defineEmits<{ ... }>()

// Store usage
const store = useMyStore()

// Computed and methods
const computed = computed(() => ...)
</script>

<template>
  <!-- Template -->
</template>
```

**Store pattern:**
```typescript
export const useMyStore = defineStore('my', {
  state: () => ({ ... }),
  getters: { ... },
  actions: { ... }
})
```

## Common Tasks

### Adding a New Setting

1. Add to `backend/schemas.py` (Settings class)
2. Add default to `backend/config.py`
3. Add to `frontend/src/types/settings.ts`
4. Add UI in appropriate settings component
5. Update `documentation/CONFIGURATION.md`

### Adding a New API Endpoint

1. Define request/response in `backend/schemas.py`
2. Add endpoint in `backend/api.py`
3. Add frontend function in `frontend/src/composables/useApi.ts`
4. Update `documentation/API.md`

### Fixing Memory Leaks

Memory management is critical. When modifying model loading/unloading:

1. Clear all references before `gc.collect()`
2. Call `torch.cuda.synchronize()` before `empty_cache()`
3. Update `clear_cache()` in `backend/model_loader.py`
4. Update `unload_model()` in `backend/processing.py`

## Testing Requirements

Before marking a task complete:

```bash
# Backend syntax check
python -m py_compile backend/api.py backend/processing.py backend/model_loader.py

# Frontend build check
cd frontend && npm run build:check

# Full test suite (if applicable)
pytest backend/tests/
cd frontend && npm run test
```

## Type Synchronization

Backend and frontend types must stay synchronized:

| Backend (Pydantic) | Frontend (TypeScript) |
|--------------------|-----------------------|
| `backend/schemas.py:Settings` | `frontend/src/types/settings.ts:Settings` |
| `backend/schemas.py:ProgressUpdate` | `frontend/src/types/progress.ts:ProgressState` |
| `backend/schemas.py:VideoInfo` | `frontend/src/types/video.ts:VideoInfo` |
| `backend/schemas.py:ProcessingStage` | `frontend/src/types/progress.ts:ProcessingStage` |
| `backend/schemas.py:WordFrequency*` | `frontend/src/types/analytics.ts:WordFrequency*` |
| `backend/schemas.py:Ngram*` | `frontend/src/types/analytics.ts:Ngram*` |
| `backend/schemas.py:Correlation*` | `frontend/src/types/analytics.ts:Correlation*` |
| `backend/schemas.py:GPUResourceMetrics` | `frontend/src/types/resources.ts:GPUMetrics` |
| `backend/schemas.py:ResourceUpdate` | `frontend/src/types/resources.ts:ResourceSnapshot` |

When changing one, change the other.

## Architecture Decisions

### Why WebSocket for Progress?
- Real-time updates without polling
- Server can push updates as they happen
- Automatic reconnection handles network issues

### Why SSE for Video Lists?
- Large video libraries (1000+ files) need progressive loading
- Reduces memory usage vs single large response
- Better perceived performance

### Why Multi-GPU Sequential Loading?
- Parallel loading can cause OOM
- Sequential is slower but reliable
- Each GPU needs ~16GB for the 8B model

## Known Limitations

1. **SageAttention**: Disabled for Qwen3-VL due to non-standard head dimensions (80 vs 64/96/128)
2. **GGUF Support**: Not implemented - requires llama.cpp server approach
3. **OpenRouter**: Not implemented - would require base64 frame encoding
4. **Authentication**: None - designed for local use only

## Removed Features

The following features were removed and should NOT be re-implemented without discussion:

- **TensorRT Compilation**: Removed due to complexity with VLMs (torch.compile doesn't serialize, TensorRT-LLM required)

## Documentation Format

When updating documentation:

1. Use tables for structured data
2. Include code examples
3. Reference file paths with line numbers when helpful
4. Keep the table of contents updated
5. Use consistent heading levels

## Questions?

If unclear about:
- Where code should go → Check ARCHITECTURE.md
- How an API works → Check API.md
- How the frontend is structured → Check FRONTEND.md
- What settings exist → Check CONFIGURATION.md
- How to set up/test → Check DEVELOPMENT.md
