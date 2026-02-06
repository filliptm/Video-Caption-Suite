# Development Guide

Complete guide for setting up, building, testing, and contributing to Video Caption Suite.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Backend runtime |
| Node.js | 18+ | Frontend build |
| npm | 9+ | Package management |
| Git | 2.x | Version control |
| CUDA | 11.8+ | GPU acceleration |
| cuDNN | 8.x | Deep learning ops |

### Hardware Requirements

**Minimum:**
- NVIDIA GPU with 16GB VRAM
- 32GB System RAM
- 50GB Storage (for model cache)

**Recommended:**
- NVIDIA RTX 4090 or A100
- 64GB System RAM
- NVMe SSD

---

## Initial Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd "Video Caption Suite"
```

### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Verify CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

---

## Running the Application

### Development Mode

**Option 1: Separate terminals**

```bash
# Terminal 1: Backend
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

**Option 2: Using start script (Windows)**

```batch
start.bat
```

### Production Mode

```bash
# Build frontend
cd frontend
npm run build
cd ..

# Run backend (serves built frontend)
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
Video Caption Suite/
├── backend/
│   ├── __init__.py
│   ├── api.py              # FastAPI application
│   ├── schemas.py          # Pydantic models
│   ├── processing.py       # Processing manager
│   ├── config.py           # Backend configuration
│   ├── model_loader.py     # Model management
│   ├── video_processor.py  # Video processing
│   ├── gpu_utils.py        # GPU utilities
│   └── tests/              # Backend tests
│       ├── conftest.py
│       ├── test_api.py
│       └── test_schemas.py
├── frontend/
│   ├── src/
│   │   ├── main.ts         # Entry point
│   │   ├── App.vue         # Root component
│   │   ├── components/     # Vue components
│   │   ├── stores/         # Pinia stores
│   │   ├── composables/    # Reusable logic
│   │   ├── types/          # TypeScript types
│   │   └── utils/          # Helpers
│   ├── public/             # Static assets
│   ├── index.html          # HTML template
│   ├── package.json        # Dependencies
│   ├── vite.config.ts      # Vite config
│   ├── tsconfig.json       # TypeScript config
│   └── tailwind.config.js  # Tailwind config
├── documentation/          # This documentation
├── models/                 # Downloaded models (git-ignored)
├── requirements.txt        # Python dependencies
├── start.bat              # Windows start script
├── CLAUDE.md              # AI assistant guidelines
└── README.md              # Quick start
```

---

## Development Workflow

### Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes**

3. **Run tests**
   ```bash
   # Backend
   pytest backend/tests/

   # Frontend
   cd frontend && npm run test
   ```

4. **Check types**
   ```bash
   cd frontend && npm run build:check
   ```

5. **Lint code**
   ```bash
   cd frontend && npm run lint
   ```

6. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add my feature"
   ```

### Code Style

**Python:**
- Follow PEP 8
- Use type hints
- Docstrings for public functions

**TypeScript/Vue:**
- Use Composition API
- Define prop types with TypeScript
- Use `<script setup lang="ts">`

---

## Testing

### Backend Tests

```bash
# Run all tests
pytest backend/tests/

# Run with coverage
pytest backend/tests/ --cov=backend

# Run specific test
pytest backend/tests/test_api.py::test_get_settings
```

**Test Structure:**
```python
# backend/tests/test_api.py
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_get_settings():
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert "model_id" in response.json()
```

### Frontend Tests

```bash
cd frontend

# Run tests
npm run test

# Run with UI
npm run test:ui

# Coverage report
npm run test:coverage
```

**Test Structure:**
```typescript
// frontend/src/components/base/__tests__/BaseButton.test.ts
import { mount } from '@vue/test-utils'
import BaseButton from '../BaseButton.vue'

describe('BaseButton', () => {
  it('renders slot content', () => {
    const wrapper = mount(BaseButton, {
      slots: { default: 'Click me' }
    })
    expect(wrapper.text()).toBe('Click me')
  })
})
```

---

## Adding Features

### Adding a New API Endpoint

1. **Define schema** (`backend/schemas.py`):
   ```python
   class MyRequest(BaseModel):
       field: str

   class MyResponse(BaseModel):
       result: str
   ```

2. **Add endpoint** (`backend/api.py`):
   ```python
   @app.post("/api/my-endpoint", response_model=MyResponse)
   async def my_endpoint(request: MyRequest):
       return MyResponse(result=f"Got: {request.field}")
   ```

3. **Add frontend API call** (`frontend/src/composables/useApi.ts`):
   ```typescript
   async function myEndpoint(field: string): Promise<MyResponse> {
     return request('/api/my-endpoint', {
       method: 'POST',
       body: JSON.stringify({ field })
     })
   }
   ```

4. **Update documentation** (`documentation/API.md`)

5. **Add tests**

### Adding a New Vue Component

1. **Create component** (`frontend/src/components/MyComponent.vue`):
   ```vue
   <script setup lang="ts">
   interface Props {
     label: string
   }

   const props = defineProps<Props>()
   const emit = defineEmits<{
     click: []
   }>()
   </script>

   <template>
     <div @click="emit('click')">
       {{ props.label }}
     </div>
   </template>
   ```

2. **Add types if needed** (`frontend/src/types/`)

3. **Add tests** (`frontend/src/components/__tests__/MyComponent.test.ts`)

4. **Update documentation** (`documentation/FRONTEND.md`)

### Adding a New Setting

1. **Update schema** (`backend/schemas.py`):
   ```python
   class Settings(BaseModel):
       # ... existing fields ...
       my_setting: int = Field(default=10, ge=1, le=100)
   ```

2. **Update config** (`backend/config.py`):
   ```python
   MY_SETTING = 10
   ```

3. **Update frontend types** (`frontend/src/types/settings.ts`):
   ```typescript
   interface Settings {
     // ... existing fields ...
     my_setting: number
   }
   ```

4. **Add UI control** (appropriate settings component)

5. **Update documentation** (`documentation/CONFIGURATION.md`)

---

## Debugging

### Backend Debugging

```python
# Add logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Processing video: {video_path}")
```

**VS Code launch.json:**
```json
{
  "name": "Backend",
  "type": "python",
  "request": "launch",
  "module": "uvicorn",
  "args": ["backend.api:app", "--reload", "--port", "8000"]
}
```

### Frontend Debugging

```typescript
// Vue Devtools (browser extension)
// Pinia devtools integration

// Console logging
console.log('State:', store.$state)
```

**VS Code launch.json:**
```json
{
  "name": "Frontend",
  "type": "chrome",
  "request": "launch",
  "url": "http://localhost:5173",
  "webRoot": "${workspaceFolder}/frontend/src"
}
```

### CUDA Debugging

```python
# Check CUDA memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Force cleanup
torch.cuda.empty_cache()
import gc
gc.collect()
```

---

## Common Issues

### CUDA Out of Memory

**Symptoms:** `CUDA out of memory` error during processing

**Solutions:**
1. Reduce `max_frames` (16-32)
2. Reduce `frame_size` (224-336)
3. Unload model before changing settings
4. Restart the server to clear VRAM

### Model Download Fails

**Symptoms:** Timeout or connection errors

**Solutions:**
1. Check internet connection
2. Try HuggingFace mirror
3. Download manually:
   ```bash
   huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
   ```

### WebSocket Disconnects

**Symptoms:** Progress updates stop, "Disconnected" in UI

**Solutions:**
1. Check if backend is running
2. Check browser console for errors
3. WebSocket auto-reconnects (wait 30s)
4. Refresh the page

### Frontend Build Errors

**Symptoms:** Type errors or build failures

**Solutions:**
1. Clear node_modules: `rm -rf node_modules && npm install`
2. Check TypeScript version compatibility
3. Run `npm run build:check` for detailed errors

---

## Performance Profiling

### Backend Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
generate_caption(model_info, frames, prompt)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### GPU Profiling

```bash
# NVIDIA profiler
nvidia-smi dmon -s pucvmet -d 1

# PyTorch profiler
python -c "
import torch
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    # Your code
    pass
print(p.key_averages().table(sort_by='cuda_time_total'))
"
```

---

## Release Process

1. **Update version** in relevant files
2. **Run full test suite**
3. **Build production frontend**
4. **Create release notes**
5. **Tag release**
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

---

## Getting Help

- Check existing documentation
- Search closed issues
- Open a new issue with:
  - OS and Python version
  - GPU model and VRAM
  - Error messages
  - Steps to reproduce
