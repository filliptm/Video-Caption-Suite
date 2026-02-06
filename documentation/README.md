# Video Caption Suite - Documentation

## Overview

Video Caption Suite is a professional-grade media captioning application that uses the **Qwen3-VL-8B** vision-language model to automatically generate detailed text descriptions for **videos and images**. It features a modern web interface, real-time progress tracking, multi-GPU support, and comprehensive optimization features.

## Table of Contents

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System design, data flow, and component relationships |
| [API.md](./API.md) | Complete REST and WebSocket API reference |
| [FRONTEND.md](./FRONTEND.md) | Vue components, Pinia stores, and UI architecture |
| [CONFIGURATION.md](./CONFIGURATION.md) | All configuration options and settings |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | Setup, building, testing, and contributing |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with 16GB+ VRAM (for 8B model)
- CUDA 11.8+ and cuDNN

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Video Caption Suite"

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running the Application

**Option 1: Using start script (Windows)**
```batch
start.bat
```

**Option 2: Manual start**
```bash
# Terminal 1: Backend
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Access the application at `http://localhost:5173`

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.10+, FastAPI, PyTorch, Transformers |
| Frontend | Vue 3, TypeScript, Pinia, Tailwind CSS, Vite |
| Model | Qwen3-VL-8B-Instruct (Vision-Language Model) |
| Video Processing | OpenCV |
| Communication | REST API + WebSocket |

## Key Features

- **Video & Image Captioning**: Generate detailed descriptions for both videos and images
- **Media Type Filters**: Toggle between videos, images, or both in directory settings
- **Multi-GPU Support**: Process media in parallel across multiple GPUs
- **Real-time Progress**: WebSocket-based live progress updates
- **Custom Prompts**: Save and reuse captioning prompts
- **Batch Processing**: Process entire folders of media files
- **Video Preview**: Hover to preview videos before processing
- **Thumbnail Caching**: Fast grid loading with cached thumbnails
- **Memory Management**: Proper VRAM cleanup on model unload

### Supported Formats

| Type | Extensions |
|------|------------|
| Videos | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv` |
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp` |

## Project Structure

```
Video Caption Suite/
├── backend/
│   ├── api.py              # FastAPI server (962 lines)
│   ├── schemas.py          # Pydantic models
│   ├── processing.py       # Processing orchestration
│   ├── config.py           # Backend configuration
│   ├── model_loader.py     # Model lifecycle management
│   ├── video_processor.py  # Video/image processing
│   └── gpu_utils.py        # GPU detection
├── frontend/
│   ├── src/
│   │   ├── App.vue         # Root component
│   │   ├── components/     # Vue components
│   │   ├── stores/         # Pinia state management
│   │   ├── composables/    # Reusable logic
│   │   └── types/          # TypeScript definitions
│   └── package.json
├── requirements.txt        # Python dependencies
├── documentation/          # This documentation
└── CLAUDE.md              # AI assistant guidelines
```

## Hardware Requirements

### Minimum (Single GPU)
- NVIDIA GPU: 16GB VRAM (RTX 4080, A4000, etc.)
- System RAM: 32GB
- Storage: 50GB (for model cache)

### Recommended (Multi-GPU)
- 2-4x NVIDIA GPUs: 16GB+ VRAM each
- System RAM: 64GB+
- NVMe SSD for faster model loading

### Quantization Options (Reduced VRAM)
- Q8_0: ~9GB VRAM
- Q4_K_M: ~5GB VRAM
- (Requires GGUF backend - see ARCHITECTURE.md)

## Support

For issues, feature requests, or contributions, please refer to [DEVELOPMENT.md](./DEVELOPMENT.md).
