"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class DtypeType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class ProcessingStage(str, Enum):
    IDLE = "idle"
    LOADING_MODEL = "loading_model"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class ProcessingSubstage(str, Enum):
    IDLE = "idle"
    EXTRACTING_FRAMES = "extracting_frames"
    ENCODING = "encoding"
    GENERATING = "generating"


class Settings(BaseModel):
    """Configuration settings for the captioner"""
    model_preset: str = "qwen3-vl-8b"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: DeviceType = DeviceType.CUDA
    dtype: DtypeType = DtypeType.BFLOAT16
    max_frames: int = Field(default=16, ge=1, le=128)
    frame_size: int = Field(default=336, ge=224, le=672)
    max_tokens: int = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    use_sage_attention: bool = False
    use_torch_compile: bool = True
    include_metadata: bool = False
    batch_size: int = Field(default=1, ge=1, le=8)
    # Gemma 4 specific — ignored by other presets.
    vision_token_budget: Optional[int] = None  # 70/140/280/560/1120
    enable_thinking: Optional[bool] = None
    prompt: str = """Describe this video in detail. Include:
- The main subject and their actions
- The setting and environment
- Any notable objects or elements
- The overall mood or atmosphere
- Any text visible in the video"""


class SettingsUpdate(BaseModel):
    """Partial settings update"""
    model_preset: Optional[str] = None
    model_id: Optional[str] = None
    device: Optional[DeviceType] = None
    dtype: Optional[DtypeType] = None
    max_frames: Optional[int] = Field(default=None, ge=1, le=128)
    frame_size: Optional[int] = Field(default=None, ge=224, le=672)
    max_tokens: Optional[int] = Field(default=None, ge=64, le=2048)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    use_sage_attention: Optional[bool] = None
    use_torch_compile: Optional[bool] = None
    include_metadata: Optional[bool] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=8)
    vision_token_budget: Optional[int] = None
    enable_thinking: Optional[bool] = None
    prompt: Optional[str] = None


class WorkerProgress(BaseModel):
    """Progress for a single GPU worker"""
    worker_id: int
    device: str
    current_video: Optional[str] = None
    substage: ProcessingSubstage = ProcessingSubstage.IDLE
    substage_progress: float = 0.0


class ProgressUpdate(BaseModel):
    """Real-time progress information sent via WebSocket"""
    stage: ProcessingStage = ProcessingStage.IDLE
    current_video: Optional[str] = None
    video_index: int = 0
    total_videos: int = 0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    model_loaded: bool = False
    vram_used_gb: float = 0.0
    substage: ProcessingSubstage = ProcessingSubstage.IDLE
    substage_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    elapsed_time: float = 0.0
    # Multi-GPU fields
    batch_size: int = 1
    workers: List[WorkerProgress] = []
    completed_videos: int = 0
    # Transient completion event fields (set only on the message after a video finishes)
    just_completed_video: Optional[str] = None
    just_completed_caption_preview: Optional[str] = None


class MediaType(str, Enum):
    """Type of media file"""
    VIDEO = "video"
    IMAGE = "image"


class VideoInfo(BaseModel):
    """Information about a media file (video or image)"""
    name: str
    path: str
    size_mb: float
    media_type: MediaType = MediaType.VIDEO
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    has_caption: bool = False
    caption_preview: Optional[str] = None


class VideoListResponse(BaseModel):
    """Response for video listing endpoint"""
    videos: List[VideoInfo]
    total_count: int


class CaptionInfo(BaseModel):
    """Information about a generated caption"""
    video_name: str
    caption_path: str
    caption_text: str
    created_at: Optional[str] = None


class CaptionListResponse(BaseModel):
    """Response for caption listing endpoint"""
    captions: List[CaptionInfo]
    total_count: int


class ProcessingRequest(BaseModel):
    """Request to start processing"""
    video_names: Optional[List[str]] = None  # None = process all


class ProcessingResponse(BaseModel):
    """Response after starting processing"""
    success: bool
    message: str
    videos_queued: int = 0


class ModelStatus(BaseModel):
    """Current model status"""
    loaded: bool = False
    model_id: Optional[str] = None
    preset_id: Optional[str] = None
    device: Optional[str] = None
    vram_used_gb: float = 0.0
    sage_attention_active: bool = False
    torch_compiled: bool = False


class ModelPresetInfo(BaseModel):
    """Public metadata for a single model preset (dropdown entry)"""
    id: str
    model_id: str
    label: str
    description: str
    approx_vram_gb: int
    default_max_frames: int
    default_frame_size: int
    supports_multi_gpu_shard: bool
    quantization: Optional[str] = None
    supports_sage_attention: bool
    supports_torch_compile: bool
    is_video_native: bool


class ModelPresetListResponse(BaseModel):
    """Response for GET /api/model-presets"""
    presets: List[ModelPresetInfo]
    default_preset_id: str


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None


class SavedPrompt(BaseModel):
    """A saved prompt in the library"""
    id: str
    name: str
    prompt: str
    created_at: str


class PromptLibrary(BaseModel):
    """Collection of saved prompts"""
    prompts: List[SavedPrompt] = []


class CreatePromptRequest(BaseModel):
    """Request to create a new saved prompt"""
    name: str = Field(..., min_length=1, max_length=100)
    prompt: str = Field(..., min_length=1)


class UpdatePromptRequest(BaseModel):
    """Request to update a saved prompt"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    prompt: Optional[str] = Field(default=None, min_length=1)


class DirectoryRequest(BaseModel):
    """Request to set working directory"""
    directory: str = Field(..., min_length=1)
    traverse_subfolders: bool = False
    include_videos: bool = True
    include_images: bool = False


class DirectoryResponse(BaseModel):
    """Response for directory operations"""
    directory: str
    video_count: Optional[int] = None
    image_count: Optional[int] = None
    traverse_subfolders: bool = False
    include_videos: bool = True
    include_images: bool = False


class DirectoryBrowseResponse(BaseModel):
    """Response for directory browsing"""
    current: str
    parent: Optional[str] = None
    directories: List[dict]  # List of {name: str, path: str}


class GPUInfoResponse(BaseModel):
    """GPU information for frontend"""
    gpu_count: int
    gpus: List[Dict[str, Any]]
    cuda_available: bool
    cuda_version: Optional[str] = None
    max_batch_size: int


# ============================================================================
# Analytics Schemas
# ============================================================================

class StopwordPreset(str, Enum):
    """Stopword filtering presets"""
    NONE = "none"
    ENGLISH = "english"
    MINIMAL = "minimal"


class WordFrequencyRequest(BaseModel):
    """Request for word frequency analysis"""
    video_names: Optional[List[str]] = None  # None = analyze all captions
    stopword_preset: StopwordPreset = StopwordPreset.ENGLISH
    custom_stopwords: Optional[List[str]] = None
    min_word_length: int = Field(default=2, ge=1, le=10)
    top_n: int = Field(default=50, ge=1, le=200)


class WordFrequencyItem(BaseModel):
    """Single word frequency entry"""
    word: str
    count: int
    frequency: float  # 0-1 percentage


class WordFrequencyResponse(BaseModel):
    """Response for word frequency analysis"""
    words: List[WordFrequencyItem]
    total_words: int
    total_unique_words: int
    captions_analyzed: int
    analysis_time_ms: float


class NgramRequest(BaseModel):
    """Request for n-gram analysis"""
    video_names: Optional[List[str]] = None
    n: int = Field(default=2, ge=2, le=4)  # 2=bigrams, 3=trigrams, 4=4-grams
    stopword_preset: StopwordPreset = StopwordPreset.ENGLISH
    top_n: int = Field(default=30, ge=1, le=100)
    min_count: int = Field(default=2, ge=1)


class NgramItem(BaseModel):
    """Single n-gram entry"""
    ngram: List[str]  # ["word1", "word2"] for bigram
    display: str  # "word1 word2" for display
    count: int
    frequency: float


class NgramResponse(BaseModel):
    """Response for n-gram analysis"""
    ngrams: List[NgramItem]
    n: int
    total_ngrams: int
    captions_analyzed: int


class CorrelationRequest(BaseModel):
    """Request for word correlation analysis"""
    video_names: Optional[List[str]] = None
    target_words: Optional[List[str]] = None  # None = auto-select top words
    window_size: int = Field(default=5, ge=2, le=20)
    min_co_occurrence: int = Field(default=3, ge=1)
    top_n: int = Field(default=50, ge=1, le=200)


class CorrelationItem(BaseModel):
    """Single correlation entry"""
    word1: str
    word2: str
    co_occurrence: int
    pmi_score: float  # Pointwise Mutual Information


class CorrelationResponse(BaseModel):
    """Response for correlation analysis"""
    correlations: List[CorrelationItem]
    nodes: List[str]  # Unique words for network visualization
    captions_analyzed: int


# ============================================================================
# Resource Monitoring Schemas
# ============================================================================

class GPUResourceMetrics(BaseModel):
    """Metrics for a single GPU"""
    index: int
    name: str
    utilization_percent: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    temperature_c: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0


class ResourceUpdate(BaseModel):
    """System resource snapshot sent via WebSocket"""
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpus: List[GPUResourceMetrics] = []
    timestamp: float = 0.0


class AnalyticsSummary(BaseModel):
    """Quick summary statistics"""
    total_captions: int
    total_words: int
    unique_words: int
    avg_words_per_caption: float
    top_words: List[WordFrequencyItem]  # Top 10 for quick display
