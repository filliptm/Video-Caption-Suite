export type DeviceType = 'cuda' | 'cpu'
export type DtypeType = 'float16' | 'bfloat16' | 'float32'

export interface Settings {
  model_preset: string
  model_id: string
  device: DeviceType
  dtype: DtypeType
  max_frames: number
  frame_size: number
  max_tokens: number
  temperature: number
  use_sage_attention: boolean
  use_torch_compile: boolean
  include_metadata: boolean
  batch_size: number
  vision_token_budget?: number | null
  enable_thinking?: boolean | null
  prompt: string
}

export interface SettingsUpdate {
  model_preset?: string
  model_id?: string
  device?: DeviceType
  dtype?: DtypeType
  max_frames?: number
  frame_size?: number
  max_tokens?: number
  temperature?: number
  use_sage_attention?: boolean
  use_torch_compile?: boolean
  include_metadata?: boolean
  batch_size?: number
  vision_token_budget?: number | null
  enable_thinking?: boolean | null
  prompt?: string
}

export const defaultSettings: Settings = {
  model_preset: 'qwen3-vl-8b',
  model_id: 'Qwen/Qwen3-VL-8B-Instruct',
  device: 'cuda',
  dtype: 'bfloat16',
  max_frames: 16,
  frame_size: 336,
  max_tokens: 512,
  temperature: 0.3,
  use_sage_attention: false,
  use_torch_compile: true,
  include_metadata: false,
  batch_size: 1,
  vision_token_budget: null,
  enable_thinking: null,
  prompt: `Describe this video in detail. Include:
- The main subject and their actions
- The setting and environment
- Any notable objects or elements
- The overall mood or atmosphere
- Any text visible in the video`,
}

// Model preset metadata from GET /api/model-presets
export interface ModelPresetInfo {
  id: string
  model_id: string
  label: string
  description: string
  approx_vram_gb: number
  default_max_frames: number
  default_frame_size: number
  supports_multi_gpu_shard: boolean
  quantization: string | null
  supports_sage_attention: boolean
  supports_torch_compile: boolean
  is_video_native: boolean
}

export interface ModelPresetListResponse {
  presets: ModelPresetInfo[]
  default_preset_id: string
}

// GPU info types
export interface GPUInfo {
  index: number
  name: string
  memory_total_gb: number
  memory_free_gb: number
  device: string
}

export interface SystemGPUInfo {
  gpu_count: number
  gpus: GPUInfo[]
  cuda_available: boolean
  cuda_version: string | null
  max_batch_size: number
}

// Prompt Library types
export interface SavedPrompt {
  id: string
  name: string
  prompt: string
  created_at: string
}

export interface PromptLibrary {
  prompts: SavedPrompt[]
}

export interface CreatePromptRequest {
  name: string
  prompt: string
}

export interface UpdatePromptRequest {
  name?: string
  prompt?: string
}

// Directory types
export interface DirectoryRequest {
  directory: string
  traverse_subfolders?: boolean
  include_videos?: boolean
  include_images?: boolean
}

export interface DirectoryResponse {
  directory: string
  video_count?: number
  image_count?: number
  traverse_subfolders?: boolean
  include_videos?: boolean
  include_images?: boolean
}

export interface DirectoryBrowseResponse {
  current: string
  parent: string | null
  directories: { name: string; path: string }[]
}
