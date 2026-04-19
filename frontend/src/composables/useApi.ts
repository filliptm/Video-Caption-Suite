import { ref } from 'vue'
import type {
  ModelStatus,
  ProcessingResponse,
  PromptLibrary,
  SavedPrompt,
  CreatePromptRequest,
  UpdatePromptRequest,
  DirectoryResponse,
  DirectoryBrowseResponse,
  ModelPresetListResponse
} from '@/types'

export function useApi() {
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function request<T>(
    url: string,
    options: RequestInit = {}
  ): Promise<T | null> {
    console.log('[useApi] request:', url, options.method || 'GET')
    loading.value = true
    error.value = null

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      console.log('[useApi] response status:', response.status, response.ok)

      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        console.log('[useApi] error response data:', data)
        throw new Error(data.detail || `HTTP ${response.status}`)
      }

      const json = await response.json()
      console.log('[useApi] success response:', json)
      return json
    } catch (e) {
      console.error('[useApi] request error:', e)
      error.value = e instanceof Error ? e.message : 'Unknown error'
      return null
    } finally {
      loading.value = false
    }
  }

  // Model operations
  async function loadModel(): Promise<boolean> {
    const result = await request<{ success: boolean }>('/api/model/load', {
      method: 'POST',
    })
    return result?.success ?? false
  }

  async function unloadModel(): Promise<boolean> {
    const result = await request<{ success: boolean }>('/api/model/unload', {
      method: 'POST',
    })
    return result?.success ?? false
  }

  async function getModelStatus(): Promise<ModelStatus | null> {
    return request<ModelStatus>('/api/model/status')
  }

  async function getModelPresets(): Promise<ModelPresetListResponse | null> {
    return request<ModelPresetListResponse>('/api/model-presets')
  }

  // Processing operations
  async function startProcessing(videoNames?: string[]): Promise<ProcessingResponse | null> {
    return request<ProcessingResponse>('/api/process/start', {
      method: 'POST',
      body: videoNames ? JSON.stringify({ video_names: videoNames }) : undefined,
    })
  }

  async function stopProcessing(): Promise<boolean> {
    const result = await request<{ success: boolean }>('/api/process/stop', {
      method: 'POST',
    })
    return result?.success ?? false
  }

  // Health check
  async function checkHealth(): Promise<boolean> {
    const result = await request<{ status: string }>('/api/health')
    return result?.status === 'healthy'
  }

  // Prompt Library operations
  async function getPrompts(): Promise<PromptLibrary | null> {
    return request<PromptLibrary>('/api/prompts')
  }

  async function createPrompt(data: CreatePromptRequest): Promise<SavedPrompt | null> {
    console.log('[useApi] createPrompt called with:', data)
    const result = await request<SavedPrompt>('/api/prompts', {
      method: 'POST',
      body: JSON.stringify(data),
    })
    console.log('[useApi] createPrompt result:', result)
    console.log('[useApi] createPrompt error:', error.value)
    return result
  }

  async function updatePrompt(id: string, data: UpdatePromptRequest): Promise<SavedPrompt | null> {
    return request<SavedPrompt>(`/api/prompts/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
  }

  async function deletePrompt(id: string): Promise<boolean> {
    const result = await request<{ success: boolean }>(`/api/prompts/${id}`, {
      method: 'DELETE',
    })
    return result?.success ?? false
  }

  // Directory operations
  async function getDirectory(): Promise<DirectoryResponse | null> {
    return request<DirectoryResponse>('/api/directory')
  }

  async function setDirectory(
    directory: string,
    traverseSubfolders: boolean = false,
    includeVideos: boolean = true,
    includeImages: boolean = false
  ): Promise<DirectoryResponse | null> {
    return request<DirectoryResponse>('/api/directory', {
      method: 'POST',
      body: JSON.stringify({
        directory,
        traverse_subfolders: traverseSubfolders,
        include_videos: includeVideos,
        include_images: includeImages,
      }),
    })
  }

  async function browseDirectory(path?: string): Promise<DirectoryBrowseResponse | null> {
    const url = path
      ? `/api/directory/browse?path=${encodeURIComponent(path)}`
      : '/api/directory/browse'
    return request<DirectoryBrowseResponse>(url)
  }

  return {
    loading,
    error,
    request,
    loadModel,
    unloadModel,
    getModelStatus,
    getModelPresets,
    startProcessing,
    stopProcessing,
    checkHealth,
    getPrompts,
    createPrompt,
    updatePrompt,
    deletePrompt,
    getDirectory,
    setDirectory,
    browseDirectory,
  }
}
