import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { VideoInfo, CaptionInfo, VideoWithStatus, VideoStatus } from '@/types'

export const useVideoStore = defineStore('video', () => {
  const videos = ref<VideoInfo[]>([])
  const captions = ref<CaptionInfo[]>([])
  const selectedVideos = ref<Set<string>>(new Set())
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Loading progress state
  const loadingTotal = ref(0)
  const loadingLoaded = ref(0)
  const loadingProgress = computed(() => {
    if (loadingTotal.value === 0) return 0
    return Math.round((loadingLoaded.value / loadingTotal.value) * 100)
  })

  const videosWithStatus = computed<VideoWithStatus[]>(() => {
    return videos.value.map(video => ({
      ...video,
      status: (video.has_caption ? 'complete' : 'pending') as VideoStatus,
    }))
  })

  const totalVideos = computed(() => videos.value.length)
  const captionedVideos = computed(() => videos.value.filter(v => v.has_caption).length)
  const pendingVideos = computed(() => videos.value.filter(v => !v.has_caption).length)

  const selectedVideosList = computed(() => {
    return videos.value.filter(v => selectedVideos.value.has(v.name))
  })

  async function fetchVideos() {
    loading.value = true
    error.value = null
    loadingTotal.value = 0
    loadingLoaded.value = 0
    videos.value = []

    try {
      const response = await fetch('/api/videos/stream')
      if (!response.ok) throw new Error('Failed to fetch videos')

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      // Throttled flush: accumulate in plain array, apply to reactive state at most every 150ms
      // This reduces Vue reactivity cascades from ~50 to ~10 for large libraries
      let pendingItems: VideoInfo[] = []
      let flushTimer: ReturnType<typeof setTimeout> | null = null

      const flushPending = () => {
        if (pendingItems.length > 0) {
          videos.value = [...videos.value, ...pendingItems]
          pendingItems = []
        }
        flushTimer = null
      }

      const scheduleFlush = () => {
        if (!flushTimer) {
          flushTimer = setTimeout(flushPending, 150)
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Process complete SSE messages
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'total') {
                loadingTotal.value = data.count
              } else if (data.type === 'batch') {
                pendingItems.push(...data.videos)
                loadingLoaded.value = data.loaded
                scheduleFlush()
              } else if (data.type === 'done') {
                // Final flush - apply everything at once
                if (flushTimer) {
                  clearTimeout(flushTimer)
                  flushTimer = null
                }
                flushPending()
              }
            } catch (e) {
              console.error('Failed to parse SSE message:', e)
            }
          }
        }
      }

      // Ensure everything is flushed after stream ends
      if (flushTimer) {
        clearTimeout(flushTimer)
      }
      flushPending()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
      // Fallback to regular endpoint if streaming fails
      try {
        const response = await fetch('/api/videos')
        if (response.ok) {
          const data = await response.json()
          videos.value = data.videos
        }
      } catch {
        // Ignore fallback errors
      }
    } finally {
      loading.value = false
    }
  }

  async function fetchCaptions() {
    try {
      const response = await fetch('/api/captions')
      if (!response.ok) throw new Error('Failed to fetch captions')
      const data = await response.json()
      captions.value = data.captions
    } catch (e) {
      console.error('Failed to fetch captions:', e)
    }
  }

  async function uploadVideo(file: File): Promise<boolean> {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/videos/upload', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Upload failed')
      }
      await fetchVideos()
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
      return false
    }
  }

  async function deleteVideo(name: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/videos/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete video')
      await fetchVideos()
      selectedVideos.value.delete(name)
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
      return false
    }
  }

  async function deleteCaption(videoName: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/captions/${encodeURIComponent(videoName)}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete caption')
      await fetchVideos()
      await fetchCaptions()
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
      return false
    }
  }

  function toggleVideoSelection(name: string) {
    if (selectedVideos.value.has(name)) {
      selectedVideos.value.delete(name)
    } else {
      selectedVideos.value.add(name)
    }
  }

  function selectAll() {
    videos.value.forEach(v => selectedVideos.value.add(v.name))
  }

  function selectNone() {
    selectedVideos.value.clear()
  }

  function selectPending() {
    selectNone()
    videos.value.filter(v => !v.has_caption).forEach(v => selectedVideos.value.add(v.name))
  }

  function getCaptionForVideo(videoName: string): CaptionInfo | undefined {
    const stem = videoName.replace(/\.[^.]+$/, '')
    return captions.value.find(c => c.video_name === stem)
  }

  function markVideoAsCaptioned(videoName: string) {
    const video = videos.value.find(v => v.name === videoName)
    if (video) {
      video.has_caption = true
    }
  }

  return {
    videos,
    captions,
    selectedVideos,
    loading,
    error,
    loadingTotal,
    loadingLoaded,
    loadingProgress,
    videosWithStatus,
    totalVideos,
    captionedVideos,
    pendingVideos,
    selectedVideosList,
    fetchVideos,
    fetchCaptions,
    uploadVideo,
    deleteVideo,
    deleteCaption,
    toggleVideoSelection,
    selectAll,
    selectNone,
    selectPending,
    getCaptionForVideo,
    markVideoAsCaptioned,
  }
})
