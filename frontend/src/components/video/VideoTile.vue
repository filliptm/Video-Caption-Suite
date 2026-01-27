<script setup lang="ts">
import { computed, ref, onUnmounted } from 'vue'
import type { VideoInfo } from '@/types'

interface Props {
  video: VideoInfo
  selected: boolean
  processing?: boolean
  showThumbnail?: boolean
  thumbnailSize?: number
}

const props = withDefaults(defineProps<Props>(), {
  processing: false,
  showThumbnail: true,
  thumbnailSize: 200,
})

const emit = defineEmits<{
  select: []
  viewCaption: []
}>()

const thumbnailError = ref(false)
const thumbnailLoaded = ref(false)
const isHovering = ref(false)
const videoRef = ref<HTMLVideoElement | null>(null)
const videoReady = ref(false)
const videoProgress = ref(0)

/**
 * Encode a video path for URL use, preserving directory structure.
 * Encodes each path segment individually to handle special characters
 * while keeping forward slashes intact for proper URL routing.
 */
function encodeVideoPath(path: string): string {
  return path.split('/').map(segment => encodeURIComponent(segment)).join('/')
}

const thumbnailUrl = computed(() => {
  if (!props.showThumbnail || thumbnailError.value) return null
  return `/api/videos/${encodeVideoPath(props.video.name)}/thumbnail?size=${props.thumbnailSize}`
})

const videoPreviewUrl = computed(() => {
  return `/api/videos/${encodeVideoPath(props.video.name)}/stream`
})

const formattedSize = computed(() => {
  const mb = props.video.size_mb
  if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`
  if (mb > 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb.toFixed(1)} MB`
})

const formattedDuration = computed(() => {
  if (!props.video.duration_sec) return null
  const sec = Math.floor(props.video.duration_sec)
  const mins = Math.floor(sec / 60)
  const secs = sec % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
})

const formattedResolution = computed(() => {
  if (!props.video.width || !props.video.height) return null
  return `${props.video.width}x${props.video.height}`
})

const formattedFps = computed(() => {
  if (!props.video.fps) return null
  return `${Math.round(props.video.fps)}fps`
})

const formattedFrameCount = computed(() => {
  if (!props.video.frame_count) return null
  return `${props.video.frame_count.toLocaleString()}f`
})

const isImage = computed(() => {
  return props.video.media_type === 'image'
})

function handleThumbnailError() {
  thumbnailError.value = true
}

function handleThumbnailLoad() {
  thumbnailLoaded.value = true
}

function handleClick(e: MouseEvent) {
  emit('select')
}

function handleViewCaption(e: MouseEvent) {
  e.stopPropagation()
  emit('viewCaption')
}

function handleMouseEnter() {
  isHovering.value = true
  videoReady.value = false
  videoProgress.value = 0
}

function handleMouseLeave() {
  isHovering.value = false
  videoReady.value = false
  videoProgress.value = 0
  if (videoRef.value) {
    videoRef.value.pause()
    videoRef.value.currentTime = 0
  }
}

function handleVideoCanPlay() {
  if (isHovering.value && videoRef.value) {
    videoReady.value = true
    videoRef.value.play().catch(() => {})
  }
}

function handleVideoLoadedData() {
  if (isHovering.value && videoRef.value) {
    videoRef.value.play().catch(() => {})
  }
}

function handleVideoTimeUpdate() {
  if (videoRef.value && videoRef.value.duration) {
    videoProgress.value = (videoRef.value.currentTime / videoRef.value.duration) * 100
  }
}

onUnmounted(() => {
  if (videoRef.value) {
    videoRef.value.pause()
  }
})
</script>

<template>
  <div
    :class="[
      'video-tile group relative h-full rounded-lg overflow-hidden cursor-pointer transition-all duration-150 flex flex-col',
      'border-2',
      selected
        ? 'border-primary-500 ring-2 ring-primary-500/30'
        : 'border-dark-700 hover:border-dark-500',
      processing && 'ring-2 ring-primary-400/50 animate-pulse',
    ]"
    @click="handleClick"
    @mouseenter="handleMouseEnter"
    @mouseleave="handleMouseLeave"
  >
    <!-- Thumbnail/Video preview area - takes remaining space -->
    <div class="relative flex-1 min-h-0 bg-dark-800 overflow-hidden">
      <!-- Thumbnail image (shown when not hovering for videos, always for images) -->
      <img
        v-if="thumbnailUrl && (!isHovering || isImage)"
        :src="thumbnailUrl"
        :alt="video.name"
        class="absolute inset-0 w-full h-full object-cover z-0 transition-transform duration-300"
        :class="[
          thumbnailLoaded ? 'opacity-100' : 'opacity-0',
          isImage && isHovering ? 'scale-110' : 'scale-100'
        ]"
        loading="lazy"
        @error="handleThumbnailError"
        @load="handleThumbnailLoad"
      />

      <!-- Video preview (shown on hover for videos only) - higher z-index -->
      <video
        v-if="isHovering && !isImage"
        ref="videoRef"
        :src="videoPreviewUrl"
        class="absolute inset-0 w-full h-full object-cover z-10"
        muted
        loop
        playsinline
        autoplay
        preload="none"
        @canplay="handleVideoCanPlay"
        @loadeddata="handleVideoLoadedData"
        @timeupdate="handleVideoTimeUpdate"
      />

      <!-- Video progress bar (only for videos) -->
      <div
        v-if="isHovering && !isImage"
        class="absolute bottom-0 left-0 right-0 h-1 bg-black/40 z-20"
      >
        <div
          class="h-full bg-primary-500 transition-[width] duration-100"
          :style="{ width: `${videoProgress}%` }"
        />
      </div>

      <!-- Fallback icon (video or image) -->
      <div
        v-if="(!thumbnailUrl || !thumbnailLoaded) && !isHovering"
        class="absolute inset-0 flex items-center justify-center bg-dark-800"
      >
        <!-- Image icon -->
        <svg v-if="isImage" class="w-1/4 max-w-12 text-dark-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <!-- Video icon -->
        <svg v-else class="w-1/4 max-w-12 text-dark-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      </div>

      <!-- Play indicator on hover (only for videos) -->
      <div
        v-if="isHovering && !isImage"
        class="absolute top-1 left-1/2 -translate-x-1/2 px-1.5 py-0.5 bg-black/60 rounded text-[9px] text-white/80 flex items-center gap-1"
      >
        <span class="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" />
        PREVIEW
      </div>

      <!-- Duration badge (videos) or Image badge (images) -->
      <div
        v-if="!isHovering && (formattedDuration || isImage)"
        class="absolute bottom-1 right-1 px-1.5 py-0.5 text-[10px] font-medium bg-black/70 text-white rounded flex items-center gap-1"
      >
        <template v-if="isImage">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          IMG
        </template>
        <template v-else>
          {{ formattedDuration }}
        </template>
      </div>

      <!-- Selection checkbox -->
      <button
        class="absolute top-1.5 left-1.5 z-20 p-0.5"
        type="button"
        title="Toggle selection"
        @click.stop="emit('select')"
      >
        <div
          :class="[
            'w-5 h-5 rounded border-2 flex items-center justify-center transition-all',
            'shadow-sm',
            selected
              ? 'bg-primary-500 border-primary-500'
              : 'bg-dark-900/80 border-dark-400 hover:border-dark-200',
          ]"
        >
          <svg
            v-if="selected"
            class="w-3 h-3 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
          </svg>
        </div>
      </button>

      <!-- Status indicator -->
      <div class="absolute top-1.5 right-1.5 z-10">
        <div
          v-if="processing"
          class="w-5 h-5 rounded-full bg-primary-500/90 flex items-center justify-center shadow-sm"
        >
          <svg class="w-3 h-3 text-white animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
        <div
          v-else-if="video.has_caption"
          class="w-5 h-5 rounded-full bg-green-500/90 flex items-center justify-center shadow-sm"
        >
          <svg class="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
          </svg>
        </div>
      </div>
    </div>

    <!-- Info area - fixed height, always visible -->
    <div class="flex-shrink-0 p-2 bg-dark-850">
      <!-- Video name -->
      <h4
        class="text-xs font-medium text-dark-200 truncate leading-tight"
        :title="video.name"
      >
        {{ video.name }}
      </h4>

      <!-- Meta row 1: Resolution, FPS, Frames (FPS/Frames for videos only) -->
      <div class="flex items-center gap-1.5 mt-1 text-[10px] text-dark-500 truncate">
        <span v-if="formattedResolution">{{ formattedResolution }}</span>
        <template v-if="!isImage">
          <span v-if="formattedResolution && formattedFps" class="text-dark-600">·</span>
          <span v-if="formattedFps">{{ formattedFps }}</span>
          <span v-if="formattedFps && formattedFrameCount" class="text-dark-600">·</span>
          <span v-if="formattedFrameCount">{{ formattedFrameCount }}</span>
        </template>
      </div>

      <!-- Meta row 2: Size and caption button -->
      <div class="flex items-center justify-between mt-0.5 gap-2">
        <span class="text-[10px] text-dark-500 truncate">{{ formattedSize }}</span>

        <!-- View caption button -->
        <button
          v-if="video.has_caption"
          class="flex-shrink-0 p-1 rounded text-dark-500 hover:text-primary-400 hover:bg-dark-700 transition-colors"
          title="View caption"
          @click="handleViewCaption"
        >
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-tile {
  contain: layout style paint;
}
</style>
