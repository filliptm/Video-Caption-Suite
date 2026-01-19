<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useWebSocket, useApi, useResizable } from '@/composables'
import { useVideoStore } from '@/stores/videoStore'
import { useProgressStore } from '@/stores/progressStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { BaseButton, VirtualGrid } from '@/components/base'
import { LayoutSidebar, ResizablePanel } from '@/components/layout'
import { VideoTile, VideoGridToolbar } from '@/components/video'
import { CaptionPanel } from '@/components/caption'
import type { VideoInfo } from '@/types'

const { connect, disconnect } = useWebSocket()
const api = useApi()
const videoStore = useVideoStore()
const progressStore = useProgressStore()
const settingsStore = useSettingsStore()

const { videos, selectedVideos, loading, loadingTotal, loadingLoaded, loadingProgress } = storeToRefs(videoStore)
const { state, wsConnected, isProcessing, isComplete, overallProgress, estimatedTimeRemaining } = storeToRefs(progressStore)

// UI state
const showCaptionPanel = ref(false)
const selectedVideoForCaption = ref<VideoInfo | null>(null)
const loadingModel = ref(false)
const startingProcess = ref(false)

// Grid columns (0 = auto)
const gridColumns = ref(0)

// Resizable caption panel using composable
const {
  width: captionPanelWidth,
  isResizing: isResizingCaption,
  startResize: startResizeCaption
} = useResizable({
  initialWidth: 384,
  minWidth: 280,
  maxWidth: 600,
  direction: 'right'
})

// Processing video name for highlighting
const processingVideoName = computed(() => {
  return isProcessing.value ? state.value.current_video : null
})

// Watch for video completion during processing
// When current_video changes, the previous video has completed
watch(
  () => state.value.current_video,
  (newVideo, oldVideo) => {
    // If we were processing a video and now moved to a different one (or null),
    // the old video completed successfully
    if (oldVideo && isProcessing.value && newVideo !== oldVideo) {
      videoStore.markVideoAsCaptioned(oldVideo)
    }
  }
)

// Watch for processing completion to mark the final video
watch(
  () => isComplete.value,
  (nowComplete, wasComplete) => {
    // When processing just completed, mark the last video as captioned
    if (nowComplete && !wasComplete && state.value.current_video) {
      videoStore.markVideoAsCaptioned(state.value.current_video)
    }
  }
)

// Selection helpers
function isSelected(name: string): boolean {
  return selectedVideos.value.has(name)
}

function toggleSelection(video: VideoInfo) {
  videoStore.toggleVideoSelection(video.name)
}

function handleVideoClick(video: VideoInfo) {
  toggleSelection(video)
}

function handleViewCaption(video: VideoInfo) {
  selectedVideoForCaption.value = video
  showCaptionPanel.value = true
}

async function handleDeleteCaption(videoName: string) {
  await videoStore.deleteCaption(videoName)
  selectedVideoForCaption.value = null
  showCaptionPanel.value = false
}

// Model controls
async function handleLoadModel() {
  loadingModel.value = true
  try {
    await api.loadModel()
  } finally {
    loadingModel.value = false
  }
}

async function handleUnloadModel() {
  loadingModel.value = true
  try {
    await api.unloadModel()
  } finally {
    loadingModel.value = false
  }
}

// Processing controls
async function handleStartProcessing() {
  startingProcess.value = true
  try {
    // Save current settings to backend before starting (ensures prompt is up-to-date)
    await settingsStore.updateSettings(settingsStore.settings)

    const videoNames = selectedVideos.value.size > 0
      ? Array.from(selectedVideos.value)
      : undefined
    await api.startProcessing(videoNames)
  } finally {
    startingProcess.value = false
  }
}

async function handleStopProcessing() {
  await api.stopProcessing()
}

async function refreshVideos() {
  await videoStore.fetchVideos()
}

onMounted(() => {
  connect()
  refreshVideos()
})

onUnmounted(() => {
  disconnect()
})
</script>

<template>
  <div class="app-container h-screen flex flex-col bg-dark-900 text-dark-100 overflow-hidden">
    <!-- Top bar -->
    <header class="flex-shrink-0 h-14 bg-dark-850 border-b border-dark-700 px-4 flex items-center justify-between">
      <!-- Left: Logo & Title -->
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
          <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        <div class="hidden sm:block">
          <h1 class="text-sm font-semibold text-dark-100">Video Caption Suite</h1>
          <p class="text-[10px] text-dark-500">Qwen3-VL-8B</p>
        </div>
      </div>

      <!-- Center: Processing status -->
      <div class="flex items-center gap-4">
        <!-- Progress when processing -->
        <div v-if="isProcessing" class="flex items-center gap-3">
          <div class="w-32 h-1.5 bg-dark-700 rounded-full overflow-hidden">
            <div
              class="h-full bg-primary-500 transition-all duration-300"
              :style="{ width: `${overallProgress}%` }"
            />
          </div>
          <span class="text-xs text-dark-400 font-mono">
            {{ state.video_index + 1 }}/{{ state.total_videos }}
          </span>
          <span v-if="estimatedTimeRemaining" class="text-xs text-dark-500">
            ~{{ estimatedTimeRemaining }}
          </span>
        </div>

        <!-- Token stats when processing/complete -->
        <div v-if="isProcessing || isComplete" class="hidden md:flex items-center gap-2 text-xs">
          <span class="text-dark-500">{{ state.tokens_generated.toLocaleString() }} tokens</span>
          <span v-if="state.tokens_per_sec > 0" class="text-primary-400 font-mono">
            {{ state.tokens_per_sec.toFixed(1) }} tok/s
          </span>
        </div>
      </div>

      <!-- Right: Status indicators -->
      <div class="flex items-center gap-3">
        <!-- VRAM indicator -->
        <div v-if="state.vram_used_gb > 0" class="hidden sm:flex items-center gap-1.5 text-xs">
          <svg class="w-3.5 h-3.5 text-dark-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          <span class="font-mono text-dark-400">{{ state.vram_used_gb.toFixed(1) }}GB</span>
        </div>

        <!-- Connection status -->
        <div class="flex items-center gap-1.5 text-xs">
          <div
            :class="[
              'w-2 h-2 rounded-full',
              wsConnected ? 'bg-green-500' : 'bg-red-500',
            ]"
          />
          <span class="text-dark-500 hidden sm:inline">
            {{ wsConnected ? 'Connected' : 'Disconnected' }}
          </span>
        </div>
      </div>
    </header>

    <!-- Main content area -->
    <div class="flex-1 flex overflow-hidden">
      <!-- Settings sidebar (always visible, resizable) -->
      <LayoutSidebar />

      <!-- Video grid area -->
      <main class="flex-1 flex flex-col overflow-hidden">
        <!-- Toolbar -->
        <VideoGridToolbar
          v-model:grid-columns="gridColumns"
          @refresh="refreshVideos"
        />

        <!-- Video grid with virtual scrolling -->
        <div class="flex-1 p-4 overflow-hidden">
          <!-- Empty state -->
          <div
            v-if="videos.length === 0 && !loading"
            class="h-full flex items-center justify-center"
          >
            <div class="text-center">
              <svg class="w-16 h-16 mx-auto text-dark-700 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <h3 class="text-lg font-medium text-dark-400 mb-1">No videos found</h3>
              <p class="text-sm text-dark-600">Add videos to the input_videos folder</p>
            </div>
          </div>

          <!-- Loading state with progress -->
          <div v-else-if="loading && videos.length === 0" class="h-full flex items-center justify-center">
            <div class="text-center">
              <svg class="w-10 h-10 mx-auto text-primary-500 animate-spin mb-4" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <p class="text-dark-300 font-medium">Loading videos...</p>
              <p v-if="loadingTotal > 0" class="text-dark-500 text-sm mt-1">
                {{ loadingLoaded.toLocaleString() }} / {{ loadingTotal.toLocaleString() }}
              </p>
            </div>
          </div>

          <!-- Virtual grid (show even while loading for progressive display) -->
          <template v-else>
            <!-- Loading progress bar at top -->
            <div v-if="loading && loadingTotal > 0" class="mb-3">
              <div class="flex items-center justify-between text-xs text-dark-400 mb-1">
                <span>Loading videos...</span>
                <span>{{ loadingLoaded.toLocaleString() }} / {{ loadingTotal.toLocaleString() }} ({{ loadingProgress }}%)</span>
              </div>
              <div class="h-1 bg-dark-700 rounded-full overflow-hidden">
                <div
                  class="h-full bg-primary-500 transition-all duration-300"
                  :style="{ width: `${loadingProgress}%` }"
                />
              </div>
            </div>

            <VirtualGrid
              :items="videos"
              :item-min-width="160"
              :gap="12"
              :columns="gridColumns"
              :class="loading && loadingTotal > 0 ? 'flex-1' : ''"
              @item-click="handleVideoClick"
              @columns-changed="(cols) => gridColumns = cols"
            >
              <template #default="{ item: video }">
                <VideoTile
                  :video="video"
                  :selected="isSelected(video.name)"
                  :processing="processingVideoName === video.name"
                  @select="toggleSelection(video)"
                  @view-caption="handleViewCaption(video)"
                />
              </template>
            </VirtualGrid>
          </template>
        </div>
      </main>

      <!-- Caption panel (collapsible, resizable) -->
      <Transition name="slide-right">
        <aside
          v-if="showCaptionPanel"
          class="flex-shrink-0 overflow-hidden relative border-l border-dark-700"
          :style="{ width: `${captionPanelWidth}px` }"
        >
          <!-- Resize handle on left side -->
          <div
            class="absolute top-0 left-0 w-2 h-full cursor-ew-resize group z-10"
            :class="isResizingCaption ? 'bg-primary-500/30' : 'hover:bg-primary-500/20'"
            @mousedown="startResizeCaption"
          >
            <!-- Visible grip dots -->
            <div
              class="absolute top-1/2 left-0 transform -translate-y-1/2 translate-x-0.5 flex flex-col gap-1 py-2 px-0.5 rounded transition-colors"
              :class="isResizingCaption ? 'bg-primary-500/40' : 'bg-dark-700 group-hover:bg-dark-600'"
            >
              <div
                v-for="i in 6"
                :key="i"
                class="w-1 h-1 rounded-full"
                :class="isResizingCaption ? 'bg-primary-400' : 'bg-dark-500 group-hover:bg-dark-400'"
              />
            </div>
          </div>
          <CaptionPanel
            :video="selectedVideoForCaption"
            @close="showCaptionPanel = false"
            @delete="handleDeleteCaption"
          />
        </aside>
      </Transition>
    </div>

    <!-- Bottom action bar -->
    <footer class="flex-shrink-0 h-14 bg-dark-850 border-t border-dark-700 px-4 flex items-center justify-between">
      <!-- Left: Model controls -->
      <div class="flex items-center gap-2">
        <div
          :class="[
            'w-2 h-2 rounded-full',
            state.model_loaded ? 'bg-green-500' : 'bg-dark-600',
          ]"
        />
        <span class="text-xs text-dark-500">
          {{ state.model_loaded ? 'Model loaded' : 'Model not loaded' }}
        </span>
        <BaseButton
          v-if="!state.model_loaded"
          variant="ghost"
          size="sm"
          :loading="loadingModel"
          :disabled="isProcessing"
          @click="handleLoadModel"
        >
          Load
        </BaseButton>
        <BaseButton
          v-else
          variant="ghost"
          size="sm"
          :loading="loadingModel"
          :disabled="isProcessing"
          @click="handleUnloadModel"
        >
          Unload
        </BaseButton>
      </div>

      <!-- Center: Selection info -->
      <div class="text-xs text-dark-500">
        <span v-if="isProcessing" class="text-primary-400">
          Processing {{ state.current_video }}...
        </span>
        <span v-else-if="isComplete" class="text-green-400">
          Completed {{ state.total_videos }} videos
        </span>
        <span v-else-if="selectedVideos.size > 0">
          {{ selectedVideos.size }} selected
        </span>
        <span v-else>
          Select videos to process
        </span>
      </div>

      <!-- Right: Process button -->
      <div class="flex items-center gap-2">
        <BaseButton
          v-if="!isProcessing"
          variant="primary"
          size="sm"
          :loading="startingProcess"
          :disabled="selectedVideos.size === 0 && pendingVideos === 0"
          @click="handleStartProcessing"
        >
          <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Start
        </BaseButton>
        <BaseButton
          v-else
          variant="danger"
          size="sm"
          @click="handleStopProcessing"
        >
          <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
          </svg>
          Stop
        </BaseButton>
      </div>
    </footer>
  </div>
</template>

<style>
/* Slide transition for right sidebar (caption panel) */
.slide-right-enter-active,
.slide-right-leave-active {
  transition: all 0.2s ease;
}

.slide-right-enter-from,
.slide-right-leave-to {
  transform: translateX(100%);
  opacity: 0;
}

/* Custom slider thumb */
.slider-thumb::-webkit-slider-thumb {
  appearance: none;
  width: 12px;
  height: 12px;
  background: #0ea5e9;
  border-radius: 50%;
  cursor: pointer;
  transition: background 0.15s;
}

.slider-thumb::-webkit-slider-thumb:hover {
  background: #38bdf8;
}

.slider-thumb::-moz-range-thumb {
  width: 12px;
  height: 12px;
  background: #0ea5e9;
  border-radius: 50%;
  cursor: pointer;
  border: none;
}
</style>
