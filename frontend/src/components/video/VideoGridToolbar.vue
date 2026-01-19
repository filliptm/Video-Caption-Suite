<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useVideoStore } from '@/stores/videoStore'

interface Props {
  gridColumns: number
  minColumns?: number
  maxColumns?: number
}

const props = withDefaults(defineProps<Props>(), {
  minColumns: 2,
  maxColumns: 12,
})

const emit = defineEmits<{
  'update:gridColumns': [value: number]
  refresh: []
}>()

const videoStore = useVideoStore()
const { loading, totalVideos, captionedVideos, pendingVideos, selectedVideos, videos } = storeToRefs(videoStore)

// Compute which selection state is active
const isAllSelected = computed(() => {
  return videos.value.length > 0 && selectedVideos.value.size === videos.value.length
})

const isNoneSelected = computed(() => {
  return selectedVideos.value.size === 0
})

const isPendingSelected = computed(() => {
  if (selectedVideos.value.size === 0) return false
  const pendingNames = new Set(videos.value.filter(v => !v.has_caption).map(v => v.name))
  if (pendingNames.size === 0) return false
  if (selectedVideos.value.size !== pendingNames.size) return false
  for (const name of selectedVideos.value) {
    if (!pendingNames.has(name)) return false
  }
  return true
})

function handleColumnChange(e: Event) {
  const value = parseInt((e.target as HTMLInputElement).value)
  emit('update:gridColumns', value)
}
</script>

<template>
  <div class="flex-shrink-0 h-12 bg-dark-850 border-b border-dark-700 px-4 flex items-center justify-between">
    <!-- Left: Video stats & selection controls -->
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 text-xs">
        <span class="text-dark-400">{{ totalVideos }} videos</span>
        <span class="text-dark-600">|</span>
        <span class="text-green-400">{{ captionedVideos }} done</span>
        <span class="text-dark-600">|</span>
        <span class="text-yellow-400">{{ pendingVideos }} pending</span>
      </div>

      <div class="hidden sm:flex items-center gap-1">
        <button
          :class="[
            'px-2 py-1 text-xs rounded transition-colors',
            isAllSelected
              ? 'bg-primary-600 text-white'
              : 'hover:bg-dark-700 text-dark-400 hover:text-dark-200'
          ]"
          @click="videoStore.selectAll()"
        >
          All
        </button>
        <button
          :class="[
            'px-2 py-1 text-xs rounded transition-colors',
            isNoneSelected
              ? 'bg-primary-600 text-white'
              : 'hover:bg-dark-700 text-dark-400 hover:text-dark-200'
          ]"
          @click="videoStore.selectNone()"
        >
          None
        </button>
        <button
          :class="[
            'px-2 py-1 text-xs rounded transition-colors',
            isPendingSelected
              ? 'bg-primary-600 text-white'
              : 'hover:bg-dark-700 text-dark-400 hover:text-dark-200'
          ]"
          @click="videoStore.selectPending()"
        >
          Pending
        </button>
      </div>
    </div>

    <!-- Right: Grid size slider & Refresh button -->
    <div class="flex items-center gap-4">
      <!-- Grid columns slider -->
      <div class="hidden md:flex items-center gap-2">
        <svg class="w-4 h-4 text-dark-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
          />
        </svg>
        <input
          type="range"
          :min="minColumns"
          :max="maxColumns"
          :value="gridColumns || 4"
          class="w-20 h-1 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-thumb"
          @input="handleColumnChange"
        />
        <span class="text-xs text-dark-500 w-4 text-center">{{ gridColumns || 'A' }}</span>
      </div>

      <!-- Refresh button -->
      <button
        class="p-1.5 rounded hover:bg-dark-700 text-dark-400 hover:text-dark-200 transition-colors"
        :class="loading && 'animate-spin'"
        :disabled="loading"
        @click="emit('refresh')"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
          />
        </svg>
      </button>
    </div>
  </div>
</template>
