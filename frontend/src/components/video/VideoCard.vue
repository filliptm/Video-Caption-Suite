<script setup lang="ts">
import { computed } from 'vue'
import type { VideoInfo } from '@/types'

interface Props {
  video: VideoInfo
  selected: boolean
  processing?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  processing: false,
})

const emit = defineEmits<{
  select: []
  delete: []
  viewCaption: []
}>()

const statusIcon = computed(() => {
  if (props.processing) return 'processing'
  if (props.video.has_caption) return 'complete'
  return 'pending'
})

const formattedSize = computed(() => {
  const mb = props.video.size_mb
  if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`
  return `${mb.toFixed(1)} MB`
})

const formattedDuration = computed(() => {
  if (!props.video.duration_sec) return null
  const sec = Math.floor(props.video.duration_sec)
  const mins = Math.floor(sec / 60)
  const secs = sec % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
})
</script>

<template>
  <div
    :class="[
      'group relative p-3 rounded-lg border transition-all cursor-pointer',
      selected
        ? 'bg-primary-900/30 border-primary-500'
        : 'bg-dark-800 border-dark-700 hover:border-dark-500',
      processing && 'animate-pulse',
    ]"
    @click="emit('select')"
  >
    <!-- Selection checkbox -->
    <button
      class="absolute top-3 left-3 p-0.5"
      type="button"
      title="Toggle selection"
      @click.stop="emit('select')"
    >
      <div
        :class="[
          'w-5 h-5 rounded border-2 flex items-center justify-center transition-colors',
          selected
            ? 'bg-primary-500 border-primary-500'
            : 'border-dark-500 hover:border-dark-300',
        ]"
      >
        <svg
          v-if="selected"
          class="w-3 h-3 text-white"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="3"
            d="M5 13l4 4L19 7"
          />
        </svg>
      </div>
    </button>

    <!-- Status indicator -->
    <div class="absolute top-3 right-3">
      <div
        v-if="statusIcon === 'complete'"
        class="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center"
      >
        <svg class="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <div
        v-else-if="statusIcon === 'processing'"
        class="w-6 h-6 rounded-full bg-primary-500/20 flex items-center justify-center"
      >
        <svg class="w-4 h-4 text-primary-400 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <div
        v-else
        class="w-6 h-6 rounded-full bg-dark-600 flex items-center justify-center"
      >
        <div class="w-2 h-2 rounded-full bg-dark-400" />
      </div>
    </div>

    <!-- Video icon -->
    <div class="flex items-center justify-center h-16 mb-2">
      <svg class="w-12 h-12 text-dark-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
          d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    </div>

    <!-- Video name -->
    <h4 class="text-sm font-medium text-dark-200 truncate mb-1" :title="video.name">
      {{ video.name }}
    </h4>

    <!-- Video info -->
    <div class="flex items-center gap-2 text-xs text-dark-400">
      <span>{{ formattedSize }}</span>
      <span v-if="formattedDuration" class="flex items-center gap-1">
        <span class="text-dark-600">|</span>
        {{ formattedDuration }}
      </span>
      <span v-if="video.width && video.height" class="flex items-center gap-1">
        <span class="text-dark-600">|</span>
        {{ video.width }}x{{ video.height }}
      </span>
    </div>

    <!-- Caption preview -->
    <div
      v-if="video.has_caption && video.caption_preview"
      class="mt-2 pt-2 border-t border-dark-700"
    >
      <p class="text-xs text-dark-400 line-clamp-2">
        {{ video.caption_preview }}
      </p>
    </div>

    <!-- Actions (shown on hover) -->
    <div class="absolute bottom-3 right-3 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
      <button
        v-if="video.has_caption"
        class="p-1.5 rounded bg-dark-700 hover:bg-dark-600 text-dark-300"
        title="View caption"
        @click.stop="emit('viewCaption')"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        </svg>
      </button>
      <button
        class="p-1.5 rounded bg-red-900/50 hover:bg-red-800 text-red-400"
        title="Delete video"
        @click.stop="emit('delete')"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
      </button>
    </div>
  </div>
</template>
