<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'
import PromptLibrary from './PromptLibrary.vue'

const settingsStore = useSettingsStore()
const { settings } = storeToRefs(settingsStore)

function updatePrompt(value: string) {
  settingsStore.setLocalSetting('prompt', value)
}

function handleLoadPrompt(prompt: string) {
  settingsStore.setLocalSetting('prompt', prompt)
}

function handleInput(e: Event) {
  const target = e.target as HTMLTextAreaElement
  updatePrompt(target.value)
}
</script>

<template>
  <div class="flex flex-col h-full">
    <!-- Prompt Library controls - fixed header -->
    <div class="flex-shrink-0 flex items-center justify-between mb-2">
      <label class="text-sm font-medium text-dark-200">Caption Prompt</label>
      <PromptLibrary :current-prompt="settings.prompt" @load-prompt="handleLoadPrompt" />
    </div>

    <!-- Prompt textarea - fills remaining space with internal scroll -->
    <div class="flex-1 min-h-0 flex flex-col">
      <textarea
        :value="settings.prompt"
        placeholder="Enter the prompt for video captioning..."
        class="flex-1 min-h-0 w-full px-3 py-2 bg-dark-800 border border-dark-600 rounded-lg text-dark-100 placeholder-dark-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
        @input="handleInput"
      />
      <p class="flex-shrink-0 text-xs text-dark-500 mt-2">
        This prompt is sent to the model with each video.
      </p>
    </div>
  </div>
</template>
