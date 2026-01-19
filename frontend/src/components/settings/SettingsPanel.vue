<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'
import { BaseCard, BaseButton } from '@/components/base'
import DirectorySettings from './DirectorySettings.vue'
import ModelSettings from './ModelSettings.vue'
import InferenceSettings from './InferenceSettings.vue'
import OptimizationSettings from './OptimizationSettings.vue'
import PromptSettings from './PromptSettings.vue'

const settingsStore = useSettingsStore()
const { loading, error, hasChanges } = storeToRefs(settingsStore)

type Tab = 'directory' | 'model' | 'inference' | 'optimization' | 'prompt'
const activeTab = ref<Tab>('directory')

const tabs: { key: Tab; label: string }[] = [
  { key: 'directory', label: 'Directory' },
  { key: 'model', label: 'Model' },
  { key: 'inference', label: 'Inference' },
  { key: 'optimization', label: 'Optimization' },
  { key: 'prompt', label: 'Prompt' },
]

async function saveSettings() {
  try {
    await settingsStore.updateSettings(settingsStore.settings)
  } catch {
    // Error is handled in store
  }
}

async function resetSettings() {
  try {
    await settingsStore.resetSettings()
  } catch {
    // Error is handled in store
  }
}

onMounted(() => {
  settingsStore.fetchSettings()
})
</script>

<template>
  <BaseCard>
    <template #header>
      <div class="flex items-center justify-between w-full">
        <h3 class="text-sm font-semibold text-dark-200">Settings</h3>
        <div class="flex items-center gap-2">
          <BaseButton
            v-if="hasChanges"
            variant="ghost"
            size="sm"
            @click="resetSettings"
          >
            Reset
          </BaseButton>
          <BaseButton
            variant="primary"
            size="sm"
            :loading="loading"
            :disabled="!hasChanges"
            @click="saveSettings"
          >
            Save
          </BaseButton>
        </div>
      </div>
    </template>

    <!-- Tabs -->
    <div class="flex flex-wrap border-b border-dark-700 mb-4 -mx-4 px-2">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        :class="[
          'flex-1 min-w-0 px-2 py-2 text-xs sm:text-sm font-medium border-b-2 transition-colors truncate',
          activeTab === tab.key
            ? 'border-primary-500 text-primary-400'
            : 'border-transparent text-dark-400 hover:text-dark-200',
        ]"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- Error message -->
    <div
      v-if="error"
      class="mb-4 p-3 bg-red-900/30 border border-red-800 rounded-lg text-sm text-red-400"
    >
      {{ error }}
    </div>

    <!-- Tab content with scroll constraint -->
    <div
      :class="[
        '-mx-4 px-4',
        activeTab === 'prompt'
          ? 'flex-1 flex flex-col min-h-0'
          : 'min-h-[200px] max-h-[50vh] overflow-y-auto'
      ]"
    >
      <DirectorySettings v-if="activeTab === 'directory'" />
      <ModelSettings v-if="activeTab === 'model'" />
      <InferenceSettings v-if="activeTab === 'inference'" />
      <OptimizationSettings v-if="activeTab === 'optimization'" />
      <PromptSettings v-if="activeTab === 'prompt'" class="flex-1 min-h-0" />
    </div>
  </BaseCard>
</template>
