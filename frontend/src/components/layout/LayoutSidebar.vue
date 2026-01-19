<script setup lang="ts">
import ResizablePanel from './ResizablePanel.vue'
import { SettingsPanel } from '@/components/settings'
import { StatusPanel } from '@/components/progress'

interface Props {
  initialWidth?: number
  minWidth?: number
  maxWidth?: number
}

withDefaults(defineProps<Props>(), {
  initialWidth: 384,
  minWidth: 320,
  maxWidth: 600,
})
</script>

<template>
  <ResizablePanel
    side="left"
    :initial-width="initialWidth"
    :min-width="minWidth"
    :max-width="maxWidth"
    class="bg-dark-850"
  >
    <!-- Settings area - SettingsPanel handles its own scrolling per tab -->
    <div class="flex-1 min-h-0 p-4 flex flex-col">
      <SettingsPanel class="flex-1 min-h-0" />
    </div>

    <!-- Fixed StatusPanel at bottom -->
    <div class="flex-shrink-0 border-t border-dark-700 p-4 bg-dark-850">
      <StatusPanel />
    </div>
  </ResizablePanel>
</template>
