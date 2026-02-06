<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useResourceStore } from '@/stores/resourceStore'

const resourceStore = useResourceStore()
const {
  snapshot,
  cpuPercent,
  ramPercent,
  ramDisplay,
  hasGPU,
  primaryGpu,
  gpuUtilPercent,
  vramDisplay,
  gpuTemp,
  tempColor,
} = storeToRefs(resourceStore)

const expanded = ref(false)
const panelRef = ref<HTMLElement | null>(null)

function toggleExpanded() {
  expanded.value = !expanded.value
}

function handleClickOutside(e: MouseEvent) {
  if (panelRef.value && !panelRef.value.contains(e.target as Node)) {
    expanded.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})

const vramPercent = computed(() => {
  if (!primaryGpu.value || primaryGpu.value.vram_total_gb === 0) return 0
  return Math.round((primaryGpu.value.vram_used_gb / primaryGpu.value.vram_total_gb) * 100)
})

function utilizationColor(percent: number): string {
  if (percent < 50) return 'bg-green-500'
  if (percent < 80) return 'bg-yellow-500'
  return 'bg-red-500'
}

function utilizationTextColor(percent: number): string {
  if (percent < 50) return 'text-green-400'
  if (percent < 80) return 'text-yellow-400'
  return 'text-red-400'
}
</script>

<template>
  <div ref="panelRef" class="relative">
    <!-- Compact indicators (always visible) -->
    <button
      class="flex items-center gap-3 px-2 py-1 rounded hover:bg-dark-700/50 transition-colors text-xs"
      @click.stop="toggleExpanded"
      title="System resources (click for details)"
    >
      <!-- CPU -->
      <div class="flex flex-col items-center gap-0.5">
        <div class="flex items-center gap-1">
          <svg class="w-3.5 h-3.5 text-dark-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          <span class="font-mono" :class="utilizationTextColor(cpuPercent)">{{ cpuPercent }}%</span>
        </div>
        <div class="w-28 h-1 bg-dark-700 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all duration-500"
            :class="utilizationColor(cpuPercent)"
            :style="{ width: `${cpuPercent}%` }"
          />
        </div>
      </div>

      <!-- RAM -->
      <div class="hidden sm:flex flex-col items-center gap-0.5">
        <div class="flex items-center gap-1">
          <svg class="w-3.5 h-3.5 text-dark-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          <span class="font-mono text-dark-400">{{ ramDisplay }}</span>
        </div>
        <div class="w-28 h-1 bg-dark-700 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all duration-500"
            :class="utilizationColor(ramPercent)"
            :style="{ width: `${ramPercent}%` }"
          />
        </div>
      </div>

      <!-- GPU (if available) -->
      <template v-if="hasGPU">
        <!-- GPU Util -->
        <div class="hidden md:flex flex-col items-center gap-0.5">
          <div class="flex items-center gap-1">
            <span class="text-dark-500">GPU</span>
            <span class="font-mono" :class="utilizationTextColor(gpuUtilPercent)">{{ gpuUtilPercent }}%</span>
          </div>
          <div class="w-28 h-1 bg-dark-700 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-all duration-500"
              :class="utilizationColor(gpuUtilPercent)"
              :style="{ width: `${gpuUtilPercent}%` }"
            />
          </div>
        </div>

        <!-- VRAM -->
        <div v-if="vramDisplay" class="hidden lg:flex flex-col items-center gap-0.5">
          <div class="flex items-center gap-1">
            <span class="text-dark-500">VRAM</span>
            <span class="font-mono text-dark-400">{{ vramDisplay }}</span>
          </div>
          <div class="w-28 h-1 bg-dark-700 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-all duration-500"
              :class="utilizationColor(vramPercent)"
              :style="{ width: `${vramPercent}%` }"
            />
          </div>
        </div>

        <!-- Temperature -->
        <div v-if="gpuTemp !== null" class="hidden xl:flex items-center gap-1">
          <div
            class="w-1.5 h-1.5 rounded-full"
            :class="gpuTemp < 60 ? 'bg-green-400' : gpuTemp < 80 ? 'bg-yellow-400' : 'bg-red-400'"
          />
          <span class="font-mono" :class="tempColor">{{ gpuTemp }}&deg;C</span>
        </div>
      </template>
    </button>

    <!-- Expanded popover -->
    <Transition
      enter-active-class="transition duration-150 ease-out"
      enter-from-class="opacity-0 translate-y-1"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition duration-100 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 translate-y-1"
    >
      <div
        v-if="expanded"
        class="absolute top-full right-0 mt-2 w-80 bg-dark-800 border border-dark-600 rounded-lg shadow-xl z-50 p-4"
      >
        <h3 class="text-sm font-medium text-dark-200 mb-3">System Resources</h3>

        <!-- CPU -->
        <div class="mb-3">
          <div class="flex items-center justify-between text-xs mb-1">
            <span class="text-dark-400">CPU Usage</span>
            <span class="font-mono" :class="utilizationTextColor(cpuPercent)">{{ cpuPercent }}%</span>
          </div>
          <div class="h-1.5 bg-dark-700 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-[width] duration-500"
              :class="utilizationColor(cpuPercent)"
              :style="{ width: `${cpuPercent}%` }"
            />
          </div>
        </div>

        <!-- RAM -->
        <div class="mb-3">
          <div class="flex items-center justify-between text-xs mb-1">
            <span class="text-dark-400">RAM</span>
            <span class="font-mono text-dark-300">{{ snapshot.ram_used_gb.toFixed(1) }} / {{ snapshot.ram_total_gb.toFixed(0) }} GB ({{ ramPercent }}%)</span>
          </div>
          <div class="h-1.5 bg-dark-700 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-[width] duration-500"
              :class="utilizationColor(ramPercent)"
              :style="{ width: `${ramPercent}%` }"
            />
          </div>
        </div>

        <!-- GPUs -->
        <template v-if="snapshot.gpus.length > 0">
          <div class="border-t border-dark-700 pt-3 mt-3">
            <div
              v-for="gpu in snapshot.gpus"
              :key="gpu.index"
              class="mb-3 last:mb-0"
            >
              <div class="text-xs text-dark-300 font-medium mb-2 truncate" :title="gpu.name">
                GPU {{ gpu.index }}: {{ gpu.name }}
              </div>

              <!-- GPU Utilization -->
              <div class="mb-2">
                <div class="flex items-center justify-between text-xs mb-1">
                  <span class="text-dark-500">Utilization</span>
                  <span class="font-mono" :class="utilizationTextColor(gpu.utilization_percent)">{{ Math.round(gpu.utilization_percent) }}%</span>
                </div>
                <div class="h-1.5 bg-dark-700 rounded-full overflow-hidden">
                  <div
                    class="h-full rounded-full transition-[width] duration-500"
                    :class="utilizationColor(gpu.utilization_percent)"
                    :style="{ width: `${gpu.utilization_percent}%` }"
                  />
                </div>
              </div>

              <!-- VRAM -->
              <div class="mb-2">
                <div class="flex items-center justify-between text-xs mb-1">
                  <span class="text-dark-500">VRAM</span>
                  <span class="font-mono text-dark-300">{{ gpu.vram_used_gb.toFixed(1) }} / {{ gpu.vram_total_gb.toFixed(0) }} GB</span>
                </div>
                <div class="h-1.5 bg-dark-700 rounded-full overflow-hidden">
                  <div
                    class="h-full rounded-full transition-[width] duration-500"
                    :class="utilizationColor((gpu.vram_used_gb / gpu.vram_total_gb) * 100)"
                    :style="{ width: `${(gpu.vram_used_gb / gpu.vram_total_gb) * 100}%` }"
                  />
                </div>
              </div>

              <!-- Temp & Power row -->
              <div class="flex items-center gap-4 text-xs">
                <div class="flex items-center gap-1">
                  <span class="text-dark-500">Temp:</span>
                  <span class="font-mono" :class="gpu.temperature_c < 60 ? 'text-green-400' : gpu.temperature_c < 80 ? 'text-yellow-400' : 'text-red-400'">
                    {{ gpu.temperature_c }}&deg;C
                  </span>
                </div>
                <div v-if="gpu.power_draw_w > 0" class="flex items-center gap-1">
                  <span class="text-dark-500">Power:</span>
                  <span class="font-mono text-dark-300">{{ gpu.power_draw_w }}W / {{ gpu.power_limit_w }}W</span>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- No GPU fallback -->
        <div v-else class="text-xs text-dark-500 italic mt-2">
          No GPU detected
        </div>
      </div>
    </Transition>
  </div>
</template>
