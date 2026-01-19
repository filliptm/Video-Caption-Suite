<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useVideoStore } from '@/stores/videoStore'
import { useApi } from '@/composables/useApi'
import { BaseButton } from '@/components/base'

const videoStore = useVideoStore()
const api = useApi()

const currentDir = ref<string>('')
const inputDir = ref<string>('')  // For the editable input field
const browseDir = ref<string>('')
const parentDir = ref<string | null>(null)
const directories = ref<{ name: string; path: string }[]>([])
const showBrowser = ref(false)
const loading = ref(false)
const errorMsg = ref<string | null>(null)
const inputError = ref<string | null>(null)

async function loadCurrentDirectory() {
  const result = await api.getDirectory()
  if (result) {
    currentDir.value = result.directory
    inputDir.value = result.directory
  }
}

async function loadFromInput() {
  const path = inputDir.value.trim()
  if (!path) {
    inputError.value = 'Please enter a directory path'
    return
  }

  // Check if it's the same as current
  if (path === currentDir.value) {
    inputError.value = null
    return
  }

  loading.value = true
  inputError.value = null

  const result = await api.setDirectory(path)
  if (result) {
    currentDir.value = result.directory
    inputDir.value = result.directory
    // Refresh video list with new directory
    await videoStore.fetchVideos()
  } else {
    inputError.value = api.error.value || 'Failed to set directory'
    // Reset input to current valid directory
    inputDir.value = currentDir.value
  }
  loading.value = false
}

async function browse(path?: string) {
  loading.value = true
  errorMsg.value = null
  const result = await api.browseDirectory(path)
  if (result) {
    browseDir.value = result.current
    parentDir.value = result.parent
    directories.value = result.directories
  } else {
    errorMsg.value = api.error.value || 'Failed to browse directory'
  }
  loading.value = false
}

async function selectDirectory(path: string) {
  loading.value = true
  errorMsg.value = null
  const result = await api.setDirectory(path)
  if (result) {
    currentDir.value = result.directory
    inputDir.value = result.directory
    showBrowser.value = false
    // Refresh video list with new directory
    await videoStore.fetchVideos()
  } else {
    errorMsg.value = api.error.value || 'Failed to set directory'
  }
  loading.value = false
}

function openBrowser() {
  showBrowser.value = true
  browse(currentDir.value || undefined)
}

function closeBrowser() {
  showBrowser.value = false
  errorMsg.value = null
}

onMounted(() => {
  loadCurrentDirectory()
})
</script>

<template>
  <div class="space-y-4">
    <!-- Current Directory Input -->
    <div>
      <label class="block text-sm font-medium text-dark-300 mb-1">
        Working Directory
      </label>
      <input
        v-model="inputDir"
        placeholder="Paste or type a directory path..."
        class="w-full px-3 py-2 bg-dark-700 border border-dark-600 rounded-lg text-dark-200 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        @keydown.enter="loadFromInput"
      />
      <!-- Buttons below input -->
      <div class="flex gap-2 mt-2">
        <BaseButton
          variant="secondary"
          :loading="loading"
          :disabled="inputDir === currentDir"
          @click="loadFromInput"
        >
          Load
        </BaseButton>
        <BaseButton @click="openBrowser">
          Browse
        </BaseButton>
      </div>
      <!-- Error message for input -->
      <p v-if="inputError" class="text-xs text-red-400 mt-2">
        {{ inputError }}
      </p>
      <p v-else class="text-xs text-dark-500 mt-2">
        Paste a path and press Enter or click Load. Videos and captions use this directory.
      </p>
    </div>

    <!-- Directory Browser Modal -->
    <Teleport to="body">
      <div
        v-if="showBrowser"
        class="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
        @click.self="closeBrowser"
      >
        <div class="bg-dark-800 rounded-lg shadow-xl w-[600px] max-h-[70vh] flex flex-col border border-dark-700">
          <!-- Header -->
          <div class="p-4 border-b border-dark-700">
            <h3 class="font-medium text-dark-100">Select Directory</h3>
            <p class="text-sm text-dark-400 truncate font-mono mt-1">{{ browseDir }}</p>
          </div>

          <!-- Error message -->
          <div v-if="errorMsg" class="px-4 py-2 bg-red-900/30 text-red-400 text-sm">
            {{ errorMsg }}
          </div>

          <!-- Directory list -->
          <div class="flex-1 overflow-y-auto p-2 min-h-[300px]">
            <!-- Loading spinner -->
            <div v-if="loading" class="flex items-center justify-center h-full">
              <svg class="animate-spin h-6 w-6 text-primary-500" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            </div>

            <template v-else>
              <!-- Parent directory -->
              <button
                v-if="parentDir"
                class="w-full flex items-center gap-3 px-3 py-2 hover:bg-dark-700 rounded-lg text-left transition-colors"
                @click="browse(parentDir)"
              >
                <svg class="w-5 h-5 text-dark-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
                </svg>
                <span class="text-dark-300">..</span>
                <span class="text-dark-500 text-sm">(parent directory)</span>
              </button>

              <!-- Subdirectories -->
              <button
                v-for="dir in directories"
                :key="dir.path"
                class="w-full flex items-center gap-3 px-3 py-2 hover:bg-dark-700 rounded-lg text-left transition-colors"
                @click="browse(dir.path)"
              >
                <svg class="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                  />
                </svg>
                <span class="text-dark-200">{{ dir.name }}</span>
              </button>

              <!-- Empty state -->
              <div
                v-if="directories.length === 0 && !parentDir"
                class="flex flex-col items-center justify-center h-full text-dark-500"
              >
                <svg class="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                  />
                </svg>
                <p>No subdirectories</p>
              </div>
            </template>
          </div>

          <!-- Footer -->
          <div class="p-4 border-t border-dark-700 flex justify-between items-center">
            <p class="text-sm text-dark-500">
              {{ directories.length }} folder{{ directories.length !== 1 ? 's' : '' }}
            </p>
            <div class="flex gap-2">
              <BaseButton variant="ghost" @click="closeBrowser">
                Cancel
              </BaseButton>
              <BaseButton :loading="loading" @click="selectDirectory(browseDir)">
                Select This Folder
              </BaseButton>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
