<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'
import { useApi } from '@/composables/useApi'
import { BaseInput, BaseSelect, BaseToggle } from '@/components/base'
import type { DeviceType, DtypeType, ModelPresetInfo } from '@/types'

const settingsStore = useSettingsStore()
const { settings } = storeToRefs(settingsStore)
const api = useApi()

const presets = ref<ModelPresetInfo[]>([])
const defaultPresetId = ref<string>('qwen3-vl-8b')
const showCustomId = ref(false)

const deviceOptions = [
  { value: 'cuda', label: 'CUDA (GPU)' },
  { value: 'cpu', label: 'CPU' },
]

const dtypeOptions = [
  { value: 'bfloat16', label: 'BFloat16 (Recommended)' },
  { value: 'float16', label: 'Float16' },
  { value: 'float32', label: 'Float32' },
]

const visionTokenBudgetOptions = [
  { value: '70', label: '70 tokens (fastest, coarse detail)' },
  { value: '140', label: '140 tokens' },
  { value: '280', label: '280 tokens (default)' },
  { value: '560', label: '560 tokens (fine detail)' },
  { value: '1120', label: '1120 tokens (OCR / dense text)' },
]

const presetOptions = computed(() =>
  presets.value.map(p => ({
    value: p.id,
    label: `${p.label} — ~${p.approx_vram_gb}GB VRAM`,
  }))
)

const activePreset = computed<ModelPresetInfo | undefined>(() =>
  presets.value.find(p => p.id === settings.value.model_preset)
)

const isGemmaPreset = computed(() => activePreset.value?.id?.startsWith('gemma-4') ?? false)

onMounted(async () => {
  const data = await api.getModelPresets()
  if (data) {
    presets.value = data.presets
    defaultPresetId.value = data.default_preset_id
  }
})

function updatePreset(value: string) {
  settingsStore.setLocalSetting('model_preset', value)
  const preset = presets.value.find(p => p.id === value)
  if (preset) {
    // Sync model_id so custom-ID field reflects the preset choice.
    settingsStore.setLocalSetting('model_id', preset.model_id)
    if (preset.supports_multi_gpu_shard) {
      settingsStore.setLocalSetting('batch_size', 1)
    }
    if (!preset.supports_torch_compile) {
      settingsStore.setLocalSetting('use_torch_compile', false)
    }
    if (!preset.supports_sage_attention) {
      settingsStore.setLocalSetting('use_sage_attention', false)
    }
  }
}

function updateDevice(value: string) {
  settingsStore.setLocalSetting('device', value as DeviceType)
}

function updateDtype(value: string) {
  settingsStore.setLocalSetting('dtype', value as DtypeType)
}

function updateModelId(value: string | number) {
  settingsStore.setLocalSetting('model_id', String(value))
}

function updateVisionTokenBudget(value: string) {
  settingsStore.setLocalSetting('vision_token_budget', parseInt(value, 10))
}

function updateEnableThinking(value: boolean) {
  settingsStore.setLocalSetting('enable_thinking', value)
}
</script>

<template>
  <div class="space-y-4">
    <BaseSelect
      :model-value="settings.model_preset"
      :options="presetOptions"
      label="Model"
      hint="Pick a preset. Switching auto-syncs related settings."
      @update:model-value="updatePreset"
    />

    <p v-if="activePreset" class="text-sm text-dark-400 -mt-2">
      {{ activePreset.description }}
      <span v-if="activePreset.supports_multi_gpu_shard" class="block mt-1 text-primary-400">
        Loads across all available GPUs — batch size forced to 1.
      </span>
      <span v-if="activePreset.quantization" class="block mt-1 text-primary-400">
        Quantization: {{ activePreset.quantization }}
      </span>
    </p>

    <!-- Gemma 4 specific controls -->
    <template v-if="isGemmaPreset">
      <BaseSelect
        :model-value="String(settings.vision_token_budget ?? 280)"
        :options="visionTokenBudgetOptions"
        label="Vision token budget (Gemma 4)"
        hint="Soft tokens per image. Lower = faster, higher = finer detail."
        @update:model-value="updateVisionTokenBudget"
      />

      <BaseToggle
        :model-value="settings.enable_thinking ?? false"
        label="Enable thinking mode"
        description="Gemma 4 can perform step-by-step reasoning before answering. Slower, sometimes better quality."
        @update:model-value="updateEnableThinking"
      />
    </template>

    <!-- Custom model id escape hatch -->
    <div>
      <BaseToggle
        :model-value="showCustomId"
        label="Advanced: custom model ID"
        description="Override the preset with an arbitrary HuggingFace repo id. Most users do not need this."
        @update:model-value="(v) => (showCustomId = v)"
      />
      <div v-if="showCustomId" class="mt-3">
        <BaseInput
          :model-value="settings.model_id"
          label="Model ID"
          placeholder="org/model-name"
          hint="Falls back to the default preset's loader strategy."
          @update:model-value="updateModelId"
        />
      </div>
    </div>

    <BaseSelect
      :model-value="settings.device"
      :options="deviceOptions"
      label="Device"
      hint="GPU (CUDA) is strongly recommended"
      @update:model-value="updateDevice"
    />

    <BaseSelect
      :model-value="settings.dtype"
      :options="dtypeOptions"
      label="Precision"
      hint="BFloat16 is fastest on modern GPUs"
      @update:model-value="updateDtype"
    />
  </div>
</template>
