<script setup lang="ts">
interface Props {
  title?: string
  collapsible?: boolean
  defaultCollapsed?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  collapsible: false,
  defaultCollapsed: false,
})

import { ref } from 'vue'

const collapsed = ref(props.defaultCollapsed)

function toggleCollapse() {
  if (props.collapsible) {
    collapsed.value = !collapsed.value
  }
}
</script>

<template>
  <div class="card flex flex-col">
    <div
      v-if="title || $slots.header"
      class="card-header flex-shrink-0 flex items-center justify-between"
      :class="{ 'cursor-pointer hover:bg-dark-700/50': collapsible }"
      @click="toggleCollapse"
    >
      <slot name="header">
        <h3 class="text-sm font-semibold text-dark-200">{{ title }}</h3>
      </slot>
      <svg
        v-if="collapsible"
        class="w-5 h-5 text-dark-400 transition-transform duration-200"
        :class="{ 'rotate-180': !collapsed }"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M19 9l-7 7-7-7"
        />
      </svg>
    </div>
    <div v-show="!collapsed" class="card-body flex-1 min-h-0 flex flex-col">
      <slot />
    </div>
    <div v-if="$slots.footer" class="flex-shrink-0 px-4 py-3 border-t border-dark-700 bg-dark-800/30">
      <slot name="footer" />
    </div>
  </div>
</template>
