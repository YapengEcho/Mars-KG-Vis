<script setup lang="ts">
defineProps<{ eyebrow: string; title: string; description?: string }>()
</script>
<template>
  <div class="page-heading">
    <span class="eyebrow">{{ eyebrow }}</span>
    <h1>{{ title }}</h1>
    <p v-if="description">{{ description }}</p>
  </div>
</template>