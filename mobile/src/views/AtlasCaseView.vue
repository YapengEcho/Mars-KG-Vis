<script setup lang="ts">
import { computed } from 'vue'
import { caseById } from '../lib/atlas'
import CatalogRecordView from './CatalogRecordView.vue'

const props = defineProps<{ id: string }>()
const item = computed(() => caseById(props.id))
</script>

<template>
  <CatalogRecordView v-if="item" :id="item.landslideId" :case-key="item.id" />
  <section v-else class="content-view">
    <RouterLink to="/atlas" class="text-button back-link">返回图鉴</RouterLink>
    <p class="notice">没有这个案例。</p>
  </section>
</template>
