<script setup lang="ts">
import { Bookmark, ArrowRight } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import { atlas } from '../stores/atlas'
</script>
<template>
  <section class="content-view">
    <div class="page-heading">
      <span class="eyebrow">YOUR COLLECTION</span>
      <h1>为下一次探索留个坐标。</h1>
      <p>已收藏 {{ atlas.saved.value.length }} 个点位，仅保存在当前设备。</p>
    </div>
    <div v-if="atlas.saved.value.length" class="record-list">
      <SlideCard v-for="slide in atlas.saved.value" :key="slide.id" :slide="slide" />
    </div>
    <div v-else class="empty-state">
      <Bookmark :size="40" :stroke-width="1" />
      <h2>这里等着你的第一个发现</h2>
      <p>在点位卡片或详情中点击收藏，方便再次查看。</p>
      <RouterLink to="/" class="secondary-button">开始探索 <ArrowRight :size="17" /></RouterLink>
    </div>
  </section>
</template>
