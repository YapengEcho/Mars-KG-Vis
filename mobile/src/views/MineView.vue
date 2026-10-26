<script setup lang="ts">
import { Bookmark, ArrowRight, Info } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import PageHeading from '../components/PageHeading.vue'
import { atlas } from '../stores/atlas'
</script>
<template>
  <section class="content-view">
    <PageHeading
      eyebrow="YOUR SPACE"
      title="我的"
      description="收藏、识别记录与设备状态。"
    />

    <div class="section-heading">
      <h2>我的收藏 <span>{{ atlas.saved.value.length }}</span></h2>
    </div>
    <div v-if="atlas.saved.value.length" class="record-list">
      <SlideCard v-for="slide in atlas.saved.value" :key="slide.id" :slide="slide" />
    </div>
    <div v-else class="empty-state">
      <Bookmark :size="40" :stroke-width="1" />
      <h2>还没有收藏</h2>
      <p>在点位卡片或详情中点击收藏，方便再次查看。</p>
      <RouterLink to="/" class="secondary-button">开始探索 <ArrowRight :size="17" /></RouterLink>
    </div>

    <div class="section-heading mine-section">
      <h2>识别记录</h2>
    </div>
    <div class="empty-inline">识别记录将在接入模型后展示，当前暂无记录。</div>

    <div class="section-heading mine-section">
      <h2>版本与服务状态</h2>
    </div>
    <div class="service-status"><span></span>识别服务尚未接入</div>
    <div class="info-card">
      <Info :size="18" />
      <p>版本 0.1.0 · 收藏与识别记录仅保存在当前设备，不会上传。</p>
    </div>
  </section>
</template>