<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Bookmark, ArrowRight, ChevronRight, Info, ScanLine } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import PageHeading from '../components/PageHeading.vue'
import { atlas } from '../stores/atlas'
import { recognitions } from '../stores/recognitions'
import { fetchCapabilities } from '../services/api'
import { formatPercent, formatRecognizedAt } from '../lib/recognition'
import { typeInfo } from '../types'

const serviceReady = ref(false)
const serviceText = ref('正在检查识别服务…')
onMounted(async () => {
  try {
    const cap = await fetchCapabilities()
    if (!cap) serviceText.value = '尚未连接后端服务，识别记录只在本机回看'
    else if (cap.classifier.available) {
      serviceReady.value = true
      serviceText.value = '单图分类服务已连接'
    } else serviceText.value = '识别服务尚未接入，识别记录只在本机回看'
  } catch {
    serviceText.value = '暂时无法连接识别服务'
  }
})
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
      <span class="save-note">仅保存在本机 · 最近收藏在前</span>
    </div>
    <p v-if="atlas.missingSaved.value.length" class="storage-warning" role="status">
      另有 {{ atlas.missingSaved.value.length }} 条收藏不在当前目录中（可能来自其他目录或已删除）。
      <button class="text-button" @click="atlas.pruneSaved()">清理失效收藏</button>
    </p>
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
      <h2>识别记录 <span>{{ recognitions.state.records.length }}</span></h2>
      <span class="save-note">仅保存在本机 · 最近识别在前</span>
    </div>
    <p v-if="recognitions.state.storageError" class="storage-warning" role="status">
      {{ recognitions.state.storageError }}
    </p>
    <div v-if="recognitions.state.records.length" class="recognition-list">
      <RouterLink
        v-for="item in recognitions.state.records"
        :key="item.id"
        :to="`/mine/recognitions/${item.id}`"
        class="recognition-item"
      >
        <img v-if="item.image" :src="item.image" alt="" />
        <span v-else class="recognition-thumb"><ScanLine :size="22" :stroke-width="1" /></span>
        <span class="recognition-info">
          <strong>{{ typeInfo[item.prediction].name }} · {{ item.prediction }}</strong>
          <em
            >{{ formatPercent(item.confidence)
            }}<template v-if="item.uncertain"> · 待复核</template></em
          >
          <span>{{ formatRecognizedAt(item.createdAt) }}</span>
        </span>
        <ChevronRight :size="17" />
      </RouterLink>
    </div>
    <div v-else class="empty-inline">
      还没有识别记录。上传单张滑坡裁图并完成识别后，可在这里回看。
      <RouterLink to="/recognize" class="text-button">去识别</RouterLink>
    </div>

    <div class="section-heading mine-section">
      <h2>版本与服务状态</h2>
    </div>
    <div class="service-status">
      <span :class="{ ready: serviceReady }"></span>{{ serviceText }}
    </div>
    <div class="info-card">
      <Info :size="18" />
      <p>版本 0.1.0 · 收藏与识别记录仅保存在当前设备，不会上传。</p>
    </div>
  </section>
</template>
