<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowLeft, ScanLine, Trash2 } from '@lucide/vue'
import PageHeading from '../components/PageHeading.vue'
import ZoomableImage from '../components/ZoomableImage.vue'
import { formatPercent, formatRecognizedAt } from '../lib/recognition'
import { recognitions } from '../stores/recognitions'
import { slideTypes, typeInfo } from '../types'

const route = useRoute()
const router = useRouter()
const record = computed(() => recognitions.get(String(route.params.id ?? '')))
function remove() {
  if (!record.value) return
  if (!window.confirm('删除这条本机识别记录？')) return
  recognitions.remove(record.value.id)
  router.replace('/mine')
}
</script>

<template>
  <section class="content-view">
    <RouterLink to="/mine" class="text-button back-link"
      ><ArrowLeft :size="16" />返回我的</RouterLink
    >
    <template v-if="record">
      <PageHeading
        eyebrow="RECOGNITION RECORD"
        title="识别记录"
        :description="`${formatRecognizedAt(record.createdAt)} · 仅保存在本机`"
      />
      <ZoomableImage
        v-if="record.image"
        :src="record.image"
        :alt="`${record.fileName} 的识别缩略图`"
      />
      <div v-else class="recognition-image-empty">
        <ScanLine :size="34" :stroke-width="1" />
        <span>该记录未保存缩略图，仅保留识别结果。</span>
      </div>
      <div class="result-card recognition-result">
        <div class="result-head">
          <div>
            <span class="eyebrow">识别结果</span>
            <h2>
              <i class="type-swatch" :style="{ background: typeInfo[record.prediction].color }"></i
              >{{ typeInfo[record.prediction].name }} · {{ record.prediction }}
            </h2>
            <p>
              置信度 {{ formatPercent(record.confidence) }} · 模型版本 {{ record.modelVersion }}
            </p>
          </div>
          <span class="result-badge" :class="{ uncertain: record.uncertain }">{{
            record.uncertain ? '待复核' : '已分类'
          }}</span>
        </div>
        <ul class="probability-list">
          <li v-for="type in slideTypes" :key="type">
            <span class="probability-name">{{ type }} · {{ typeInfo[type].name }}</span>
            <span class="probability-value">{{ formatPercent(record.probabilities[type]) }}</span>
            <span class="probability-track"
              ><i
                :style="{
                  width: formatPercent(record.probabilities[type]),
                  background: typeInfo[type].color,
                }"
              ></i
            ></span>
          </li>
        </ul>
        <p class="result-note">
          概率为模型输出分数，未经校准，不作为地质真实性的百分比；待复核记录请结合影像与资料判断。
        </p>
        <p v-if="record.fileName" class="result-meta">原图文件名：{{ record.fileName }}</p>
        <div class="result-actions">
          <RouterLink :to="`/atlas/type/${record.prediction}`" class="secondary-button"
            >查看{{ typeInfo[record.prediction].name }}图鉴</RouterLink
          ><RouterLink to="/recognize" class="secondary-button">再识别一张</RouterLink>
        </div>
        <button class="text-button danger-text" @click="remove">
          <Trash2 :size="15" />删除这条记录
        </button>
      </div>
    </template>
    <div v-else class="empty-state">
      <ScanLine :size="40" :stroke-width="1" />
      <h2>记录不存在</h2>
      <p>这条识别记录可能已被删除，或保存在其他设备上。</p>
      <RouterLink to="/mine" class="secondary-button">返回我的</RouterLink>
    </div>
  </section>
</template>
