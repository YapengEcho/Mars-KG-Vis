<script setup lang="ts">
import { computed } from 'vue'
import { ArrowLeft, ArrowRight, BookOpen, Bookmark, Crosshair, ExternalLink } from '@lucide/vue'
import { useRouter } from 'vue-router'
import { atlas } from '../stores/atlas'
import { atlasContent, caseById, caseRecord, casesOfType, typeEntry } from '../lib/atlas'
import { coordinateLabel, typeInfo } from '../types'

const props = defineProps<{ id: string }>()
const router = useRouter()
const item = computed(() => caseById(props.id))
const entry = computed(() => (item.value ? typeEntry(item.value.type) : undefined))
const record = computed(() =>
  item.value ? caseRecord(item.value, atlas.state.catalog.records) : undefined,
)
const siblings = computed(() => {
  if (!item.value) return []
  const list = casesOfType(item.value.type)
  const index = list.findIndex((c) => c.id === item.value?.id)
  return [list[index - 1], list[index + 1]].filter((c) => Boolean(c))
})
function locate() {
  const id = record.value?.id
  if (!id) return
  atlas.dismiss()
  void router.push({ path: '/', query: { focus: id, visit: String(Date.now()) } })
}
</script>
<template>
  <section class="content-view">
    <RouterLink :to="item ? `/atlas/type/${item.type}` : '/atlas'" class="text-button back-link"
      ><ArrowLeft :size="16" />{{ item ? `返回${typeInfo[item.type].name}图鉴` : '返回图鉴' }}</RouterLink
    >
    <p v-if="!item" class="notice">没有这个案例：{{ props.id }}</p>
    <template v-else-if="entry">
      <div class="page-heading atlas-heading">
        <span class="eyebrow">{{ item.type }} · {{ entry.name }} · {{ item.tile }}</span>
        <h1>{{ item.title }}</h1>
        <p>{{ item.subtitle }}</p>
      </div>

      <div class="case-meta">
        <span class="type-badge" :style="{ color: typeInfo[item.type].color }">
          {{ item.type }} / {{ entry.name }}
        </span>
        <span class="data-badge">{{ item.region }}</span>
        <span class="data-badge">{{ item.previewScale }}</span>
        <span v-for="tag in item.tags" :key="tag" class="data-badge">{{ tag }}</span>
      </div>

      <div class="section-heading"><h2>观察要点</h2></div>
      <ol class="atlas-points case-points">
        <li v-for="point in item.observations" :key="point">{{ point }}</li>
      </ol>

      <div class="section-heading"><h2>目录记录</h2></div>
      <dl v-if="record" class="atlas-facts record-facts">
        <div>
          <dt>编号</dt>
          <dd>{{ record.id }} · 原始编号 {{ record.sourceId ?? '—' }}</dd>
        </div>
        <div>
          <dt>火星经纬度</dt>
          <dd>{{ coordinateLabel(record.longitude, record.latitude) }}</dd>
        </div>
        <div>
          <dt>面积</dt>
          <dd>{{ record.areaKm2 === null ? '暂无可靠记录' : `${record.areaKm2.toFixed(2)} km²` }}</dd>
        </div>
        <div>
          <dt>资料来源</dt>
          <dd>
            <a
              v-if="record.source.url?.startsWith('https://')"
              :href="record.source.url"
              target="_blank"
              rel="noopener noreferrer"
              >{{ record.source.title }} <ExternalLink :size="12" /></a
            ><span v-else>{{ record.source.title }}</span>
          </dd>
        </div>
      </dl>
      <p v-else class="notice">
        当前目录里没有 {{ item.landslideId }}（示例目录与公开子集只含部分记录），
        因此这一页只显示图鉴文案；接入完整目录后会自动显示坐标、面积与来源。
      </p>

      <div class="case-actions">
        <button class="primary-button" :disabled="!record" @click="locate">
          <Crosshair :size="18" />定位到火星球
        </button>
        <button
          class="secondary-button"
          :disabled="!record"
          :aria-pressed="record ? atlas.isSaved(record.id) : false"
          @click="record && atlas.toggleSaved(record.id)"
        >
          <Bookmark :size="17" :fill="record && atlas.isSaved(record.id) ? 'currentColor' : 'none'" />
          {{ record && atlas.isSaved(record.id) ? '已收藏' : '收藏这一条' }}
        </button>
        <RouterLink class="text-button" :to="`/atlas/type/${item.type}`">同类案例</RouterLink>
      </div>

      <div class="section-heading"><h2>引用与出处</h2></div>
      <ul class="citation-list">
        <li>
          <BookOpen :size="15" /><span>{{ item.citationLocator }}</span>
        </li>
        <li>
          <BookOpen :size="15" /><span
            >{{ atlasContent.sourceDocument.title }}（{{ atlasContent.sourceDocument.note }}）</span
          >
        </li>
      </ul>

      <nav class="case-nav" aria-label="同类案例切换">
        <RouterLink v-for="(sib, i) in siblings" :key="sib!.id" :to="`/atlas/case/${sib!.id}`">
          <component :is="i === 0 ? ArrowLeft : ArrowRight" :size="15" />
          <span>{{ sib!.title }}</span>
        </RouterLink>
      </nav>
    </template>
  </section>
</template>
