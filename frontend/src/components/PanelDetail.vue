<template>
  <div class="panel side-panel detail-panel">
    <div class="panel-title">详情 · {{ selected.id }}</div>
    <dl class="detail-list">
      <template v-if="detail">
        <div class="d-item"><dt>名称</dt><dd>{{ detail.name }}</dd></div>
        <div class="d-item"><dt>类型</dt><dd><el-tag size="small">{{ detail.type }}</el-tag></dd></div>
        <div class="d-item"><dt>经纬度</dt><dd>{{ detail.lat }}, {{ detail.lon }}</dd></div>
        <div class="d-item"><dt>长度</dt><dd>{{ detail.length_m }}m</dd></div>
        <div class="d-item"><dt>任务</dt><dd>{{ detail.mission }}</dd></div>
        <div class="d-item"><dt>特征</dt><dd>{{ (detail.features || []).join('、') }}</dd></div>
        <div class="d-item d-desc"><dt>描述</dt><dd>{{ detail.description }}</dd></div>
      </template>
    </dl>
    <div class="panel-title sub">图谱 · 邻接关系</div>
    <GraphView
      v-if="subgraph"
      :nodes="subgraph.nodes"
      :edges="subgraph.edges"
      :degraded="subgraph.degraded"
      :height="'200px'"
    />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import api from '@/api'
import GraphView from '@/components/GraphView.vue'

const props = defineProps({
  selected: { type: Object, default: null },
  subgraph: { type: Object, default: null },
})

const detail = ref(null)

watch(
  () => props.selected?.id,
  async (id) => {
    if (!id) {
      detail.value = null
      return
    }
    try {
      detail.value = await api.getLandslide(id)
    } catch (e) {
      detail.value = props.selected
    }
  },
  { immediate: true }
)
</script>

<style scoped>
.detail-panel {
  padding-bottom: 12px;
}
.detail-list {
  padding: 12px 14px;
}
.d-item {
  display: flex;
  margin-bottom: 10px;
  font-size: 13px;
}
.d-item dt {
  width: 56px;
  color: var(--mars-dim);
  flex: 0 0 auto;
}
.d-item dd {
  flex: 1;
  color: var(--mars-text);
}
.d-desc dd {
  line-height: 1.6;
}
.sub {
  margin-top: 6px;
}
</style>