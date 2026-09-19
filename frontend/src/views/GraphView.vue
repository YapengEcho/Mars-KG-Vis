<template>
  <div class="graph-page panel">
    <div class="graph-header">
      <span class="panel-title">知识图谱</span>
      <el-select
        v-model="selectedId"
        placeholder="选择滑坡"
        size="small"
        style="width: 220px"
        @change="loadGraph"
      >
        <el-option
          v-for="l in landslides"
          :key="l.id"
          :label="`${l.name} (${l.id})`"
          :value="l.id"
        />
      </el-select>
    </div>
    <div class="graph-body">
      <GraphView
        v-if="subgraph"
        :nodes="subgraph.nodes"
        :edges="subgraph.edges"
        :degraded="subgraph.degraded"
        :height="'calc(100vh - 140px)'"
      />
      <el-empty v-else description="请在上方选择一个滑坡" />
    </div>
    <el-alert
      v-if="subgraph && subgraph.degraded"
      title="图谱降级：当前使用回退数据（Neo4j 不可用）"
      type="warning"
      :closable="false"
      show-icon
    />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import api from '@/api'
import GraphView from '@/components/GraphView.vue'

const landslides = ref([])
const selectedId = ref('')
const subgraph = ref(null)

async function loadGraph() {
  if (!selectedId.value) return
  const res = await api.getSubgraph(selectedId.value)
  subgraph.value = res
}

onMounted(async () => {
  const res = await api.listLandslides()
  landslides.value = res.results
})
</script>

<style scoped>
.graph-page {
  height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
}
.graph-header {
  display: flex;
  align-items: center;
  gap: 16px;
}
.graph-body {
  flex: 1;
  min-height: 0;
}
</style>