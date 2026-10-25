<template>
  <div class="dashboard">
    <header class="dash-header">
      <h1>{{ title }}</h1>
      <nav class="dash-nav">
        <el-button
          size="small"
          :type="currentTab === 'overview' ? 'primary' : 'default'"
          @click="currentTab = 'overview'"
        >
          概览
        </el-button>
        <el-button
          size="small"
          :type="currentTab === 'list' ? 'primary' : 'default'"
          @click="currentTab = 'list'"
        >
          列表
        </el-button>
      </nav>
    </header>

    <main class="dash-body">
      <section class="dash-map panel">
        <MarsMap :landslides="landslides" @select="onSelect" />
      </section>

      <aside class="dash-side">
        <!-- 概览侧栏 -->
        <template v-if="currentTab === 'overview'">
          <div class="panel side-panel">
            <div class="panel-title">统计概览</div>
            <div class="stat-grid">
              <div class="stat-item">
                <div class="stat-num">{{ stats.total }}</div>
                <div class="stat-label">滑坡总数</div>
              </div>
              <div class="stat-item">
                <div class="stat-num">{{ stats.by_mission.length }}</div>
                <div class="stat-label">任务数</div>
              </div>
            </div>
            <ChartBar :data="stats.by_type" title="按类型" />
          </div>
          <div class="panel side-panel">
            <div class="panel-title">类型分布</div>
            <ChartPie :data="stats.by_type" />
          </div>
        </template>

        <!-- 列表侧栏 -->
        <div v-else class="panel side-panel list-panel">
          <div class="panel-title">滑坡列表（{{ landslides.length }}）</div>
          <ul class="landslide-list">
            <li
              v-for="item in landslides"
              :key="item.id"
              :class="{ active: item.id === selectedId }"
              @click="onSelect(item)"
            >
              <div class="ls-name">{{ item.name }}</div>
              <div class="ls-meta">
                <el-tag size="small">{{ item.type }}</el-tag>
                <span class="ls-id">{{ item.id }}</span>
              </div>
            </li>
          </ul>
        </div>
      </aside>

      <aside class="dash-side dash-side-right">
        <PanelDetail
          v-if="selected"
          :selected="selected"
          :subgraph="subgraph"
          @request-subgraph="loadSubgraph"
        />
        <div v-else class="panel side-panel empty-tip">点击地图标记，查看详情</div>
      </aside>
    </main>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import api from '@/api'
import MarsMap from '@/components/MarsMap.vue'
import ChartBar from '@/components/ChartBar.vue'
import ChartPie from '@/components/ChartPie.vue'
import PanelDetail from '@/components/PanelDetail.vue'

const title = '火星滑坡分类知识图谱大屏'

const currentTab = ref('overview')
const landslides = ref([])
const stats = ref({ total: 0, by_type: [], by_mission: [] })
const selected = ref(null)
const selectedId = ref(null)
const subgraph = ref(null)

async function loadLandslides() {
  const res = await api.listLandslides()
  landslides.value = res.results
}

async function loadStats() {
  const res = await api.getStats()
  stats.value = res
}

async function onSelect(item) {
  selected.value = item
  selectedId.value = item.id
  await loadSubgraph()
}

async function loadSubgraph() {
  if (!selectedId.value) return
  const res = await api.getSubgraph(selectedId.value)
  subgraph.value = res
}

onMounted(() => {
  loadLandslides()
  loadStats()
})
</script>

<style scoped>
.dashboard {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.dash-header {
  height: 60px;
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  background: linear-gradient(
    90deg,
    rgba(226, 87, 76, 0.9),
    rgba(20, 26, 38, 0.9)
  );
  border-bottom: 1px solid var(--mars-border);
}
.dash-header h1 {
  font-size: 22px;
  color: #fff;
  letter-spacing: 2px;
}
.dash-nav {
  display: flex;
  gap: 8px;
}
.dash-body {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 300px 320px;
  gap: 12px;
  padding: 12px;
  min-height: 0;
}
.dash-map {
  min-height: 0;
  min-width: 0;
  overflow: hidden;
}
.dash-side {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  overflow-y: auto;
}
.dash-side-right {
  overflow-y: auto;
}
.stat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  padding: 12px;
}
.stat-item {
  background: rgba(79, 176, 224, 0.08);
  border: 1px solid var(--mars-border);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.stat-num {
  font-size: 28px;
  font-weight: 700;
  color: var(--mars-blue);
}
.stat-label {
  color: var(--mars-dim);
  font-size: 13px;
  margin-top: 4px;
}
.side-panel {
  flex: 1;
  min-height: 180px;
}
.list-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.landslide-list {
  list-style: none;
  overflow-y: auto;
  flex: 1;
  padding: 4px 8px 8px;
}
.landslide-list li {
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid transparent;
  margin-bottom: 6px;
}
.landslide-list li:hover,
.landslide-list li.active {
  background: rgba(79, 176, 224, 0.12);
  border-color: var(--mars-blue);
}
.ls-name {
  font-size: 14px;
}
.ls-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 6px;
}
.ls-id {
  color: var(--mars-dim);
  font-size: 12px;
}
.empty-tip {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--mars-dim);
}

@media (max-width: 1024px) {
  .dash-body {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr 200px 240px;
  }
}
</style>