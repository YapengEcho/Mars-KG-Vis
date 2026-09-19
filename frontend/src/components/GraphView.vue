<template>
  <div ref="el" class="graph-box" :style="{ height }"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  nodes: { type: Array, default: () => [] },
  edges: { type: Array, default: () => [] },
  degraded: { type: Boolean, default: false },
  height: { type: String, default: '260px' },
})

const el = ref(null)
let chart = null

const KIND_COLOR = {
  Landslide: '#e2574c',
  LandslideType: '#f0b04d',
  Region: '#4fb0e0',
  Feature: '#7ecb5f',
  Paper: '#b58ce0',
}

function render() {
  if (!chart) return
  const nodes = props.nodes.map((n) => ({
    id: n.id,
    name: n.label || n.id,
    category: n.kind,
    symbolSize: n.kind === 'Landslide' ? 34 : 26,
    itemStyle: { color: KIND_COLOR[n.kind] || '#8b95a8' },
    label: { color: '#dce3f0', fontSize: 11 },
  }))
  const categories = [...new Set(props.nodes.map((n) => n.kind))].map((k) => ({
    name: k,
  }))

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item' },
    legend: [
      {
        bottom: 0,
        data: categories.map((c) => c.name),
        textStyle: { color: '#dce3f0' },
      },
    ],
    series: [
      {
        type: 'graph',
        layout: 'force',
        data: nodes,
        links: props.edges.map((e) => ({
          source: e.source,
          target: e.target,
          label: { show: true, formatter: e.rel, color: '#8b95a8', fontSize: 9 },
        })),
        categories,
        roam: true,
        draggable: true,
        force: { repulsion: 180, edgeLength: 90 },
        lineStyle: { color: '#2f3b52', width: 1.5 },
        emphasis: { focus: 'adjacency' },
      },
    ],
  })
}

watch(
  () => [props.nodes, props.edges],
  () => render(),
  { deep: true }
)

onMounted(() => {
  chart = echarts.init(el.value)
  render()
})
onBeforeUnmount(() => chart && chart.dispose())
</script>

<style scoped>
.graph-box {
  width: 100%;
}
</style>