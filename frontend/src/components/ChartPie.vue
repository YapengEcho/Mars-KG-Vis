<template>
  <div ref="el" class="chart-box" :style="{ height }"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  data: { type: Array, default: () => [] },
  height: { type: String, default: '240px' },
})

const el = ref(null)
let chart = null

const PIE_COLORS = ['#e2574c', '#f0b04d', '#4fb0e0', '#7ecb5f', '#b58ce0']

function render() {
  if (!chart) return
  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item' },
    legend: {
      bottom: 0,
      textStyle: { color: '#dce3f0' },
    },
    series: [
      {
        type: 'pie',
        radius: ['45%', '72%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: true,
        itemStyle: { borderRadius: 6, borderColor: '#0b0e14', borderWidth: 2 },
        label: { color: '#dce3f0', formatter: '{b}\n{c}' },
        data: props.data.map((d, i) => ({
          name: d.type || d.name,
          value: d.count,
          itemStyle: { color: PIE_COLORS[i % PIE_COLORS.length] },
        })),
      },
    ],
  })
}

watch(
  () => props.data,
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
.chart-box {
  width: 100%;
}
</style>