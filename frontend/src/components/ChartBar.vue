<template>
  <div ref="el" class="chart-box" :style="{ height }"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  data: { type: Array, default: () => [] },
  height: { type: String, default: '220px' },
})

const el = ref(null)
let chart = null

function render() {
  if (!chart) return
  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {},
    xAxis: {
      type: 'category',
      data: props.data.map((d) => d.type || d.name),
      axisLabel: { color: '#dce3f0' },
      axisLine: { lineStyle: { color: '#2f3b52' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#8b95a8' },
      splitLine: { lineStyle: { color: '#1b2232' } },
    },
    series: [
      {
        type: 'bar',
        data: props.data.map((d) => d.count),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#f0b04d' },
            { offset: 1, color: '#e2574c' },
          ]),
        },
        barWidth: '55%',
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