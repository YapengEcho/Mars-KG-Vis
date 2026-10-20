<script setup lang="ts">
import { MessageSquare, Info } from '@lucide/vue'
import PageHeading from '../components/PageHeading.vue'
</script>
<template>
  <section class="content-view">
    <PageHeading
      eyebrow="ASK THE ATLAS"
      title="火星滑坡助手"
      description="对话式查询目录、资料与识别结果。"
    />
    <div class="empty-state">
      <MessageSquare :size="40" :stroke-width="1" />
      <h2>助手正在接入</h2>
      <p>目录与资料问答、识别结果解释由周璇实现，当前为待接入的占位框架。</p>
    </div>
    <div class="info-card">
      <Info :size="18" />
      <p>助手将支持文字对话、资料检索、统计与地图定位，并给出带来源的回答。</p>
    </div>
  </section>
</template>