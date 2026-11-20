<script setup lang="ts">
import { computed } from 'vue'
import { renderChatMarkdown } from '../lib/chatMarkdown'

const props = defineProps<{ content: string }>()
const html = computed(() => renderChatMarkdown(props.content))
</script>

<template>
  <div class="assistant-markdown" v-html="html"></div>
</template>

<style scoped>
.assistant-markdown {
  font-size: 13px;
  line-height: 1.65;
  overflow-wrap: anywhere;
}
.assistant-markdown :deep(p),
.assistant-markdown :deep(ol),
.assistant-markdown :deep(ul),
.assistant-markdown :deep(blockquote),
.assistant-markdown :deep(pre),
.assistant-markdown :deep(table) {
  margin: 0 0 0.75em;
}
.assistant-markdown :deep(h1),
.assistant-markdown :deep(h2),
.assistant-markdown :deep(h3),
.assistant-markdown :deep(h4),
.assistant-markdown :deep(h5),
.assistant-markdown :deep(h6) {
  margin: 0.85em 0 0.45em;
  font-size: 1.08em;
  font-weight: 650;
  line-height: 1.45;
}
.assistant-markdown :deep(ol),
.assistant-markdown :deep(ul) {
  padding-inline-start: 1.45em;
}
.assistant-markdown :deep(li + li) {
  margin-top: 0.2em;
}
.assistant-markdown :deep(li > ol),
.assistant-markdown :deep(li > ul) {
  margin: 0.2em 0 0.45em;
}
.assistant-markdown :deep(blockquote) {
  border-inline-start: 2px solid var(--line);
  padding-inline-start: 0.75em;
  color: var(--muted);
}
.assistant-markdown :deep(a) {
  color: var(--accent);
  text-decoration: underline;
  text-underline-offset: 0.16em;
}
.assistant-markdown :deep(code) {
  border-radius: 4px;
  padding: 0.08em 0.24em;
  background: var(--bg);
  font-size: 0.94em;
}
.assistant-markdown :deep(pre) {
  overflow-x: auto;
  border-radius: 6px;
  padding: 0.65em 0.75em;
  background: var(--bg);
}
.assistant-markdown :deep(pre code) {
  padding: 0;
}
.assistant-markdown :deep(table) {
  display: block;
  overflow-x: auto;
  border-collapse: collapse;
}
.assistant-markdown :deep(th),
.assistant-markdown :deep(td) {
  border: 1px solid var(--line);
  padding: 0.3em 0.55em;
}
.assistant-markdown :deep(> :first-child) {
  margin-top: 0;
}
.assistant-markdown :deep(> :last-child),
.assistant-markdown :deep(li > :last-child),
.assistant-markdown :deep(blockquote > :last-child) {
  margin-bottom: 0;
}
</style>
