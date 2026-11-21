<script setup lang="ts">
import { computed, ref } from 'vue'
import { Settings2, X } from '@lucide/vue'
import { modelPresets } from '../services/externalChat'
import type { ExternalProvider } from '../services/externalChat'
import { chat } from '../stores/chat'

const dialog = ref<HTMLDialogElement>()
const provider = ref<'builtin' | ExternalProvider>('builtin')
const endpoint = ref('')
const model = ref('')
const apiKey = ref('')
const error = ref('')
const canKeepKey = computed(
  () =>
    chat.state.model?.provider === provider.value &&
    chat.state.model.endpoint === endpoint.value.trim(),
)

function open() {
  error.value = ''
  apiKey.value = ''
  provider.value = chat.state.model?.provider ?? 'builtin'
  endpoint.value = chat.state.model?.endpoint ?? ''
  model.value = chat.state.model?.model ?? ''
  dialog.value?.showModal()
}

function chooseProvider() {
  error.value = ''
  apiKey.value = ''
  const selected = provider.value === 'builtin' ? undefined : modelPresets[provider.value]
  endpoint.value = selected?.endpoint ?? ''
  model.value = selected?.model ?? ''
}

function save() {
  try {
    if (provider.value === 'builtin') chat.setModel(null)
    else
      chat.setModel({
        provider: provider.value,
        endpoint: endpoint.value,
        model: model.value,
        apiKey: apiKey.value,
      })
    dialog.value?.close()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '无法应用此模型配置。'
  }
}
</script>

<template>
  <button class="model-settings-button" type="button" :disabled="chat.state.busy" @click="open">
    <Settings2 :size="16" aria-hidden="true" />模型设置
  </button>
  <dialog
    ref="dialog"
    class="model-settings"
    aria-labelledby="assistant-model-settings-title"
    @close="apiKey = ''"
  >
    <div class="model-settings-header">
      <h2 id="assistant-model-settings-title">模型设置</h2>
      <button class="model-close" type="button" aria-label="关闭模型设置" @click="dialog?.close()">
        <X :size="19" aria-hidden="true" />
      </button>
    </div>
    <form class="model-form" autocomplete="off" @submit.prevent="save">
      <label>
        <span>模型服务</span>
        <select v-model="provider" autofocus @change="chooseProvider">
          <option value="builtin">内置助手</option>
          <option v-for="(preset, id) in modelPresets" :key="id" :value="id">
            {{ preset.label }}
          </option>
        </select>
      </label>
      <template v-if="provider !== 'builtin'">
        <label>
          <span>模型 ID</span>
          <input v-model="model" type="text" required maxlength="160" spellcheck="false" />
        </label>
        <label>
          <span>接口地址</span>
          <input
            v-model="endpoint"
            type="url"
            required
            :readonly="provider !== 'custom'"
            placeholder="https://example.com/v1"
            spellcheck="false"
            @input="apiKey = ''"
          />
        </label>
        <p v-if="provider === 'custom'" class="model-hint">
          使用支持 OpenAI Chat Completions 和网页跨域访问的 HTTPS 接口。
        </p>
        <label>
          <span>API Key</span>
          <input
            v-model="apiKey"
            type="password"
            autocomplete="off"
            :placeholder="canKeepKey ? '留空沿用当前密钥' : '粘贴此接口的密钥'"
            :required="!canKeepKey"
          />
        </label>
        <p class="model-hint">
          密钥刷新后清除，只发往上方接口。应用用当前问题查询目录或图鉴，有需要时再将对话交给所选模型。
        </p>
      </template>
      <p v-if="error" class="model-error" role="alert">{{ error }}</p>
      <p v-if="provider !== 'builtin' || chat.state.model" class="model-hint">
        更改配置会开启新对话。
      </p>
      <div class="model-form-actions">
        <button class="secondary-button" type="button" @click="dialog?.close()">取消</button>
        <button class="primary-button" type="submit">应用</button>
      </div>
    </form>
  </dialog>
</template>

<style scoped>
.model-settings-button {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 44px;
  padding-inline: var(--space-xs);
  border: 1px solid var(--line);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
.model-settings-button:enabled:hover {
  border-color: var(--accent);
  color: var(--accent-hover);
}
.model-settings {
  width: min(478px, calc(100vw - 32px));
  max-height: calc(100dvh - 32px);
  padding: var(--space-lg);
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
  color: var(--text);
}
.model-settings::backdrop {
  background: var(--color-backdrop);
}
.model-settings-header,
.model-form-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-sm);
}
.model-settings-header h2 {
  font-size: var(--text-lg);
  font-weight: 600;
}
.model-close {
  display: grid;
  min-width: 44px;
  min-height: 44px;
  place-items: center;
}
.model-form,
.model-form label {
  display: grid;
  gap: var(--space-xs);
}
.model-form {
  margin-block-start: var(--space-sm);
  gap: var(--space-sm);
}
.model-form label {
  font-size: var(--text-sm);
}
.model-form input,
.model-form select {
  width: 100%;
  min-width: 0;
  min-height: 44px;
  padding: 0 var(--space-sm);
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text);
  font: inherit;
}
.model-form input[readonly] {
  color: var(--muted);
}
.model-hint,
.model-error {
  font-size: var(--text-xs);
  line-height: 1.6;
}
.model-hint {
  color: var(--muted);
}
.model-error {
  color: var(--color-error);
}
.model-form-actions {
  justify-content: flex-end;
  margin-block-start: var(--space-xs);
}
</style>
