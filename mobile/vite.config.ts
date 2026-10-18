import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: { port: 5173, strictPort: true, proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: {
    target: 'es2022',
    rollupOptions: { output: { manualChunks: { globe: ['three'], ui: ['vue', 'vue-router'] } } },
  },
})
