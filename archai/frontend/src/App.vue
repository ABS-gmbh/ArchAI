<template>
  <div id="app">
    <header>
      <h1>ArchAI - Manuscript OCR</h1>
    </header>
    <main>
      <div class="container">
        <h2>Document Processing Pipeline</h2>
        <p>Upload a manuscript image to begin processing...</p>
        <button @click="checkBackend">Backend Status: {{ backendStatus }}</button>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const backendStatus = ref('Checking...')

const checkBackend = async () => {
  try {
    const response = await fetch('/api/health')
    if (response.ok) {
      backendStatus.value = '✓ Online'
    } else {
      backendStatus.value = '✗ Offline'
    }
  } catch (error) {
    backendStatus.value = '✗ Error'
  }
}

onMounted(() => {
  checkBackend()
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

#app {
  min-height: 100vh;
}

header {
  background: rgba(0, 0, 0, 0.3);
  color: white;
  padding: 2rem;
  text-align: center;
}

header h1 {
  font-size: 2.5rem;
  font-weight: 700;
}

main {
  padding: 2rem;
  display: flex;
  justify-content: center;
}

.container {
  background: white;
  border-radius: 12px;
  padding: 3rem;
  max-width: 600px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

h2 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.8rem;
}

p {
  color: #666;
  margin-bottom: 2rem;
  line-height: 1.6;
}

button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
}

button:active {
  transform: translateY(0);
}
</style>
