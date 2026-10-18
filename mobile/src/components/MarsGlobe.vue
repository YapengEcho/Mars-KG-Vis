<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import {
  formatMarsCoordinate,
  isMarsCoordinate,
  lonLatToCartesian,
  normalizeLongitude,
} from '../lib/marsCoordinates'
import { typeInfo, type SlideType } from '../types'

interface GlobePoint {
  id: string
  longitude: number
  latitude: number
  type: string
  name: string
}

const props = defineProps<{ points: GlobePoint[]; selectedId?: string }>()
const emit = defineEmits<{ select: [id: string] }>()
const host = ref<HTMLDivElement | null>(null)
const failed = ref(false)
const ready = ref(false)
const usablePoints = computed(() => props.points.filter(isMarsCoordinate))
const selected = computed(() => usablePoints.value.find((point) => point.id === props.selectedId))
const radius = 2

function colorForType(type: string): string {
  return type in typeInfo ? typeInfo[type as SlideType].color : '#e5bd8e'
}

let renderer: THREE.WebGLRenderer | undefined
let scene: THREE.Scene | undefined
let camera: THREE.PerspectiveCamera | undefined
let controls: OrbitControls | undefined
let markers: THREE.InstancedMesh | undefined
let selectedRing: THREE.Mesh | undefined
let selectionDot: THREE.Mesh | undefined
let resizeObserver: ResizeObserver | undefined
let intersectionObserver: IntersectionObserver | undefined
let motionPreference: MediaQueryList | undefined
let frame = 0
let inView = true
let reducedMotion = false
let pointerStart: { x: number; y: number } | undefined
let focusAnimation: { start: THREE.Vector3; end: THREE.Vector3; began: number } | undefined
const raycaster = new THREE.Raycaster()
const pointer = new THREE.Vector2()
const trackedTextures: THREE.Texture[] = []

// An original procedural illustration, without geographic imagery or elevation claims.
function makeIllustrativeTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 768
  canvas.height = 384
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas is unavailable')
  let seed = 0x47a329
  const random = (): number => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    return seed / 4294967296
  }
  const field = new Float32Array(32 * 32 * 32)
  for (let index = 0; index < field.length; index++) field[index] = random()
  const lattice = (x: number, y: number, z: number): number =>
    field[(x & 31) + ((y & 31) << 5) + ((z & 31) << 10)]!
  const blend = (start: number, end: number, amount: number): number =>
    start + (end - start) * amount
  const fade = (value: number): number => value * value * (3 - 2 * value)
  const noise = (x: number, y: number, z: number): number => {
    const ix = Math.floor(x),
      iy = Math.floor(y),
      iz = Math.floor(z)
    const fx = fade(x - ix),
      fy = fade(y - iy),
      fz = fade(z - iz)
    return blend(
      blend(
        blend(lattice(ix, iy, iz), lattice(ix + 1, iy, iz), fx),
        blend(lattice(ix, iy + 1, iz), lattice(ix + 1, iy + 1, iz), fx),
        fy,
      ),
      blend(
        blend(lattice(ix, iy, iz + 1), lattice(ix + 1, iy, iz + 1), fx),
        blend(lattice(ix, iy + 1, iz + 1), lattice(ix + 1, iy + 1, iz + 1), fx),
        fy,
      ),
      fz,
    )
  }
  const pixels = context.createImageData(canvas.width, canvas.height)
  for (let y = 0; y < canvas.height; y++) {
    const latitude = (0.5 - y / canvas.height) * Math.PI
    const cosLat = Math.cos(latitude)
    const ny = Math.sin(latitude)
    for (let x = 0; x < canvas.width; x++) {
      // SphereGeometry maps its u=0.25 line onto longitude zero.
      const longitude = (x / canvas.width) * Math.PI * 2 - Math.PI / 2
      const nx = Math.sin(longitude) * cosLat
      const nz = Math.cos(longitude) * cosLat
      // Sample noise in three dimensions so the longitude seam remains continuous.
      const continental = noise(nx * 2.7 + 13.2, ny * 2.7 + 7.8, nz * 2.7 + 17.3)
      const regional = noise(nx * 6.8 + continental * 1.8 + 5.1, ny * 6.8 + 11.7, nz * 6.8 + 2.4)
      const detail = noise(nx * 19.3 + 7.2, ny * 19.3 + 21.1, nz * 19.3 + 9.5)
      const grain = noise(nx * 71.7 + 4.3, ny * 71.7 + 2.7, nz * 71.7 + 1.6)
      const region = continental * 0.73 + regional * 0.27
      const dark = fade(Math.max(0, Math.min(1, (region - 0.43) / 0.29))) * 0.82
      const shade = (regional - 0.5) * 19 + (detail - 0.5) * 14 + (grain - 0.5) * 5
      const polar = Math.max(0, (Math.abs(ny) - 0.988 - (regional - 0.5) * 0.006) * 65)
      const cap = Math.min(0.68, polar)
      const index = (y * canvas.width + x) * 4
      pixels.data[index] = (blend(186, 104, dark) + shade) * (1 - cap) + 221 * cap
      pixels.data[index + 1] = (blend(115, 74, dark) + shade * 0.82) * (1 - cap) + 207 * cap
      pixels.data[index + 2] = (blend(78, 56, dark) + shade * 0.66) * (1 - cap) + 185 * cap
      pixels.data[index + 3] = 255
    }
  }
  context.putImageData(pixels, 0, 0)
  // A few shallow, illustrative crater rims; these are not mapped features.
  const craters = [
    { lon: -29, lat: 24, size: 6.5 },
    { lon: 35, lat: -13, size: 8.2 },
    { lon: 74, lat: 37, size: 4.7 },
    { lon: -56, lat: -33, size: 4.2 },
  ]
  for (let index = 0; index < 20; index++)
    craters.push({
      lon: random() * 360 - 180,
      lat: random() * 124 - 62,
      size: 1.3 + random() * 3.8,
    })
  for (const crater of craters) {
    const latitude = (crater.lat * Math.PI) / 180
    const centerX = (((crater.lon + 90 + 360) % 360) / 360) * canvas.width
    const centerY = ((90 - crater.lat) / 180) * canvas.height
    const verticalRadius = (crater.size / 180) * canvas.height
    const horizontalRadius = verticalRadius / Math.cos(latitude)
    for (const offset of [-canvas.width, 0, canvas.width]) {
      context.save()
      context.translate(centerX + offset, centerY)
      context.scale(horizontalRadius, verticalRadius)
      const basin = context.createRadialGradient(-0.1, -0.12, 0.03, 0, 0, 1)
      basin.addColorStop(0, 'rgba(58, 35, 23, 0.07)')
      basin.addColorStop(0.63, 'rgba(58, 35, 23, 0.12)')
      basin.addColorStop(0.79, 'rgba(47, 29, 20, 0.21)')
      basin.addColorStop(0.89, 'rgba(232, 177, 123, 0.18)')
      basin.addColorStop(1, 'rgba(186, 115, 78, 0)')
      context.fillStyle = basin
      context.beginPath()
      context.arc(0, 0, 1, 0, Math.PI * 2)
      context.fill()
      context.restore()
    }
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.wrapS = THREE.RepeatWrapping
  trackedTextures.push(texture)
  return texture
}

function addGlobe(): void {
  if (!scene) return
  const surfaceTexture = makeIllustrativeTexture()
  const surface = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 80, 48),
    new THREE.MeshStandardMaterial({
      map: surfaceTexture,
      bumpMap: surfaceTexture,
      bumpScale: 0.024,
      roughness: 1,
      metalness: 0,
    }),
  )
  scene.add(surface)
  const atmosphere = new THREE.Mesh(
    new THREE.SphereGeometry(radius * 1.028, 64, 40),
    new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide,
      uniforms: { glowColor: { value: new THREE.Color('#d39268') } },
      vertexShader: `varying vec3 vNormal; varying vec3 vView;
        void main() {
          vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
          vNormal = normalize(normalMatrix * normal);
          vView = normalize(-viewPosition.xyz);
          gl_Position = projectionMatrix * viewPosition;
        }`,
      fragmentShader: `uniform vec3 glowColor; varying vec3 vNormal; varying vec3 vView;
        void main() {
          float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 3.4);
          gl_FragColor = vec4(glowColor, rim * 0.13);
        }`,
    }),
  )
  scene.add(atmosphere)
  const keyLight = new THREE.DirectionalLight('#fff0df', 2.7)
  keyLight.position.set(-5, 3, 4)
  scene.add(keyLight)
  scene.add(new THREE.AmbientLight('#ddd0c4', 0.28))
  const fill = new THREE.DirectionalLight('#94a3ad', 0.1)
  fill.position.set(4, -1, -2)
  scene.add(fill)
  selectedRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.066, 0.007, 8, 40),
    new THREE.MeshBasicMaterial({ color: '#fff0ca' }),
  )
  selectionDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.028, 12, 8),
    new THREE.MeshBasicMaterial({ color: '#fff3cf' }),
  )
  selectedRing.visible = selectionDot.visible = false
  scene.add(selectedRing, selectionDot)
}

function disposeMesh(mesh: THREE.Mesh): void {
  mesh.geometry.dispose()
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
  materials.forEach((material) => material.dispose())
}

function rebuildMarkers(): void {
  if (!scene) return
  if (markers) {
    scene.remove(markers)
    disposeMesh(markers)
    markers.dispose()
    markers = undefined
  }
  const points = usablePoints.value
  if (points.length) {
    markers = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.018, 8, 6),
      new THREE.MeshBasicMaterial({ color: '#ffffff' }),
      points.length,
    )
    const matrix = new THREE.Matrix4()
    points.forEach((point, index) => {
      matrix.makeTranslation(...lonLatToCartesian(point.longitude, point.latitude, radius + 0.018))
      markers!.setMatrixAt(index, matrix)
      markers!.setColorAt(index, new THREE.Color(colorForType(point.type)))
    })
    markers.instanceMatrix.needsUpdate = true
    if (markers.instanceColor) markers.instanceColor.needsUpdate = true
    markers.computeBoundingSphere()
    scene.add(markers)
  }
  updateSelection(false)
}

function focusPoint(id: string): void {
  const point = usablePoints.value.find((item) => item.id === id)
  if (!point || !camera) return
  const direction = new THREE.Vector3(...lonLatToCartesian(point.longitude, point.latitude))
  const end = direction.multiplyScalar(camera.position.length())
  // Exact poles make a Y-up camera singular; keep a small stable offset.
  if (Math.abs(point.latitude) > 89.5) end.z = 0.035
  if (reducedMotion) {
    camera.position.copy(end)
    controls?.update()
    focusAnimation = undefined
  } else {
    focusAnimation = { start: camera.position.clone(), end, began: performance.now() }
  }
}

function updateSelection(shouldFocus = true): void {
  if (!selectedRing || !selectionDot) return
  const point = selected.value
  selectedRing.visible = selectionDot.visible = Boolean(point)
  if (!point) return
  const position = new THREE.Vector3(
    ...lonLatToCartesian(point.longitude, point.latitude, radius + 0.035),
  )
  selectedRing.position.copy(position)
  selectedRing.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    position.clone().normalize(),
  )
  selectionDot.position.copy(position)
  if (shouldFocus) focusPoint(point.id)
}

function zoom(factor: number): void {
  if (!camera || !controls) return
  focusAnimation = undefined
  camera.position.setLength(
    THREE.MathUtils.clamp(
      camera.position.length() * factor,
      controls.minDistance,
      controls.maxDistance,
    ),
  )
  controls.update()
}

function zoomIn(): void {
  zoom(0.84)
}
function zoomOut(): void {
  zoom(1.19)
}
function resetView(): void {
  if (!camera) return
  focusAnimation = undefined
  camera.position.set(0, 0.65, 5.65)
  controls?.target.set(0, 0, 0)
  controls?.update()
}

defineExpose({ zoomIn, zoomOut, resetView, focusPoint })

function onKeydown(event: KeyboardEvent): void {
  if (!camera) return
  if (event.key === '+' || event.key === '=') zoomIn()
  else if (event.key === '-') zoomOut()
  else if (event.key === 'Home') resetView()
  else if (event.key.startsWith('Arrow')) {
    focusAnimation = undefined
    const spherical = new THREE.Spherical().setFromVector3(camera.position)
    if (event.key === 'ArrowLeft') spherical.theta -= 0.12
    else if (event.key === 'ArrowRight') spherical.theta += 0.12
    else if (event.key === 'ArrowUp') spherical.phi -= 0.12
    else if (event.key === 'ArrowDown') spherical.phi += 0.12
    spherical.makeSafe()
    camera.position.setFromSpherical(spherical)
    controls?.update()
  } else return
  event.preventDefault()
}

function onPointerDown(event: PointerEvent): void {
  pointerStart = { x: event.clientX, y: event.clientY }
  focusAnimation = undefined
}

function onPointerUp(event: PointerEvent): void {
  if (!pointerStart || !renderer || !camera || !markers) return
  const distance = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y)
  pointerStart = undefined
  if (distance > 7) return
  const rectangle = renderer.domElement.getBoundingClientRect()
  pointer.set(
    ((event.clientX - rectangle.left) / rectangle.width) * 2 - 1,
    (-(event.clientY - rectangle.top) / rectangle.height) * 2 + 1,
  )
  raycaster.setFromCamera(pointer, camera)
  const intersection = raycaster.intersectObject(markers)[0]
  if (!intersection || intersection.instanceId === undefined) return
  const point = usablePoints.value[intersection.instanceId]
  if (!point) return
  // Raycast the surface too: markers on the hidden hemisphere must not be clickable.
  const markerPosition = new THREE.Vector3(...lonLatToCartesian(point.longitude, point.latitude))
  if (markerPosition.dot(camera.position.clone().normalize()) < radius / camera.position.length())
    return
  emit('select', point.id)
}

function resize(): void {
  if (!host.value || !renderer || !camera) return
  const width = host.value.clientWidth
  const height = host.value.clientHeight
  if (!width || !height) return
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height, false)
}

function animate(now: number): void {
  frame = 0
  if (!renderer || !scene || !camera || !inView || document.hidden || failed.value) return
  if (focusAnimation) {
    const progress = Math.min(1, (now - focusAnimation.began) / 550)
    const eased = 1 - Math.pow(1 - progress, 3)
    const distance = focusAnimation.start.length()
    // Blend directions without letting the camera pass through the planet.
    camera.position.copy(focusAnimation.start).lerp(focusAnimation.end, eased)
    if (camera.position.lengthSq() < 0.01) camera.position.copy(focusAnimation.end)
    camera.position.setLength(distance)
    if (progress === 1) focusAnimation = undefined
  }
  controls?.update()
  renderer.render(scene, camera)
  ready.value = true
  frame = requestAnimationFrame(animate)
}

function resume(): void {
  if (!frame && !document.hidden && inView && !failed.value) frame = requestAnimationFrame(animate)
}

function onVisibilityChange(): void {
  if (document.hidden && frame) {
    cancelAnimationFrame(frame)
    frame = 0
  } else resume()
}

function updateMotionPreference(): void {
  reducedMotion = motionPreference?.matches ?? false
  if (controls) controls.enableDamping = !reducedMotion
  if (reducedMotion && focusAnimation && camera) {
    camera.position.copy(focusAnimation.end)
    focusAnimation = undefined
  }
}

function onContextLost(event: Event): void {
  event.preventDefault()
  failed.value = true
  if (frame) cancelAnimationFrame(frame)
  frame = 0
}

onMounted(() => {
  if (!host.value) return
  try {
    renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: 'low-power',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0x000000, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.domElement.setAttribute('aria-hidden', 'true')
    host.value.prepend(renderer.domElement)
    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(43, 1, 0.1, 50)
    camera.position.set(0, 0.65, 5.65)
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enablePan = false
    controls.enableDamping = true
    controls.dampingFactor = 0.09
    controls.rotateSpeed = 0.6
    controls.zoomSpeed = 0.8
    controls.minDistance = 3.2
    controls.maxDistance = 8.5
    controls.update()
    motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
    updateMotionPreference()
    motionPreference.addEventListener('change', updateMotionPreference)
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('webglcontextlost', onContextLost)
    addGlobe()
    rebuildMarkers()
    updateSelection()
    resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(host.value)
    if ('IntersectionObserver' in window) {
      intersectionObserver = new IntersectionObserver((entries) => {
        inView = entries[0]?.isIntersecting ?? true
        if (!inView && frame) {
          cancelAnimationFrame(frame)
          frame = 0
        } else resume()
      })
      intersectionObserver.observe(host.value)
    }
    document.addEventListener('visibilitychange', onVisibilityChange)
    resize()
    resume()
  } catch {
    failed.value = true
  }
})

watch(() => props.points, rebuildMarkers, { deep: true })
watch(
  () => props.selectedId,
  () => updateSelection(),
)

onBeforeUnmount(() => {
  cancelAnimationFrame(frame)
  resizeObserver?.disconnect()
  intersectionObserver?.disconnect()
  document.removeEventListener('visibilitychange', onVisibilityChange)
  motionPreference?.removeEventListener('change', updateMotionPreference)
  controls?.dispose()
  if (renderer) {
    renderer.domElement.removeEventListener('pointerdown', onPointerDown)
    renderer.domElement.removeEventListener('pointerup', onPointerUp)
    renderer.domElement.removeEventListener('webglcontextlost', onContextLost)
  }
  scene?.traverse((object) => {
    if (object instanceof THREE.Mesh) disposeMesh(object)
  })
  markers?.dispose()
  trackedTextures.forEach((texture) => texture.dispose())
  renderer?.dispose()
  renderer?.forceContextLoss()
  renderer?.domElement.remove()
})

function fallbackPosition(point: GlobePoint): Record<string, string> {
  return {
    left: `${((normalizeLongitude(point.longitude) + 180) / 360) * 100}%`,
    top: `${((90 - point.latitude) / 180) * 100}%`,
    backgroundColor: colorForType(point.type),
  }
}
</script>

<template>
  <div
    ref="host"
    class="mars-globe"
    :class="{ 'is-ready': ready, 'has-fallback': failed }"
    tabindex="0"
    role="group"
    aria-label="火星点位示意球。拖动旋转，双指或滚轮缩放，方向键调整视角。"
    @keydown="onKeydown"
  >
    <div v-if="failed" class="globe-fallback">
      <p>当前设备无法显示三维球体，已切换为坐标示意。</p>
      <div class="coordinate-map" aria-label="滑坡经纬度示意图">
        <span class="map-north">90°N</span>
        <span class="map-equator">0°</span>
        <span class="map-south">90°S</span>
        <button
          v-for="point in usablePoints"
          :key="point.id"
          type="button"
          class="map-point"
          :class="{ 'is-selected': point.id === selectedId }"
          :style="fallbackPosition(point)"
          :aria-label="`${point.name || point.id}，${formatMarsCoordinate(point)}`"
          :title="`${point.name || point.id} · ${formatMarsCoordinate(point)}`"
          @click="emit('select', point.id)"
        />
      </div>
      <div class="longitude-labels"><span>180°W</span><span>0°</span><span>180°E</span></div>
    </div>
    <div v-else-if="!ready" class="globe-loading" role="status">正在准备火星球…</div>
    <div v-if="selected" class="globe-coordinate" aria-live="polite">
      {{ formatMarsCoordinate(selected) }}
    </div>
  </div>
</template>

<style scoped>
.mars-globe {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 320px;
  overflow: hidden;
  isolation: isolate;
  outline: none;
}
.mars-globe:focus-visible {
  outline: 2px solid var(--accent, #86b6ff);
  outline-offset: -3px;
  border-radius: 16px;
}
.mars-globe :deep(canvas) {
  display: block;
  width: 100%;
  height: 100%;
  touch-action: none;
  cursor: grab;
}
.mars-globe :deep(canvas:active) {
  cursor: grabbing;
}
.globe-loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #b7c6db;
  font-size: 12px;
  pointer-events: none;
}
.globe-coordinate {
  position: absolute;
  z-index: 1;
  bottom: 50px;
  right: 18px;
  pointer-events: none;
  color: #c1ccdb;
  font-size: 10px;
  line-height: 1.5;
  font-variant-numeric: tabular-nums;
}
.globe-fallback {
  position: absolute;
  inset: 18% 10% 23%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.globe-fallback > p {
  margin: 0 0 18px;
  color: #9ba9bd;
  font-size: 11px;
  line-height: 1.6;
  text-align: center;
}
.coordinate-map {
  position: relative;
  width: 100%;
  aspect-ratio: 2;
  border: 1px solid #425672;
  border-radius: 5px;
  background-color: #182333;
  background-image:
    linear-gradient(#819cbe33 1px, transparent 1px),
    linear-gradient(90deg, #819cbe33 1px, transparent 1px);
  background-size: 16.666% 25%;
}
.map-point {
  position: absolute;
  width: 9px;
  height: 9px;
  padding: 0;
  border: 1px solid #101823;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  cursor: pointer;
}
.map-point:focus-visible,
.map-point.is-selected {
  outline: 2px solid #d7e7ff;
  outline-offset: 3px;
  z-index: 1;
}
.map-north,
.map-equator,
.map-south {
  position: absolute;
  left: -29px;
  color: #9ba9bd;
  font-size: 9px;
  transform: translateY(-50%);
}
.map-north {
  top: 0;
}
.map-equator {
  top: 50%;
}
.map-south {
  top: 100%;
}
.longitude-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  color: #9ba9bd;
  font-size: 9px;
}
@media (max-width: 380px) {
  .globe-coordinate {
    bottom: 52px;
    left: 18px;
    right: auto;
  }
}
</style>
