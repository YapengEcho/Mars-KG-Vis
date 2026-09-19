<template>
  <div ref="mapEl" class="mars-map"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const props = defineProps({
  landslides: { type: Array, default: () => [] },
})
const emit = defineEmits(['select'])

const mapEl = ref(null)
let map = null
let layer = null

// ---- 火星专用简单圆柱投影 CRS ----
// 火星全球图是等距圆柱投影（equirectangular），经度铺满 [-180,180]，纬度 [-90,90]，
// 与地球 Web Mercator 不同。这里用平等的经纬度线性映射，保证经度跨度 = 2x 纬度跨度。
const TileSize = 256
const MarsCRS = L.extend({}, L.CRS.Earth, {
  code: 'EPSG:4326-mars',
  projection: {
    project: (latlng) => new L.Point(latlng.lng, -latlng.lat),
    unproject: (point) => new L.LatLng(-point.y, point.x),
    bounds: L.bounds([-180, -90], [180, 90]),
  },
  // 与等距圆柱匹配的瓦片变换：x=(lon+180)/360, y=(90-lat)/180
  transformation: new L.Transformation(1 / 360, 0.5, 1 / 180, 0.5),
  scale: (zoom) => Math.pow(2, zoom) * TileSize,
  // 平铺方向让世界显示为约 2:1 的矩形
  wrapLng: [-180, 180],
})

const USGS_URL =
  'https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map'
const WMS_LAYER = 'MOLA_color' // 彩色火星地貌；可换 MOLA_THEMIS_blend / MOLA_bw

function initMap() {
  map = L.map(mapEl.value, {
    crs: MarsCRS,
    center: [0, 0],
    zoom: 0,
    minZoom: 0,
    maxZoom: 5,
    zoomControl: true,
  })

  // 火星 WMS 底图（等距圆柱投影）
  L.tileLayer.wms(USGS_URL, {
    layers: WMS_LAYER,
    crs: L.CRS.EPSG4326,
    format: 'image/png',
    transparent: false,
    maxZoom: 5,
    tileSize: TileSize,
    attribution:
      '© <a href="https://astrogeology.usgs.gov/maps/mars-mars-cyc-cyl">USGS Astrogeology / NASA</a> · Mars-KG',
  }).addTo(map)
}

function renderPoints() {
  if (layer) {
    map.removeLayer(layer)
  }
  layer = L.layerGroup().addTo(map)

  props.landslides.forEach((item) => {
    if (item.lat == null || item.lon == null) return
    const marker = L.circleMarker(
      [item.lat, item.lon],
      {
        radius: 6,
        color: '#e2574c',
        weight: 1,
        fillColor: '#f0b04d',
        fillOpacity: 0.85,
      }
    )
      .bindPopup(
        `<b>${item.name || item.id}</b><br/>${item.type} · ${item.mission} · ${item.length_m}m`
      )
      .on('click', () => emit('select', item))
    layer.addLayer(marker)
  })

  // 缩放适配到点集
  if (props.landslides.length) {
    const bounds = L.latLngBounds(
      props.landslides.map((d) => [d.lat, d.lon])
    )
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 5 })
  }
}

watch(
  () => props.landslides,
  (v) => v && renderPoints(),
  { deep: false }
)

onMounted(() => {
  initMap()
  renderPoints()
})

onBeforeUnmount(() => {
  if (map) map.remove()
})
</script>

<style scoped>
.mars-map {
  width: 100%;
  height: 100%;
  min-height: 400px;
  background: var(--mars-bg);
}
:deep(.leaflet-container) {
  background: #0b0e14;
  width: 100%;
  height: 100%;
}
</style>