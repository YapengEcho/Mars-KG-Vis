import source from '../../../examples/catalog.real-subset.json'
import { samples } from './samples'
import { parseCatalog } from './catalog'

// Original numbers remain unchanged. Reference mode is distinct from a verified catalog.
export const publicCatalog = parseCatalog({
  version: 'public-reference-20261110',
  mode: 'reference',
  coordinateSystem: {
    body: source.coordinateStatus.body,
    longitude: source.coordinateStatus.longitude,
    latitude: source.coordinateStatus.latitude,
  },
  records: source.records.map((record) => ({
    ...record,
    imageUrl: samples.find((sample) => sample.id === record.id)?.imageUrl ?? null,
  })),
})
