import source from '../../../examples/catalog.real-subset.json'
import type { Catalog } from '../types'

/** Image alignment is separate from a formally documented source CRS. */
export function coordinateStatus(catalog: Catalog): { label: string; note: string } {
  if (catalog.mode !== 'reference') return { label: '火星经纬度', note: '' }

  // The 2026-11-10 CTX check applies to these published coordinate values only.
  // API/custom catalogs must not inherit the claim merely by using reference mode.
  const aligned =
    catalog.records.length > 0 &&
    catalog.records.every((record) =>
      source.records.some(
        (checked) =>
          record.id === checked.id &&
          record.sourceId === checked.sourceId &&
          record.longitude === checked.longitude &&
          record.latitude === checked.latitude,
      ),
    )

  return aligned
    ? { label: '已与 CTX 影像对齐', note: source.coordinateStatus.notice }
    : {
        label: '原始编目坐标',
        note: '保留原始经纬度，原始坐标说明待补；尚未完成影像对齐核验。',
      }
}
