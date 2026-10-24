# 交接包校验报告

由 `scripts/handoff_check.py` 生成。`fail` 表示阻断，`warn` 表示需人工确认。

| 检查项 | 结果 | 说明 |
|---|---|---|
| `catalog.records` | pass | 2971 records (expected 2971) |
| `catalog.ids.unique` | pass | 2971 unique ids out of 2971 |
| `catalog.source_id.continuous` | pass | min=1 max=2971 |
| `catalog.type.enum` | pass | {'SP': 321, 'SL': 1787, 'FL': 50, 'CL': 813} |
| `catalog.type.counts_vs_manifest` | pass | catalog={'SP': 321, 'SL': 1787, 'FL': 50, 'CL': 813} manifest={'SP': 321, 'SL': 1787, 'FL': 50, 'CL': 813} |
| `catalog.coordinates.range` | pass | all inside lon[-180,180] lat[-90,90] |
| `catalog.optional_fields` | warn | area null=0, name empty=2971, desc empty=2971 |
| `catalog.type.mapping` | pass | aliases {'RL': 'SL', 'DF': 'FL'} folded into display types; raw counts {'SP': 321, 'SL': 1785, 'FL': 49, 'CL': 813, 'RL': 2, 'DF': 1} |
| `index.records` | pass | 2971 entries for 2971 catalog records |
| `index.identity_matches_catalog` | pass | source_id, catalog_id and display_type agree |
| `index.record_indices` | pass | point and boundary indices cover 0..n-1 without gaps |
| `images.files_present` | pass | 1083 files present |
| `images.size` | pass | all 224x224 |
| `images.type_matches_catalog` | pass | image folder type equals catalog type |
| `images.coverage` | warn | 1083 of 2971 catalog records have a crop (36.5%) |
| `images.sha256` | pass | all checksums match |
| `sample.records` | pass | 20 records (expected 20) |
| `sample.matches_catalog` | pass | all 20 trace back to catalog rows and indexed crops |
| `sample.type_balance` | pass | {'SL': 5, 'SP': 5, 'FL': 5, 'CL': 5} |
| `sample.contract_fields` | pass | all contract fields present |
| `sample.coordinate_status` | warn | coordinate status marked: ['latitude_convention_pending_verification'] |
| `cases.records` | pass | 12 candidates (expected 12) |
| `cases.type_balance` | pass | {'SL': 3, 'SP': 3, 'FL': 3, 'CL': 3} |
| `cases.traceable` | pass | all candidates trace to catalog, crops and gap lists |
| `shapes.counts` | pass | points=2971 boundaries=2971 catalog=2971 |
| `shapes.boundary_fields` | pass | fields=['序号', '类型', 'area', 'Perimeter', '经度', '纬度'] |
| `shapes.case_geometry` | pass | all 12 candidates have point and boundary records |
| `shapes.point_vs_catalog` | pass | point geometry equals catalog lon/lat |
| `shapes.dms_vs_geometry` | warn | 93 records lose the sign in DMS text when the degree part is 0 (e.g. [686, 687, 688, 689, 690]); geometry and catalog keep the correct sign |
| `shapes.boundary_area` | pass | all boundary areas positive |
| `coordinate.latitude_range` | pass | lat min=-45.7975 max=77.9780; equator records=0 |
| `coordinate.convention` | warn | planetocentric is NOT verified: planetographic alternative would move points by up to 0.34° at these latitudes (median 0.10°). PRJ declares Earth WGS84 and must not be used as evidence for Mars. |

汇总：{'pass': 27, 'warn': 5}
