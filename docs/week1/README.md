# 第 1 周交付（2026-10-19 — 10-25）

负责人：滕佳杰｜数据来源：`teng_jiajie_week1_20261024` 交接包（2971 条编目、2971 点、2971 面、1083 裁片）

## 交付清单

| 本周要求 | 交付物 | 状态 |
|---|---|---|
| 字段／坐标说明 v1 | [`contracts/data-dictionary.md`](../../contracts/data-dictionary.md) | 完成；纬度约定写“待核准”，附核准步骤 |
| 约 20 条真实记录样例 | [`records20.md`](records20.md)（JSON 随交付包单独提交） | 完成，20 条全部回溯到编目与裁片 |
| 12 例清单与缺项表 | [`12-cases.md`](12-cases.md)（JSON 随交付包单独提交） | 完成；12 条为候选，缺高清图/典型性确认 |
| 底图资源核对表 | [`basemap.md`](basemap.md) | 完成；首选 MDIM 2.1，已核实服务地址 |
| 四类图鉴资料与文案 | [`atlas-types.md`](atlas-types.md) | 草稿，待逐句核定 |
| 所用脚本与运行说明 | [`../../scripts/handoff_check.py`](../../scripts/handoff_check.py)、[`../../scripts/build_week1_artifacts.py`](../../scripts/build_week1_artifacts.py)、[`run.md`](run.md) | 完成，含 9 个单元测试 |
| 校验报告 | [`handoff-check.md`](handoff-check.md) | 27 项 PASS / 5 项 WARN / 0 项 FAIL |

## 一句话结论

**数据链路是通的**：2971 条目录、点位、边界、1083 张裁片、12 个案例全部按 `source_id` 对齐，图片逐张校验通过；**卡点只有三个**——纬度约定未核准、案例典型性未确认、高清展示图未申请。这三项分别需要老师（1、2）和崔亚鹏（3）。

## 本周发现的两个数据问题

1. **纬度约定无认证依据**：原始 `.prj` 写地球 WGS84（错误标注），论文未声明，只能证明面积是用火星球体算的；planetographic 与 planetocentric 在本数据纬度上最大差 0.34°（中位 0.10°），足以造成明显错位。核准前真实目录不上架。
2. **度分秒文本丢负号 93 条**：度为 0 时 `-0°25′30″` 写成 `0°25′30″`，数值字段正确，展示必须由数值重新格式化。

## 复现方式

```bash
python scripts/handoff_check.py --handoff <交接包路径> --md docs/week1/handoff-check.md
python scripts/build_week1_artifacts.py --handoff <交接包路径> --out <输出目录>
python scripts/test_handoff_check.py
```

依赖：Python 3.11+；几何核对需要 `pyshp`（`pip install pyshp`）。

## 为什么仓库里没有 `records20.json` / `cases12.json`

main 上的 `scripts/check_public_assets.py` 规定：仓库内任何 JSON 只允许出现 8 条已批准的真实记录（`sourceId` 482、483、493、505、528、529、533、597），其余真实记录不得进入公开仓库。
20 条样例与 12 例候选超出该边界，因此**只随交付包提交**（本地 `data\第1周_2026-10-19_10-25\`），仓库里保留 Markdown 说明。
需要公开的 8 条样例见 `examples/catalog.real-subset.json`。

## 下周（第 2 周）衔接

- 崔亚鹏：给 12 个编号导高清 CTX 图；确认 224 裁片的重采样规则；启动真实目录入库（先解决纬度约定）
- 梁先楷：接全球底图，按 `basemap.md` 第 3 节修 90° 偏移
- 滕佳杰：纬度核验（5 个地标法）、12 例典型性终审、四类文案定稿、结构化关联数据（类型—案例—来源）交周璇
- 周璇：拿 `atlas-types.md` 切分后的片段建检索索引
