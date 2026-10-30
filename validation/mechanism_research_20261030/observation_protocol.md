# 辅助字段观测接入协议

`observation_intake.py` 检查预先计划的辅助字段是否具备独立复核所需的采集记录。它不读取预测文件中的数值，不调用 `compare_display`，不评分，也不判断 0AMV 公式是否复刻成功。

本轮冻结计划有 18 个字段，即 3 只股票、3 个周期和 2 个指标。分母来自独立输入的完整计划。缺失、不可采集或有歧义的字段不能从分母删除，也不能补成零。

## 输入

命令接受三个独立输入。

- `--plan` 是在采集标签前保存的 JSON 计划。
- `--packet` 是采集后保存的 JSON 观测包。
- `--evidence-root` 是私有本地证据目录。预测文件、契约依据和观测来源都必须位于此目录内。

`--as-of` 是调用者给定的检查时刻，格式为含时区的 ISO 时间。省略时使用机器当前 UTC 时间。它不是可信时间戳服务，也不证明记录在声明的时刻已经存在。

输入对象拒绝缺少字段和额外字段。JSON 重复键会被拒绝。下面的所有数值与名称都是合成格式示例，不是官方读数。

### 计划

计划包含以下字段。

```json
{
  "planned_at": "2026-10-30T09:00:00+00:00",
  "prediction_frozen_at": "2026-10-30T08:30:00+00:00",
  "prediction": {
    "source_id": "synthetic-prediction",
    "path": "synthetic-prediction.txt",
    "sha256": "填写本地文件实际的64位小写SHA256"
  },
  "purpose": "contract_calibration",
  "expected": [
    {
      "symbol": "SYNTHETIC-A",
      "date": "2026-10-30",
      "indicator": "CYF",
      "period": 13,
      "field_label": null,
      "unit": null,
      "display_contract": {
        "status": "unknown",
        "reason": "尚未独立确定显示契约",
        "quantum": null,
        "origin": null,
        "rounding": "unknown"
      }
    }
  ]
}
```

这段示例只有一个字段，不能替代本轮完整的 18 项计划。`expected` 必须非空且不能重复。股票、日期、指标、周期组成唯一键。指标只接受 `CYF` 和 `CYC`，周期必须为正整数，日期必须严格为 `YYYY-MM-DD`。程序不推测周期，也不转换单位。

`field_label` 和 `unit` 是预先声明的实际字段文字与单位。未知值必须为 `null`，不能用观测值反填后继续沿用原计划哈希。修改计划需要新包绑定，不能把同一批已见标签变成新的盲测。

`purpose` 只接受 `contract_calibration` 或 `blind_validation`。观测包必须与计划一致。用于确定显示规则的观测必须标为 `contract_calibration`。工具检查声明的一致性，不能证明调用者没有把校准数据错误地声明为盲测。

### 显示契约

未知契约采用上述五字段对象。`status` 也可以为 `ambiguous`，并保留原因。两种状态都要求 `quantum`、`origin` 为 `null`，`rounding` 为 `unknown`，并阻止就绪。

已知契约采用以下字段。

```json
{
  "status": "known",
  "quantum": "0.1",
  "origin": "0",
  "rounding": "nearest",
  "allows_cyc_zero": false,
  "basis": "independent_calibration",
  "evidence": {
    "source_id": "synthetic-contract",
    "path": "synthetic-contract.txt",
    "sha256": "填写本地文件实际的64位小写SHA256"
  },
  "established_at": "2026-10-30T08:00:00+00:00"
}
```

这是人为约定的合成契约，不是对官方软件采用 `nearest` 的断言。截图显示几位小数不能证明舍入规则，候选残差也不能用于选择 `quantum`。

`quantum` 必须是正的精确十进制字符串。`origin` 也必须是精确十进制字符串。`rounding` 接受 `nearest`、`floor`、`ceiling`、`truncate`。`basis` 接受 `documentation`、`independent_calibration`。依据文件必须通过字节哈希检查，`established_at` 不得晚于 `planned_at`。

现有 `display_measurement.py` 只支持零原点、最近网格契约。其他已知规则会保留，但结果为 `needs_contract`，本工具不会增加评分规则或容差。已知契约会检查录入值是否在声明的网格上，这不是预测匹配。

### 观测包

包的顶层字段如下。

```json
{
  "plan_sha256": "填写plan_sha256(plan)的实际结果",
  "prediction_sha256": "与计划中的prediction.sha256完全相同",
  "prediction_frozen_at": "2026-10-30T08:30:00+00:00",
  "purpose": "contract_calibration",
  "assembled_at": "2026-10-30T11:00:00+00:00",
  "observations": []
}
```

`plan_sha256(plan)` 对完整计划作确定性 JSON 编码，再计算 SHA256。编码使用 UTF-8、按键排序、无空白分隔符，并按唯一键排序 `expected`。计划和观测条目的顺序不影响检查结果。它不是原始 JSON 文件的字节哈希。

`observations` 必须逐项覆盖计划，不能保持空数组。每项先带有 `symbol`、`date`、`indicator`、`period`，再按状态采用下列字段。

- `unavailable` 或 `ambiguous` 只另加 `status` 和非空 `reason`。不能额外夹带 `value`。
- `observed` 另加 `status`、`value`、`display_text`、`source`、`software`、`field`、`captured_at`。

一个合成 `observed` 记录的附加字段如下。

```json
{
  "status": "observed",
  "value": "50.0",
  "display_text": "50.0",
  "source": {
    "source_id": "synthetic-capture",
    "path": "synthetic-capture.txt",
    "sha256": "填写本地文件实际的64位小写SHA256"
  },
  "software": {"name": "SYNTHETIC fixture", "version": "fixture-1"},
  "field": {"label": "CYF", "period": 13, "unit": "percent"},
  "captured_at": "2026-10-30T10:00:00+00:00"
}
```

`display_text` 是原始数字标记，必须与 `value` 逐字相同，包含末尾的零。字段文字和单位另存于 `field`。不支持无记录地移除千位分隔符、转换百分数、按显示位数四舍五入。无法确定数字标记时保留 `ambiguous` 状态和原因。

数值只接受精确十进制字符串，拒绝浮点数、布尔值和非有限值。`CYF` 的有效域为 `[0, 100]`。`CYC` 必须为正。只有显式已知的零原点、最近网格契约声明 `allows_cyc_zero: true` 时，才接受 `CYC` 显示零；这表示正数被显示为零，不表示内部真值可以为零。未知契约不能授权零值。

`field.period` 必须等于计划周期。已知的计划字段名和单位必须逐字匹配。若计划字段名或单位仍为 `null`，本工具不会据此次观测补齐它，也不会就绪。观测时仍未知的软件版本、实际字段名或单位不能伪装成 `observed`。

`planned_at`、冻结时刻、采集时刻、组包时刻、检查时刻都必须含时区。每次采集不得早于计划及冻结，不得晚于组包；组包不得晚于检查时刻。声明的冻结时间必须与计划字符串完全相同。

### 本地证据引用

每个引用均含 `source_id`、`path`、`sha256`。`source_id` 只能使用字母、数字、点、下划线和连字符，最长 128 字符，以字母或数字开头。同一 ID 不能绑定不同路径或哈希。

`path` 必须是证据根目录内的相对路径。绝对路径、空路径段、`.`、`..`、反斜杠以及任何符号链接都被拒绝。目标必须是普通文件；缺失引用、不可读引用和哈希变化均导致无效。输出只含来源 ID 和哈希，不复制文件内容或原始路径。来源 ID 本身也不能包含私有路径。

## 结果

每个有效计划项都保留在 `cases`。计划本身无效时，程序拒绝给出可信分母。观测包无效时，仍返回所有计划项，并将可用覆盖归零。

状态按下列优先级产生。

| 状态 | 含义 |
| --- | --- |
| `invalid` | 身份、结构、数值、证据或时序检查失败。 |
| `collection_incomplete` | 至少一项为 `unavailable` 或 `ambiguous`。 |
| `needs_metadata` | 已采集，但预先声明的字段名或单位未知。 |
| `needs_contract` | 已采集且元数据匹配，但显示契约未知、有歧义或不受现有辅助函数支持。 |
| `ready_for_independent_review` | 所有计划项通过本工具的采集就绪检查，仍需独立复核。 |

`coverage.planned` 始终包含所有计划字段。`coverage.observed` 是有效录入记录数。`coverage.usable` 是元数据、证据和支持契约齐备的记录数，不是通过评分的记录数。全缺失结果必须为 `0/18`，不能称为完全一致。部分可用记录也不能使全包就绪。

输出中的 `metadata_checked` 仅表示声明的元数据与计划一致，`evidence_checked` 仅表示本地引用和字节哈希通过检查。它们不证明图片含有该数值、来源真实、转录正确或人员已经复核。声明哈希只绑定字节，不证明时间真实性或观测独立性。

所有结果的 `official_formula_replication` 固定为 `not_established`，`label_truth` 固定为 `not_verified`，`mechanism_parity` 固定为 `not_evaluated`，`blindness_verified` 固定为 `false`。本工具不能满足 64 日或两个独立 20 日窗口的公式复刻验收。

## 命令

在仓库根目录运行以下模块。路径为调用者的私有文件，命令不写结果文件。

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.11 -m validation.mechanism_research_20261030.observation_intake \
  --plan /private/intake-plan.json \
  --packet /private/intake-packet.json \
  --evidence-root /private \
  --as-of 2026-10-30T12:00:00+00:00
```

有效包退出码为 `0`，包括明确不完整的包。无效输入退出码为 `2`。退出码 `0` 不表示就绪，调用者必须读取 JSON 的 `status` 和 `ready_for_independent_review`。结果只写 stdout，调用者可将结果保存为新的追加式证据，不能覆盖旧证据。
