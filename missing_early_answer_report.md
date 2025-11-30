# MMLU Early Answer 欠損レポート

## Summary

`results/mmlu/original` を正解として `results/mmlu/early_answer` を検証した結果、一部のタスクで欠損が確認されました。

- **サブディレクトリ数**: 10 (00-09)
- **欠損があるタスク**: 10 / 57
- **欠損レコード数**: 各サブディレクトリ30件（全サブディレクトリ共通）

## 欠損の原因

### 根本原因

`original` で `predicted_answer` が空文字で登録されているレコードが、`early_answer` で除外されている。

### なぜ `predicted_answer` が空になるか

`model/call_gpt_oss.py:102-128` の回答抽出ロジック:

```python
analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'

if result["analysis"] == "" or result["final"] == "":
    return {"analysis": "", "final": ""}  # ← 空を返す
```

**LLMが期待されるフォーマットで回答しなかった場合、抽出に失敗して空になる。**

## GPUメモリオーバー仮説の検証

### 仮説

推論が長くなるとGPUメモリをオーバーして生成に失敗する。

### 検証結果

| ai_time閾値 | 失敗件数 | 成功件数 | 備考 |
|-------------|---------|---------|------|
| 22秒以上 | 1 | 0 | |
| 21秒以上 | 7 | 0 | **100%失敗** |
| 20秒以上 | 7 | 3 | |
| 19秒以上 | 9 | 5 | |
| 18秒以上 | 10 | 7 | |

**21秒を超える処理はすべて失敗**しており、成功例が存在しない。

### 失敗のカテゴリ分類

| カテゴリ | 件数 | ai_time | 推定原因 |
|---------|------|---------|----------|
| 長時間失敗 | 7 | 21秒超 | GPUメモリオーバー |
| 中間失敗 | 19 | 5-21秒 | メモリ圧迫の境界ゾーン |
| 短時間失敗 | 4 | 5秒未満 | フォーマット不備など別原因 |

### 短時間失敗の詳細（4件）

| タスク | question_number | ai_time |
|--------|-----------------|---------|
| miscellaneous | 316 | 0.60s |
| miscellaneous | 341 | 0.45s |
| miscellaneous | 388 | 0.47s |
| high_school_macroeconomics | 216 | 1.50s |

### 結論

- **GPUメモリオーバー仮説は一定の説得力がある**
- 21秒超えで100%失敗、成功の最大は20.43秒
- ただし短時間での失敗も存在するため、複合的な原因がある

## 欠損詳細

| タスク | Original | Early Answer | 欠損数 | 欠損 question_number |
|--------|----------|--------------|--------|----------------------|
| professional_law | 1,534 | 1,519 | 15 | 52, 246, 284, 433, 460, 482, 596, 886, 919, 1024, 1122, 1159, 1210, 1293, 1515 |
| college_computer_science | 100 | 97 | 3 | 6, 49, 78 |
| miscellaneous | 783 | 780 | 3 | 316, 341, 388 |
| college_chemistry | 100 | 98 | 2 | 18, 35 |
| college_mathematics | 100 | 98 | 2 | 14, 92 |
| college_medicine | 173 | 172 | 1 | 74 |
| high_school_macroeconomics | 390 | 389 | 1 | 216 |
| high_school_mathematics | 270 | 269 | 1 | 82 |
| high_school_us_history | 204 | 203 | 1 | 41 |
| professional_accounting | 282 | 281 | 1 | 7 |

## 完了しているタスク（47タスク）

abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge, college_biology, college_physics, computer_security, conceptual_physics, econometrics, electrical_engineering, elementary_mathematics, formal_logic, global_facts, high_school_biology, high_school_chemistry, high_school_computer_science, high_school_european_history, high_school_geography, high_school_government_and_politics, high_school_microeconomics, high_school_physics, high_school_psychology, high_school_statistics, high_school_world_history, human_aging, human_sexuality, international_law, jurisprudence, logical_fallacies, machine_learning, management, marketing, medical_genetics, moral_disputes, moral_scenarios, nutrition, philosophy, prehistory, professional_medicine, professional_psychology, public_relations, security_studies, sociology, us_foreign_policy, virology, world_religions

---
*Generated: 2025-11-30*
