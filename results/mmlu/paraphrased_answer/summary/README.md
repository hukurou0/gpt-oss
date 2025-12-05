# Paraphrased Answer 分析結果

選択肢を言い換えた場合のモデルの回答変化を分析した結果をまとめる。

## 概要

| 項目 | Original | Paraphrased Answer |
|------|----------|-------------------|
| 正答率 | 82.84% | 82.73% |
| 正解数 | 11,633 | 11,617 |
| 総問題数 | 14,042 | 14,042 |

差は16問（-0.11ポイント）。ただし、この差は言い換え処理の技術的な失敗に起因するものであり、言い換え自体の影響ではない（後述）。

## 言い換え失敗の判定基準

言い換え処理が正常に行われたかどうかを以下の基準で判定した。

### 失敗と判定する基準

| 基準 | 説明 | 該当件数 |
|------|------|----------|
| too_short | 言い換え後のanalysisが元のanalysisの**10%未満**の長さ | 225件 |

### 失敗の主な原因

言い換えモデルが元のanalysis全体（数百〜数千文字）を言い換えず、回答文字（"A"など）だけを出力してしまうケースが大半。

例:
```
anatomy Q98: 元2379文字 → 言い換え後1文字 "C"
business_ethics Q80: 元4605文字 → 言い換え後1文字 "A"
```

## データ分類

| 分類 | 問題数 | 割合 |
|------|--------|------|
| 全データ | 14,042 | 100% |
| 正常な言い換え | 13,817 | 98.40% |
| 言い換え失敗 | 225 | 1.60% |

## 正答率の比較

| データセット | Original | Paraphrased | 差 |
|-------------|----------|-------------|-----|
| 全データ (14,042問) | 82.84% | 82.73% | -0.11pt |
| **正常な言い換えのみ (13,817問)** | 82.84% | **82.89%** | **+0.05pt** |
| 言い換え失敗のみ (225問) | 83.11% | 72.89% | -10.22pt |

## 回答変化パターン

### 全データ (14,042問)

| パターン | 件数 | 割合 |
|----------|------|------|
| both_correct | 11,604 | 82.64% |
| both_wrong | 2,396 | 17.06% |
| wrong_to_correct | 13 | 0.09% |
| correct_to_wrong | 29 | 0.21% |

### 正常な言い換えのみ (13,817問)

| パターン | 件数 | 割合 |
|----------|------|------|
| both_correct | 11,441 | 82.80% |
| both_wrong | 2,359 | 17.07% |
| wrong_to_correct | 12 | 0.09% |
| correct_to_wrong | 5 | 0.04% |

### 言い換え失敗のみ (225問)

| パターン | 件数 | 割合 |
|----------|------|------|
| both_correct | 163 | 72.44% |
| both_wrong | 37 | 16.44% |
| wrong_to_correct | 1 | 0.44% |
| correct_to_wrong | 24 | 10.67% |

## 正常な言い換えでの回答変化の詳細

### wrong_to_correct (12件)

| Subject | Q# | Original | Paraphrased | Correct |
|---------|-----|----------|-------------|---------|
| college_medicine | 127 | D | C | C |
| high_school_macroeconomics | 288 | A | B | B |
| medical_genetics | 46 | D | A | A |
| miscellaneous | 251 | D | A | A |
| miscellaneous | 755 | Y | C | C |
| moral_scenarios | 708 | A | B | B |
| philosophy | 20 | B | D | D |
| prehistory | 168 | A | B | B |
| professional_accounting | 31 | D | B | B |
| professional_accounting | 246 | B | A | A |
| professional_psychology | 140 | C | A | A |
| professional_psychology | 242 | B | C | C |

### correct_to_wrong (5件)

| Subject | Q# | Original | Paraphrased | Correct |
|---------|-----|----------|-------------|---------|
| global_facts | 16 | B | C | B |
| professional_law | 743 | B | * | B |
| professional_law | 1038 | D | A | D |
| clinical_knowledge | 35 | B | K | B |
| formal_logic | 13 | D | * | D |

## 言い換えによるAnalysis結論の変化

言い換え処理によって、analysis内の結論（回答）自体が変わったケースを分析した。

### 抽出方法

analysisの末尾から結論パターン（"answer is X", "output X" 等）を抽出し、元のanalysisと言い換え後のanalysisで結論が異なるケースを特定。

### 結果

総問題数 14,042件のうち、両方のanalysisから結論を抽出でき、かつ抽出した結論が予測回答と一致した信頼できるケースは3件。

| Subject | Q# | Orig結論 | Para結論 | 正解 | 変化 |
|---------|-----|----------|----------|------|------|
| econometrics | 55 | B | D | C | wrong→wrong |
| professional_accounting | 246 | B | A | A | wrong→correct |
| professional_psychology | 140 | C | A | A | wrong→correct |

### サマリー

| 変化 | 件数 |
|------|------|
| wrong_to_correct（改善） | 2 |
| correct_to_wrong（悪化） | 0 |
| wrong_to_wrong | 1 |

**言い換えによってanalysisの結論が変わったケースでは、悪化は0件、改善は2件。**

## 洞察

### 1. 言い換え失敗の影響

- 言い換え失敗は全体の**1.6%（225件）**に過ぎない
- しかし、correct_to_wrong 29件のうち**24件（83%）**がこれに起因している
- 言い換え失敗のデータでは正答率が**-10.22pt**悪化

### 2. 正常な言い換えの効果

- 正答率: 82.84% → 82.89%（**+0.05pt**）
- correct_to_wrong: **5件**のみ
- wrong_to_correct: **12件**
- ネット: **+7問の改善**

### 3. 言い換えによる結論変化の効果

- 言い換えによってanalysisの結論自体が変わったケースは少数（3件）
- そのうち悪化は**0件**、改善は**2件**
- 言い換えがモデルの推論を改善する効果がある可能性を示唆

### 4. 結論

**選択肢の言い換え自体はモデルの判断に悪影響を与えず、むしろわずかながら改善効果がある。**

観測されたcorrect_to_wrongの大部分（83%）は、言い換えプロセスの技術的問題（言い換えモデルが指示を誤解し、分析全体を言い換えず回答だけを返した）が原因であり、言い換え内容の品質の問題ではない。

正常に言い換えが行われた場合、言い換えによってモデルが異なる視点から問題を見ることができ、元の誤った推論から脱却できる可能性がある。
