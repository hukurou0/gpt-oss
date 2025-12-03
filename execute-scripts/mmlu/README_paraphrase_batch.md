# MMLU Paraphrase Batch API

OpenAI Batch APIを使用して、MMLUデータセットの分析結果を言い換えるスクリプト群です。
Batch APIを使用することで、通常のAPI料金の**50%割引**で処理できます。

## 概要

| スクリプト | 説明 |
|-----------|------|
| `paraphrase_batch_prepare.py` | Batch API用のリクエストJSONLファイルを生成 |
| `paraphrase_batch_submit.py` | バッチをOpenAIにアップロード・実行 |
| `paraphrase_batch_download.py` | 完了したバッチの結果をダウンロード |

## ディレクトリ構造

```
results/mmlu/
├── original/                    # 元データ（入力）
│   ├── abstract_algebra.jsonl
│   ├── anatomy.jsonl
│   └── ...
├── paraphrase_batch/            # Batch API関連ファイル
│   ├── abstract_algebra.jsonl   # リクエストファイル
│   ├── anatomy.jsonl
│   ├── ...
│   ├── _metadata.json           # 生成時のメタデータ
│   ├── _batch_status.json       # バッチ実行状態
│   └── results/                 # ダウンロードした生結果
│       ├── abstract_algebra.jsonl
│       └── ...
└── paraphrased/                 # マージ済み最終結果
    ├── abstract_algebra.jsonl
    └── ...
```

## 使い方

### 前提条件

```bash
export OPENAI_API_KEY="your-api-key"
```

### Step 1: リクエストファイルの生成

```bash
python execute-scripts/mmlu/paraphrase_batch_prepare.py
```

`results/mmlu/original/` の各科目ファイルから、Batch API形式のJSONLファイルを生成します。

### Step 2: バッチの送信

```bash
# 全科目を送信
python execute-scripts/mmlu/paraphrase_batch_submit.py

# 特定の科目のみ送信
python execute-scripts/mmlu/paraphrase_batch_submit.py --subjects abstract_algebra anatomy
```

### Step 3: 状態の確認

```bash
python execute-scripts/mmlu/paraphrase_batch_submit.py --status
```

出力例:
```
Subject                        Status          Progress
============================================================
abstract_algebra               completed       100/100
anatomy                        in_progress     50/135
astronomy                      failed          0/0

Summary: 1 completed, 1 in_progress, 1 failed
```

### Step 4: failedバッチの再送信

同時実行バッチ数の制限により失敗した場合、完了後に再送信できます。

```bash
# 全failedバッチを再送信
python execute-scripts/mmlu/paraphrase_batch_submit.py --retry

# 特定の科目のみ再送信
python execute-scripts/mmlu/paraphrase_batch_submit.py --retry --subjects abstract_algebra
```

### Step 5: 結果のダウンロード

```bash
# 完了したバッチの結果をダウンロード
python execute-scripts/mmlu/paraphrase_batch_download.py

# 元データとマージして保存（推奨）
python execute-scripts/mmlu/paraphrase_batch_download.py --merge

# ダウンロード済み結果のサマリー表示
python execute-scripts/mmlu/paraphrase_batch_download.py --summary
```

## コマンドオプション一覧

### paraphrase_batch_submit.py

| オプション | 説明 |
|-----------|------|
| (なし) | 未送信のバッチを送信 |
| `--status` | 全バッチの状態を確認 |
| `--retry` | failedバッチを再送信 |
| `--cancel` | 実行中のバッチをキャンセル |
| `--subjects` | 処理対象を特定科目に限定 |

### paraphrase_batch_download.py

| オプション | 説明 |
|-----------|------|
| (なし) | 完了したバッチの結果をダウンロード |
| `--merge` | 元データとマージして `paraphrased/` に保存 |
| `--summary` | ダウンロード済み結果のサマリー表示 |
| `--subjects` | 処理対象を特定科目に限定 |

## 出力データ形式

### マージ後のデータ (`--merge` オプション使用時)

```json
{
  "question_number": 1,
  "analysis": "元の分析テキスト...",
  "paraphrased_analysis": "言い換えた分析テキスト...",
  "paraphrase_tokens": {
    "input": 686,
    "output": 417,
    "total": 1103
  }
}
```

## 費用見積もり

| 項目 | 値 |
|------|-----|
| 総問題数 | 約14,000問 |
| 推定inputトークン | 約6.1M |
| 推定outputトークン | 約3.3M |
| 通常料金 | 約$39 |
| **Batch API料金（50%割引）** | **約$19** |

※ gpt-4.1 (input: $2.00/1M, output: $8.00/1M) での見積もり

## トラブルシューティング

### 多くのバッチがfailedになる

**原因**: OpenAI Batch APIの同時実行バッチ数制限

**対処法**:
1. `--status` で完了を待つ
2. `--retry` で再送信
3. 必要に応じて `--subjects` で少数ずつ再送信

### 環境変数エラー

```
Error: OPENAI_API_KEY environment variable is not set
```

**対処法**:
```bash
export OPENAI_API_KEY="your-api-key"
```

## 処理フロー図

```
┌─────────────────────────────────────────────────────────────┐
│  results/mmlu/original/*.jsonl                              │
│  (元データ)                                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  paraphrase_batch_prepare.py                                │
│  → Batch API形式のJSONLを生成                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  results/mmlu/paraphrase_batch/*.jsonl                      │
│  (リクエストファイル)                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  paraphrase_batch_submit.py                                 │
│  → OpenAI Batch APIにアップロード・実行                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  OpenAI Batch API (24h以内に処理)                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  paraphrase_batch_download.py --merge                       │
│  → 結果をダウンロードしてマージ                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  results/mmlu/paraphrased/*.jsonl                           │
│  (最終結果: 元データ + 言い換えテキスト)                       │
└─────────────────────────────────────────────────────────────┘
```
