import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

def load_predictions(directory):
    """指定されたディレクトリから予測結果を読み込む"""
    predictions = {}

    jsonl_files = list(Path(directory).glob("*.jsonl"))

    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                key = (data['subject'], data['question_number'])
                predictions[key] = data['predicted_answer']

    return predictions

def calculate_agreement(original_preds, early_preds):
    """2つの予測結果の一致率を計算"""
    common_keys = set(original_preds.keys()) & set(early_preds.keys())

    if not common_keys:
        return 0.0

    agreements = sum(1 for key in common_keys if original_preds[key] == early_preds[key])
    return agreements / len(common_keys)

# originalの予測を読み込む
print("Loading original predictions...")
original_predictions = load_predictions("results/original/mmlu")
print(f"Loaded {len(original_predictions)} predictions from original")

# 各p値についてearly_answerの予測を読み込み、一致率を計算
p_values = []
agreement_rates = []

early_answer_base = Path("results/early_answer/mmlu")
p_dirs = sorted([d for d in early_answer_base.iterdir() if d.is_dir()])

for p_dir in p_dirs:
    p_value = int(p_dir.name) / 10.0
    p_values.append(p_value)

    print(f"Processing p={p_value}...")
    early_predictions = load_predictions(p_dir)
    print(f"  Loaded {len(early_predictions)} predictions")

    agreement = calculate_agreement(original_predictions, early_predictions)
    agreement_rates.append(agreement)
    print(f"  Agreement rate: {agreement:.3f}")

# グラフの作成
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(p_values, agreement_rates, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax.set_xlabel('p value', fontsize=12)
ax.set_ylabel('Agreement Rate with Original', fontsize=12)
ax.set_title('Answer Agreement Rate: Early Answer vs Original', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

# パーセンテージ表示
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
plt.savefig('mmlu_results.png', dpi=300, bbox_inches='tight')
print("\nグラフを 'mmlu_results.png' として保存しました")

# 結果のサマリーを表示
print("\n=== Summary ===")
for p, rate in zip(p_values, agreement_rates):
    print(f"p={p:.1f}: Agreement={rate:.1%}")
