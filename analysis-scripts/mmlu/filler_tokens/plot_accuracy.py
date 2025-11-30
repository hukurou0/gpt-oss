import matplotlib.pyplot as plt
import numpy as np

# データ
folders = ['original', '00', '01', '02', '03', '04', '05', '06', '07']
accuracies = [82.71, 82.91, 86.06, 85.97, 85.88, 86.05, 85.93, 85.93, 86.93]
correct_counts = [11614, 11617, 10742, 10657, 10438, 10563, 10361, 10362, 8750]
total_counts = [14042, 14012, 12482, 12396, 12154, 12275, 12058, 12058, 10066]

# グラフのスタイル設定
plt.figure(figsize=(14, 6))

# 正解率の棒グラフ (originalだけ色を変える)
colors = ['coral'] + ['steelblue'] * 8  # originalはcoral、それ以外はsteelblue
bars = plt.bar(folders, accuracies, color=colors, alpha=0.8, edgecolor='black')

# 各バーの上に正解率とカウントを表示
for i, (folder, acc, correct, total) in enumerate(zip(folders, accuracies, correct_counts, total_counts)):
    plt.text(i, acc + 0.3, f'{acc}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.text(i, acc - 1.5, f'{correct}/{total}', ha='center', va='top', fontsize=8, color='white')

# グラフの設定
plt.xlabel('フォルダ', fontsize=12, fontweight='bold')
plt.ylabel('正解率 (%)', fontsize=12, fontweight='bold')
plt.title('フォルダごとの正解率', fontsize=14, fontweight='bold')
plt.ylim(80, 90)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# 平均線を追加
avg_accuracy = np.mean(accuracies)
plt.axhline(y=avg_accuracy, color='red', linestyle='--', linewidth=2, label=f'平均: {avg_accuracy:.2f}%')
plt.legend()

plt.tight_layout()
plt.savefig('/Users/huku/Desktop/program/study/gpt-oss/results/mmlu/accuracy_by_folder.png', dpi=300, bbox_inches='tight')
print(f"グラフを保存しました: results/mmlu/accuracy_by_folder.png")
print(f"平均正解率: {avg_accuracy:.2f}%")

plt.show()
