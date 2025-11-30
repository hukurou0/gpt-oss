import matplotlib.pyplot as plt

# データ
p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracy = [0.503, 0.524, 0.553, 0.594, 0.611, 0.635, 0.667, 0.699, 0.744, 0.795, 0.827]

# グラフの作成
fig, ax = plt.subplots(figsize=(10, 6))

# Accuracy vs p
ax.plot(p_values, accuracy, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax.set_xlabel('p value', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy vs p value', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.45, 0.85)

plt.tight_layout()
plt.savefig('mmlu_results.png', dpi=300, bbox_inches='tight')
print("グラフを 'mmlu_results.png' として保存しました")
