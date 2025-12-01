#!/usr/bin/env python3
"""
results/mmlu/originalとresults/mmlu/early_answerの各フォルダを比較し、
フォルダごとに回答の一致率をグラフにするスクリプト
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_answers(folder_path):
    """
    フォルダ内のJSONLファイルから回答を読み込む

    Returns:
        dict: {(subject, question_number): predicted_answer}
    """
    answers = {}
    folder = Path(folder_path)

    for jsonl_file in folder.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    subject = data.get("subject", "")
                    question_number = data.get("question_number", 0)
                    predicted_answer = data.get("predicted_answer", "")
                    answers[(subject, question_number)] = predicted_answer
                except json.JSONDecodeError:
                    continue

    return answers


def calculate_match_rate(original_answers, target_answers):
    """
    2つの回答セットを比較し、一致率を計算する

    Returns:
        dict: 一致率の統計情報
    """
    common_keys = set(original_answers.keys()) & set(target_answers.keys())
    total = len(common_keys)

    if total == 0:
        return {"total": 0, "matched": 0, "match_rate": 0.0}

    matched = sum(
        1 for key in common_keys
        if original_answers[key] == target_answers[key]
    )

    return {
        "total": total,
        "matched": matched,
        "match_rate": (matched / total * 100) if total > 0 else 0.0
    }


def calculate_match_rate_by_subject(original_answers, target_answers):
    """
    サブジェクトごとに一致率を計算する

    Returns:
        dict: {subject: {"total": int, "matched": int, "match_rate": float}}
    """
    common_keys = set(original_answers.keys()) & set(target_answers.keys())
    subject_stats = defaultdict(lambda: {"total": 0, "matched": 0})

    for key in common_keys:
        subject = key[0]
        subject_stats[subject]["total"] += 1
        if original_answers[key] == target_answers[key]:
            subject_stats[subject]["matched"] += 1

    # 一致率を計算
    for subject in subject_stats:
        stats = subject_stats[subject]
        stats["match_rate"] = (
            stats["matched"] / stats["total"] * 100
            if stats["total"] > 0 else 0.0
        )

    return dict(subject_stats)


def main():
    base_path = Path("results/mmlu")
    original_path = base_path / "original"
    early_answer_path = base_path / "early_answer"
    output_dir = Path("results/mmlu/early_answer/summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        print(f"エラー: {original_path} が見つかりません")
        return

    if not early_answer_path.exists():
        print(f"エラー: {early_answer_path} が見つかりません")
        return

    # Originalの回答を読み込む
    print("Originalの回答を読み込み中...")
    original_answers = load_answers(original_path)
    print(f"  {len(original_answers)} 件の回答を読み込みました")

    # Early answerの各フォルダを処理
    folders = sorted([
        d for d in early_answer_path.iterdir()
        if d.is_dir() and d.name != "summary"
    ])

    folder_results = {}
    all_subject_results = {}

    print("\nEarly answerの各フォルダを比較中...")
    for folder in folders:
        folder_name = folder.name
        print(f"  処理中: {folder_name}")

        target_answers = load_answers(folder)

        # 全体の一致率
        overall_stats = calculate_match_rate(original_answers, target_answers)
        folder_results[folder_name] = overall_stats

        # サブジェクトごとの一致率
        subject_stats = calculate_match_rate_by_subject(original_answers, target_answers)
        all_subject_results[folder_name] = subject_stats

        print(f"    一致率: {overall_stats['match_rate']:.2f}% "
              f"({overall_stats['matched']}/{overall_stats['total']})")

    # グラフ1: フォルダごとの全体一致率
    print("\nグラフを作成中...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左側: フォルダごとの一致率（棒グラフ）
    folder_names = list(folder_results.keys())
    match_rates = [folder_results[f]["match_rate"] for f in folder_names]

    ax1 = axes[0]
    bars = ax1.bar(folder_names, match_rates, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Early Answer Folder', fontsize=12)
    ax1.set_ylabel('Answer Match Rate (%)', fontsize=12)
    ax1.set_title('Answer Match Rate by Folder\n(Original vs Early Answer)', fontsize=14)
    ax1.set_ylim(0, max(match_rates) * 1.2 if match_rates else 100)

    # 棒グラフに値を表示
    for bar, rate in zip(bars, match_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # 右側: フォルダごとの一致率（折れ線グラフ）
    ax2 = axes[1]
    ax2.plot(folder_names, match_rates, marker='o', linewidth=2,
             markersize=8, color='steelblue')
    ax2.fill_between(folder_names, match_rates, alpha=0.3, color='steelblue')
    ax2.set_xlabel('Early Answer Folder', fontsize=12)
    ax2.set_ylabel('Answer Match Rate (%)', fontsize=12)
    ax2.set_title('Answer Match Rate Trend', fontsize=14)
    ax2.set_ylim(0, max(match_rates) * 1.2 if match_rates else 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    graph_path = output_dir / "answer_match_rate_by_folder.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    print(f"  グラフを保存しました: {graph_path}")
    plt.close()

    # グラフ3: 先行研究との比較グラフ
    # 先行研究データ (MMLU, X軸: 理由サンプル提供割合%)
    prior_research = {
        "3-Step CoTs": {"x": [0, 25, 50, 75, 100], "y": [80, 84, 92, 97, 100]},
        "4-Step CoTs": {"x": [0, 25, 50, 75, 100], "y": [80, 82, 89, 96, 100]},
        "5-Step CoTs": {"x": [0, 25, 50, 75, 100], "y": [79, 79, 85, 94, 100]},
        "6-Step CoTs": {"x": [0, 25, 50, 75, 100], "y": [73, 76, 80, 89, 100]},
    }

    # 現在のデータをX軸0-100%にマッピング (フォルダ00-09 → 0-90%, 100%は100%一致)
    our_x = [int(f) * 10 for f in folder_names] + [100]  # 00→0%, 01→10%, ..., 100%
    our_y = match_rates + [100.0]  # 100%の推論提供時は100%一致

    fig, ax = plt.subplots(figsize=(12, 8))

    # 先行研究のデータをプロット
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    markers = ['s', '^', 'D', 'v']
    for (label, data), color, marker in zip(prior_research.items(), colors, markers):
        ax.plot(data["x"], data["y"], marker=marker, linewidth=2, markersize=8,
                color=color, label=f'Prior Research: {label}', linestyle='--', alpha=0.7)

    # 現在のデータをプロット
    ax.plot(our_x, our_y, marker='o', linewidth=3, markersize=10,
            color='steelblue', label='Ours (GPT-OSS-20b-F16)', linestyle='-')

    ax.set_xlabel('Reasoning Sample Ratio (%)', fontsize=14)
    ax.set_ylabel('Answer Match Rate (%)', fontsize=14)
    ax.set_title('Comparison with Prior Research\n(Answer Match Rate vs Reasoning Ratio)', fontsize=16)
    ax.set_xlim(-5, 105)
    ax.set_ylim(40, 105)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    comparison_path = output_dir / "comparison_with_prior_research.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"  先行研究比較グラフを保存しました: {comparison_path}")
    plt.close()

    # グラフ2: サブジェクトごとの一致率ヒートマップ
    if all_subject_results and folder_names:
        # すべてのサブジェクトを取得
        all_subjects = set()
        for folder_data in all_subject_results.values():
            all_subjects.update(folder_data.keys())
        all_subjects = sorted(all_subjects)

        # ヒートマップ用のデータを作成
        heatmap_data = np.zeros((len(all_subjects), len(folder_names)))
        for j, folder in enumerate(folder_names):
            for i, subject in enumerate(all_subjects):
                if subject in all_subject_results[folder]:
                    heatmap_data[i, j] = all_subject_results[folder][subject]["match_rate"]

        fig, ax = plt.subplots(figsize=(12, max(10, len(all_subjects) * 0.3)))
        im = ax.imshow(heatmap_data, cmap='YlGn', aspect='auto')

        ax.set_xticks(np.arange(len(folder_names)))
        ax.set_yticks(np.arange(len(all_subjects)))
        ax.set_xticklabels(folder_names)
        ax.set_yticklabels(all_subjects)

        plt.setp(ax.get_yticklabels(), fontsize=8)
        ax.set_xlabel('Early Answer Folder', fontsize=12)
        ax.set_ylabel('Subject', fontsize=12)
        ax.set_title('Answer Match Rate by Subject and Folder (%)', fontsize=14)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Match Rate (%)', fontsize=10)

        plt.tight_layout()
        heatmap_path = output_dir / "answer_match_rate_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"  ヒートマップを保存しました: {heatmap_path}")
        plt.close()

    # 結果をJSONに保存
    result_data = {
        "overall_by_folder": folder_results,
        "by_subject": all_subject_results
    }

    json_path = output_dir / "answer_change_comparison.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"  詳細結果を保存しました: {json_path}")

    # サマリーを表示
    print("\n" + "=" * 60)
    print("回答一致率サマリー (Original vs Early Answer)")
    print("=" * 60)
    print(f"{'Folder':<10} {'Match Rate':>15} {'Matched/Total':>20}")
    print("-" * 60)
    for folder in folder_names:
        stats = folder_results[folder]
        print(f"{folder:<10} {stats['match_rate']:>14.2f}% "
              f"{stats['matched']:>8}/{stats['total']:<8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
