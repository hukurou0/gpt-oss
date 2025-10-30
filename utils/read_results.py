"""
JSON Lines形式の結果ファイルを読み込むためのユーティリティスクリプト

使用例:
    python utils/read_results.py results/mmlu_results_20251030_163248.jsonl
"""

import json
import sys
import pandas as pd


def read_jsonl(filepath: str) -> pd.DataFrame:
    """
    JSON Lines形式のファイルを読み込んでDataFrameに変換

    Args:
        filepath: .jsonlファイルのパス

    Returns:
        pd.DataFrame: 結果のデータフレーム
    """
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 空行をスキップ
                results.append(json.loads(line))

    return pd.DataFrame(results)


def main():
    if len(sys.argv) < 2:
        print("Usage: python utils/read_results.py <jsonl_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    # データを読み込み
    df = read_jsonl(filepath)

    # 基本統計を表示
    print("="*60)
    print("Results Summary")
    print("="*60)
    print(f"Total questions: {len(df)}")
    print(f"Correct answers: {df['is_correct'].sum()}")
    print(f"Accuracy: {df['is_correct'].mean():.3f}")
    print(f"Average AI time: {df['ai_time_seconds'].mean():.2f}s")
    print("="*60)

    # サブジェクト別の統計
    print("\nAccuracy by subject:")
    subject_stats = df.groupby('subject').agg({
        'is_correct': ['sum', 'count', 'mean']
    }).round(3)
    print(subject_stats)

    # 最初の5行を表示
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
