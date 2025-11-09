"""
JSON Lines形式の結果ファイルを読み込むためのユーティリティスクリプト

使用例:
    python utils/read_results.py results/mmlu_results_20251030_163248.jsonl
    python utils/read_results.py results/  # フォルダ指定で全ファイル処理
"""

import json
import sys
import pandas as pd
from pathlib import Path


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


def get_jsonl_files(path: str) -> list:
    """
    パスがディレクトリの場合は全ての.jsonlファイルを取得、
    ファイルの場合はそのファイルを返す

    Args:
        path: ファイルまたはディレクトリのパス

    Returns:
        list: .jsonlファイルのパスのリスト
    """
    path_obj = Path(path)

    if path_obj.is_dir():
        # ディレクトリの場合、全ての.jsonlファイルを取得
        return sorted(path_obj.glob("*.jsonl"))
    elif path_obj.is_file():
        # ファイルの場合、そのファイルを返す
        return [path_obj]
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python utils/read_results.py <jsonl_file_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    # .jsonlファイルのリストを取得
    jsonl_files = get_jsonl_files(input_path)

    if not jsonl_files:
        print(f"No .jsonl files found in: {input_path}")
        sys.exit(1)

    print(f"\nFound {len(jsonl_files)} file(s) to process")

    # 全ファイルのデータを結合
    all_dfs = []
    file_stats = []

    for filepath in jsonl_files:
        print(f"Reading: {filepath.name}")
        df = read_jsonl(str(filepath))
        df['source_file'] = filepath.name  # ファイル名を記録
        all_dfs.append(df)

        # ファイルごとの統計を記録
        file_stats.append({
            'file': filepath.name,
            'total': len(df),
            'correct': df['is_correct'].sum(),
            'accuracy': df['is_correct'].mean(),
            'avg_time': df['ai_time_seconds'].mean()
        })

    # 全データを結合
    df = pd.concat(all_dfs, ignore_index=True)

    # 複数ファイルの場合、ファイル別の統計を表示
    if len(jsonl_files) > 1:
        print("\nResults by file:")
        file_stats_df = pd.DataFrame(file_stats)
        print(file_stats_df[['file', 'accuracy', 'avg_time']].to_string(index=False))
        print()

    # 全体の結果
    print(f"Overall Accuracy: {df['is_correct'].mean():.3f}")
    print(f"Average Time: {df['ai_time_seconds'].mean():.2f}s")


if __name__ == "__main__":
    main()
