import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

"""
paraphrased_analysisを使って最終回答を取得するスクリプト

results/mmlu/paraphrase_batch/results/ のバッチ結果を直接読み込んで、
最終解答を生成します。

使用例:
    python run_mmlu_paraphrased.py
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from model.experiment_call import generate_paraphrased_answer
from dataset.mmlu.experiment_evaluate import eval
from logs.utils.logger import setup_logger
from logs.utils.result_saver import ResultSaver

# -------- パラメータ設定 --------
BATCH_RESULTS_DIR = "results/mmlu/paraphrase_batch/results"  # バッチ結果ディレクトリ（入力）
OUTPUT_DIR = "results/mmlu/paraphrased_answer"  # 出力ディレクトリ
DATA_DIR = "dataset/mmlu/data"  # MMLUデータディレクトリ


def load_paraphrased_analysis_map(batch_result_file: str, subject: str) -> dict:
    """
    バッチ結果ファイルからparaphrased_analysisマップを作成

    Args:
        batch_result_file: バッチ結果のJSON Linesファイルのパス
        subject: サブジェクト名

    Returns:
        dict: (subject, question_number) -> paraphrased_analysis のマップ
    """
    analysis_map = {}
    with open(batch_result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]  # e.g., "abstract_algebra_q1"

            # custom_idからquestion_numberを抽出
            # format: {subject}_q{number}
            if custom_id.startswith(f"{subject}_q"):
                question_number = int(custom_id.split("_q")[-1])

                if data.get("error"):
                    continue

                response = data.get("response", {})
                if response.get("status_code") == 200:
                    content = response["body"]["choices"][0]["message"]["content"]
                    key = (subject, question_number)
                    analysis_map[key] = content

    return analysis_map


def process_subject(subject: str, batch_result_file: str, generate_func, output_dir: str, logger):
    """
    単一のsubjectを処理する

    Args:
        subject: サブジェクト名
        batch_result_file: バッチ結果ファイルのパス
        generate_func: generate関数
        output_dir: 出力ディレクトリ
        logger: ロガー

    Returns:
        tuple: (cors, acc, subject_time, ai_times, result_filepath)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing subject: {subject}")
    logger.info(f"{'='*60}")

    # バッチ結果からparaphrased_analysis_mapをロード
    logger.info(f"Loading paraphrased analysis from: {batch_result_file}")
    analysis_map = load_paraphrased_analysis_map(batch_result_file, subject)
    logger.info(f"Loaded {len(analysis_map)} paraphrased analyses")

    # 結果保存用のインスタンスを作成（subject別）
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{subject}.jsonl")
    result_saver = ResultSaver(output_dir=output_dir)
    logger.info(f"Results will be saved to: {output_file}")

    # データをロード
    dev_df = pd.read_csv(os.path.join(DATA_DIR, "dev", f"{subject}_dev.csv"), header=None)[:5]
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test", f"{subject}_test.csv"), header=None)

    # 評価を実行
    cors, acc, subject_time, ai_times = eval(
        generate_func, subject, dev_df, test_df, result_saver, analysis_map
    )

    logger.info(f"Subject {subject} completed: accuracy={acc:.3f}, time={subject_time:.2f}s")

    return cors, acc, subject_time, ai_times, output_file


if __name__ == "__main__":
    # ロガーの初期化
    logger = setup_logger()

    logger.info(f"Batch results directory: {BATCH_RESULTS_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # results/mmlu/paraphrase_batch/results ディレクトリ内の全.jsonlファイルを取得
    results_dir = Path(BATCH_RESULTS_DIR)
    if not results_dir.exists():
        logger.error(f"Directory not found: {BATCH_RESULTS_DIR}")
        raise FileNotFoundError(f"Directory not found: {BATCH_RESULTS_DIR}")

    # エラーファイルを除外
    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    jsonl_files = [f for f in jsonl_files if not f.name.endswith("_errors.jsonl")]
    logger.info(f"Found {len(jsonl_files)} subject files to process")

    if len(jsonl_files) == 0:
        logger.error("No .jsonl files found in the directory")
        raise FileNotFoundError("No .jsonl files found")

    # generate関数: paraphrased_analysisを使用
    def generate_wrapper(prompt: str, analysis: str):
        return generate_paraphrased_answer(prompt, analysis)

    # 全体の統計を記録
    all_cors = []
    all_subject_times = []
    all_ai_times = []
    processed_subjects = []

    total_start_time = time.time()

    # 各ファイルを処理
    for jsonl_file in jsonl_files:
        subject = jsonl_file.stem  # ファイル名から拡張子を除いた部分

        try:
            cors, acc, subject_time, ai_times, output_file = process_subject(
                subject, str(jsonl_file), generate_wrapper, OUTPUT_DIR, logger
            )

            all_cors.append(cors)
            all_subject_times.append(subject_time)
            all_ai_times.extend(ai_times)
            processed_subjects.append(subject)

            logger.info(f"Saved results to: {output_file}")

        except Exception as e:
            logger.error(f"Error processing subject {subject}: {str(e)}")
            # エラーが発生しても次のsubjectを処理
            continue

    total_elapsed_time = time.time() - total_start_time

    # サマリーを出力
    if len(all_cors) > 0:
        weighted_acc = np.mean(np.concatenate(all_cors))

        separator = "="*60
        logger.info("\n" + separator)
        logger.info("SUMMARY (PARAPHRASED ANALYSIS)")
        logger.info(separator)
        logger.info(f"Processed subjects: {len(processed_subjects)}/{len(jsonl_files)}")
        logger.info(f"Average accuracy: {weighted_acc:.3f}")
        logger.info(f"Total execution time: {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f}min)")
        logger.info(f"Average time per subject: {np.mean(all_subject_times):.2f}s")
        logger.info(f"Average time per AI call: {np.mean(all_ai_times):.2f}s")
        logger.info(f"Total AI calls: {len(all_ai_times)}")
        logger.info(separator)

        print("\n" + separator)
        print("SUMMARY (PARAPHRASED ANALYSIS)")
        print(separator)
        print(f"Processed subjects: {len(processed_subjects)}/{len(jsonl_files)}")
        print(f"Average accuracy: {weighted_acc:.3f}")
        print(f"Total execution time: {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f}min)")
        print(f"Average time per subject: {np.mean(all_subject_times):.2f}s")
        print(f"Average time per AI call: {np.mean(all_ai_times):.2f}s")
        print(f"Total AI calls: {len(all_ai_times)}")
        print(separator)
        print(f"\nResults saved to: {OUTPUT_DIR}/")
    else:
        logger.error("No subjects were successfully processed")
        print("No subjects were successfully processed")
