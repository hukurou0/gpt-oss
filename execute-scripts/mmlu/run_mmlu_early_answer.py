import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

"""
run_mmlu.pyで生成された結果を使ってearly_answerで再評価するスクリプト

使用例:
    python run_mmlu_early_answer.py
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from model.experiment_call import generate_early_answer
from dataset.mmlu.experiment_evaluate import eval, load_analysis_map
from logs.utils.logger import setup_logger, get_logger
from logs.utils.result_saver import ResultSaver

# -------- パラメータ設定 --------
ORIGINAL_RESULTS_DIR = "results/mmlu/original"  # 入力結果ディレクトリ
OUTPUT_BASE_DIR = "results/mmlu/early_answer"  # 出力ベースディレクトリ
ANALYSIS_PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 使用するanalysisの割合リスト
DATA_DIR = "dataset/mmlu/data"  # MMLUデータディレクトリ

def process_subject(subject: str, results_file: str, generate_func, output_dir: str, logger):
    """
    単一のsubjectを処理する

    Args:
        subject: サブジェクト名
        results_file: 元の結果ファイルのパス
        generate_func: generate関数
        output_dir: 出力ディレクトリ
        logger: ロガー

    Returns:
        tuple: (cors, acc, subject_time, ai_times, result_filepath)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing subject: {subject}")
    logger.info(f"{'='*60}")

    # analysis_mapをロード
    logger.info(f"Loading analysis from: {results_file}")
    analysis_map = load_analysis_map(results_file)
    logger.info(f"Loaded {len(analysis_map)} analyses")

    # 結果保存用のインスタンスを作成（subject別）
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{subject}.jsonl")
    result_saver = ResultSaver(output_dir=output_dir, filename=f"{subject}.jsonl")
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

    logger.info(f"Original results directory: {ORIGINAL_RESULTS_DIR}")
    logger.info(f"Output base directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Analysis percentages: {ANALYSIS_PERCENTAGES}")

    # results/original/mmlu ディレクトリ内の全.jsonlファイルを取得
    results_dir = Path(ORIGINAL_RESULTS_DIR)
    if not results_dir.exists():
        logger.error(f"Directory not found: {ORIGINAL_RESULTS_DIR}")
        raise FileNotFoundError(f"Directory not found: {ORIGINAL_RESULTS_DIR}")

    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} subject files to process")

    if len(jsonl_files) == 0:
        logger.error("No .jsonl files found in the directory")
        raise FileNotFoundError("No .jsonl files found")

    # 全パーセンテージの統計を記録
    all_percentage_results = {}

    grand_total_start_time = time.time()

    # analysis_percentageごとに処理
    for analysis_percentage in ANALYSIS_PERCENTAGES:
        percentage_int = int(analysis_percentage * 10)  # 0.1 -> 1, 0.9 -> 9
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"{percentage_int:02d}")

        logger.info(f"\n{'#'*80}")
        logger.info(f"Starting processing with analysis_percentage={analysis_percentage:.1f} ({percentage_int:02d})")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'#'*80}")

        # generate関数をラップしてanalysis_percentageを渡す
        def generate_wrapper(prompt: str, analysis: str):
            return generate_early_answer(prompt, analysis, analysis_percentage)

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
                    subject, str(jsonl_file), generate_wrapper, output_dir, logger
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

        # 各percentageごとのサマリーを出力
        if len(all_cors) > 0:
            weighted_acc = np.mean(np.concatenate(all_cors))

            separator = "="*60
            logger.info("\n" + separator)
            logger.info(f"SUMMARY FOR ANALYSIS_PERCENTAGE={analysis_percentage:.1f}")
            logger.info(separator)
            logger.info(f"Processed subjects: {len(processed_subjects)}/{len(jsonl_files)}")
            logger.info(f"Average accuracy: {weighted_acc:.3f}")
            logger.info(f"Total execution time: {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f}min)")
            logger.info(f"Average time per subject: {np.mean(all_subject_times):.2f}s")
            logger.info(f"Average time per AI call: {np.mean(all_ai_times):.2f}s")
            logger.info(f"Total AI calls: {len(all_ai_times)}")
            logger.info(separator)

            print("\n" + separator)
            print(f"SUMMARY FOR ANALYSIS_PERCENTAGE={analysis_percentage:.1f}")
            print(separator)
            print(f"Processed subjects: {len(processed_subjects)}/{len(jsonl_files)}")
            print(f"Average accuracy: {weighted_acc:.3f}")
            print(f"Total execution time: {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f}min)")
            print(f"Average time per subject: {np.mean(all_subject_times):.2f}s")
            print(f"Average time per AI call: {np.mean(all_ai_times):.2f}s")
            print(f"Total AI calls: {len(all_ai_times)}")
            print(separator)
            print(f"\nResults saved to: {output_dir}/")

            # 統計を保存
            all_percentage_results[analysis_percentage] = {
                'accuracy': weighted_acc,
                'processed_subjects': len(processed_subjects),
                'total_time': total_elapsed_time
            }
        else:
            logger.error(f"No subjects were successfully processed for percentage {analysis_percentage:.1f}")
            print(f"No subjects were successfully processed for percentage {analysis_percentage:.1f}")

    grand_total_elapsed_time = time.time() - grand_total_start_time

    # 全体の最終サマリーを出力
    if all_percentage_results:
        separator = "="*80
        logger.info("\n\n" + separator)
        logger.info("FINAL OVERALL SUMMARY")
        logger.info(separator)

        print("\n\n" + separator)
        print("FINAL OVERALL SUMMARY")
        print(separator)

        for percentage, results in sorted(all_percentage_results.items()):
            percentage_int = int(percentage * 10)
            summary_line = f"  {percentage_int:02d} (p={percentage:.1f}): Accuracy={results['accuracy']:.3f}, Subjects={results['processed_subjects']}, Time={results['total_time']:.1f}s"
            logger.info(summary_line)
            print(summary_line)

        logger.info(separator)
        logger.info(f"Grand total execution time: {grand_total_elapsed_time:.2f}s ({grand_total_elapsed_time/60:.2f}min)")
        logger.info(separator)

        print(separator)
        print(f"Grand total execution time: {grand_total_elapsed_time:.2f}s ({grand_total_elapsed_time/60:.2f}min)")
        print(separator)
    else:
        logger.error("No results were successfully processed")
        print("No results were successfully processed")