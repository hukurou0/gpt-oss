"""
run_mmlu.pyで生成された結果を使ってearly_answerで再評価するスクリプト

使用例:
    python run_mmlu_early_answer.py
"""

from dataset.mmlu.early_answer_evaluate import main
from model.early_answer import generate_early_answer

# -------- パラメータ設定 --------
RESULTS_FILE = "results/mmlu_results_20251030_163851.jsonl"  # 入力結果ファイル
ANALYSIS_PERCENTAGE = 0.9  # 使用するanalysisの割合（0.0 ~ 1.0）

if __name__ == "__main__":
    print(f"Results file: {RESULTS_FILE}")
    print(f"Using {ANALYSIS_PERCENTAGE*100}% of analysis")

    # generate関数をラップしてanalysis_percentageを渡す
    def generate_wrapper(prompt: str, analysis: str):
        return generate_early_answer(prompt, analysis, ANALYSIS_PERCENTAGE)

    # main関数を実行
    main(generate_wrapper, results_file=RESULTS_FILE)