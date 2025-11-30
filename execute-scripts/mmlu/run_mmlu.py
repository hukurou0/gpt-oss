import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataset.mmlu.original_evaluate import main
from model.call_gpt_oss import generate

# ========== 設定 ==========
# 特定のsubjectから開始する（None の場合は最初から）
# 例: "anatomy", "college_chemistry" など
START_SUBJECT = "college_biology"

# 特定の問題番号から開始する（1ベース、1 の場合は最初から）
# 例: 145 → 145問目から開始
START_QUESTION = 145

# 結果を保存するディレクトリ
OUTPUT_DIR = "results/mmlu/original"
# ==========================

if __name__ == "__main__":
    main(generate,
         start_subject=START_SUBJECT,
         start_question=START_QUESTION,
         output_dir=OUTPUT_DIR)