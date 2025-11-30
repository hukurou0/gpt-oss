from dataset.mmlu.original_evaluate import main
from model.call_gpt_oss import generate

# ========== 設定 ==========
# 特定のsubjectから開始する（None の場合は最初から）
# 例: "anatomy", "college_chemistry" など
START_FROM = None

# 特定のインデックスから開始する（0ベース、None の場合は最初から）
# 例: 10 → 11番目のsubjectから開始
START_INDEX = None

# 結果を保存するディレクトリ
OUTPUT_DIR = "results/original/mmlu"
# ==========================

if __name__ == "__main__":
    main(generate,
         start_from=START_FROM,
         start_index=START_INDEX,
         output_dir=OUTPUT_DIR)