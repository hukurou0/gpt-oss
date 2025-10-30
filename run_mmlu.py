from dataset.mmlu.original_evaluate import main
from model.call_gpt_oss import generate

if __name__ == "__main__":
    main(generate)