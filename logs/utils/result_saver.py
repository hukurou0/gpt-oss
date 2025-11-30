import json
import os
from typing import List


class ResultSaver:
    """MMLU評価結果を科目ごとのJSON Lines形式で逐次保存するクラス"""

    def __init__(self, output_dir: str = "results/mmlu/original"):
        """
        Args:
            output_dir: 結果を保存するディレクトリ
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 現在書き込み中の科目
        self.current_subject = None

    def add_result(
        self,
        subject: str,
        question_number: int,
        question: str,
        choices: List[str],
        analysis: str,
        predicted_answer: str,
        correct_answer: str,
        is_correct: bool,
        ai_time: float
    ):
        """
        1つの質問の結果を科目ごとのJSON Linesファイルに即座に書き込む

        Args:
            subject: サブジェクト名
            question_number: 質問番号
            question: 質問文（保存しない）
            choices: 選択肢のリスト（保存しない）
            analysis: モデルの分析
            predicted_answer: モデルの予測答え
            correct_answer: 正解
            is_correct: 正解かどうか
            ai_time: AI処理時間（秒）
        """
        result = {
            "subject": subject,
            "question_number": question_number,
            "analysis": analysis,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "ai_time_seconds": round(ai_time, 2)
        }

        # 科目ごとのJSONLファイルパス
        subject_filepath = os.path.join(self.output_dir, f"{subject}.jsonl")

        # JSON Lines形式で追記（1行1JSON）
        with open(subject_filepath, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

        self.current_subject = subject

    def get_subject_filepath(self, subject: str) -> str:
        """科目のファイルパスを取得"""
        return os.path.join(self.output_dir, f"{subject}.jsonl")
