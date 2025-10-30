from model.call_gpt_oss import create_harmony_prompt, run_llm, parse_llm_output


# グローバル変数でanalysis_percentageを保持
_global_analysis_percentage = 0.5


def set_analysis_percentage(percentage: float):
    """
    グローバルなanalysis_percentageを設定

    Args:
        percentage: 使用する割合（0.0 ~ 1.0）
    """
    global _global_analysis_percentage
    _global_analysis_percentage = percentage


def _truncate_analysis(analysis: str, percentage: float) -> str:
    """
    analysisの前半X%を切り出す

    Args:
        analysis: 元のanalysis文字列
        percentage: 使用する割合（0.0 ~ 1.0）

    Returns:
        str: 切り出されたanalysis
    """
    if percentage <= 0:
        return ""
    if percentage >= 1.0:
        return analysis

    # 文字数ベースで切り出し
    target_length = int(len(analysis) * percentage)
    return analysis[:target_length]


def _add_cut_assistant_message(prompt: str, analysis: str):
    """
    プロンプトにanalysisを追加してfinalチャンネルに続くようにする

    Args:
        prompt: 元のプロンプト
        analysis: 追加するanalysis（既に切り出し済み）

    Returns:
        str: 構築されたプロンプト
    """
    return prompt + "<|end|><|start|>assistant<|channel|>analysis<|message|>" + analysis + "<|end|><|start|>assistant<|channel|>final<|message|>"


def generate_early_answer(prompt: str, analysis: str, analysis_percentage: float = None) -> dict:
    """
    analysisの一部を使って答えを生成する

    Args:
        prompt: ユーザープロンプト
        analysis: 既存のanalysis
        analysis_percentage: analysisの前半何%を使用するか（0.0 ~ 1.0）
                           Noneの場合はグローバル設定を使用
                           例: 0.5 = 50%, 1.0 = 100%

    Returns:
        dict: parse_llm_outputの結果
    """
    # percentageが指定されていない場合はグローバル設定を使用
    if analysis_percentage is None:
        analysis_percentage = _global_analysis_percentage

    # analysisを指定された割合で切り出し
    truncated_analysis = _truncate_analysis(analysis, analysis_percentage)

    harmony_prompt = create_harmony_prompt(prompt)
    input_prompt = _add_cut_assistant_message(harmony_prompt, truncated_analysis)
    output = run_llm(input_prompt)
    return output[0]