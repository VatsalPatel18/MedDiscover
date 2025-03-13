from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_response(ground_truth, response):
    """
    Evaluate an LLM response against a ground truth using ROUGE and BLEU scores.
    
    Parameters:
        ground_truth (str): The expected answer.
        response (str): The LLM-generated answer.
        
    Returns:
        dict: A dictionary with ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores.
    """
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing_function = SmoothingFunction().method4
    
    rouge_scores = rouge_scorer_instance.score(ground_truth, response)
    bleu_score = sentence_bleu([ground_truth.split()], response.split(), smoothing_function=smoothing_function)
    return {
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "bleu": bleu_score
    }
