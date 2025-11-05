"""
Evaluate generated answers using various metrics.

This script loads generated answers and evaluates them using simple metrics
like length, keyword presence, and structure.
"""
import argparse
import pandas as pd
import mlflow
import re
from typing import Dict


def evaluate_answer(answer: str, question: str, category: str, weight: int) -> Dict[str, float]:
    """
    Evaluate a single answer using various metrics.
    
    Args:
        answer: Generated answer text
        question: Original question
        category: Question category
        weight: Question weight/difficulty
        
    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}
    
    # Length metrics
    metrics['answer_length'] = len(answer)
    metrics['word_count'] = len(answer.split())
    metrics['sentence_count'] = len(re.split(r'[.!?]+', answer))
    
    # Check for error
    metrics['is_error'] = 1.0 if answer.startswith('Error:') else 0.0
    
    # Check if answer contains question keywords (simple relevance check)
    question_keywords = set(re.findall(r'\b\w{4,}\b', question.lower()))
    answer_keywords = set(re.findall(r'\b\w{4,}\b', answer.lower()))
    common_keywords = question_keywords.intersection(answer_keywords)
    metrics['keyword_overlap'] = len(common_keywords) / max(len(question_keywords), 1)
    
    # Structure checks
    metrics['has_multiple_sentences'] = 1.0 if metrics['sentence_count'] > 1 else 0.0
    metrics['is_detailed'] = 1.0 if metrics['word_count'] >= 50 else 0.0
    metrics['is_very_detailed'] = 1.0 if metrics['word_count'] >= 100 else 0.0
    
    # Category-specific expectations (simple heuristics)
    if category == "Definition":
        # Definitions should be clear and concise
        metrics['meets_category_expectation'] = 1.0 if 20 <= metrics['word_count'] <= 150 else 0.5
    elif category == "Commonsense":
        # Commonsense should be explanatory
        metrics['meets_category_expectation'] = 1.0 if metrics['word_count'] >= 30 else 0.5
    else:
        metrics['meets_category_expectation'] = 1.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers"
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default="answers.csv",
        help="CSV file with generated answers"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation.csv",
        help="Output CSV file for evaluation results"
    )
    
    args = parser.parse_args()
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("answers_file", args.answers_file)
        
        # Load answers
        answers_df = pd.read_csv(args.answers_file)
        print(f"Loaded {len(answers_df)} answers from {args.answers_file}")
        
        # Evaluate each answer
        evaluations = []
        
        for idx, row in answers_df.iterrows():
            metrics = evaluate_answer(
                answer=row['Answer'],
                question=row['Question'],
                category=row['Category'],
                weight=row['Weight']
            )
            
            evaluation = {
                'Category': row['Category'],
                'Question': row['Question'],
                'Weight': row['Weight'],
                **metrics
            }
            evaluations.append(evaluation)
        
        # Create evaluation DataFrame
        eval_df = pd.DataFrame(evaluations)
        eval_df.to_csv(args.output_file, index=False)
        print(f"\nEvaluation results saved to {args.output_file}")
        
        # Log aggregate metrics
        mlflow.log_metric("avg_answer_length", eval_df['answer_length'].mean())
        mlflow.log_metric("avg_word_count", eval_df['word_count'].mean())
        mlflow.log_metric("avg_sentence_count", eval_df['sentence_count'].mean())
        mlflow.log_metric("error_rate", eval_df['is_error'].mean())
        mlflow.log_metric("avg_keyword_overlap", eval_df['keyword_overlap'].mean())
        mlflow.log_metric("detailed_answer_rate", eval_df['is_detailed'].mean())
        mlflow.log_metric("category_expectation_score", eval_df['meets_category_expectation'].mean())
        
        # Log artifact
        mlflow.log_artifact(args.output_file)
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Average word count: {eval_df['word_count'].mean():.1f}")
        print(f"Average keyword overlap: {eval_df['keyword_overlap'].mean():.2%}")
        print(f"Detailed answers: {eval_df['is_detailed'].mean():.2%}")
        print(f"Error rate: {eval_df['is_error'].mean():.2%}")
        print(f"Category expectation score: {eval_df['meets_category_expectation'].mean():.2%}")
        
        print("\nMLflow run completed!")


if __name__ == "__main__":
    main()
