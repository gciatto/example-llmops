"""
Compare different prompt templates and log results to MLflow.

This script runs answer generation with multiple prompt templates,
evaluates the results, and logs everything to MLflow for comparison.
"""
import argparse
import pandas as pd
import mlflow
from openai import OpenAI
import tempfile
from pathlib import Path
from evaluate_responses import evaluate_answer
from utils import (
    load_questions,
    search_web,
    format_search_results,
    load_prompt_template,
    get_openai_api_key
)


def generate_answer_with_prompt(
    client: OpenAI,
    question: str,
    category: str,
    weight: int,
    prompt_template: str,
    model: str,
    use_search: bool,
    search_results_count: int
) -> str:
    """Generate an answer using specified prompt template."""
    search_context = ""
    if use_search:
        results = search_web(question, max_results=search_results_count)
        search_context = format_search_results(results)
    
    prompt = prompt_template.format(
        category=category,
        question=question,
        weight=weight,
        search_results=search_context
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in software engineering education."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description="Compare different prompt templates"
    )
    parser.add_argument(
        "--prompt-templates",
        type=str,
        required=True,
        help="Comma-separated list of prompt template files"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Maximum number of questions to process"
    )
    parser.add_argument(
        "--use-search",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to use web search"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    
    args = parser.parse_args()
    use_search = args.use_search.lower() == "true"
    
    # Parse prompt templates
    template_paths = [t.strip() for t in args.prompt_templates.split(',')]
    
    # Initialize OpenAI client
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    
    # Load questions (limited subset for comparison)
    questions_df = load_questions()
    if args.max_questions > 0:
        questions_df = questions_df.head(args.max_questions)
    
    print(f"Comparing {len(template_paths)} prompts on {len(questions_df)} questions...\n")
    
    # Parent run for comparison
    with mlflow.start_run(run_name="prompt_comparison"):
        mlflow.log_param("num_prompts", len(template_paths))
        mlflow.log_param("num_questions", len(questions_df))
        mlflow.log_param("model", args.model)
        mlflow.log_param("use_search", use_search)
        
        results_summary = []
        
        # Test each prompt template
        for template_path in template_paths:
            template_name = Path(template_path).stem
            print(f"\n{'='*60}")
            print(f"Testing prompt: {template_name}")
            print(f"{'='*60}")
            
            # Child run for this prompt
            with mlflow.start_run(run_name=f"prompt_{template_name}", nested=True):
                mlflow.log_param("prompt_template", template_path)
                mlflow.log_param("model", args.model)
                mlflow.log_param("use_search", use_search)
                
                # Load template
                prompt_template = load_prompt_template(template_path)
                mlflow.log_text(prompt_template, "prompt_template.txt")
                
                # Generate answers
                answers = []
                for idx, row in questions_df.iterrows():
                    try:
                        answer = generate_answer_with_prompt(
                            client=client,
                            question=row['Question'],
                            category=row['Category'],
                            weight=row['Weight'],
                            prompt_template=prompt_template,
                            model=args.model,
                            use_search=use_search,
                            search_results_count=3
                        )
                        
                        answers.append({
                            'Category': row['Category'],
                            'Question': row['Question'],
                            'Weight': row['Weight'],
                            'Answer': answer
                        })
                    except Exception as e:
                        print(f"Error: {e}")
                        answers.append({
                            'Category': row['Category'],
                            'Question': row['Question'],
                            'Weight': row['Weight'],
                            'Answer': f"Error: {str(e)}"
                        })
                
                # Evaluate answers
                evaluations = []
                for answer_data in answers:
                    metrics = evaluate_answer(
                        answer=answer_data['Answer'],
                        question=answer_data['Question'],
                        category=answer_data['Category'],
                        weight=answer_data['Weight']
                    )
                    evaluations.append(metrics)
                
                eval_df = pd.DataFrame(evaluations)
                
                # Log metrics
                avg_word_count = eval_df['word_count'].mean()
                avg_keyword_overlap = eval_df['keyword_overlap'].mean()
                error_rate = eval_df['is_error'].mean()
                detailed_rate = eval_df['is_detailed'].mean()
                category_score = eval_df['meets_category_expectation'].mean()
                
                mlflow.log_metric("avg_word_count", avg_word_count)
                mlflow.log_metric("avg_keyword_overlap", avg_keyword_overlap)
                mlflow.log_metric("error_rate", error_rate)
                mlflow.log_metric("detailed_answer_rate", detailed_rate)
                mlflow.log_metric("category_expectation_score", category_score)
                
                # Save results
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    answers_df = pd.DataFrame(answers)
                    answers_df.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, f"answers_{template_name}.csv")
                    Path(f.name).unlink()
                
                # Print summary
                print(f"\nResults for {template_name}:")
                print(f"  Avg word count: {avg_word_count:.1f}")
                print(f"  Keyword overlap: {avg_keyword_overlap:.2%}")
                print(f"  Detailed answers: {detailed_rate:.2%}")
                print(f"  Error rate: {error_rate:.2%}")
                print(f"  Category score: {category_score:.2%}")
                
                # Store for comparison
                results_summary.append({
                    'prompt': template_name,
                    'avg_word_count': avg_word_count,
                    'keyword_overlap': avg_keyword_overlap,
                    'detailed_rate': detailed_rate,
                    'error_rate': error_rate,
                    'category_score': category_score
                })
        
        # Log comparison summary
        summary_df = pd.DataFrame(results_summary)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            summary_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "comparison_summary.csv")
            Path(f.name).unlink()
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print("\nComparison completed! Check MLflow UI for detailed results.")


if __name__ == "__main__":
    main()
