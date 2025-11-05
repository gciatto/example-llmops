"""
Generate answers for quiz questions using OpenAI API.

This script loads questions from a CSV file, optionally enriches them with
web search results, and generates answers using OpenAI's API with a 
configurable prompt template.
"""
import argparse
import pandas as pd
from openai import OpenAI
import mlflow
from utils import *
import tempfile


def generate_answer(
    client: OpenAI,
    question: str,
    category: str,
    weight: int,
    prompt_template_name: str,
    model: str = "gpt-4o-mini",
    use_search: bool = False,
    search_results_count: int = 3
) -> str:
    """
    Generate an answer for a single question.
    
    Args:
        client: OpenAI client instance
        question: Question text
        category: Question category
        weight: Question weight/difficulty
        prompt_template_name: Name of the prompt template file (without extension)
        model: OpenAI model to use
        use_search: Whether to enrich with web search
        search_results_count: Number of search results to include
        
    Returns:
        Generated answer text
    """
    # Optionally get search results
    search_context = ""
    if use_search:
        results = search_web(question, max_results=search_results_count)
        search_context = format_search_results(results)

    prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_template_name}@latest")
    
    # Format the prompt
    prompt = prompt.template.format(
        category=category,
        question=question,
        weight=weight,
        search_results=search_context
    )
    
    # Call OpenAI API
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
        description="Generate answers for quiz questions using OpenAI"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="basic",
        help="Name of the prompt template file inside `prompts/`, without extension (e.g., 'basic' for `basic.txt`)"
    )
    parser.add_argument(
        "--use-search",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to use web search for enrichment"
    )
    parser.add_argument(
        "--search-results-count",
        type=int,
        default=3,
        help="Number of search results to include"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=-1,
        help="Maximum number of questions to process (-1 for all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    
    args = parser.parse_args()
    use_search = args.use_search.lower() == "true"
    
    # Start MLflow run
    with mlflow.start_run(run_name="generate_answers"):
        mlflow.set_tag("mlflow.runName", "generate_answers")

        mlflow.autolog()
        
        # Initialize OpenAI client
        api_key = get_openai_api_key()
        client = OpenAI(api_key=api_key)
        
        # Load questions
        questions_df = load_questions()
        
        # Limit questions if specified
        if args.max_questions > 0:
            questions_df = questions_df.head(args.max_questions)
        
        mlflow.log_metric("total_questions", len(questions_df))
        
        # Generate answers
        answers = []
        print(f"Generating answers for {len(questions_df)} questions...")
        
        for idx, row in questions_df.iterrows():
            print(f"Processing question {idx + 1}/{len(questions_df)}: {row['Question'][:50]}...")
            
            try:
                answer = generate_answer(
                    client=client,
                    question=row['Question'],
                    category=row['Category'],
                    weight=row['Weight'],
                    prompt_template_name=args.prompt_template,
                    model=args.model,
                    use_search=use_search,
                    search_results_count=args.search_results_count
                )
                
                answers.append({
                    'Category': row['Category'],
                    'Question': row['Question'],
                    'Weight': row['Weight'],
                    'Answer': answer
                })
                
            except Exception as e:
                print(f"Error processing question: {e}")
                answers.append({
                    'Category': row['Category'],
                    'Question': row['Question'],
                    'Weight': row['Weight'],
                    'Answer': f"Error: {str(e)}"
                })
        
        # Save results
        results_df = pd.DataFrame(answers)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            temp_output_path = tmp_dir / "answers.csv"
            results_df.to_csv(temp_output_path, index=False)
            print(f"\nAnswers saved to {temp_output_path}")

            # Log artifact
            mlflow.log_artifact(temp_output_path)
            # Print artifact URI on MLflow server
            print(f"Answers artifact logged at: {mlflow.get_artifact_uri('answers.csv')}")
            mlflow.log_metric("successful_generations", len([a for a in answers if not a['Answer'].startswith('Error:')]))
            mlflow.log_metric("total_generations", len(answers))
            mlflow.log_metric("failed_generations", len([a for a in answers if a['Answer'].startswith('Error:')]))
            mlflow.log_metric("success_rate", len([a for a in answers if not a['Answer'].startswith('Error:')]) / len(answers) if len(answers) > 0 else 0)
        
        print("MLflow run completed!")


if __name__ == "__main__":
    main()
