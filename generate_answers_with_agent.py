"""
Generate answers for quiz questions using LangChain and OpenAI API.

This script loads questions from a CSV file and generates answers using 
LangChain's API with configurable prompt templates and optional tool usage.
"""
import argparse
import pandas as pd
import mlflow
from utils import *
import tempfile
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for information related to a query. Use this when you need additional context or references to answer the question accurately.
    
    Args:
        query: The search query to look up on the web
        max_results: Maximum number of search results to return (default: 3)
    
    Returns:
        Formatted string with search results including titles, snippets, and URLs
    """
    return web_search_tool(query, max_results)


def generate_answer(
    question: str,
    category: str,
    weight: int,
    prompt_template_name: str,
    model: str = "gpt-4o-mini",
    search_results_count: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """
    Generate an answer for a single question using LangChain.
    
    Args:
        question: Question text
        category: Question category
        weight: Question weight/difficulty
        prompt_template_name: Name of the prompt template file (without extension)
        model: OpenAI model to use
        search_results_count: Number of search results to include when tool is called
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens in generated response
        
    Returns:
        Generated answer text
    """

    # Load prompt template
    prompt_text = mlflow.genai.load_prompt(f"prompts:/{prompt_template_name}@latest")
    
    # Format the base prompt
    formatted_prompt = prompt_text.format(
        category=category,
        question=question,
        weight=weight,
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=get_openai_api_key()
    )
    
    if search_results_count > 0:
        # Create agent with tools
        tools = [web_search]
        
        # Create prompt template for agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
                "You are an expert in software engineering education.\n"
                "Use the provided web search tool to find relevant information when needed."
            ),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True
        )
        
        # Invoke agent
        result = agent_executor.invoke({
            "input": formatted_prompt,
            "max_results": search_results_count
        })
        
        return result["output"]
    else:
        # Direct LLM call without tools
        messages = [
            ("system", "You are an expert in software engineering education."),
            ("human", formatted_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers for quiz questions using LangChain and OpenAI"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="basic",
        help="Name of the prompt template file inside `prompts/`, without extension (e.g., 'basic' for `basic.txt`)"
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
        help="Maximum number of questions to process (-1 for unlimited)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (0.0 to 1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens in generated responses"
    )
    
    args = parser.parse_args()
    
    # Start MLflow run
    with mlflow.start_run(run_name="generate_answers_with_agent") as run:
        mlflow.set_tag("mlflow.runName", "generate_answers_with_agent")

        mlflow.autolog()
        
        # Load questions
        questions_df = load_questions()
        
        # Limit questions if specified
        if args.max_questions > 0:
            questions_df = questions_df.head(args.max_questions)
        
        mlflow.log_metric("total_questions", len(questions_df))
        
        # Generate answers
        answers = []
        print(f"Generating answers for {len(questions_df)} questions using LangChain...")
        
        for idx, row in questions_df.iterrows():
            print(f"Processing question {idx + 1}/{len(questions_df)}: {row['Question'][:50]}...")
            
            try:
                answer = generate_answer(
                    question=row['Question'],
                    category=row['Category'],
                    weight=row['Weight'],
                    prompt_template_name=args.prompt_template,
                    model=args.model,
                    search_results_count=args.search_results_count,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
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
        print("\nTo evaluate the generated answers, run:")
        print(
            "\tmlflow run -e evaluate_responses --env-manager=local", 
            f"--experiment-id {run.info.experiment_id}",
            f". -P generation_run_id={run.info.run_id}\n"
        )


if __name__ == "__main__":
    main()
