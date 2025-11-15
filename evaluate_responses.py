import argparse
import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import Guidelines, scorer, RelevanceToQuery


@scorer
def enough_words(outputs: dict) -> Feedback:
    text = outputs['choices'][-1]['message']['content']
    word_count = len(text.split())
    score = word_count >= 10
    rationale = (
        f"The response has more than 10 words: {word_count}"
        if score
        else f"The response does not have enough words because it has less than 10 words: {word_count}."
    )
    return Feedback(value=score, rationale=rationale)

@scorer
def not_too_many_words(outputs: dict) -> Feedback:
    text = outputs['choices'][-1]['message']['content']
    word_count = len(text.split())
    score = word_count <= 1000
    rationale = (
        f"The response has less than 1000 words: {word_count}"
        if score
        else f"The response has too many words: {word_count}."
    )
    return Feedback(value=score, rationale=rationale)


def guidelines_model(model: str = None):
    yield Guidelines(
        name="english",
        guidelines="The answer should be in English.",
        model=model
    )
    yield Guidelines(
        name="software_engineering_related",
        guidelines="The answer is correctly contextualizing the question within the domain of software engineering.",
        model=model
    )
    yield Guidelines(
        name="reference_to_definition",
        guidelines="The answer should reference and/or quote relevant definitions for the concepts mentioned in the question.",
        model=model
    )
    yield RelevanceToQuery(
        model=model
    )
    yield enough_words
    yield not_too_many_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers using MLflow GenAI scorers"
    )
    parser.add_argument(
        "--generation-run-id",
        type=str,
        default=None,
        help="MLflow run ID of the generation run to evaluate. If not provided, evaluates all traces in the experiment."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model to use as judge for evaluation (e.g., gpt-4.1-mini, gpt-4o)"
    )
    
    args = parser.parse_args()

    if (run_id := args.generation_run_id) is None or run_id == "none":
        traces = mlflow.search_traces()  # all traces in the experiment
    else:
        traces = mlflow.search_traces(run_id=args.generation_run_id)
    
    mlflow.genai.evaluate(
        data=traces,
        scorers=list(guidelines_model(model=args.judge_model)),
    )
