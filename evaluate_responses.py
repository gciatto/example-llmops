import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import Guidelines, scorer, RelevanceToQuery
import sys


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

    generation_run_id = sys.argv[1] if len(sys.argv) > 1 else None
    if generation_run_id:
        traces = mlflow.search_traces(run_id=generation_run_id)
    else:
        traces = mlflow.search_traces() # all traces in the experiment

    judge_model = sys.argv[2] if len(sys.argv) > 2 else None
    
    mlflow.genai.evaluate(
        data=traces,
        scorers=list(guidelines_model(model=judge_model)),
    )
