# Quiz Answer Generator - MLflow Project

A minimal yet realistic MLflow project demonstrating LLMOps practices for automated quiz answer generation using OpenAI's API.

## Project Overview

This project automates the generation of exam answers for software engineering questions, featuring:

- **Multiple prompt templates** for different answer styles (basic, academic, concise, practical)
- **Web search enrichment** using DuckDuckGo to provide additional context
- **Answer evaluation** with metrics for quality assessment
- **Prompt comparison** to identify the best-performing templates
- **MLflow tracking** for experiment management and reproducibility

## Project Structure

```
.
├── MLproject                  # MLflow project definition
├── python_env.yaml           # Python environment specification
├── requirements.txt          # Python dependencies
├── questions.csv             # Input: quiz questions with categories and weights
├── utils.py                  # Shared utility functions
├── generate_answers.py       # Entry point: generate answers
├── evaluate_responses.py     # Entry point: evaluate answers
├── compare_prompts.py        # Entry point: compare prompt templates
└── prompts/                  # Prompt template directory
    ├── basic.txt            # Simple, direct answers
    ├── academic.txt         # Rigorous academic style
    ├── concise.txt          # Brief, bullet-point style
    └── practical.txt        # Example-focused answers
```

## Prerequisites

1. **Python 3.11+**
2. **OpenAI API Key**: Set as environment variable
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Installation

### Option 1: Using pip
```bash
pip install -r requirements.txt
```

### Option 2: Using MLflow (recommended)
MLflow will automatically create the environment when running the project.

## Usage

### 1. Generate Answers

Generate answers for all questions using a specific prompt template:

```bash
# Basic usage
mlflow run . -e generate_answers

# With custom parameters
mlflow run . -e generate_answers \
  -P prompt_template=prompts/academic.txt \
  -P output_file=answers_academic.csv \
  -P max_questions=20 \
  -P model=gpt-4o-mini

# With web search enrichment
mlflow run . -e generate_answers \
  -P use_search=true \
  -P search_results_count=5 \
  -P max_questions=10
```

**Parameters:**
- `prompt_template`: Path to prompt template file (default: `prompts/basic.txt`)
- `output_file`: Output CSV file name (default: `answers.csv`)
- `use_search`: Enable DuckDuckGo search enrichment (`true`/`false`, default: `false`)
- `search_results_count`: Number of search results to include (default: `3`)
- `max_questions`: Limit number of questions to process (default: `-1` for all)
- `model`: OpenAI model to use (default: `gpt-4o-mini`)

### 2. Evaluate Responses

Evaluate generated answers using various metrics:

```bash
# Evaluate default answers.csv
mlflow run . -e evaluate_responses

# Evaluate specific file
mlflow run . -e evaluate_responses \
  -P answers_file=answers_academic.csv \
  -P output_file=evaluation_academic.csv
```

**Parameters:**
- `answers_file`: Input CSV with generated answers (default: `answers.csv`)
- `output_file`: Output CSV with evaluation metrics (default: `evaluation.csv`)

**Metrics computed:**
- Answer length and word count
- Keyword overlap with question
- Structure quality (sentences, detail level)
- Category-specific expectations
- Error rate

### 3. Compare Prompts

Compare multiple prompt templates systematically:

```bash
# Compare all prompts on 10 questions
mlflow run . -e compare_prompts \
  -P prompt_templates="prompts/basic.txt,prompts/academic.txt,prompts/concise.txt" \
  -P max_questions=10

# Compare with search enrichment
mlflow run . -e compare_prompts \
  -P prompt_templates="prompts/basic.txt,prompts/practical.txt" \
  -P max_questions=15 \
  -P use_search=true
```

**Parameters:**
- `prompt_templates`: Comma-separated list of prompt template paths (required)
- `max_questions`: Limit questions for comparison (default: `10`)
- `use_search`: Enable web search (`true`/`false`, default: `false`)
- `model`: OpenAI model to use (default: `gpt-4o-mini`)

### 4. View Results in MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to:
- Compare experiment runs
- View logged metrics and parameters
- Download generated artifacts (answers, evaluations)
- Analyze prompt performance

## Direct Python Execution

You can also run the scripts directly:

```bash
# Generate answers
python generate_answers.py \
  --prompt-template prompts/basic.txt \
  --output-file answers.csv \
  --max-questions 10

# Evaluate
python evaluate_responses.py \
  --answers-file answers.csv \
  --output-file evaluation.csv

# Compare prompts
python compare_prompts.py \
  --prompt-templates "prompts/basic.txt,prompts/academic.txt" \
  --max-questions 10
```

## Example Workflow

Here's a complete workflow for experimenting with different approaches:

```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY='sk-...'

# 2. Generate answers with different prompts
mlflow run . -e generate_answers -P prompt_template=prompts/basic.txt -P output_file=answers_basic.csv -P max_questions=20
mlflow run . -e generate_answers -P prompt_template=prompts/academic.txt -P output_file=answers_academic.csv -P max_questions=20

# 3. Evaluate each set
mlflow run . -e evaluate_responses -P answers_file=answers_basic.csv -P output_file=eval_basic.csv
mlflow run . -e evaluate_responses -P answers_file=answers_academic.csv -P output_file=eval_academic.csv

# 4. Compare prompts systematically
mlflow run . -e compare_prompts -P prompt_templates="prompts/basic.txt,prompts/academic.txt,prompts/concise.txt,prompts/practical.txt" -P max_questions=15

# 5. View results
mlflow ui
```

## Key Features for MLflow Demonstration

### 1. **Multiple Entry Points**
Three distinct entry points showing different stages of an ML pipeline:
- Data generation (`generate_answers`)
- Evaluation (`evaluate_responses`)
- Experimentation (`compare_prompts`)

### 2. **Tool Integration**
Demonstrates function calling with DuckDuckGo search API to enrich answers with web results.

### 3. **Prompt Engineering**
Four different prompt templates showcasing:
- Basic: simple and direct
- Academic: formal and comprehensive
- Concise: brief and focused
- Practical: example-driven

### 4. **Comprehensive Tracking**
All runs log:
- Parameters (prompt, model, settings)
- Metrics (quality scores, error rates)
- Artifacts (generated answers, evaluations)
- Prompt templates for reproducibility

### 5. **Real-world Scenario**
Based on actual exam questions, making it relatable and practical for educational contexts.

## Customization

### Adding New Prompts

Create a new file in the `prompts/` directory:

```txt
# prompts/custom.txt
Your custom prompt template here.

Question: {question}
Category: {category}
Weight: {weight}

{search_results}

Instructions for the LLM...
```

Then use it:
```bash
mlflow run . -e generate_answers -P prompt_template=prompts/custom.txt
```

### Modifying Evaluation Metrics

Edit `evaluate_responses.py` to add custom metrics in the `evaluate_answer()` function.

### Using Different Models

```bash
mlflow run . -e generate_answers -P model=gpt-4o
```

## Troubleshooting

**Issue**: `OPENAI_API_KEY not set`
- **Solution**: Export your API key: `export OPENAI_API_KEY='your-key'`

**Issue**: Import errors when running directly
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Rate limiting errors
- **Solution**: Reduce `max_questions` or add delays in the code

**Issue**: Search results empty
- **Solution**: DuckDuckGo may be rate-limiting; try without search first

## License

This is an educational example project for demonstrating MLflow and LLMOps concepts.

## Notes

- The evaluation metrics are intentionally simple for demonstration purposes
- In production, you'd want more sophisticated evaluation (e.g., LLM-as-judge, human evaluation)
- The web search integration is optional and can be disabled for faster/cheaper runs
- All results are tracked in MLflow for reproducibility and comparison
