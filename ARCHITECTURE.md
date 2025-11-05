# Project Architecture Overview

## Core Concepts Demonstrated

This MLflow project demonstrates key LLMOps practices:

### 1. **MLproject Convention**
- Multiple entry points for different pipeline stages
- Parameterized execution
- Environment specification
- Reproducible runs

### 2. **Modular Design**
Each Python file serves a specific purpose:

```
utils.py              → Shared utilities (loading data, search, prompts)
generate_answers.py   → Answer generation with configurable prompts
evaluate_responses.py → Quality metrics and evaluation
compare_prompts.py    → Systematic prompt comparison
```

### 3. **Tool Integration**
The DuckDuckGo search integration demonstrates:
- External tool/API integration
- Context enrichment for LLM responses
- Optional feature toggling

### 4. **Prompt Engineering**
Four distinct prompt styles show:
- Template-based prompting
- Prompt iteration and comparison
- Style adaptation for different use cases

### 5. **Experiment Tracking**
MLflow integration provides:
- Parameter logging (prompts, models, settings)
- Metric tracking (quality scores, error rates)
- Artifact storage (answers, evaluations, templates)
- Run comparison and analysis

## Data Flow

```
questions.csv
    ↓
[generate_answers.py]
    ├─→ Load questions
    ├─→ Apply prompt template
    ├─→ Optional: enrich with web search
    ├─→ Call OpenAI API
    └─→ Save answers.csv
         ↓
    [evaluate_responses.py]
         ├─→ Load answers
         ├─→ Compute metrics
         └─→ Save evaluation.csv

[compare_prompts.py]
    ├─→ Load multiple prompt templates
    ├─→ Generate answers for each
    ├─→ Evaluate all results
    ├─→ Log comparative metrics to MLflow
    └─→ Create summary report
```

## MLflow Entry Points

### Entry Point 1: `generate_answers`
**Purpose**: Generate answers using OpenAI with customizable prompts

**Parameters**:
- `prompt_template`: Which prompt style to use
- `use_search`: Enable web search enrichment
- `model`: OpenAI model selection
- `max_questions`: Limit for testing

**Outputs**:
- CSV with generated answers
- MLflow metrics: question count, success rate

### Entry Point 2: `evaluate_responses`
**Purpose**: Evaluate answer quality with multiple metrics

**Parameters**:
- `answers_file`: File to evaluate
- `output_file`: Where to save metrics

**Outputs**:
- CSV with per-question metrics
- MLflow metrics: averages, rates, scores

### Entry Point 3: `compare_prompts`
**Purpose**: Systematically compare prompt templates

**Parameters**:
- `prompt_templates`: Comma-separated list of prompts
- `max_questions`: Questions to test on
- `use_search`: Enable for all prompts

**Outputs**:
- Nested MLflow runs (one per prompt)
- Comparison summary artifact
- Comprehensive metrics for analysis

## Tool Function: Web Search

The `search_web()` function demonstrates tool integration:

**Input**: Question text
**Process**: 
1. Query DuckDuckGo API
2. Extract top N results
3. Format as structured text

**Output**: Enriched context for LLM

**Integration Point**: Injected into prompt via `{search_results}` placeholder

This simulates real-world scenarios where LLMs need external information sources.

## Evaluation Metrics

Simple but illustrative metrics for demo purposes:

1. **Length Metrics**: word count, character count, sentences
2. **Relevance**: keyword overlap between question and answer
3. **Structure**: sentence count, detail level
4. **Category Fit**: does answer match expected style for category?
5. **Error Rate**: API failures or malformed responses

In production, you'd add:
- LLM-as-judge evaluation
- Human evaluation scores
- Task-specific metrics (accuracy, completeness)
- Cost tracking (tokens, API calls)

## Extensibility

### Adding New Prompts
1. Create `prompts/newstyle.txt`
2. Use placeholders: `{question}`, `{category}`, `{weight}`, `{search_results}`
3. Run with `-P prompt_template=prompts/newstyle.txt`

### Adding New Metrics
1. Edit `evaluate_answer()` in `evaluate_responses.py`
2. Add new metric calculation
3. Return in metrics dict
4. Automatically logged to MLflow

### Adding New Tools
1. Add function to `utils.py`
2. Integrate in `generate_answer()`
3. Add parameter to control usage
4. Include output in prompt template

### Custom Models
Just pass `-P model=gpt-4` or any OpenAI model ID

## Real-World Problem Context

This project addresses a genuine educational challenge:

**Problem**: Instructors need model answers for hundreds of exam questions
**Manual Approach**: Time-consuming, inconsistent quality
**Automated Solution**: LLM generation with quality control

**Key Requirements Met**:
- ✅ Handle diverse question types (definition, commonsense, terminal commands)
- ✅ Adjust difficulty based on weight
- ✅ Maintain consistency across answers
- ✅ Enable quality evaluation
- ✅ Support iteration and improvement
- ✅ Track experiments for reproducibility

## Usage Patterns

### Pattern 1: Quick Test
```bash
mlflow run . -e generate_answers -P max_questions=5
```

### Pattern 2: Production Run
```bash
mlflow run . -e generate_answers \
  -P prompt_template=prompts/academic.txt \
  -P use_search=true \
  -P model=gpt-4o
```

### Pattern 3: Experiment Comparison
```bash
mlflow run . -e compare_prompts \
  -P prompt_templates="prompts/basic.txt,prompts/academic.txt,prompts/practical.txt" \
  -P max_questions=20
```

### Pattern 4: Pipeline Execution
```bash
# Generate
mlflow run . -e generate_answers -P output_file=v1.csv

# Evaluate
mlflow run . -e evaluate_responses -P answers_file=v1.csv

# Compare with previous version...
```

## Next Steps for Enhancement

1. **Advanced Evaluation**: Add LLM-as-judge for semantic quality
2. **Cost Tracking**: Log token usage and API costs
3. **Caching**: Store/reuse results for identical questions
4. **Batch Processing**: Optimize API calls with batching
5. **Human Feedback**: Integrate human ratings into metrics
6. **A/B Testing**: Statistical comparison of prompt variants
7. **Fine-tuning**: Use best answers as training data
8. **Multi-modal**: Add diagram/code generation for technical questions

## MLflow Best Practices Demonstrated

✅ **Parameterization**: Everything configurable via CLI
✅ **Environment Management**: Explicit Python dependencies
✅ **Artifact Logging**: All outputs saved and versioned
✅ **Nested Runs**: Hierarchical organization for comparisons
✅ **Metric Tracking**: Quantitative evaluation logged
✅ **Reproducibility**: Complete parameter and artifact tracking
✅ **Documentation**: Clear README and code comments
