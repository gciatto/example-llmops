# Quick Start Guide

## 5-Minute Setup

### 1. Set OpenAI API Key
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run a Quick Test
```bash
mlflow run . -e generate_answers -P max_questions=3
```

### 4. View Results
```bash
mlflow ui
# Open http://localhost:5000
```

## Example Commands

### Generate with Different Prompts
```bash
# Basic style
mlflow run . -e generate_answers -P prompt_template=prompts/basic.txt -P max_questions=10

# Academic style  
mlflow run . -e generate_answers -P prompt_template=prompts/academic.txt -P max_questions=10

# With web search
mlflow run . -e generate_answers -P use_search=true -P max_questions=5
```

### Evaluate Results
```bash
mlflow run . -e evaluate_responses -P answers_file=answers.csv
```

### Compare Prompts
```bash
mlflow run . -e compare_prompts \
  -P prompt_templates="prompts/basic.txt,prompts/academic.txt" \
  -P max_questions=5
```

## What to Expect

### First Run (~2 min)
- Processes 3 questions by default
- Generates `answers.csv`
- Logs to MLflow (creates `mlruns/` directory)

### MLflow UI
- Navigate to http://localhost:5000
- See all runs with parameters and metrics
- Download generated artifacts
- Compare different prompt templates

## Troubleshooting

**"OPENAI_API_KEY not set"**
→ Run: `export OPENAI_API_KEY='your-key'`

**"Import errors"**
→ Run: `pip install -r requirements.txt`

**"Rate limit exceeded"**
→ Use `-P max_questions=1` to test with fewer questions

## Cost Estimation

Using `gpt-4o-mini`:
- ~100-200 tokens per question
- ~$0.0001-0.0002 per question
- 10 questions ≈ $0.001-0.002

Full dataset (295 questions): ~$0.30-0.60

## Next Steps

1. ✅ Run quick test (3 questions)
2. ✅ View results in MLflow UI
3. ✅ Try different prompt templates
4. ✅ Compare prompts with `compare_prompts`
5. ✅ Evaluate with web search enabled

## Files You'll Generate

- `answers.csv` - Generated answers
- `evaluation.csv` - Quality metrics
- `mlruns/` - MLflow tracking data
- `mlartifacts/` - Stored artifacts

All are git-ignored automatically.

## Demo Script

For a complete automated demo:
```bash
./demo.sh
```

This runs through all entry points with small samples.
