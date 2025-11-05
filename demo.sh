#!/usr/bin/env bash
# Quick demo script to test the MLflow project

set -e

echo "==================================="
echo "Quiz Answer Generator - Quick Demo"
echo "==================================="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Please run: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✓ OpenAI API key detected"
echo ""

# Demo 1: Generate answers with basic prompt
echo "Step 1: Generating answers with basic prompt (5 questions)..."
mlflow run . -e generate_answers \
  -P prompt_template=prompts/basic.txt \
  -P output_file=demo_basic.csv \
  -P max_questions=5 \
  -P model=gpt-4o-mini

echo ""
echo "Step 2: Evaluating generated answers..."
mlflow run . -e evaluate_responses \
  -P answers_file=demo_basic.csv \
  -P output_file=demo_eval.csv

echo ""
echo "Step 3: Comparing different prompts (3 questions each)..."
mlflow run . -e compare_prompts \
  -P prompt_templates="prompts/basic.txt,prompts/concise.txt" \
  -P max_questions=3 \
  -P model=gpt-4o-mini

echo ""
echo "==================================="
echo "Demo completed successfully!"
echo "==================================="
echo ""
echo "View results:"
echo "  1. Check generated files: demo_basic.csv, demo_eval.csv"
echo "  2. Launch MLflow UI: mlflow ui"
echo "  3. Open browser: http://localhost:5000"
echo ""
