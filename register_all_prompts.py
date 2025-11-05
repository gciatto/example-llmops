from utils import *
import mlflow.genai


if __name__ == "__main__":
    for file in DIR_PROMPTS.glob("*.txt"):
        template_name = file.stem
        prompt_template = load_prompt_template(template_name)
        mlflow.genai.register_prompt(template_name, prompt_template)
        print(f"Registered prompt template: {template_name}")

    print("\nAll prompt templates registered.")