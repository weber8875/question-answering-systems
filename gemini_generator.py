from dotenv import load_dotenv
import os
import google.generativeai as genai
from pathlib import Path 

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def load_prompt(path: str) -> str:
    return Path(path).read_text()

def generate_answer(context: str, question: str, model_name="models/gemini-2.5-pro"):
    prompt_template = load_prompt("prompts/qa_prompt.txt")  
    prompt = prompt_template.format(context=context, question=question)

    model = genai.GenerativeModel(model_name=model_name)
    return model.generate_content(prompt).text
