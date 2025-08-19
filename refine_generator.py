from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path 
import os

REFINE_PROMPT_PATH = "prompts/refine_prompt.txt"
QA_PROMPT_PATH = "prompts/qa_prompt.txt"

model_name="models/gemini-2.5-pro"

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_prompt(path: str) -> str:
    return Path(path).read_text()

def refine_answer(existing_answer: str, new_context: str, question: str, model_name: str):
    prompt_template = load_prompt(REFINE_PROMPT_PATH)
    prompt = prompt_template.format(
        existing_answer=existing_answer,
        new_context=new_context,
        question=question
    )
    model = genai.GenerativeModel(model_name=model_name)
    return model.generate_content(prompt).text



def generate_refined_answer(chunks, question,model_name=model_name):

    # 第一段
    initial_context = chunks[0].page_content
    base_prompt = load_prompt("prompts/qa_prompt.txt")
    first_prompt = base_prompt.format(context=initial_context, question=question)
    
    model = genai.GenerativeModel(model_name=model_name)
    answer = model.generate_content(first_prompt).text


    for chunk in chunks[1:]:
        answer = refine_answer(
            existing_answer=answer, 
            new_context=chunk.page_content,
            question=question,
            model_name=model_name
        )
    
    return answer

