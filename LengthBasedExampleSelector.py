from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector
from dotenv import load_dotenv
import re

load_dotenv()
examples = [
    {"question": "1+1等于多少？", "answer": "答案是2"},
    {"question": "2+2等于多少？", "answer": "答案是4"},
    {"question": "3+3等于多少？", "answer": "答案是6"},
    {"question": "4+4等于多少？", "answer": "答案是8"},
    {"question": "5+5等于多少？", "answer": "答案是10"},
]

example_prompt = PromptTemplate.from_template("问题:{question}\n答案:{answer}")
lengthBasedExampleSelector = LengthBasedExampleSelector(
    examples=examples, example_prompt=example_prompt, max_length=10
)
fewShotPromptTemplate = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=lengthBasedExampleSelector,
    prefix="你是一数学专家",
    suffix="问题:{user_question}\n AI:",
)
formatted_prompt = fewShotPromptTemplate.format(user_question="8 plus 7 等于多少?")
print(formatted_prompt)
print("=" * 60)
llm = ChatDeepSeek(
    model="deepseek-chat", temperature=0.5, base_url="https://api.deepseek.com"
)
result = llm.invoke(formatted_prompt)
print(result)
text = "1 + 1"
tokens = re.split(r"\s+", text.strip())
print(tokens)
