from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate.from_template("你是一个{role},用户{username}问:{question}")

print("原始输入", template.input_variables)

partial_template = template.partial(role="AI助手", username="张三")
print("部分输入", partial_template.input_variables)

formatted_prompt = partial_template.format(question="你好,你是谁", username="李四")

llm = ChatDeepSeek(
    model="deepseek-chat", temperature=0.5, base_url="https://api.deepseek.com"
)
result = llm.invoke(formatted_prompt)
print(result)
