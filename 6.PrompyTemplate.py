from smartchain.prompts import PromptTemplate
from smartchain.chat_models import ChatDeepSeek

prompt_template = PromptTemplate.from_template("你好,我叫{name},你是谁?")
print(prompt_template, type(prompt_template))

llm = ChatDeepSeek()
formatted_prompt = prompt_template.format(name="张三")

print(formatted_prompt)
result = llm.invoke(formatted_prompt)
print(result.content)
