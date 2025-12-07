from langchain_core.prompts import load_prompt
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="qwq-plus", temperature=0.5)
prompt = load_prompt("prompts/system_prompt.yaml")
result = llm.invoke(prompt)
print(result)
