from smartchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="qwq-plus", temperature=0.5)
result = llm.invoke("你好,你是谁", temperature=0.5)
print(result)
print(result.content)
