from dotenv import load_dotenv
from smartchain.chat_models import ChatAI

load_dotenv()
llm = ChatAI(model="qwq-plus", temperature=0.5)
result = llm.invoke("你好,你是谁", temperature=0.5)
print(result)
