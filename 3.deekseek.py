from smartchain.chat_models import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat", api_key="sk-bb99cf132b184a169b5e053b346a7c25")
result = llm.invoke("你好,你是谁", temperature=0.8)
print(result, type(result))
print(result.content)
