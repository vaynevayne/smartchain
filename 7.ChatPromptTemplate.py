from smartchain import prompts
from smartchain.chat_models import ChatDeepSeek
from smartchain.prompts import ChatPromptTemplate

llm = ChatDeepSeek()
template = ChatPromptTemplate(
    [
        ("System", "你是一个乐于助人的AI助手,你的名字是{name}"),
        ("HUMAN", "你还,最近怎么样?"),
        ("ai", "我很好,谢谢你的关心"),
        ("user", "{user_input}"),
    ]
)

prompt_value = template.invoke({"name": "小助", "user_input": "你叫什么名字?"})

# print(prompt_value.to_string())
# print(prompt_value.to_messages())
result = llm.invoke(prompt_value)
print("+++", result.content)
