from smartchain.chat_models import ChatDeepSeek
from smartchain.message import HumanMessage
from smartchain.prompts import ChatPromptTemplate, MessagesPlaceholder


class CustomMessage:
    def __init__(self):
        self.type = "human"
        self.content = "CustomMessage"

    def __str__(self):
        return self.content


customMessage = CustomMessage()
history = [
    HumanMessage(content="你好"),
    customMessage,
    {"role": "user", "content": "你好"},
]

template = ChatPromptTemplate(
    [
        ("system", "你是一个ai助手"),
        # 占位符,将填充历史消息
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

llm = ChatDeepSeek()
prompt_messages = template.format_messages(history=history, question="请介绍一下你自己")

for msg in prompt_messages:
    print(f"{getattr(msg, 'type', 'human')}:{getattr(msg, 'content', str(msg))}")

result = llm.invoke(prompt_messages)
print(result.content)
