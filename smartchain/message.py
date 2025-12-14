class BaseMessage:
    def __init__(self, content: str, **kwargs):
        self.content = content
        self.type = kwargs.get("type", "base")

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"{self.__class__.__name__}(content={self.content})"


class AIMessage(BaseMessage):
    def __init__(self, content, **kwargs):
        super().__init__(content, type="ai", **kwargs)


class HumanMessage(BaseMessage):
    def __init__(self, content, **kwargs):
        super().__init__(content, type="human", **kwargs)


class SystemMessage(BaseMessage):
    def __init__(self, content, **kwargs):
        super().__init__(content, type="system", **kwargs)

