class BaseMessage:
    def __init__(self, content: str, **kwargs):
        self.content = content
        self.type = kwargs.get("type","base")

        for key,value in kwargs.items():
            if key !== 'type'
            


class AIMessage(BaseMessage):
    def __innit__(self,content,**kwargs):
        super().__init__(content,type="ai",**kwargs)

class HumanMessage(BaseMessage):
    def __innit__(self,content,**kwargs):
        super().__init__(content,type="human",**kwargs)


class SystemMessage(BaseMessage):
    def __innit__(self,content,**kwargs):
        super().__init__(content,type="human",**kwargs)