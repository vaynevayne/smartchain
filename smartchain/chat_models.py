import os
from typing import Any, List, Tuple

from openai import openai
from .message import AIMessage


class ChatOpenAI:
    def __init__(self, model: str = "qwq-plus", **kwargs):
        self.model = model
        # 优先使用传入的 api_key，其次读取环境变量（同时支持 DASHSCOPE 和 OPENAI 变量名）
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供API密钥(OPENAI_API_KEY)")
        # 其他模型参数传递给底层 ChatQwQ
        self.model_kwargs = {k: v for k, v in kwargs.items() if k != "api_key"}
        self.client = openai.OpenAI(api_key=self.api_key)

    def invoke(self, input: Any, **kwargs) -> str:
        messages = self._convert_input(input)
         # 构建 API 请求参数字典
        params = {
            "model": self.model,
            "messages": messages,
            **self.model_kwargs,
            **kwargs
        }
        # 直接调用底层 llm 的 invoke，传入消息与额外参数（如 temperature）
        response = self.client.chat.completions.create(**params)
          # 取出返回结果中的第一个选项
        choice = response.choices[0]
        message = choice.message.content or ""
        # 返回纯文本内容，便于 print 与后续处理
        return AIMessage(content=content)

    def _convert_input(self, input: Any) -> List[Tuple[str, str]]:
            """
        将输入转换为 OpenAI API 需要的消息格式

        Args:
            input: 字符串或消息列表

        Returns:
            list[dict]: OpenAI API 格式的消息列表
        """
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        # 输入为列表时，需逐个元素处理
        return [("human", str(input))]

    def stream(self,input,**kwargs):
        messages = self._convert_input(input)
        params = {
            "model":self.model,
            "messages":message,
            "stream":True,
            **self.model_kwargs,
            **kwargs
        }
        stream = self.    
