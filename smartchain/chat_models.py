import os
from typing import Any, List, Tuple

from openai import OpenAI
from .message import AIMessage, HumanMessage, SystemMessage
from .prompts import ChatPromptValue


class ChatOpenAI:
    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.model = model
        # 优先使用传入的 api_key，其次读取环境变量（同时支持 DASHSCOPE 和 OPENAI 变量名）
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供API密钥(OPENAI_API_KEY)")
        # 其他模型参数传递给底层 ChatQwQ
        self.model_kwargs = {k: v for k, v in kwargs.items() if k != "api_key"}
        self.client = OpenAI(api_key=self.api_key)

    def invoke(self, input: Any, **kwargs) -> str:
        messages = self._convert_input(input)
        # 构建 API 请求参数字典
        params = {
            "model": self.model,
            "messages": messages,
            **self.model_kwargs,
            **kwargs,
        }
        # 直接调用底层 llm 的 invoke，传入消息与额外参数（如 temperature）
        response = self.client.chat.completions.create(**params)
        # 取出返回结果中的第一个选项
        choice = response.choices[0]
        content = choice.message.content or ""
        # 返回纯文本内容，便于 print 与后续处理
        return AIMessage(content=content)

    def stream(self, input, **kwargs):
        messages = self._convert_input(input)
        params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.model_kwargs,
            **kwargs,
        }
        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield AIMessage(content=delta.content)

    def _convert_input(self, input: Any) -> List[Tuple[str, str]]:
        if isinstance(input, ChatPromptValue):
            input.to_messages()
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = []
            for msg in input:
                if isinstance(msg, str):
                    messages.append({"role": "user", "content": msg})
                elif isinstance(msg, (AIMessage, HumanMessage, SystemMessage)):
                    if isinstance(msg, AIMessage):
                        role = "assistant"
                    elif isinstance(msg, HumanMessage):
                        role = "user"
                    elif isinstance(msg, SystemMessage):
                        role = "system"
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    messages.append({"role": role, content: content})
                elif isinstance(msg, dict):
                    messages.append(msg)
                elif isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg
                    messages.append({"role": role, "content": content})
                return messages
        else:
            return [{"role": "user", "content": str(input)}]


class ChatDeepSeek:
    def __init__(self, model: str = "deepseek-chat", **kwargs):
        self.model = model
        self.api_key = kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(f"需要提供api_key或者设置DEEPSEEK_API_KEY环境变量")
        # 保存除了api_key 之外的参数
        self.model_kwargs = {k: v for k, v in kwargs.items() if k != "api_key"}
        base_url = kwargs.get("base_url", "https://api.deepseek.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    # 调用模型生成回复的方法
    def invoke(self, input, **kwargs):
        # 将输入的数据转换成openai期望的消息的格式
        messages = self._convert_input(input)
        params = {
            "model": self.model,
            "messages": messages,
            **self.model_kwargs,
            **kwargs,
        }
        # 使用OpenAI的客端发起完成请求并获取回复
        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        content = choice.message.content or ""
        return AIMessage(content=content)

    def _convert_input(self, input):
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        else:
            return [{"role": "user", "content": str(input)}]


class ChatTongyi:
    def __init__(self, model: str = "qwen-max", **kwargs):
        self.model = model
        self.api_key = kwargs.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(f"需要提供api_key或者设置DASHSCOPE_API_KEY环境变量")
        # 保存着除了api_key之外其它的额外参数，供API调用
        self.model_kwargs = {k: v for k, v in kwargs.items() if k != "api_key"}
        base_url = kwargs.get(
            "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

    # 调用模型生成回复的方法
    def invoke(self, input, **kwargs):
        # 将输入的数据转换成openai期望的消息的格式
        messages = self._convert_input(input)
        params = {
            "model": self.model,
            "messages": messages,
            **self.model_kwargs,
            **kwargs,
        }
        # 使用OpenAI的客端发起完成请求并获取回复
        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        content = choice.message.content or ""
        return AIMessage(content=content)

    def _convert_input(self, input):
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        else:
            return [{"role": "user", "content": str(input)}]
