import re

from smartchain.message import AIMessage, HumanMessage, SystemMessage


class PromptTemplate:
    def __init__(self, template: str, partial_variables: dict = None):
        self.template = template
        self.partial_variables = partial_variables or {}
        # 从模板中提取全部变量的变量名,到列表中
        all_variables = self._extract_variables(template)
        # 最后阶段 调用format时,需要接受的变量名列表
        # 待填充变量 = 等于全部的变量减去部分填充的变量
        self.input_variables = [
            v for v in all_variables if v not in self.partial_variables
        ]

    def _extract_variables(self, template):
        # 正则提取,生成字段,返回列表
        pattern = r"\{([^}:]+)(?::[^}]+)?\}"
        matches = re.findall(pattern, template)
        return list(dict.fromkeys(matches))

    @classmethod
    def from_template(cls, template: str):
        return cls(template=template)

    def partial(self, **kwargs):
        new_partial_variables = {**self.partial_variables, **kwargs}
        new_template = PromptTemplate(
            template=self.template, partial_variables=new_partial_variables
        )
        return new_template

    def format(self, **kwargs):
        all_vars = {**self.partial_variables, **kwargs}
        # 注入变量到模板中
        # 缺失变量 = 剩余待填充变量 - 传入变量
        missing_vars = set(self.input_variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"缺失变量:{missing_vars}")
        return self.template.format(**all_vars)


class ChatPromptValue:
    def __init__(self, message):
        self.messages = message

    def to_messages(self):
        return self.messages

    def to_string(self):
        parts = []
        for msg in self.messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role_map = {"system": "System", "human": "Human", "ai": "AI"}
                role = role_map.get(msg.type, msg.type.capitalize())
                parts.append(f"{role}:{msg.content}")
            else:
                parts.append(str(msg))
        return "\n".join(parts)


class ChatPromptTemplate:

    def __init__(self, messages):
        self.messages = messages
        self.input_variables = self._extract_input_variables()

    def _extract_input_variables(self):
        variables = set()
        for msg in self.messages:
            if isinstance(msg, tuple) and len(msg) == 2:
                _, template_str = msg
                prompt = PromptTemplate.from_template(template_str)
                variables.update(prompt.input_variables)

        return list(variables)

    def invoke(self, input_variables):
        formatted_messages = self._format_all_messages(input_variables)
        return ChatPromptValue(message=formatted_messages)

    def _format_all_messages(self, variables):
        """
        把传入的 variables 字典值,注入到模板中, 并返回新的模板列表
        """
        formatted_message = []
        for msg in self.messages:
            if isinstance(msg, tuple) and len(msg) == 2:
                role, template_str = msg
                prompt = PromptTemplate.from_template(template_str)
                content = prompt.format(**variables)
                formatted_message.append(self._create_message_from_role(role, content))

            elif isinstance(msg, BaseMessagePromptTemplate):
                formatted_message.append(msg.format(**variables))
            elif isinstance(msg, MessagesPlaceholder):
                pass
                # placeholder_messages = self._get_placeholder_value()
            else:
                formatted_message.append(msg)
        return formatted_message

    def _create_message_from_role(self, role, content):
        normalized_role = str(role).lower()
        if normalized_role == "system":
            return SystemMessage(content=content)
        elif normalized_role in ["human", "user"]:
            return HumanMessage(content=content)
        elif normalized_role in ("ai", "assistant"):
            return AIMessage(content=content)
        else:
            raise ValueError(f"未知的角色{role}")

    @classmethod
    def from_messages(cls, messages):
        return cls(messages=messages)


def format_messages(self, **kwargs):
    return self._format_all_message(kwargs)


class BaseMessagePromptTemplate:
    def __init__(self, prompt) -> None:
        self.prompt = prompt

    def from_template(cls, template: str):
        prompt = PromptTemplate.from_template(template)
        return cls(prompt=prompt)

    def format(self, **kwargs):
        content = self.prompt.format(**kwargs)
        return self._create_message(content)

    def _create_message(self, content):
        raise NotImplementedError


class SystemMessagePromptTemplate(BaseMessagePromptTemplate):
    def _create_message(self, content):
        return SystemMessage(content=content)


class HumanMessagePromptTemplate(BaseMessagePromptTemplate):
    def _create_message(self, content):
        return HumanMessage(content=content)


class AIMessagePromptTemplate(BaseMessagePromptTemplate):
    def _create_message(self, content):
        return AIMessage(content=content)


class MessagesPlaceholder:
    def __init__(self, variable_name):
        self.variable_name = variable_name
