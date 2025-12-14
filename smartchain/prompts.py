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
                role = role_map.get(msg.role, msg.type.capitalize())
                parts.append(f"{role}:{msg.content}")
            else:
                parts.append(str(msg))
        return "\n".join(parts)
