# app/services/rag/memory_context.py
from collections import deque

class ConversationMemory:
    def __init__(self, max_turns=5):
        self.history = deque(maxlen=max_turns)

    def add_turn(self, question, answer):
        self.history.append({"q": question, "a": answer})

    def get_context(self):
        return "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in self.history])
