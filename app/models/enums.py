from enum import Enum

class AgentMode(str, Enum):

    REACT = "react"
    PLAN_EXECUTE = "plan_execute"

class RetrievalMode(str, Enum):

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class MessageRole(str, Enum):

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class TaskStatus(str, Enum):

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
