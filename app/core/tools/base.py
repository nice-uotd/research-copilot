from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel
class ToolParameter(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
class BaseTool(ABC):
    name: str = "base_tool"
    description: str = "基础工具"
    def __init__(self) -> None:
        self.parameters: list[ToolParameter] = []
    def schema_parameters(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {"type": "object", "properties": properties, "required": required}
    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        pass
