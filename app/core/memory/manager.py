from __future__ import annotations

from loguru import logger

from app.core.memory.long_term import LongTermMemory
from app.core.memory.short_term import ShortTermMemory
from app.models.schemas import MemoryContext, Message

class MemoryManager:

    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory) -> None:
\
\
\

        self._stm = short_term
        self._ltm = long_term

    async def get_context(self, session_id: str, query: str) -> MemoryContext:

        try:
            short_msgs = await self._stm.get_history(session_id)
        except Exception as e:
            logger.exception("读取短期记忆失败: {}", e)
            short_msgs = []

        try:
            long_items = await self._ltm.recall(query, session_id, top_k=5)
        except Exception as e:
            logger.exception("长期记忆召回失败: {}", e)
            long_items = []

        return MemoryContext(
            session_id=session_id,
            short_term_messages=short_msgs,
            long_term_items=long_items,
        )

    async def save(self, session_id: str, message: Message) -> None:

        try:
            await self._stm.add_message(session_id, message)
        except Exception as e:
            logger.exception("保存短期记忆失败: {}", e)
            raise RuntimeError(f"save 失败: {e}") from e
