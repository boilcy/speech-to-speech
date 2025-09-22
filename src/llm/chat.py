from typing import List, Dict, Optional, Any
from loguru import logger


class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(
        self,
        max_history_pairs: int = 4,
        max_tokens: Optional[int] = None,
        preserve_system: bool = True,
        preserve_recent: int = 1,
    ):
        self.max_history_pairs = max_history_pairs
        self.max_tokens = max_tokens
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent

        # 分离存储系统消息和对话历史
        self.system_message: Optional[Dict[str, Any]] = None
        self.conversation_history: List[Dict[str, Any]] = []

        # 用于标记重要消息
        self.protected_indices: set = set()

    def init_chat(self, message: Dict[str, Any]) -> None:
        """
        设置初始系统消息。

        Args:
            message: 系统消息字典，通常包含role和content
        """
        if self.preserve_system:
            self.system_message = message
            logger.debug(
                f"System message initialized: {message.get('role', 'unknown')}"
            )
        else:
            self.conversation_history.append(message)

    def append(self, message: Dict[str, Any], protected: bool = False) -> None:
        """
        添加新消息到对话历史。

        Args:
            message: 消息字典
            protected: 是否标记为受保护消息
        """
        self.conversation_history.append(message)

        if protected:
            self.protected_indices.add(len(self.conversation_history) - 1)

        # 执行窗口管理
        self._manage_context_window()

    def _manage_context_window(self) -> None:
        """
        智能管理上下文窗口，删除旧的非重要消息。
        """
        if len(self.conversation_history) <= 2:  # 至少保留一轮对话
            return

        # 计算当前对话轮数（用户-助手配对）
        current_pairs = self._count_conversation_pairs()

        if current_pairs <= self.max_history_pairs:
            return

        # 需要删除一些历史消息
        pairs_to_remove = current_pairs - self.max_history_pairs
        self._remove_old_pairs(pairs_to_remove)

    def _count_conversation_pairs(self) -> int:
        """计算当前的对话轮数。"""
        return len(self.conversation_history) // 2

    def _remove_old_pairs(self, pairs_to_remove: int) -> None:
        """
        删除旧的对话对，但保护重要消息和最近的对话。

        Args:
            pairs_to_remove: 需要删除的对话轮数
        """
        messages_to_remove = pairs_to_remove * 2
        protected_count = len(self.protected_indices)
        recent_protected = self.preserve_recent * 2  # 最近N轮对话的消息数

        # 计算实际可删除的消息数量
        total_messages = len(self.conversation_history)
        deletable_end_index = max(0, total_messages - recent_protected)

        removed_count = 0
        i = 0

        while i < deletable_end_index and removed_count < messages_to_remove:
            if i not in self.protected_indices:
                # 删除消息并更新受保护索引
                self.conversation_history.pop(i)
                self._update_protected_indices_after_removal(i)
                removed_count += 1
                deletable_end_index -= 1  # 列表长度减少了
            else:
                i += 1

        if removed_count > 0:
            logger.debug(f"Removed {removed_count} old messages from chat history")

    def _update_protected_indices_after_removal(self, removed_index: int) -> None:
        """
        在删除消息后更新受保护消息的索引。

        Args:
            removed_index: 被删除消息的原始索引
        """
        new_protected_indices = set()
        for idx in self.protected_indices:
            if idx > removed_index:
                new_protected_indices.add(idx - 1)
            elif idx < removed_index:
                new_protected_indices.add(idx)
            # idx == removed_index 的情况不添加（已被删除）

        self.protected_indices = new_protected_indices

    def to_list(self) -> List[Dict[str, Any]]:
        """
        返回完整的聊天消息列表，包括系统消息。

        Returns:
            完整的消息列表
        """
        messages = []

        if self.system_message:
            messages.append(self.system_message)

        messages.extend(self.conversation_history)

        return messages

    def get_recent_messages(self, count: int) -> List[Dict[str, Any]]:
        """
        获取最近的N条消息。

        Args:
            count: 要获取的消息数量

        Returns:
            最近的消息列表
        """
        if count <= 0:
            return []

        recent = self.conversation_history[-count:] if self.conversation_history else []

        # 如果需要系统消息且存在，则添加到开头
        if self.system_message:
            return [self.system_message] + recent

        return recent

    def clear_history(self, keep_system: bool = True) -> None:
        """
        清除对话历史。

        Args:
            keep_system: 是否保留系统消息
        """
        self.conversation_history.clear()
        self.protected_indices.clear()

        if not keep_system:
            self.system_message = None

        logger.debug("Chat history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取聊天统计信息。

        Returns:
            包含统计信息的字典
        """
        return {
            "total_messages": len(self.conversation_history),
            "conversation_pairs": self._count_conversation_pairs(),
            "protected_messages": len(self.protected_indices),
            "has_system_message": self.system_message is not None,
            "max_history_pairs": self.max_history_pairs,
        }

    def protect_message(self, index: int) -> bool:
        """
        标记指定索引的消息为受保护。

        Args:
            index: 消息在conversation_history中的索引

        Returns:
            是否成功标记
        """
        if 0 <= index < len(self.conversation_history):
            self.protected_indices.add(index)
            logger.debug(f"Message at index {index} marked as protected")
            return True
        return False

    def unprotect_message(self, index: int) -> bool:
        """
        取消指定索引消息的保护状态。

        Args:
            index: 消息在conversation_history中的索引

        Returns:
            是否成功取消保护
        """
        if index in self.protected_indices:
            self.protected_indices.remove(index)
            logger.debug(f"Message at index {index} protection removed")
            return True
        return False
