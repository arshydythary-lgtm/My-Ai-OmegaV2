# memory.py - نظام الذاكرة المحسّن والموثوق
from typing import List, Dict, Optional, Callable
from collections import deque
import json
from datetime import datetime


def estimate_tokens(text: str) -> int:
    """
    تقدير عدد الرموز بناءً على طول النص.
    الافتراض: 1 token ≈ 4 أحرف (متوسط للعربية والإنجليزية)
    """
    if not text:
        return 0
    return max(1, len(text.split()) + len(text) // 10)


class Memory:
    """
    نظام الذاكرة القصيرة المدى للمحادثة
    - يحافظ على عدد محدود من الرسائل
    - يدعم حدود الرموز (tokens)
    - يدعم قوالب مختلفة للتنسيق
    - يدعم الحفظ والتحميل
    """

    def __init__(
            self,
            max_history: int = 50,
            max_tokens: Optional[int] = 2000,
            token_counter: Optional[Callable[[str], int]] = None,
    ):
        """
        Args:
            max_history: أقصى عدد رسائل (الأقدم يُحذف)
            max_tokens: أقصى عدد رموز (اختياري)
            token_counter: دالة مخصصة لعد الرموز
        """
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.token_counter = token_counter or estimate_tokens

        # استخدام deque لكفاءة أفضل
        self.messages: deque = deque(maxlen=max_history)
        self.total_tokens = 0

    def add(self, role: str, content: str) -> bool:
        """
        إضافة رسالة جديدة

        Returns:
            True إذا تمت الإضافة بنجاح، False إذا تم رفضها
        """
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"❌ دور غير معروف: {role}")

        content = content.strip()
        if not content:
            return False

        # حساب الرموز
        token_count = self.token_counter(content)

        # فحص حد الرموز
        if self.max_tokens and (self.total_tokens + token_count) > self.max_tokens:
            # حذف الرسائل الأقدم حتى نتسع للرسالة الجديدة
            while len(self.messages) > 0 and self.total_tokens + token_count > self.max_tokens:
                old_msg = self.messages.popleft()
                self.total_tokens -= self.token_counter(old_msg["content"])

        # إضافة الرسالة
        message = {
            "role": role,
            "content": content,
            "tokens": token_count,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.total_tokens += token_count

        return True

    def add_user(self, content: str) -> bool:
        """إضافة رسالة مستخدم"""
        return self.add("user", content)

    def add_assistant(self, content: str) -> bool:
        """إضافة رسالة مساعد"""
        return self.add("assistant", content)

    def add_system(self, content: str) -> bool:
        """إضافة رسالة نظام"""
        return self.add("system", content)

    def get(self) -> List[Dict]:
        """الحصول على نسخة من جميع الرسائل"""
        return list(self.messages)

    def get_last_n(self, n: int) -> List[Dict]:
        """الحصول على آخر n رسالة"""
        if n <= 0:
            return []
        return list(self.messages)[-n:]

    def get_first_n(self, n: int) -> List[Dict]:
        """الحصول على أول n رسالة"""
        if n <= 0:
            return []
        return list(self.messages)[:n]

    def remove_last(self, count: int = 1) -> int:
        """
        حذف آخر n رسالة

        Returns:
            عدد الرسائل المحذوفة
        """
        removed = 0
        for _ in range(count):
            if self.messages:
                msg = self.messages.pop()
                self.total_tokens -= msg.get("tokens", 0)
                removed += 1
        return removed

    def clear(self) -> None:
        """مسح جميع الرسائل"""
        self.messages.clear()
        self.total_tokens = 0

    def get_token_count(self) -> int:
        """الحصول على إجمالي الرموز"""
        return self.total_tokens

    def get_message_count(self) -> int:
        """الحصول على عدد الرسائل"""
        return len(self.messages)

    def get_stats(self) -> Dict:
        """إحصائيات الذاكرة"""
        total_messages = len(self.messages)

        user_messages = sum(1 for m in self.messages if m["role"] == "user")
        assistant_messages = sum(1 for m in self.messages if m["role"] == "assistant")
        system_messages = sum(1 for m in self.messages if m["role"] == "system")

        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "system_messages": system_messages,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "max_history": self.max_history,
            "memory_usage_percent": (self.total_tokens / self.max_tokens * 100) if self.max_tokens else 0,
        }

    def trim_to_token_limit(self, limit: Optional[int] = None) -> int:
        """
        تقليم الذاكرة إلى حد معين من الرموز

        Returns:
            عدد الرسائل المحذوفة
        """
        limit = limit or self.max_tokens
        if not limit:
            return 0

        removed = 0
        while self.total_tokens > limit and len(self.messages) > 1:
            msg = self.messages.popleft()
            self.total_tokens -= msg.get("tokens", 0)
            removed += 1

        return removed

    def format_as_prompt(
            self,
            system_prompt: Optional[str] = None,
            template: str = "simple",
            max_messages: Optional[int] = None,
            include_timestamps: bool = False
    ) -> str:
        """
        تحويل الذاكرة إلى نص جاهز للنموذج

        Templates:
            - simple: "User: ... Assistant: ..."
            - qa: "السؤال: ... الإجابة: ..."
            - llama: "[INST] ... [/INST]"
        """
        messages_to_use = list(self.messages)

        if max_messages and max_messages > 0:
            messages_to_use = messages_to_use[-max_messages:]

        lines = []

        # إضافة System Prompt
        if system_prompt:
            if template == "simple":
                lines.append(f"[System: {system_prompt}]")
            elif template == "qa":
                lines.append(f"النظام: {system_prompt}")
            elif template == "llama":
                lines.append(f"<<SYS>>\n{system_prompt}\n<</SYS>>\n")

        # إضافة الرسائل
        for msg in messages_to_use:
            role = msg["role"]
            content = msg["content"]
            timestamp = f" [{msg.get('timestamp', '')}]" if include_timestamps else ""

            if template == "simple":
                if role == "user":
                    lines.append(f"User: {content}{timestamp}")
                elif role == "assistant":
                    lines.append(f"Assistant: {content}{timestamp}")
                elif role == "system":
                    lines.append(f"[System: {content}]{timestamp}")

            elif template == "qa":
                if role == "user":
                    lines.append(f"السؤال: {content}{timestamp}")
                elif role == "assistant":
                    lines.append(f"الإجابة: {content}{timestamp}")
                elif role == "system":
                    lines.append(f"النظام: {content}{timestamp}")

            elif template == "llama":
                if role == "user":
                    lines.append(f"[INST] {content} [/INST]")
                elif role == "assistant":
                    lines.append(f"{content}")

        return "\n".join(lines)

    def export_to_json(self, filepath: str) -> bool:
        """حفظ الذاكرة إلى ملف JSON"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(list(self.messages), f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ الذاكرة إلى {filepath}")
            return True
        except Exception as e:
            print(f"❌ خطأ في الحفظ: {e}")
            return False

    def import_from_json(self, filepath: str) -> bool:
        """تحميل الذاكرة من ملف JSON"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.clear()
            for msg in data:
                if "role" in msg and "content" in msg:
                    self.add(msg["role"], msg["content"])

            print(f"✅ تم تحميل الذاكرة من {filepath}")
            return True
        except Exception as e:
            print(f"❌ خطأ في التحميل: {e}")
            return False

    def get_conversation_summary(self) -> str:
        """ملخص سريع للمحادثة"""
        stats = self.get_stats()
        return (
            f"📊 ملخص المحادثة:\n"
            f"   • الرسائل: {stats['user_messages']} سؤال + {stats['assistant_messages']} إجابة\n"
            f"   • الرموز: {stats['total_tokens']}/{stats['max_tokens']}\n"
            f"   • النسبة: {stats['memory_usage_percent']:.1f}%"
        )

    def __len__(self) -> int:
        """عدد الرسائل"""
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"Memory(messages={len(self.messages)}/{self.max_history}, "
            f"tokens={self.total_tokens}/{self.max_tokens})"
        )

    def __str__(self) -> str:
        return self.get_conversation_summary()