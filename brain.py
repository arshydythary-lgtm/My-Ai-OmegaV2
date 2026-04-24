# brain.py - نظام الذكاء والقرارات المحسّن
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import OrderedDict
import json


class Brain:
    """
    نظام الذاكرة والذكاء للمساعد
    - اتخاذ قرارات ذكية
    - إدارة السياق
    - إدارة الذاكرة والـ cache
    """

    def __init__(
            self,
            model_name: str = "أوميقا",
            max_context_length: int = 2000,
            max_history: int = 20,
            enable_cache: bool = True
    ):
        """
        Args:
            model_name: اسم المساعد
            max_context_length: أقصى طول للسياق
            max_history: أقصى عدد رسائل سابقة
            enable_cache: تفعيل الـ cache
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.max_history = max_history
        self.enable_cache = enable_cache

        # النصوص الافتراضية
        self.system_prompt = self._get_default_system_prompt()
        self.custom_instructions = ""

        # الـ Cache
        self.cache = {} if enable_cache else None
        self.cache_hits = 0
        self.cache_misses = 0

        # الإحصائيات
        self.stats = {
            "total_prompts_built": 0,
            "search_queries": 0,
            "file_reads": 0,
            "context_merges": 0,
        }

        # الكلمات المفتاحية
        self.keywords = self._load_keywords()

    def _get_default_system_prompt(self) -> str:
        """النص الافتراضي للنظام"""
        return f"""أنت مساعد ذكي اسمك {self.model_name}، دقيق، وتفكر قبل أن تجيب.

المميزات:
- تجاوب بشكل منطقي ومفيد
- تعتمد على المعلومات المتاحة فقط
- إذا لم تعرف الإجابة، قل "لا أعرف" بوضوح
- لا تختلق معلومات
- تتحدث باللغة العربية بطلاقة
- تجيب بإيجاز وفائدة

التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

    def _load_keywords(self) -> Dict[str, List[str]]:
        """تحميل الكلمات المفتاحية للقرارات"""
        return {
            "search": [
                "من هو", "ما هو", "كم", "متى", "أين", "كيف", "لماذا",
                "سعر", "أخبار", "آخر", "تحديث", "حالي", "اليوم", "الآن",
                "تعريف", "معنى", "شرح", "معلومات عن", "بحث عن",
                "حقائق", "إحصائيات", "أرقام", "نتائج", "بيانات حديثة"
            ],
            "file": [
                "ملف", "data", "csv", "اقرأ", "تحليل", "بيانات", "جداول",
                "المرفق", "المستند", "sheet", "excel", "pdf", "json",
                "قراءة", "معالجة", "استخراج", "من الملف", "الملف"
            ],
            "code": [
                "كود", "برنامج", "python", "javascript", "java", "sql",
                "اكتب", "حل", "خطأ", "debug", "compile", "run", "function"
            ],
            "math": [
                "حساب", "رياضيات", "معادلة", "احسب", "ناتج", "جذر",
                "مشتقة", "تكامل", "تربيع", "قسمة", "ضرب"
            ]
        }

    # ============================================================
    # اتخاذ القرار
    # ============================================================

    def decide(self, user_input: str, context: str = "") -> Dict[str, bool]:
        """
        تقرير نوع الإجابة المطلوبة بناءً على الإدخال والسياق

        Returns:
            قاموس يحتوي على القرارات
        """
        text = user_input.lower()
        context_lower = context.lower()

        decisions = {
            "use_search": False,
            "use_file": False,
            "use_code": False,
            "use_math": False,
            "formal": False,
            "detailed": False,
        }

        # فحص الكلمات المفتاحية
        for keyword in self.keywords["search"]:
            if keyword in text or keyword in context_lower:
                decisions["use_search"] = True
                break

        for keyword in self.keywords["file"]:
            if keyword in text or keyword in context_lower:
                decisions["use_file"] = True
                break

        for keyword in self.keywords["code"]:
            if keyword in text or keyword in context_lower:
                decisions["use_code"] = True
                break

        for keyword in self.keywords["math"]:
            if keyword in text or keyword in context_lower:
                decisions["use_math"] = True
                break

        # فحص العلامات
        if "؟" in user_input or "?" in user_input:
            decisions["formal"] = True

        if len(user_input.split()) > 20 or "شرح" in text or "أشرح" in text:
            decisions["detailed"] = True

        return decisions

    # ============================================================
    # إدارة السياق
    # ============================================================

    def _clean_text(self, text: str) -> str:
        """تنظيف النص"""
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text)
        # إزالة الأسطر الفارغة المتعددة
        text = re.sub(r'\n{3,}', '\n\n', text)
        # إزالة الرموز الغريبة
        text = re.sub(r'[^\w\s\.\,\!\؟\-\:\(\)،\—]', '', text)
        return text.strip()

    def _truncate_text(self, text: str, max_length: int) -> str:
        """قص النص بذكاء"""
        if len(text) <= max_length:
            return text

        # محاولة القص عند أقرب مسافة أو علامة ترقيم
        truncated = text[:max_length]

        # البحث عن آخر مسافة
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:  # إذا كانت قريبة بما يكفي
            truncated = truncated[:last_space]

        # البحث عن آخر علامة ترقيم
        last_punct = max(
            truncated.rfind('.'),
            truncated.rfind(','),
            truncated.rfind('!'),
            truncated.rfind('؟'),
        )
        if last_punct > max_length * 0.7:
            truncated = truncated[:last_punct + 1]

        return truncated.rstrip() + "..."

    def _merge_context(self, sections: List[str], separator: str = "\n\n---\n\n") -> str:
        """
        دمج أقسام السياق مع احترام الحد الأقصى للطول
        """
        self.stats["context_merges"] += 1

        merged = ""
        for section in sections:
            if not section or not section.strip():
                continue

            # التحقق من الحد الأقصى
            potential_length = len(merged) + len(separator) + len(section)
            if potential_length > self.max_context_length:
                # محاولة إضافة الجزء الأول فقط
                remaining = self.max_context_length - len(merged) - len(separator)
                if remaining > 100:  # إذا بقي متسع معقول
                    section = self._truncate_text(section, remaining)
                    merged += separator + section
                break

            merged += separator + section

        return merged.lstrip(separator)

    def _get_cache_key(self, data: Any) -> str:
        """توليد مفتاح للـ cache"""
        if isinstance(data, str):
            return hash(data) % 10 ** 9
        return hash(str(data)) % 10 ** 9

    def cache_get(self, key: str) -> Optional[str]:
        """الحصول من الـ cache"""
        if not self.enable_cache or not self.cache:
            return None

        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]["value"]

        self.cache_misses += 1
        return None

    def cache_set(self, key: str, value: str, ttl: int = 3600) -> None:
        """حفظ في الـ cache"""
        if not self.enable_cache or not self.cache:
            return

        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "ttl": ttl
        }

        # تنظيف الـ cache إذا كبر جداً
        if len(self.cache) > 1000:
            self.cache.clear()

    # ============================================================
    # بناء الـ Prompt
    # ============================================================

    def build_prompt(
            self,
            user_input: str,
            conversation_history: Optional[List[Dict]] = None,
            additional_context: Optional[str] = None,
    ) -> str:
        """
        بناء prompt شامل وذكي

        Args:
            user_input: السؤال الحالي
            conversation_history: سجل المحادثة
            additional_context: سياق إضافي

        Returns:
            prompt جاهز للنموذج
        """
        self.stats["total_prompts_built"] += 1

        # اتخاذ الق��ار
        decision = self.decide(user_input, additional_context or "")

        # بناء الأقسام
        sections = []

        # 1. نص النظام
        system_section = self.system_prompt
        if self.custom_instructions:
            system_section += f"\n\nتعليمات خاصة:\n{self.custom_instructions}"
        sections.append(("النظام", system_section))

        # 2. السياق الإضافي
        if additional_context:
            sections.append(("السياق الإضافي", additional_context))

        # 3. المحادثة السابقة
        if conversation_history:
            recent_msgs = conversation_history[-self.max_history:]
            conv_text = ""
            for msg in recent_msgs:
                role = msg.get("role", "unknown")
                content = msg.get("content", "").strip()
                if content:
                    content = self._clean_text(content)
                    conv_text += f"{role}: {content}\n"

            if conv_text:
                sections.append(("المحادثة السابقة", conv_text))

        # 4. بناء الـ prompt النهائي
        prompt = self.system_prompt.strip() + "\n\n"

        # إضافة السياق
        if additional_context:
            prompt += "=== معلومات إضافية ===\n"
            prompt += self._truncate_text(additional_context, self.max_context_length // 3)
            prompt += "\n\n"

        # إضافة المحادثة السابقة
        if conversation_history:
            prompt += "=== المحادثة ===\n"
            for msg in conversation_history[-self.max_history:]:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if content:
                    content = self._clean_text(content)
                    display_name = "أنت" if role == "user" else self.model_name
                    prompt += f"{display_name}: {content}\n"
            prompt += "\n"

        # إضافة السؤال الحالي
        prompt += f"=== السؤال ===\nأنت: {user_input}\n\n"
        prompt += f"{self.model_name}: "

        return prompt

    # ============================================================
    # إدارة الإعدادات
    # ============================================================

    def set_system_prompt(self, prompt: str) -> None:
        """تعيين نص النظام المخصص"""
        if prompt.strip():
            self.system_prompt = prompt

    def reset_system_prompt(self) -> None:
        """إعادة تعيين النص الافتراضي"""
        self.system_prompt = self._get_default_system_prompt()

    def set_custom_instructions(self, instructions: str) -> None:
        """إضافة تعليمات خاصة"""
        self.custom_instructions = instructions.strip()

    def get_personality(self) -> Dict[str, Any]:
        """الحصول على معلومات الشخصية"""
        return {
            "name": self.model_name,
            "system_prompt": self.system_prompt,
            "custom_instructions": self.custom_instructions,
            "max_context_length": self.max_context_length,
            "max_history": self.max_history,
        }

    def set_personality(self, config: Dict[str, Any]) -> None:
        """تعيين إعدادات الشخصية"""
        if "name" in config:
            self.model_name = config["name"]
        if "system_prompt" in config:
            self.set_system_prompt(config["system_prompt"])
        if "custom_instructions" in config:
            self.set_custom_instructions(config["custom_instructions"])

    # ============================================================
    # الإحصائيات والمعلومات
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات"""
        cache_stats = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.cache) if self.cache else 0,
        }

        if self.cache_hits + self.cache_misses > 0:
            cache_stats["hit_rate"] = (
                    self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            )

        return {
            **self.stats,
            **cache_stats
        }

    def clear_cache(self) -> None:
        """مسح الـ cache"""
        if self.cache:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0

    def export_config(self, filepath: str) -> bool:
        """حفظ الإعدادات إلى ملف"""
        try:
            config = {
                "name": self.model_name,
                "system_prompt": self.system_prompt,
                "custom_instructions": self.custom_instructions,
                "max_context_length": self.max_context_length,
                "max_history": self.max_history,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ خطأ في الحفظ: {e}")
            return False

    def import_config(self, filepath: str) -> bool:
        """تحميل الإعدادات من ملف"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.set_personality(config)
            return True
        except Exception as e:
            print(f"❌ خطأ في التحميل: {e}")
            return False

    def __repr__(self) -> str:
        return f"Brain(name='{self.model_name}', cache={'enabled' if self.enable_cache else 'disabled'})"

    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"🧠 {self.model_name}\n"
            f"   Prompts: {stats['total_prompts_built']}\n"
            f"   Cache Hit Rate: {stats.get('hit_rate', 0):.1f}%\n"
            f"   Cache Size: {stats['cache_size']}"
        )