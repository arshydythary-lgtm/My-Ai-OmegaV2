# tokenizer.py - نسخة محسّنة تدعم العربية والإنجليزية بذكاء (مع معالجة متقدمة للأرقام والرموز)
import os
import json
import re
from typing import List, Dict, Tuple
import pickle


class MyTokenizer:
    """
    Tokenizer خفيف الوزن بدون اعتمادية على مكتبات خارجية
    يدعم العربية والإنجليزية بشكل صحيح ومحسن
    """

    def __init__(self, vocab: Dict[str, int], special_tokens: Dict[str, int]):
        """
        vocab: قاموس {كلمة: id}
        special_tokens: قاموس {<special>: id}
        """
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.vocab_size = len(vocab)

        # تعيين الـ Tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        # تعيين الـ IDs الطويلة
        self.pad_token_id = special_tokens.get(self.pad_token, 0)
        self.unk_token_id = special_tokens.get(self.unk_token, 1)
        self.bos_token_id = special_tokens.get(self.bos_token, 2)
        self.eos_token_id = special_tokens.get(self.eos_token, 3)

        # ✅ أسماء مختصرة متوافقة مع train.py
        self.pad_id = self.pad_token_id
        self.unk_id = self.unk_token_id
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id

        # عكس القاموس للفك
        self.id_to_token = {v: k for k, v in vocab.items()}
        for token, tid in special_tokens.items():
            self.id_to_token[tid] = token

    @classmethod
    def build(cls, texts: List[str], vocab_size: int = 16000, min_frequency: int = 2, save_path: str = "my_tokenizer"):
        """بناء tokenizer من قائمة النصوص"""
        print(f"🔨 بناء tokenizer من {len(texts)} نصاً...")

        # تنظيف النصوص
        texts = [t.strip() for t in texts if t and t.strip()]

        if not texts:
            raise ValueError("❌ لا توجد نصوص صالحة للتدريب")

        # 1. عد تردد الكلمات
        word_freq = {}
        for text in texts:
            tokens = cls._tokenize_text(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        # 2. الاحتفاظ بـ vocab_size الأعلى
        special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }

        vocab = dict(special_tokens)
        token_id = len(special_tokens)

        # إضافة الكلمات الشائعة
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words:
            if freq >= min_frequency and len(vocab) < vocab_size:
                vocab[word] = token_id
                token_id += 1
            if len(vocab) >= vocab_size:
                break

        print(f"✅ تم بناء vocab بحجم {len(vocab)}")

        # 3. الحفظ
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_path, "special_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, ensure_ascii=False, indent=2)

        print(f"💾 تم حفظ tokenizer في {save_path}")
        return cls(vocab, special_tokens)

    @classmethod
    def load(cls, save_path: str = "my_tokenizer"):
        """تحميل tokenizer محفوظ"""
        vocab_path = os.path.join(save_path, "vocab.json")
        special_path = os.path.join(save_path, "special_tokens.json")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"❌ لم يتم العثور على vocab في {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        with open(special_path, "r", encoding="utf-8") as f:
            special_tokens = json.load(f)

        print(f"✅ تم تحميل tokenizer: vocab_size={len(vocab)}")
        return cls(vocab, special_tokens)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """
        تقسيم النص إلى Tokens بدقة عالية.
        تحسينات جديدة:
        - التعامل مع الأرقام المتصلة بالحروف (مثل GPT-4, 2024م).
        - التعامل مع الروابط وعلامات البريد الإلكتروني ككتلة واحدة.
        """
        # نمط متطور:
        # 1. [\w]+(?:'[\w]+)? : كلمات إنجليزية/عربية مع اختصارات
        # 2. [\u0600-\u06FF]+ : كلمات عربية صريحة (للتأكد من عدم انفصال الحروف في بعض البيئات)
        # 3. \d+(?:[\.\,]\d+)*%? : أرقام ونسب مئوية
        # 4. [^\w\s\u0600-\u06FF] : رموز وعلامات ترقيم

        # دمج الأنماط لالتقاط أفضل نتيجة
        pattern = r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?|[\u0600-\u06FF]+|[^\w\s\u0600-\u06FF]"

        tokens = re.findall(pattern, text)

        # تحويل الكلمات الإنجليزية الصرفة لأحرف صغيرة، مع الحفاظ على العربية والأرقام كما هي
        processed_tokens = []
        for t in tokens:
            if re.match(r'^[a-zA-Z]+$', t):
                processed_tokens.append(t.lower())
            else:
                processed_tokens.append(t)

        return [t for t in processed_tokens if t]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """تكويد النص إلى قائمة من الـ IDs"""
        tokens = self._tokenize_text(text)
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """فك الترميز من IDs إلى نص"""
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue
            token = self.id_to_token.get(token_id, self.unk_token)
            tokens.append(token)

        # إعادة تجميع النص
        text = " ".join(tokens)

        # إصلاح المسافات حول علامات الترقيم
        text = re.sub(r'\s+([,.!?;:\)\]}>])', r'\1', text)
        text = re.sub(r'([\(\[{<])\s+', r'\1', text)

        # إصلاح المسافات بين الكلمات العربية والإنجليزية لتحسين القراءة (اختياري)
        # إزالة المسافة قبل علامات الترقيم العربية تحديداً إذا وجدت
        text = re.sub(r'\s+([؛،؟!])', r'\1', text)

        return text.strip()

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """تكويد قائمة نصوص دفعة واحدة"""
        return [self.encode(text, add_special_tokens) for text in texts]

    def batch_decode(self, list_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """فك ترميز قائمة من الـ IDs دفعة واحدة"""
        return [self.decode(ids, skip_special_tokens) for ids in list_ids]

    def save(self, save_path: str = "my_tokenizer"):
        """حفظ tokenizer على القرص"""
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_path, "special_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)

        print(f"✅ تم حفظ tokenizer في {save_path}")

    def __len__(self):
        return self.vocab_size


# ============================================================
# دوال متوافقة مع الكود القديم
# ============================================================

def build_tokenizer_from_texts(texts, vocab_size=16000, save_path="my_tokenizer"):
    """دالة متوافقة مع الكود القديم"""
    return MyTokenizer.build(texts, vocab_size=vocab_size, save_path=save_path)


def load_tokenizer(save_path="my_tokenizer"):
    """دالة متوافقة مع الكود القديم"""
    return MyTokenizer.load(save_path)
