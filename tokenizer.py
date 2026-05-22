# tokenizer.py - نسخة محسّنة تدعم العربية والإنجليزية ولغات البرمجة بذكاء
import os
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class MyTokenizer:
    """
    Tokenizer خفيف الوزن بدون اعتمادية على مكتبات خارجية
    يدعم العربية والإنجليزية ولغات البرمجة بشكل صحيح ومحسن
    مع تحسينات BPE مصغرة للأداء الاحترافي
    """

    def __init__(self, vocab: Dict[str, int], special_tokens: Dict[str, int],
                 merges: Optional[List[Tuple[str, str]]] = None):
        """
        vocab: قاموس {كلمة/توكن: id}
        special_tokens: قاموس {<special>: id}
        merges: قائمة عمليات الدمج لـ BPE (اختياري)
        """
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.merges = merges or []
        self.vocab_size = len(vocab)

        # تعيين الـ Tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        # تعيين الـ IDs
        self.pad_token_id = special_tokens.get(self.pad_token, 0)
        self.unk_token_id = special_tokens.get(self.unk_token, 1)
        self.bos_token_id = special_tokens.get(self.bos_token, 2)
        self.eos_token_id = special_tokens.get(self.eos_token, 3)

        # أسماء مختصرة متوافقة مع train.py
        self.pad_id = self.pad_token_id
        self.unk_id = self.unk_token_id
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id

        # عكس القاموس للفك
        self.id_to_token = {v: k for k, v in vocab.items()}
        for token, tid in special_tokens.items():
            self.id_to_token[tid] = token

        # بناء جدول merges سريع
        self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def build(cls, texts: List[str], vocab_size: int = 32000, min_frequency: int = 2,
              save_path: str = "my_tokenizer", bpe_merges: int = 5000):
        """بناء tokenizer من قائمة النصوص مع دعم BPE"""
        print(f"🔨 بناء tokenizer من {len(texts)} نصاً...")

        texts = [t.strip() for t in texts if t and t.strip()]

        if not texts:
            raise ValueError("❌ لا توجد نصوص صالحة للتدريب")

        # عد تردد الكلمات والرموز
        word_freq = defaultdict(int)
        char_pairs = defaultdict(int)

        for text in texts:
            tokens = cls._tokenize_text(text)
            for token in tokens:
                word_freq[token] += 1

            # جمع أزواج الحروف لـ BPE
            for token in tokens:
                chars = list(token)
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    char_pairs[pair] += 1

        # بناء الـ BPE merges
        merges = []
        if bpe_merges > 0:
            print(f"🔄 بناء {bpe_merges} عملية دمج BPE...")
            for _ in range(bpe_merges):
                if not char_pairs:
                    break
                best_pair = max(char_pairs.items(), key=lambda x: x[1])[0]
                merges.append(best_pair)

                new_freq = {}
                for token, freq in word_freq.items():
                    new_token = cls._apply_merge(token, best_pair)
                    new_freq[new_token] = new_freq.get(new_token, 0) + freq

                char_pairs.clear()
                for token in new_freq.keys():
                    chars = list(token)
                    for i in range(len(chars) - 1):
                        pair = (chars[i], chars[i + 1])
                        char_pairs[pair] += 1

                word_freq = new_freq

        # بناء الـ vocab النهائي
        special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }

        vocab = dict(special_tokens)
        token_id = len(special_tokens)

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words:
            if freq >= min_frequency and len(vocab) < vocab_size:
                vocab[word] = token_id
                token_id += 1
            if len(vocab) >= vocab_size:
                break

        print(f"✅ تم بناء vocab بحجم {len(vocab)} مع {len(merges)} عملية دمج BPE")

        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_path, "special_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_path, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(merges, f, ensure_ascii=False, indent=2)

        print(f"💾 تم حفظ tokenizer في {save_path}")
        return cls(vocab, special_tokens, merges)

    @classmethod
    def load(cls, save_path: str = "my_tokenizer"):
        """تحميل tokenizer محفوظ"""
        vocab_path = os.path.join(save_path, "vocab.json")
        special_path = os.path.join(save_path, "special_tokens.json")
        merges_path = os.path.join(save_path, "merges.json")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"❌ لم يتم العثور على vocab في {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        with open(special_path, "r", encoding="utf-8") as f:
            special_tokens = json.load(f)

        merges = []
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                merges = json.load(f)

        print(f"✅ تم تحميل tokenizer: vocab_size={len(vocab)}, merges={len(merges)}")
        return cls(vocab, special_tokens, merges)

    @staticmethod
    def _apply_merge(token: str, pair: Tuple[str, str]) -> str:
        """تطبيق دمج BPE على توكن"""
        first, second = pair
        return token.replace(first + second, first + second)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """
        تقسيم النص إلى Tokens بدقة عالية.
        دعم ممتاز للعربية والإنجليزية ولغات البرمجة
        """
        patterns = [
            # روابط وبريد إلكتروني
            r'https?://[^\s]+|[\w\.-]+@[\w\.-]+\.\w+',
            # كلمات برمجية ومعرفيات (متغيرات، دوال، كلاسات)
            r'[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*',
            # أرقام مركبة (عشرية، نسب، عملات، تواريخ)
            r'\d+(?:[\.]\d+)*(?:%|م|هـ|€|\$|£|¥)?',
            # كلمات عربية
            r'[\u0600-\u06FF]+(?:[\u0600-\u06FF]|[\u064B-\u065F])*',
            # رموز برمجة وعمليات
            r'[+\-*/%=<>!&|^~?:]+',
            # أقواس ورموز خاصة
            r'[\(\)\[\]{};,\.\'\"]',
            # أي كلمة أخرى
            r'\w+|[^\w\s]',
        ]

        pattern = '|'.join(f'({p})' for p in patterns)
        tokens = []

        for match in re.finditer(pattern, text):
            token = match.group(0)
            if token:
                tokens.append(token)

        # lowercase للإنجليزية فقط (ليس للكود)
        processed_tokens = []
        for t in tokens:
            if re.match(r'^[a-zA-Z]+$', t) and '_' not in t and '.' not in t:
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
        """فك الترميز من IDs إلى نص مع تحسينات للعربية ولغات البرمجة"""
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue
            token = self.id_to_token.get(token_id, self.unk_token)
            tokens.append(token)

        # إعادة تجميع النص بدقة
        text_parts = []
        prev_was_arabic = False
        prev_was_code_symbol = False

        for i, token in enumerate(tokens):
            is_arabic = bool(re.match(r'^[\u0600-\u06FF]+$', token))
            is_code_symbol = bool(re.match(r'^[+\-*/%=<>!&|^~?:;\(\)\[\]{}]+$', token))
            is_punctuation = bool(re.match(r'^[,.!?;:\'\"]+$', token))

            if i > 0:
                if is_code_symbol or is_punctuation:
                    text_parts.append(token)
                elif prev_was_code_symbol:
                    text_parts.append(token)
                elif prev_was_arabic and is_arabic:
                    text_parts.append(' ')
                    text_parts.append(token)
                else:
                    text_parts.append(' ')
                    text_parts.append(token)
            else:
                text_parts.append(token)

            prev_was_arabic = is_arabic
            prev_was_code_symbol = is_code_symbol

        text = ''.join(text_parts)

        # إصلاح المسافات حول علامات الترقيم
        text = re.sub(r'\s+([,.!?;:\)\]}>])', r'\1', text)
        text = re.sub(r'([\(\[{<])\s+', r'\1', text)
        text = re.sub(r'\s+([؛،?!])', r'\1', text)

        # إصلاح المسافات حول رموز البرمجة
        text = re.sub(r'\s+([+\-*/%=<>!&|^~?:])', r'\1', text)
        text = re.sub(r'([+\-*/%=<>!&|^~?:])\s+', r'\1', text)

        return text.strip()

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True,
                     padding: bool = False, max_length: Optional[int] = None) -> List[List[int]]:
        """تكويد قائمة نصوص دفعة واحدة مع تحسينات الأداء"""
        encoded = []
        for text in texts:
            ids = self.encode(text, add_special_tokens)

            if max_length and len(ids) > max_length:
                ids = ids[:max_length]

            encoded.append(ids)

        if padding and encoded:
            max_len = max(len(seq) for seq in encoded)
            if max_length:
                max_len = min(max_len, max_length)

            for i, seq in enumerate(encoded):
                pad_length = max_len - len(seq)
                if pad_length > 0:
                    encoded[i] = seq + [self.pad_token_id] * pad_length

        return encoded

    def batch_decode(self, list_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """فك ترميز قائمة من الـ IDs دفعة واحدة"""
        return [self.decode(ids, skip_special_tokens) for ids in list_ids]

    def validate(self, texts: List[str], sample_size: int = 100) -> Dict:
        """فحص جودة التوكنيزر وحساب نسبة OOV"""
        if not texts:
            return {"error": "لا توجد نصوص للتحقق"}

        sample = texts[:sample_size] if len(texts) > sample_size else texts
        total_tokens = 0
        oov_tokens = 0
        oov_examples = []

        for text in sample:
            tokens = self._tokenize_text(text)
            for token in tokens:
                total_tokens += 1
                if token not in self.vocab:
                    oov_tokens += 1
                    if len(oov_examples) < 20:
                        oov_examples.append(token)

        oov_rate = (oov_tokens / total_tokens * 100) if total_tokens > 0 else 0

        return {
            "total_tokens": total_tokens,
            "oov_tokens": oov_tokens,
            "oov_rate": f"{oov_rate:.2f}%",
            "vocab_size": len(self.vocab),
            "oov_examples": oov_examples[:10],
            "quality": "ممتاز" if oov_rate < 5 else "جيد" if oov_rate < 15 else "يحتاج تحسين"
        }

    def save(self, save_path: str = "my_tokenizer"):
        """حفظ tokenizer على القرص"""
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_path, "special_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_path, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False, indent=2)

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
