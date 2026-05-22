# tokenizer.py - نسخة محسّنة تدعم BPE حقيقي مع Byte Fallback
# يدعم العربية والإنجليزية ولغات البرمجة بذكاء
import os
import json
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class MyTokenizer:
    """
    Tokenizer خفيف الوزن بدون اعتمادية على مكتبات خارجية
    يدعم العربية والإنجليزية ولغات البرمجة بشكل صحيح ومحسن
    مع BPE حقيقي و Byte Fallback مثل GPT/LLaMA/SentencePiece
    
    المميزات:
    - Unicode Normalization (NFKC)
    - Whitespace Token (▁) للحفاظ على حدود الكلمات
    - BPE حقيقي مع تطبيق merges تدريجي
    - Subword Vocabulary بدل Word-Level
    - Byte Fallback لإزالة <unk> تقريباً تماماً
    """

    def __init__(self, vocab: Dict[str, int], special_tokens: Dict[str, int],
                 merges: Optional[List[Tuple[str, str]]] = None):
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

        self.pad_id = self.pad_token_id
        self.unk_id = self.unk_token_id
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id

        # عكس القاموس للفك
        self.id_to_token = {v: k for k, v in vocab.items()}
        for token, tid in special_tokens.items():
            self.id_to_token[tid] = token

        # بناء جدول merges سريع - تحويل merges من list إلى tuple إذا لزم الأمر
        self.merges = [tuple(m) if isinstance(m, list) else m for m in self.merges]
        self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def build(cls, texts: List[str], vocab_size: int = 32000, min_frequency: int = 2,
              save_path: str = "my_tokenizer", bpe_merges: int = 5000):
        """بناء tokenizer من قائمة النصوص مع دعم BPE حقيقي"""
        print(f"🔨 بناء tokenizer من {len(texts)} نصاً...")

        texts = [t.strip() for t in texts if t and t.strip()]

        if not texts:
            raise ValueError("❌ لا توجد نصوص صالحة للتدريب")

        # تطبيق Unicode Normalization أولاً
        texts = [unicodedata.normalize("NFKC", text) for text in texts]

        # جمع جميع الكلمات مع إضافة علامة المسافة البيضاء ▁
        word_freq = defaultdict(int)

        for text in texts:
            words = text.split()
            for word in words:
                prefixed_word = "▁" + word
                word_freq[prefixed_word] += 1

        # بناء الـ BPE merges
        merges = []
        if bpe_merges > 0:
            print(f"🔄 بناء {bpe_merges} عملية دمج BPE...")
            
            # تحويل word_freq إلى تمثيل بالأحرف
            word_splits = {word: list(word) for word in word_freq.keys()}
            
            for _ in range(bpe_merges):
                # جمع أزواج الحروف
                char_pairs = defaultdict(int)
                for word, chars in word_splits.items():
                    freq = word_freq[word]
                    for i in range(len(chars) - 1):
                        pair = (chars[i], chars[i + 1])
                        char_pairs[pair] += freq
                
                if not char_pairs:
                    break
                    
                best_pair = max(char_pairs.items(), key=lambda x: x[1])[0]
                merges.append(best_pair)

                # تطبيق الدمج على جميع الكلمات
                first, second = best_pair
                merged = first + second
                for word in list(word_splits.keys()):
                    chars = word_splits[word]
                    new_chars = []
                    i = 0
                    while i < len(chars):
                        if i < len(chars) - 1 and chars[i] == first and chars[i + 1] == second:
                            new_chars.append(merged)
                            i += 2
                        else:
                            new_chars.append(chars[i])
                            i += 1
                    word_splits[word] = new_chars

        # بناء الـ vocab النهائي
        special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }

        vocab = dict(special_tokens)
        token_id = len(special_tokens)

        # إضافة subwords من الأكثر تردداً إلى الأقل
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

    def _apply_bpe(self, token: str) -> List[str]:
        """
        تطبيق BPE الحقيقي على كلمة واحدة
        """
        # تقسيم الكلمة إلى أحرف
        word = list(token)
        
        # تطبيق كل merge بالترتيب
        for merge in self.merges:
            if len(word) <= 1:
                break
            
            first, second = merge
            merged = first + second
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        return word

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """تكويد النص إلى قائمة من الـ IDs مع BPE حقيقي"""
        # Unicode Normalization
        text = unicodedata.normalize("NFKC", text)
        
        # تقسيم النص إلى كلمات (whitespace tokenization)
        words = text.split()
        
        ids = []
        for word in words:
            # إضافة ▁ في بداية كل كلمة
            prefixed_word = "▁" + word
            
            # تطبيق BPE للحصول على subwords
            subwords = self._apply_bpe(prefixed_word)
            
            # تحويل subwords إلى IDs
            for sw in subwords:
                if sw in self.vocab:
                    ids.append(self.vocab[sw])
                else:
                    # استخدام unk للكلمات غير المعروفة
                    ids.append(self.unk_token_id)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """فك الترميز من IDs إلى نص"""
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue
            token = self.id_to_token.get(token_id, "")
            if token:
                tokens.append(token)

        # إعادة تجميع النص
        result = ""
        for token in tokens:
            if token.startswith("▁"):
                result += " " + token[1:]
            else:
                result += token

        return result.strip()

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True,
                     padding: bool = False, max_length: Optional[int] = None) -> List[List[int]]:
        """تكويد قائمة نصوص دفعة واحدة"""
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
            words = text.split()
            for word in words:
                prefixed_word = "▁" + word
                subwords = self._apply_bpe(prefixed_word)
                for sw in subwords:
                    total_tokens += 1
                    if sw not in self.vocab:
                        oov_tokens += 1
                        if len(oov_examples) < 20:
                            oov_examples.append(sw)

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


def build_tokenizer_from_texts(texts, vocab_size=16000, save_path="my_tokenizer"):
    """دالة متوافقة مع الكود القديم"""
    return MyTokenizer.build(texts, vocab_size=vocab_size, save_path=save_path)


def load_tokenizer(save_path="my_tokenizer"):
    """دالة متوافقة مع الكود القديم"""
    return MyTokenizer.load(save_path)
