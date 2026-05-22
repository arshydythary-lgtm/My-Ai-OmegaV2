# model_optimized.py - نسخة مصححة تماماً (RoPE مفعّل، LoRA r=32, d_model=1024)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


########################################
# 1. Quantization (محفوظ للمستقبل - غير مفعل افتراضياً في التدريب العادي)
########################################
class QuantizedLinear(nn.Module):
    """Linear layer مع quantization"""

    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('weight_scale', torch.ones(out_features))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # في التدريب نستخدم الأوزان الكاملة لضمان استقرار التدرجات
        return F.linear(x, self.weight, self.bias)


########################################
# 2. LoRA - Low-Rank Adaptation
########################################
class LoRALinear(nn.Module):
    """Linear layer مع LoRA adaptation"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        # الطبقة الأساسية مجمدة
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.requires_grad_(False)

        # محولات LoRA
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # تهيئة صحيحة لـ LoRA (صفر لـ B وتوزيع طبيعي لـ A)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = lora_alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))
        return base_out + lora_out * self.scaling

    def get_trainable_params(self):
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())


########################################
# 3. RoPE - Rotary Position Embedding
########################################
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - يدعم max_seq_len ديناميكي"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.zeros(max_seq_len, dim), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(max_seq_len, dim), persistent=False)

    def _update_cos_sin_tables(self, seq_len: int, device: str, dtype: torch.dtype):
        if seq_len > self._seq_len_cached:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
            self._seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        self._update_cos_sin_tables(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """تطبيق Rotary Position Embedding على Q و K"""
    # x shape: [B, H, T, D]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)

    # إضافة أبعاد للبث (Broadcasting) ليتوافق مع [B, H, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + rotated * sin


########################################
# 4. Flash Attention مع دعم RoPE الصريح
########################################
class FlashAttention(nn.Module):
    """Efficient attention mechanism مع تطبيق RoPE داخلي"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        self.qkv = LoRALinear(
            d_model,
            d_model * 3,
            r=8,
            lora_alpha=16,
            lora_dropout=dropout
        )
        self.out_proj = LoRALinear(
            d_model,
            d_model,
            r=8,
            lora_alpha=16,
            lora_dropout=dropout
        )
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # إعادة التشكيل: [B, T, C] -> [B, H, T, HeadDim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # >>> التصحيح الجوهري: تطبيق RoPE هنا قبل الانتباه <<<
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=causal,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            if causal:
                mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


########################################
# 5. Gradient Checkpointing Wrapper (مصحح)
########################################
class GradCheckpointBlock(nn.Module):
    """Block مع gradient checkpointing محدد المدخلات"""

    def __init__(self, layer: nn.Module, use_checkpoint: bool = True):
        super().__init__()
        self.layer = layer
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # تمرير المدخلات بشكل صريح لضمان عمل checkpoint بشكل صحيح
            return torch.utils.checkpoint.checkpoint(
                self.layer,
                x, cos, sin,
                use_reentrant=False
            )
        return self.layer(x, cos, sin)


########################################
# 6. Transformer Block المحسّن
########################################
class OptimizedTransformerBlock(nn.Module):
    """Transformer block محسّن مع Flash Attention و RoPE"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FlashAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # تمرير cos/sin للانتباه
        x = x + self.attn(self.ln1(x), cos, sin, causal=True)
        x = x + self.ff(self.ln2(x))
        return x


########################################
# 7. Student Model (للـ Distillation)
########################################
class StudentModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, num_layers: int = 3,
                 max_seq_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_seq_len)
        self.blocks = nn.ModuleList([
            OptimizedTransformerBlock(d_model, n_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        cos, sin = self.rope(x, seq_len)
        x = self.token_emb(x) * math.sqrt(self.d_model if hasattr(self, 'd_model') else 256)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln(x)
        return self.lm_head(x)


########################################
# 8. النموذج الرئيسي OptimizedMiniLLM
########################################
class OptimizedMiniLLM(nn.Module):
    """نموذج LLM محسّن مع كل التقنيات و max_seq_len ديناميكي"""

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 1024,
            n_heads: int = 16,
            num_layers: int = 16,
            max_seq_len: int = 512,
            use_lora: bool = True,
            lora_r: int = 8,
            use_gradient_checkpoint: bool = True,
            dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads

        # Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # RoPE
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

        # Layers
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = OptimizedTransformerBlock(d_model, n_heads, dropout=dropout)
            if use_gradient_checkpoint:
                block = GradCheckpointBlock(block, use_checkpoint=True)
            self.blocks.append(block)

        self.ln_final = nn.LayerNorm(d_model)

        # Output Head
        if use_lora:
            self.lm_head = LoRALinear(d_model, vocab_size, r=lora_r, lora_alpha=lora_r * 16, lora_dropout=dropout)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

        # freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # unfreeze LoRA فقط
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.weight.requires_grad = True
                module.lora_B.weight.requires_grad = True

        # LayerNorm optional
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]

        # التعامل مع التسلسلات الأطول من الحد المسموح
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        # حساب جداول الموقع مرة واحدة لكل الباتش
        cos, sin = self.rope(x, seq_len)

        # Embedding + Scaling
        x = self.token_emb(x) * math.sqrt(self.d_model)

        # المرور عبر الطبقات (مع تمرير cos/sin)
        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.ln_final(x)
        return self.lm_head(x)

    def get_trainable_params_count(self) -> int:
        if isinstance(self.lm_head, LoRALinear):
            trainable = sum(p.numel() for p in self.lm_head.get_trainable_params())
            # إضافة أي معاملات أخرى قابلة للتدريب (مثل LayerNorm)
            trainable += sum(p.numel() for name, p in self.named_parameters()
                             if p.requires_grad and 'lora' not in name and 'base_layer' not in name)
            return trainable
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def update_max_seq_len(self, new_max_seq_len: int):
        if new_max_seq_len > self.max_seq_len:
            self.max_seq_len = new_max_seq_len
            self.rope.max_seq_len = new_max_seq_len
            # إعادة تهيئة المخزن المؤقت لـ RoPE
            self.rope._seq_len_cached = 0
            print(f"✅ تم تحديث max_seq_len إلى {new_max_seq_len}")

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 100,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            device: str = "cpu"
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.max_seq_len:
                x = input_ids[:, -self.max_seq_len:]
            else:
                x = input_ids

            logits = self(x)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_idx, top_k_vals)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumsum_probs <= top_p
                sorted_mask[..., 0] = True
                sorted_logits[~sorted_mask] = float('-inf')
                logits = torch.scatter(logits, -1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


########################################
# مقارنة الأداء والتحقق
########################################
def compare_models():
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4  # تقليل الحجم للاختبار السريع
    seq_len = 128

    print("=" * 60)
    print("🔬 فحص النموذج المصحح (RoPE + LoRA + Checkpoint)")
    print("=" * 60)

    try:
        model = OptimizedMiniLLM(
            vocab_size=16000,
            d_model=1024,
            n_heads=16,
            num_layers=4,  # عدد قليل للاختبار
            use_lora=True,
            lora_r=8,
            use_gradient_checkpoint=True
        ).to(device)

        total_params = model.get_total_params_count()
        train_params = model.get_trainable_params_count()

        print(f"✅ تم إنشاء النموذج بنجاح على {device}")
        print(f"📊 المعاملات الكلية: {total_params:,}")
        print(f"🎯 المعاملات القابلة للتدريب (LoRA): {train_params:,}")
        print(f"💾 نسبة التدريب: {(train_params / total_params) * 100:.2f}%")

        x = torch.randint(0, 16000, (batch_size, seq_len)).to(device)

        # اختبار Forward Pass
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        start = time.time()
        with torch.no_grad():
            out = model(x)
        elapsed = time.time() - start
        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

        print(f"⚡ وقت الاستدلال (Batch {batch_size}): {elapsed:.4f}s")
        print(f"🧠 استهلاك الذاكرة: {mem:.2f} GB")
        print(f"📝 شكل المخرج: {out.shape} (المتوقع: [{batch_size}, {seq_len}, 16000])")

        # اختبار بسيط للتدريب (Backward)
        model.train()
        loss = model(x).sum()
        loss.backward()
        print("✅ نجح حساب التدرجات (Backward Pass) مع Gradient Checkpointing!")

    except Exception as e:
        print(f"❌ حدث خطأ: {e}")


if __name__ == "__main__":
    compare_models()
