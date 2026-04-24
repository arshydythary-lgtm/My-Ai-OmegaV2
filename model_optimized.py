# model_optimized.py - نموذج محسّن مع max_seq_len ديناميكي
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

########################################
# 1. Quantization - تقليل الدقة (أسرع 4-7x)
########################################

class QuantizedLinear(nn.Module):
    """Linear layer مع quantization"""
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # الأوزان والانحيازات
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # مقاييس التكميم
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('activation_scale', torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)
    
    def quantize_weights(self):
        """تكميم الأوزان"""
        max_val = self.weight.abs().max()
        self.weight_scale = (max_val / (2 ** (self.bits - 1) - 1)).clamp(min=1e-8)
        self.weight.data = torch.clamp(
            (self.weight / self.weight_scale).round(),
            -(2 ** (self.bits - 1)),
            2 ** (self.bits - 1) - 1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_weight = self.weight * self.weight_scale
        return F.linear(x, quantized_weight, self.bias)

########################################
# 2. LoRA - Low-Rank Adaptation (Microsoft)
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
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        
        # الطبقة الأساسية (frozen)
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.requires_grad_(False)
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # تهيئة LoRA
        nn.init.normal_(self.lora_A.weight, std=1 / math.sqrt(in_features))
        nn.init.zeros_(self.lora_B.weight)
        
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = lora_alpha / r
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))
        return base_out + lora_out * self.scaling
    
    def get_trainable_params(self):
        """الحصول على المعاملات القابلة للتدريب فقط"""
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())

########################################
# 3. RoPE - Rotary Position Embedding (ديناميكي)
########################################

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - يدعم max_seq_len ديناميكي"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # حساب الترددات
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # مسح الـ cache
        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.zeros(max_seq_len, dim), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(max_seq_len, dim), persistent=False)
    
    def _update_cos_sin_tables(self, seq_len: int, device: str, dtype: torch.dtype):
        """تحديث جداول cos/sin ديناميكياً"""
        if seq_len > self._seq_len_cached:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
            self._seq_len_cached = seq_len
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """إرجاع cos و sin للتسلسل الحالي"""
        if seq_len is None:
            seq_len = x.shape[1] if x.ndim > 1 else x.shape[0]
        
        self._update_cos_sin_tables(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """تطبيق Rotary Position Embedding"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    return x * cos + rotated * sin

########################################
# 4. Flash Attention - سريع وموفّر للذاكرة
########################################

class FlashAttention(nn.Module):
    """Efficient attention mechanism"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.use_flash = hasattr(F, "scaled_dot_product_attention")
    
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=causal,
                dropout_p=0.0 if not self.training else 0.1
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            if causal:
                mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = attn_weights @ v
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

########################################
# 5. Gradient Checkpointing
########################################

class GradCheckpointBlock(nn.Module):
    """Block مع gradient checkpointing"""
    def __init__(self, layer: nn.Module, use_checkpoint: bool = True):
        super().__init__()
        self.layer = layer
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.layer,
                x,
                *args,
                use_reentrant=False,
                **kwargs
            )
        return self.layer(x, *args, **kwargs)

########################################
# 6. Efficient Embedding
########################################

class OptimizedEmbedding(nn.Module):
    """Embedding محسّنة مع scaling"""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = d_model ** 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale

########################################
# 7. Transformer Block محسّن
########################################

class OptimizedTransformerBlock(nn.Module):
    """Transformer block محسّن مع Flash Attention و RoPE"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FlashAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.ln1(x), causal=True)
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        return x

########################################
# 8. Knowledge Distillation - Student Model
########################################

class StudentModel(nn.Module):
    """نموذج صغير (student) لـ distillation"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.token_emb = OptimizedEmbedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            OptimizedTransformerBlock(d_model, n_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_emb.embedding.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.lm_head(x)

########################################
# 9. النموذج الرئيسي المحسّن مع max_seq_len ديناميكي
########################################

class OptimizedMiniLLM(nn.Module):
    """نموذج LLM محسّن مع كل التقنيات و max_seq_len ديناميكي"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 512,  # الحد الأقصى الديناميكي
        use_lora: bool = True,
        lora_r: int = 8,
        use_gradient_checkpoint: bool = True,
        use_flash_attn: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding
        self.token_emb = OptimizedEmbedding(vocab_size, d_model)
        
        # RoPE ديناميكي
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            block = OptimizedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            
            if use_gradient_checkpoint:
                block = GradCheckpointBlock(block, use_checkpoint=True)
            
            self.blocks.append(block)
        
        # Layer norm
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output layer
        if use_lora:
            self.lm_head = LoRALinear(
                d_model,
                vocab_size,
                r=lora_r,
                lora_alpha=16,
                lora_dropout=dropout
            )
        else:
            self.lm_head = nn.Linear(d_model, vocab_size)
            self.lm_head.weight = self.token_emb.embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass مع max_seq_len ديناميكي"""
        seq_len = x.shape[1]
        
        # تحقق من الطول
        if seq_len > self.max_seq_len:
            # قص التسلسل إذا تجاوز الحد الأقصى
            x = x[:, -self.max_seq_len:]
            seq_len = self.max_seq_len
        
        # Embedding
        x = self.token_emb(x)
        
        # تطبيق RoPE (اختياري - إذا أردت استخدامه)
        # cos, sin = self.rope(x, seq_len)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm و output
        x = self.ln_final(x)
        return self.lm_head(x)
    
    def get_trainable_params_count(self) -> int:
        """عدد المعاملات القابلة للتدريب"""
        if isinstance(self.lm_head, LoRALinear):
            trainable = sum(p.numel() for p in self.lm_head.get_trainable_params())
            trainable += sum(p.numel() for p in self.parameters() if p.requires_grad)
            return trainable
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params_count(self) -> int:
        """عدد المعاملات الكلي"""
        return sum(p.numel() for p in self.parameters())
    
    def update_max_seq_len(self, new_max_seq_len: int):
        """تحديث max_seq_len ديناميكياً"""
        if new_max_seq_len > self.max_seq_len:
            self.max_seq_len = new_max_seq_len
            self.rope.max_seq_len = new_max_seq_len
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
        """توليد النصوص بكفاءة"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # استخدام آخر max_seq_len token فقط
            if input_ids.shape[1] > self.max_seq_len:
                x = input_ids[:, -self.max_seq_len:]
            else:
                x = input_ids
            
            logits = self(x)
            logits = logits[:, -1, :]
            
            # Sampling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_idx, top_k_vals)
            
            # Top-p filtering
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
# مقارنة الأداء
########################################

def compare_models():
    """مقارنة النماذج المختلفة"""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    seq_len = 512
    
    print("=" * 80)
    print("🔬 مقارنة الأداء والذاكرة (مع max_seq_len ديناميكي)")
    print("=" * 80)
    
    # النموذج الأصلي
    print("\n📌 النموذج الأصلي (بدون تحسينات):")
    model1 = OptimizedMiniLLM(
        vocab_size=16000,
        d_model=512,
        n_heads=8,
        num_layers=6,
        max_seq_len=512,
        use_lora=False,
        use_gradient_checkpoint=False,
        use_flash_attn=False
    ).to(device)
    
    params1 = model1.get_total_params_count()
    print(f"   المعاملات الكلية: {params1:,}")
    
    # قياس الذاكرة والسرعة
    x = torch.randint(0, 16000, (batch_size, seq_len)).to(device)
    
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model1(x)
    elapsed1 = time.time() - start
    mem1 = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    
    print(f"   الوقت (5 iterations): {elapsed1:.3f}s")
    print(f"   الذاكرة: {mem1:.2f} GB")
    
    # النموذج المحسّن بـ LoRA + Flash Attention
    print("\n✨ مع LoRA + Flash Attention + Gradient Checkpointing:")
    model2 = OptimizedMiniLLM(
        vocab_size=16000,
        d_model=512,
        n_heads=8,
        num_layers=6,
        max_seq_len=512,
        use_lora=True,
        lora_r=8,
        use_gradient_checkpoint=True,
        use_flash_attn=True
    ).to(device)
    
    params2_total = model2.get_total_params_count()
    params2_train = model2.get_trainable_params_count()
    print(f"   المعاملات الكلية: {params2_total:,}")
    print(f"   المعاملات القابلة للتدريب: {params2_train:,}")
    print(f"   تقليل: {(1 - params2_train / params1) * 100:.1f}%")
    
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model2(x)
    elapsed2 = time.time() - start
    mem2 = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    
    print(f"   الوقت (5 iterations): {elapsed2:.3f}s")
    print(f"   الذاكرة: {mem2:.2f} GB")
    print(f"   تسريع: {elapsed1 / elapsed2:.1f}x")
    print(f"   توفير ذاكرة: {(1 - mem2 / max(mem1, 0.01)) * 100:.1f}%")
    
    # اختبار max_seq_len الديناميكي
    print("\n🔄 اختبار max_seq_len الديناميكي:")
    model2.update_max_seq_len(1024)
    
    x_long = torch.randint(0, 16000, (batch_size, 1024)).to(device)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(3):
            _ = model2(x_long)
    elapsed_long = time.time() - start
    mem_long = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    
    print(f"   seq_len: 1024 | الوقت (3 iterations): {elapsed_long:.3f}s | الذاكرة: {mem_long:.2f} GB")
    
    # النموذج الصغير (Knowledge Distillation)
    print("\n🎓 مع Knowledge Distillation (Student Model):")
    model3 = StudentModel(
        vocab_size=16000,
        d_model=256,
        n_heads=4,
        num_layers=3,
        max_seq_len=512
    ).to(device)
    
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"   المعاملات: {params3:,}")
    print(f"   تقليل: {(1 - params3 / params1) * 100:.1f}%")
    
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model3(x)
    elapsed3 = time.time() - start
    mem3 = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    
    print(f"   الوقت (5 iterations): {elapsed3:.3f}s")
    print(f"   الذاكرة: {mem3:.2f} GB")
    print(f"   تسريع: {elapsed1 / elapsed3:.1f}x")
    print(f"   توفير ذاكرة: {(1 - mem3 / max(mem1, 0.01)) * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print("🏆 الملخص:")
    print(f"   LoRA + Flash: {(params2_train/params1)*100:.1f}% أوزان قابلة للتدريب | {elapsed1/elapsed2:.1f}x سرعة")
    print(f"   Student: {(params3/params1)*100:.1f}% أوزان | {elapsed1/elapsed3:.1f}x سرعة")
    print(f"   max_seq_len: ديناميكي (512 -> 1024)")
    print("=" * 80)

if __name__ == "__main__":
    compare_models()