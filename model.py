# model.py - نموذج MiniLLM محسّن (متوافق مع الجميع)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


########################################
# RoPE (Rotary Position Embedding) - ديناميكي
########################################

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding مع دعم ديناميكي"""

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
        """تحديث جداول cos/sin ديناميكياً"""
        if seq_len > self._seq_len_cached:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
            self._seq_len_cached = seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """إرجاع cos و sin للتسلسل الحالي"""
        self._update_cos_sin_tables(seq_len, self.inv_freq.device, self.inv_freq.dtype)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """تطبيق Rotary Position Embedding"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + rotated * sin


########################################
# Multi-Head Attention مع Flash Attention
########################################

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) يجب أن يكون قابل للقسمة على n_heads ({n_heads})"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_prob = dropout

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Causal mask
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        # Flash Attention support
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
            self,
            x: torch.Tensor,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(seq_len=T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV cache
        if past_kv is not None:
            k_prev, v_prev = past_kv
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_kv = (k.detach(), v.detach()) if use_cache else None

        # Attention (Flash أو Standard)
        if self.use_flash and not use_cache:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=False
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)
            attn_output = attn_weights @ v

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output), new_kv


########################################
# Feed-Forward Network
########################################

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


########################################
# Gradient Checkpointing Block
########################################

class GradCheckpointBlock(nn.Module):
    """Block مع Gradient Checkpointing"""

    def __init__(self, layer: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.layer = layer
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, *args, **kwargs):
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
# Transformer Block
########################################

class Block(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            max_seq_len: int,
            dropout: float = 0.1,
            use_gradient_checkpoint: bool = False
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

        self.use_gradient_checkpoint = use_gradient_checkpoint

        if use_gradient_checkpoint:
            self.checkpoint_attn = GradCheckpointBlock(self._attn_block, use_checkpoint=True)
            self.checkpoint_ff = GradCheckpointBlock(self._ff_block, use_checkpoint=True)

    def _attn_block(self, x: torch.Tensor, past_kv=None, use_cache=False):
        """Attention block للـ checkpointing"""
        attn_out, new_kv = self.attn(self.ln1(x), past_kv, use_cache)
        x = x + attn_out
        if use_cache:
            return x, new_kv
        return x

    def _ff_block(self, x: torch.Tensor):
        """FF block للـ checkpointing"""
        return x + self.ff(self.ln2(x))

    def forward(
            self,
            x: torch.Tensor,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False
    ):
        if self.use_gradient_checkpoint:
            # Self-attention with checkpointing
            if use_cache:
                x, new_kv = self.checkpoint_attn(x, past_kv, use_cache)
            else:
                x = self.checkpoint_attn(x, past_kv, use_cache)

            # Feed-forward with checkpointing
            x = self.checkpoint_ff(x)

            if use_cache:
                return x, new_kv
            return x
        else:
            # بدون checkpointing
            attn_out, new_kv = self.attn(self.ln1(x), past_kv, use_cache)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))

            if use_cache:
                return x, new_kv
            return x


########################################
# MiniLLM Model - محسّن
########################################

class MiniLLM(nn.Module):
    """نموذج MiniLLM محسّن مع جميع التحسينات"""

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 1024,
            n_heads: int = 16,
            num_layers: int = 16,
            max_seq_len: int = 512,
            dropout: float = 0.1,
            use_gradient_checkpoint: bool = False,
            use_flash_attn: bool = True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            Block(
                d_model,
                n_heads,
                max_seq_len,
                dropout,
                use_gradient_checkpoint=use_gradient_checkpoint
            )
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """تهيئة الأوزان"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """للتدريب"""
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_new_tokens: int = 200,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            repetition_penalty: float = 1.2,
            device: str = "cpu"
    ) -> torch.Tensor:
        """توليد نص من prompt"""
        self.eval()

        batch_size = prompt_ids.shape[0]
        generated = prompt_ids.clone()
        past_kvs = [None] * self.num_layers

        for _ in range(max_new_tokens):
            # فقط آخر token
            if generated.shape[1] > self.max_seq_len:
                x = self.token_emb(generated[:, -1:])
            else:
                x = self.token_emb(generated[:, -1:])

            new_kvs = []
            for i, block in enumerate(self.blocks):
                x, kv = block(x, past_kvs[i], use_cache=True)
                new_kvs.append(kv)

            x = self.ln_final(x)
            logits = self.lm_head(x)[:, -1, :]

            # Apply repetition penalty
            for token_id in set(generated[0].tolist()):
                logits[:, token_id] /= repetition_penalty

            # Sampling
            next_token = self._sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            generated = torch.cat([generated, next_token], dim=1)

            # Check for max length
            if generated.shape[1] >= self.max_seq_len:
                break

            past_kvs = new_kvs

        return generated

    @staticmethod
    def _sample(logits, temperature=0.7, top_k=50, top_p=0.9):
        """استخراج token التالي"""
        logits = logits / max(temperature, 1e-6)

        # Top-K filtering
        if top_k > 0:
            top_k_vals, top_k_idx = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_idx, top_k_vals)

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_mask = cumsum_probs <= top_p
            sorted_mask[..., 0] = True

            sorted_logits[~sorted_mask] = float('-inf')
            logits = torch.scatter(logits, -1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def get_total_params(self) -> int:
        """إجمالي المعاملات"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """المعاملات القابلة للتدريب"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)