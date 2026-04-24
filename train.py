# train.py - برنامج التدريب المحسّن والموثوق
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import gc
import argparse
from torch.utils.tensorboard import SummaryWriter

from model import MiniLLM
from tokenizer import MyTokenizer
from memory import Memory
from brain import Brain

warnings.filterwarnings("ignore")


# ============================================================
# 1. معاملات سطر الأوامر
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="تدريب نموذج MiniLLM")

    # معاملات التدريب الأساسية
    parser.add_argument("--epochs", type=int, default=50, help="عدد الحقبات")
    parser.add_argument("--batch_size", type=int, default=2, help="حجم الدفعة")
    parser.add_argument("--lr", type=float, default=5e-4, help="معدل التعلم")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="تحلل الأوزان")
    parser.add_argument("--grad_accum", type=int, default=4, help="تجميع التدرجات")
    parser.add_argument("--patience", type=int, default=10, help="صبر Early Stopping")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="تمويه التسمية")

    # معاملات النموذج
    parser.add_argument("--block_size", type=int, default=512, help="طول التسلسل")
    parser.add_argument("--d_model", type=int, default=512, help="بعد النموذج")
    parser.add_argument("--n_heads", type=int, default=8, help="عدد رؤوس الانتباه")
    parser.add_argument("--num_layers", type=int, default=6, help="عدد طبقات النموذج")
    parser.add_argument("--dropout", type=float, default=0.1, help="معدل الـ Dropout")

    # معاملات المسارات والتخزين
    parser.add_argument("--checkpoint_dir", type=str, default="try", help="مجلد الـ checkpoints")
    parser.add_argument("--tokenizer_path", type=str, default="my_tokenizer", help="مسار الـ tokenizer")
    parser.add_argument("--data_path", type=str, default="data.csv", help="مسار البيانات")
    parser.add_argument("--tensorboard", action="store_true", help="تفعيل TensorBoard")

    # معاملات متقدمة
    parser.add_argument("--seed", type=int, default=42, help="البذرة العشوائية")
    parser.add_argument("--warmup_steps", type=int, default=500, help="خطوات الإحماء")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="قص التدرجات")
    parser.add_argument("--val_split", type=float, default=0.1, help="نسبة التحقق")
    parser.add_argument("--num_workers", type=int, default=-1, help="عمال DataLoader (-1 = تلقائي)")
    parser.add_argument("--resume", action="store_true", help="استئناف التدريب")

    return parser.parse_args()


args = parse_args()


# ============================================================
# 2. إعداد الجهاز والبيئة
# ============================================================
def setup_device():
    """إعداد الجهاز والإعدادات الأمثل"""
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
        print("✅ Intel XPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple MPS")
    else:
        device = "cpu"
        print(f"✅ CPU ({os.cpu_count()} cores)")

    return device


device = setup_device()
print(f"🎯 Device: {device.upper()}\n")

# تعيين البذرة
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device == "cuda":
    torch.cuda.manual_seed_all(args.seed)

# ============================================================
# 3. تحميل/بناء الـ Tokenizer
# ============================================================
print("📂 إعداد الـ Tokenizer...")
try:
    tokenizer = MyTokenizer.load(args.tokenizer_path)
    print(f"✅ تم تحميل tokenizer من {args.tokenizer_path}")
except FileNotFoundError:
    print(f"⚠️  لم يتم العثور على tokenizer، جاري البناء من {args.data_path}...")

    try:
        df = pd.read_csv(args.data_path, nrows=10000)  # حد أقصى 10000 سطر للتدريب السريع
        texts = []

        for col in df.columns:
            texts.extend(df[col].dropna().astype(str).tolist())

        if not texts:
            raise ValueError("لا توجد نصوص صالحة في البيانات")

        tokenizer = MyTokenizer.build(
            texts,
            vocab_size=16000,
            min_frequency=2,
            save_path=args.tokenizer_path
        )
        print(f"✅ تم بناء وحفظ tokenizer جديد\n")

    except Exception as e:
        print(f"❌ خطأ: {e}")
        exit(1)

vocab_size = tokenizer.vocab_size
pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

print(f"✅ Tokenizer Config:")
print(f"   vocab_size: {vocab_size}")
print(f"   pad_id: {pad_id}, bos_id: {bos_id}, eos_id: {eos_id}\n")

# ============================================================
# 4. تحضير البيانات
# ============================================================
print("📊 تحضير البيانات...")

try:
    df = pd.read_csv(args.data_path)
except Exception as e:
    print(f"❌ خطأ في قراءة البيانات: {e}")
    exit(1)

# التحقق من الأعمدة
required_cols = ["context", "answer"]
if not all(col in df.columns for col in required_cols):
    print(f"❌ خطأ: البيانات يجب أن تحتوي على الأعمدة: {required_cols}")
    print(f"   الأعمدة الموجودة: {list(df.columns)}")
    exit(1)

# تنظيف البيانات
df = df.dropna(subset=required_cols)
df = df[(df['context'].str.len() > 5) & (df['answer'].str.len() > 5)]

print(f"✅ البيانات المحملة: {len(df)} سطر")

# تقسيم البيانات
indices = np.random.permutation(len(df))
split_idx = int((1 - args.val_split) * len(df))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

print(f"✅ التقسيم: {len(train_idx)} تدريب، {len(val_idx)} تحقق\n")


def build_sequence(context: str, answer: str):
    """بناء تسلسل مع loss mask صحيح"""
    prompt = f"السؤال: {context}\nالإجابة: "

    # ترميز السؤال والإجابة
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(f"{answer}", add_special_tokens=False)

    # دمج: [BOS] + prompt + answer + [EOS]
    full_ids = [bos_id] + prompt_ids + answer_ids + [eos_id]

    # Loss mask: فقط على الإجابة
    loss_mask = [0] * (1 + len(prompt_ids)) + [1] * (len(answer_ids) + 1)

    return full_ids, loss_mask


# بناء البيانات
train_pairs = []
val_pairs = []

print("🔄 بناء بيانات التدريب...")
for idx in tqdm(train_idx, desc="Train", leave=False):
    try:
        ctx = str(df.iloc[idx]["context"]).strip()
        ans = str(df.iloc[idx]["answer"]).strip()
        if ctx and ans:
            train_pairs.append(build_sequence(ctx, ans))
    except Exception as e:
        continue

print("🔄 بناء بيانات التحقق...")
for idx in tqdm(val_idx, desc="Val", leave=False):
    try:
        ctx = str(df.iloc[idx]["context"]).strip()
        ans = str(df.iloc[idx]["answer"]).strip()
        if ctx and ans:
            val_pairs.append(build_sequence(ctx, ans))
    except Exception as e:
        continue

print(f"\n✅ البيانات النهائية:")
print(f"   Train pairs: {len(train_pairs)}")
print(f"   Val pairs: {len(val_pairs)}\n")

if len(train_pairs) == 0 or len(val_pairs) == 0:
    print("❌ لا توجد بيانات صالحة!")
    exit(1)


# ============================================================
# 5. Dataset و DataLoader
# ============================================================
class TextDataset(Dataset):
    """Dataset للبيانات النصية"""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ids, mask = self.pairs[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def collate_fn(batch, block_size, pad_id):
    """دمج الدفعات مع padding ذكي"""
    max_len = min(max(len(ids) for ids, _ in batch), block_size)

    padded_ids, targets, target_masks = [], [], []

    for ids, mask in batch:
        # قص إذا لزم الأمر
        if len(ids) > max_len:
            ids = ids[:max_len]
            mask = mask[:max_len]

        # Padding
        pad_len = max_len - len(ids)
        ids = ids.tolist() + [pad_id] * pad_len
        mask = mask.tolist() + [0] * pad_len

        # Input/Target split
        x = ids[:-1]
        y = ids[1:]
        m = mask[1:]

        padded_ids.append(x)
        targets.append(y)
        target_masks.append(m)

    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(target_masks, dtype=torch.float32)
    )


# إنشاء Datasets و Loaders
train_dataset = TextDataset(train_pairs)
val_dataset = TextDataset(val_pairs)

num_workers = 0 if device == "cpu" else (args.num_workers if args.num_workers > 0 else min(4, os.cpu_count() // 2))

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    collate_fn=lambda b: collate_fn(b, args.block_size, pad_id),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    collate_fn=lambda b: collate_fn(b, args.block_size, pad_id),
)

# ============================================================
# 6. بناء النموذج
# ============================================================
print("🤖 بناء النموذج...")
model = MiniLLM(
    vocab_size=vocab_size,
    d_model=args.d_model,
    n_heads=args.n_heads,
    num_layers=args.num_layers,
    max_seq_len=args.block_size,
    dropout=args.dropout
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ النموذج:")
print(f"   إجمالي المعاملات: {total_params:,}")
print(f"   المعاملات القابلة للتدريب: {trainable_params:,}\n")


# ============================================================
# 7. دالة الخسارة
# ============================================================
def masked_cross_entropy(logits, targets, mask, label_smoothing=0.0):
    """حساب الخسارة مع masking"""
    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction='none',
        label_smoothing=label_smoothing
    )

    loss = loss.reshape(B, T)
    mask_sum = mask.sum()

    if mask_sum == 0:
        return torch.tensor(0.0, device=logits.device)

    return (loss * mask).sum() / mask_sum


# ============================================================
# 8. التحسين والمجدول
# ============================================================
print("⚙️  إعداد التحسين...")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.95),
    weight_decay=args.weight_decay,
    eps=1e-8
)

total_steps = len(train_loader) * args.epochs // args.grad_accum

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy='cos',
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95
)

scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
use_amp = device == "cuda"

print(f"✅ Optimizer: AdamW")
print(f"   LR: {args.lr}, Warmup: {args.warmup_steps}")
print(f"   Gradient Accumulation: {args.grad_accum}")
print(f"   AMP: {'Enabled' if use_amp else 'Disabled'}\n")

# ============================================================
# 9. إدارة Checkpoint
# ============================================================
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(os.path.join(args.checkpoint_dir, "logs"), exist_ok=True)

start_epoch = 0
global_step = 0
best_val_loss = float('inf')
early_stop_counter = 0

latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")

if args.resume and os.path.exists(latest_ckpt_path):
    print("📌 استئناف التدريب...")
    ckpt = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    start_epoch = ckpt['epoch'] + 1
    global_step = ckpt.get('global_step', 0)
    best_val_loss = ckpt.get('best_val_loss', float('inf'))

    print(f"   Epoch: {start_epoch}")
    print(f"   Best Loss: {best_val_loss:.4f}")
    print(f"   Global Step: {global_step}\n")

# TensorBoard
writer = None
if args.tensorboard:
    log_dir = os.path.join(args.checkpoint_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard: {log_dir}\n")


# ============================================================
# 10. دوال مساعدة
# ============================================================
def compute_accuracy(logits, targets, mask):
    """حساب الدقة"""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float() * mask
    return correct.sum().item() / (mask.sum().item() + 1e-8)


def format_time(seconds):
    """تنسيق الوقت"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================
# 11. حلقة التدريب الرئيسية
# ============================================================
print("🚀 بدء التدريب...\n")
print("=" * 70)

import time

training_start = time.time()

for epoch in range(start_epoch, args.epochs):
    epoch_start = time.time()

    # ============================================================
    # التدريب
    # ============================================================
    model.train()
    total_loss = 0
    total_acc = 0
    optimizer.zero_grad()

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{args.epochs} [TRAIN]",
        ncols=80
    )

    for step, (x, y, mask) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = masked_cross_entropy(logits, y, mask, args.label_smoothing)

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        else:
            logits = model(x)
            loss = masked_cross_entropy(logits, y, mask, args.label_smoothing)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        acc = compute_accuracy(logits, y, mask)
        total_loss += loss.item()
        total_acc += acc

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{acc:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / len(train_loader)

    # ============================================================
    # التقييم
    # ============================================================
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        pbar_val = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [VAL]",
            ncols=80,
            leave=False
        )

        for x_val, y_val, mask_val in pbar_val:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            mask_val = mask_val.to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits_val = model(x_val)
                    loss_val = masked_cross_entropy(logits_val, y_val, mask_val)
            else:
                logits_val = model(x_val)
                loss_val = masked_cross_entropy(logits_val, y_val, mask_val)

            val_loss += loss_val.item()
            val_acc += compute_accuracy(logits_val, y_val, mask_val)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    perplexity = np.exp(min(avg_val_loss, 10))

    epoch_time = time.time() - epoch_start
    total_time = time.time() - training_start

    # ============================================================
    # طباعة النتائج
    # ============================================================
    print(f"\n📊 Epoch {epoch + 1}/{args.epochs}")
    print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Time: {format_time(epoch_time)} (Total: {format_time(total_time)})")

    # ============================================================
    # حفظ النتائج
    # ============================================================
    if writer:
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
        writer.add_scalar("Accuracy/val", avg_val_acc, epoch)
        writer.add_scalar("Perplexity/val", perplexity, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    # حفظ أفضل نموذج
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        early_stop_counter = 0
        print(f"   ✅ أفضل نموذج محفوظ! (loss: {best_val_loss:.4f})")
    else:
        early_stop_counter += 1
        print(f"   ⚠️  No improvement ({early_stop_counter}/{args.patience})")

        if early_stop_counter >= args.patience:
            print(f"\n🛑 Early Stopping at Epoch {epoch + 1}")
            break

    # حفظ آخر checkpoint
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': vocab_size,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'num_layers': args.num_layers,
            'max_seq_len': args.block_size,
            'dropout': args.dropout,
        }
    }, latest_ckpt_path)

    # حفظ checkpoint دوري
    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), periodic_path)

    # تنظيف الذاكرة
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print("=" * 70)

# ============================================================
# النهاية
# ============================================================
print("\n✅ اكتمل التدريب بنجاح!")

# حفظ النموذج النهائي
final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"💾 النموذج النهائي: {final_model_path}")

# حفظ الـ tokenizer
tokenizer.save(args.checkpoint_dir)
print(f"💾 الـ Tokenizer: {args.checkpoint_dir}")

# إغلاق TensorBoard
if writer:
    writer.close()

print(f"\n🎉 الوقت الكلي: {format_time(time.time() - training_start)}")
print(f"📊 أفضل Validation Loss: {best_val_loss:.4f}")
print("=" * 70)