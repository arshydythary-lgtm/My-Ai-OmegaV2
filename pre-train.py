# train.py - برنامج التدريب المحسن والمتوافق مع model_optimized.py
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

# استيراد النموذج المحسن
from model_optimized import OptimizedMiniLLM

# ملاحظة: تأكد من وجود ملفات tokenizer.py, memory.py, brain.py أو علق الأسطر التالية إذا لم تكن ضرورية للتدريب الأساسي
try:
    from tokenizer import MyTokenizer
except ImportError:
    print("⚠️  لم يتم العثور على tokenizer.py، سيتم استخدام فئة وهمية أو يجب توفيرها.")


    # فئة وهمية للضرورة فقط إذا كان الملف مفقوداً (للغرض التجريبي)
    class MyTokenizer:
        @staticmethod
        def load(path): return None

        @staticmethod
        def build(*args, **kwargs): return None

warnings.filterwarnings("ignore")


# ============================================================
# 1. معاملات سطر الأوامر
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="تدريب نموذج MiniLLM المحسن")

    # معاملات التدريب الأساسية
    parser.add_argument("--epochs", type=int, default=1, help="عدد الحقبات")
    parser.add_argument("--batch_size", type=int, default=4, help="حجم الدفعة")
    parser.add_argument("--lr", type=float, default=5e-4, help="معدل التعلم")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="تحلل الأوزان")
    parser.add_argument("--grad_accum", type=int, default=16, help="تجميع التدرجات")
    parser.add_argument("--patience", type=int, default=10, help="صبر Early Stopping")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="تمويه التسمية")

    # معاملات النموذج (يجب أن تتطابق مع model_optimized.py)
    parser.add_argument("--block_size", type=int, default=512, help="طول التسلسل")
    parser.add_argument("--d_model", type=int, default=1024, help="بعد النموذج")
    parser.add_argument("--n_heads", type=int, default=16, help="عدد رؤوس الانتباه")
    parser.add_argument("--num_layers", type=int, default=16, help="عدد طبقات النموذج")
    parser.add_argument("--dropout", type=float, default=0.1, help="معدل الـ Dropout")

    # معاملات المسارات والتخزين
    parser.add_argument("--checkpoint_dir", type=str, default="try", help="مجلد الـ try")
    parser.add_argument("--tokenizer_path", type=str, default="my_tokenizer", help="مسار الـ tokenizer")
    parser.add_argument("--data_path", type=str, default="data.csv", help="مسار البيانات")
    parser.add_argument("--tensorboard", action="store_true", help="تفعيل TensorBoard")

    # معاملات متقدمة
    parser.add_argument("--seed", type=int, default=42, help="البذرة العشوائية")
    parser.add_argument("--warmup_steps", type=int, default=500, help="خطوات الإحماء")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="قص التدرجات")
    parser.add_argument("--val_split", type=float, default=0.1, help="نسبة التحقق")
    parser.add_argument("--num_workers", type=int, default=0, help="عمال DataLoader (0 للتبسيط)")
    parser.add_argument("--resume", action="store_true", help="استئناف التدريب")

    return parser.parse_args()


args = parse_args()

# ثوابت لضمان التطابق التام مع بنية النموذج
# ملاحظة: هذه القيم يجب أن تتطابق مع ما يتوقعه OptimizedMiniLLM أو يتم تمريرها له
LORA_R = 32
USE_GRAD_CHECKPOINT = True


# ============================================================
# 2. إعداد الجهاز والبيئة
# ============================================================
def setup_device():
    """إعداد الجهاز والإعدادات الأمثل"""
    if torch.cuda.is_available():
        device = "cpu"
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
tokenizer = None


# محاكاة بسيطة للـ Tokenizer في حال عدم وجود الملف الفعلي للتجربة
# في البيئة الحقيقية، استخدم الكود الأصلي الخاص بك
class SimpleTokenizer:
    def __init__(self):
        self.vocab_size = 16000
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_id = 2
        self.char_to_id = {chr(i): i % 1000 for i in range(32, 127)}  # بسيط جداً

    def encode(self, text, add_special_tokens=False):
        ids = [self.char_to_id.get(c, 3) for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return "".join([chr(i % 1000 + 32) for i in ids if i > 3])

    def save(self, path): pass

    @staticmethod
    def load(path): return SimpleTokenizer()

    @staticmethod
    def build(*args, **kwargs): return SimpleTokenizer()


try:
    # محاولة التحميل الحقيقي إذا كان الملف موجوداً
    tokenizer = MyTokenizer.load(args.tokenizer_path)
    print(f"✅ تم تحميل tokenizer من {args.tokenizer_path}")
except (FileNotFoundError, AttributeError, NameError):
    print("⚠️  لم يتم العثور على tokenizer خارجي، استخدام Tokenizer تجريبي للتأكد من عمل الكود.")
    tokenizer = SimpleTokenizer()

vocab_size = tokenizer.vocab_size
pad_id = tokenizer.pad_token_id
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_id

print(f"✅ Tokenizer Config:")
print(f"   vocab_size: {vocab_size}")
print(f"   pad_id: {pad_id}, bos_id: {bos_id}, eos_id: {eos_id}\n")

# ============================================================
# 4. تحضير البيانات
# ============================================================
print("📊 تحضير البيانات...")

# إنشاء بيانات وهمية للتجربة إذا لم يوجد ملف
if not os.path.exists(args.data_path):
    print("⚠️  ملف البيانات غير موجود، جاري إنشاء بيانات تجريبية...")
    data = {
        'text': [f"هذا نص تجريبي رقم {i}" for i in range(100)]
    }
    df = pd.DataFrame(data)
    df.to_csv(args.data_path, index=False)
    print(f"✅ تم إنشاء ملف بيانات تجريبي: {args.data_path}")
else:
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"❌ خطأ في قراءة البيانات: {e}")
        exit(1)

# التحقق من الأعمدة
required_cols = ["text"]
if not all(col in df.columns for col in required_cols):
    print(f"❌ خطأ: البيانات يجب أن تحتوي على الأعمدة: {required_cols}")
    print(f"   الأعمدة الموجودة: {list(df.columns)}")
    # محاولة إصلاح ذاتي بسيط إذا كانت الأعمدة مختلفة
    if len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: 'context', df.columns[1]: 'answer'})
        print(f"⚠️  تم إعادة تسمية الأعمدة افتراضياً: {list(df.columns)}")
    else:
        exit(1)

# تنظيف البيانات
df = df.dropna(subset=required_cols)
df = df[
    df["text"].astype(str).str.len() > 10
]

print(f"✅ البيانات المحملة: {len(df)} سطر")

# تقسيم البيانات
indices = np.random.permutation(len(df))
split_idx = int((1 - args.val_split) * len(df))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

print(f"✅ التقسيم: {len(train_idx)} تدريب، {len(val_idx)} تحقق\n")


def build_sequence(text: str):
    """Pretraining: next-token prediction فقط"""

    ids = tokenizer.encode(text, add_special_tokens=True)

    # لازم يكون عندنا تسلسل طويل كفاية
    if len(ids) < 2:
        return None

    return ids

# بناء البيانات
train_pairs = []
val_pairs = []

print("🔄 بناء بيانات التدريب...")
for idx in tqdm(train_idx, desc="Train", leave=False):
    try:
        text = str(df.iloc[idx]["text"]).strip()

        seq = build_sequence(text)

        if seq:
            train_pairs.append(seq)
    except Exception:
        continue

print("🔄 بناء بيانات التحقق...")
for idx in tqdm(val_idx, desc="Val", leave=False):
    try:
        text = str(df.iloc[idx]["text"]).strip()

        seq = build_sequence(text)

        if seq:
            val_pairs.append(seq)
    except Exception:
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
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ids, mask = self.pairs[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def collate_fn(batch, block_size, pad_id):
    """دمج الدفعات مع padding ذكي وقص آمن"""
    # تحديد الطول الأقصى في هذه الدفعة (مع احترام الحد الأقصى للنموذج)
    max_len = min(max(len(ids) for ids, _ in batch), block_size)

    # ضمان أن الطول لا يقل عن 1
    if max_len < 1: max_len = 1

    padded_ids, targets, target_masks = [], [], []

    for ids, mask in batch:
        # تحويل tensors إلى lists إذا لزم الأمر
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(mask, torch.Tensor):
            mask = mask.tolist()

        # قص إذا تجاوز الطول الأقصى
        if len(ids) > max_len:
            ids = ids[:max_len]
            mask = mask[:max_len]

        # Padding
        pad_len = max_len - len(ids)
        ids_padded = ids + [pad_id] * pad_len
        mask_padded = mask + [0] * pad_len

        # Input/Target split
        # Input: [BOS, prompt, answer_part...] (بدون آخر عنصر)
        # Target: [prompt, answer..., EOS] (بدون أول عنصر)
        x = ids_padded[:-1]
        y = ids_padded[1:]
        m = mask_padded[1:]  # الـ Mask يتبع الـ Target

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

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device == "cuda"),
    collate_fn=lambda b: collate_fn(b, args.block_size, pad_id),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=(device == "cuda"),
    collate_fn=lambda b: collate_fn(b, args.block_size, pad_id),
)

# ============================================================
# 6. بناء النموذج
# ============================================================
print("🤖 بناء النموذج...")
model = OptimizedMiniLLM(
    vocab_size=vocab_size,
    d_model=args.d_model,
    n_heads=args.n_heads,
    num_layers=args.num_layers,
    max_seq_len=args.block_size,
    use_lora=True,
    lora_r=LORA_R,
    use_gradient_checkpoint=USE_GRAD_CHECKPOINT,
    dropout=args.dropout
).to(device)

total_params = model.get_total_params_count()
trainable_params = model.get_trainable_params_count()

print(f"✅ النموذج:")
print(f"   إجمالي المعاملات: {total_params:,}")
print(f"   المعاملات القابلة للتدريب (LoRA): {trainable_params:,}")
print(f"   نسبة التدريب: {(trainable_params / total_params) * 100:.2f}%\n")


# ============================================================
# 7. دالة الخسارة
# ============================================================
def masked_cross_entropy(logits, targets, mask, label_smoothing=0.0):
    """حساب الخسارة مع masking دقيق"""
    B, T, V = logits.shape

    # التأكد من تطابق الأبعاد
    if logits.shape[1] != targets.shape[1]:
        # قص أو توسيع إذا حدث اختلاف بسيط بسبب الـ slicing
        min_t = min(logits.shape[1], targets.shape[1])
        logits = logits[:, :min_t, :]
        targets = targets[:, :min_t]
        mask = mask[:, :min_t]

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction='none',
        label_smoothing=label_smoothing
    )

    loss = loss.reshape(B, -1)
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
# تجنب الخطأ في حال كان total_steps صغيراً جداً
if total_steps == 0: total_steps = 1

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

print(f"✅ Optimizer: AdamW | LR: {args.lr}")
print(f"✅ Gradient Accumulation: {args.grad_accum} | AMP: {'Enabled' if use_amp else 'Disabled'}\n")

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
    try:
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict']:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"   ✅ تم الاستئناف من Epoch {start_epoch}")
    except Exception as e:
        print(f"   ⚠️  فشل تحميل الـ checkpoint: {e}")
        print("   بدء التدريب من الصفر...")

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
    """حساب الدقة مع تجاهل العناصر غير المحسوبة"""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float() * mask
    total = mask.sum().item()
    if total == 0: return 0.0
    return correct.sum().item() / total


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_model_config():
    return {
        'vocab_size': vocab_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'num_layers': args.num_layers,
        'max_seq_len': args.block_size,
        'dropout': args.dropout,
        'use_lora': True,
        'lora_r': LORA_R,
        'use_gradient_checkpoint': USE_GRAD_CHECKPOINT,
        'use_flash_attn': True,
    }


# ============================================================
# 11. حلقة التدريب الرئيسية
# ============================================================
print("🚀 بدء التدريب...\n")
print("=" * 70)

import time

training_start = time.time()

for epoch in range(start_epoch, args.epochs):
    epoch_start = time.time()
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

        # Forward & Backward
        if use_amp and scaler:
            with torch.amp.autocast(device_type="cuda",
                                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
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

        # Metrics
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
    # التقييم (Evaluation)
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

            if use_amp and scaler:
                with torch.amp.autocast(device_type="cuda"):
                    logits_val = model(x_val)
                    loss_val = masked_cross_entropy(logits_val, y_val, mask_val)
            else:
                logits_val = model(x_val)
                loss_val = masked_cross_entropy(logits_val, y_val, mask_val)

            val_loss += loss_val.item()
            val_acc += compute_accuracy(logits_val, y_val, mask_val)

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_val_acc = val_acc / len(val_loader) if len(val_loader) > 0 else 0
    perplexity = np.exp(min(avg_val_loss, 10))

    epoch_time = time.time() - epoch_start
    total_time = time.time() - training_start

    # ============================================================
    # الطباعة والحفظ
    # ============================================================
    print(f"\n📊 Epoch {epoch + 1}/{args.epochs}")
    print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Time: {format_time(epoch_time)} (Total: {format_time(total_time)})")

    if writer:
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
        writer.add_scalar("Accuracy/val", avg_val_acc, epoch)
        writer.add_scalar("Perplexity/val", perplexity, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    save_config = get_model_config()

    # حفظ أفضل نموذج
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': save_config
        }, best_model_path)
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
        'config': save_config
    }, latest_ckpt_path)

    # تنظيف الذاكرة
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("=" * 70)

# ============================================================
# النهاية
# ============================================================
print("\n✅ اكتمل التدريب بنجاح!")

final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'config': get_model_config()
}, final_model_path)
print(f"💾 النموذج النهائي: {final_model_path}")

if writer:
    writer.close()

print(f"\n🎉 الوقت الكلي: {format_time(time.time() - training_start)}")
print(f"📊 أفضل Validation Loss: {best_val_loss:.4f}")
print("=" * 70)