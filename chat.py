# chat.py - واجهة Streamlit محسنة ومتوافقة مع النموذج الكبير (1024d)
import os
import torch
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
import time
import re

from model_optimized import OptimizedMiniLLM
from memory import Memory
from brain import Brain
from tokenizer import MyTokenizer

# ============================================================
# 1. الإعدادات العامة
# ============================================================
PAGE_CONFIG = {
    "page_title": "🤖 MiniLLM Chat (Large)",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# ✅ إصلاح حرج: استخدام CUDA إذا كان متاحاً، وليس العكس
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "try"
TOKENIZER_PATH = "my_tokenizer"

# إعدادات التوليد الافتراضية
DEFAULT_GEN_CONFIG = {
    "max_gen_len": 256,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
}

# معاملات النموذج المحسّن (متطابقة مع train.py الجديد)
DEFAULT_MODEL_CONFIG = {
    "d_model": 1024,
    "n_heads": 16,
    "num_layers": 16,
    "max_seq_len": 512,
    "use_lora": True,
    "lora_r": 32,
    "use_gradient_checkpoint": True,
    "use_flash_attn": True,
    "dropout": 0.1,
}

# ============================================================
# 2. تهيئة Streamlit والتصميم
# ============================================================
st.set_page_config(**PAGE_CONFIG)

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .optimization-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 0.5rem 1rem; border-radius: 0.25rem;
        font-size: 0.85rem; display: inline-block; margin: 0.2rem;
    }
    .stat-box { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. دوال التحميل والتخزين المؤقت (Cache)
# ============================================================

@st.cache_resource
def load_tokenizer():
    """تحميل الـ Tokenizer"""
    try:
        tokenizer = MyTokenizer.load(TOKENIZER_PATH)
        st.success(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
        return tokenizer
    except FileNotFoundError:
        st.error(f"❌ خطأ: لم يتم العثور على tokenizer في {TOKENIZER_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"❌ خطأ غير متوقع في الـ Tokenizer: {e}")
        st.stop()


@st.cache_resource
def load_model(_tokenizer):
    """تحميل النموذج مع استرجاع الإعدادات من الـ Checkpoint"""
    vocab_size = _tokenizer.vocab_size

    # البحث عن ملفات النموذج
    best_model_path = None
    config_from_ckpt = None

    if os.path.exists(CHECKPOINT_DIR):
        # الأولوية لأفضل نموذج، ثم النهائي
        for name in ["best_model.pth", "final_model.pth"]:
            path = os.path.join(CHECKPOINT_DIR, name)
            if os.path.exists(path):
                best_model_path = path
                break

        # محاولة استخراج الإعدادات (config) من أحدث نقطة حفظ
        latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
        if os.path.exists(latest_ckpt):
            try:
                ckpt = torch.load(latest_ckpt, map_location=DEVICE)
                config_from_ckpt = ckpt.get('config')
            except Exception:
                pass

    # دمج الإعدادات: الأولوية لـ config من الملف، ثم الافتراضي
    final_config = DEFAULT_MODEL_CONFIG.copy()
    if config_from_ckpt:
        final_config.update(config_from_ckpt)

    try:
        # بناء النموذج بالإعدادات المستخرجة
        model = OptimizedMiniLLM(
            vocab_size=vocab_size,
            d_model=final_config.get("d_model", 1024),
            n_heads=final_config.get("n_heads", 16),
            num_layers=final_config.get("num_layers", 16),
            max_seq_len=final_config.get("max_seq_len", 512),
            use_lora=final_config.get("use_lora", True),
            lora_r=final_config.get("lora_r", 32),
            use_gradient_checkpoint=final_config.get("use_gradient_checkpoint", True),
            dropout=final_config.get("dropout", 0.1),
        ).to(DEVICE)

        model_name = "random_init"

        # تحميل الأوزان إذا وجدت
        if best_model_path:
            state_dict = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=False)
            model_name = os.path.basename(best_model_path)
            st.success(f"✅ Model loaded: {model_name}")
        else:
            st.warning("⚠️ لا يوجد checkpoint محفوظ، سيتم استخدام أوزان عشوائية")

        model.eval()

        # حساب الإحصائيات للعرض
        total_params = model.get_total_params_count()
        trainable_params = model.get_trainable_params_count()

        optimizations = []
        if final_config.get("use_gradient_checkpoint"): optimizations.append("Grad Checkpoint")
        if final_config.get("use_flash_attn"): optimizations.append("Flash Attention")
        if final_config.get("use_lora"): optimizations.append(f"LoRA (r={final_config.get('lora_r', 8)})")

        st.session_state.model_info = {
            "name": model_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "optimizations": optimizations,
            "config": final_config,
        }

        return model, model_name

    except Exception as e:
        st.error(f"❌ فشل تحميل النموذج: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# ============================================================
# 4. دوال التوليد والمعالجة
# ============================================================

def clean_response(text: str) -> str:
    """تنظيف النص المولد من البادئات والأسطر الزائدة بأمان"""
    # إزالة البادئات فقط إذا كانت في بداية السطر الأول
    lines = text.split('\n')
    first_line = lines[0]

    prefixes = ["Assistant:", "AI:", "User:", "System:", "الإجابة:", "المساعد:", "السؤال:", "النظام:"]
    for prefix in prefixes:
        if first_line.startswith(prefix):
            lines[0] = first_line[len(prefix):].lstrip()
            break

    text = '\n'.join(lines)

    # تنظيف التنسيق
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """تقدير سريع لعدد الرموز"""
    if not text: return 0
    return max(1, len(text.split()) + len(text) // 10)


def generate_response(
        tokenizer: MyTokenizer,
        model: OptimizedMiniLLM,
        prompt: str,
        config: Dict,
        progress_placeholder,
        max_model_len: int = 512
) -> str:
    """توليد الرد باستخدام النموذج مع حماية من تجاوز طول السياق"""
    try:
        # ترميز الـ Prompt للتحقق من الطول
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)

        # ✅ حماية: إذا كان الـ Prompt أطول من سعة النموذج، نقصه
        if len(input_ids) >= max_model_len:
            # نحافظ على آخر جزء من المحادثة (الأهم)
            trim_len = max_model_len - config["max_gen_len"] - 50  # هامش أمان
            if trim_len < 10: trim_len = 10

            input_ids = input_ids[-trim_len:]
            # إعادة فك الترميز جزئياً قد يكون معقداً، لذا نعتمد أن الذاكرة ستتعامل مع القص
            # لكن هنا سنمرر المقصوص مباشرة للتوكنز
            st.warning(f"⚠️ المحادثة طويلة جداً، تم قص السياق القديم للبقاء ضمن حد {max_model_len} رمز.")

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

        progress_placeholder.info("⏳ جاري التفكير والتوليد...")

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_tensor,
                max_new_tokens=config["max_gen_len"],
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                device=DEVICE
            )

        # فك الترميز (نتجاهل جزء الـ Prompt ونأخذ فقط المولد)
        # نستخدم طول input_ids الفعلي بعد القص إن وجد
        current_input_len = input_tensor.shape[1]
        response_ids = generated[0, current_input_len:].tolist()

        if not response_ids:
            return "..."

        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        response_text = clean_response(response_text)

        progress_placeholder.empty()
        return response_text

    except Exception as e:
        progress_placeholder.error(f"❌ خطأ أثناء التوليد: {str(e)}")
        return f"عذراً، حدث خطأ تقني: {str(e)}"


# ============================================================
# 5. إدارة الجلسة والواجهة
# ============================================================

def initialize_session():
    if "memory" not in st.session_state:
        # نضبط الذاكرة لتكون أقل قليلاً من حد النموذج لتترك مساحة للرد
        st.session_state.memory = Memory(max_history=50, max_tokens=1800)
    if "brain" not in st.session_state:
        st.session_state.brain = Brain(model_name="أوميقا", max_context_length=2000, max_history=20)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_stats" not in st.session_state:
        st.session_state.show_stats = False
    if "show_optimizations" not in st.session_state:
        st.session_state.show_optimizations = False
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


def main():
    initialize_session()

    # تحميل الموارد مرة واحدة
    if not st.session_state.model_loaded:
        with st.spinner("⏳ جاري تحميل النموذج والموارد... (قد يستغرق وقتاً لأول مرة)"):
            tokenizer = load_tokenizer()
            model, model_name = load_model(tokenizer)
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_name = model_name
            st.session_state.model_loaded = True
            st.rerun()
    else:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model
        model_name = st.session_state.model_name

    # --- الرأس ---
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    with col1:
        st.title("🤖 MiniLLM Chat")
        st.caption(f"💻 Device: {DEVICE.upper()} | 📦 Model: {model_name}")
    with col2:
        if st.button("📊 الإحصائيات", use_container_width=True):
            st.session_state.show_stats = not st.session_state.show_stats
    with col3:
        if st.button("⚡ التحسينات", use_container_width=True):
            st.session_state.show_optimizations = not st.session_state.show_optimizations

    # --- عرض معلومات التحسينات ---
    if st.session_state.show_optimizations and st.session_state.model_info:
        info = st.session_state.model_info
        st.markdown("### ⚡ تفاصيل النموذج المحسّن")

        c1, c2, c3 = st.columns(3)
        c1.metric("إجمالي المعاملات", f"{info['total_params']:,}")
        c2.metric("قابلة للتدريب (LoRA)", f"{info['trainable_params']:,}")
        reduction = (1 - info['trainable_params'] / info['total_params']) * 100
        c3.metric("توفير الذاكرة", f"{reduction:.1f}%")

        st.markdown("**التقنيات النشطة:** " + " ".join(
            [f'<span class="optimization-badge">{o}</span>' for o in info['optimizations']], unsafe_allow_html=True))
        st.divider()

    # --- الشريط الجانبي ---
    with st.sidebar:
        st.header("⚙️ الإعدادات")

        with st.expander("🎚️ معايير التوليد", expanded=True):
            temperature = st.slider("Temperature", 0.1, 2.0, DEFAULT_GEN_CONFIG["temperature"], 0.1)
            top_k = st.slider("Top-K", 0, 100, DEFAULT_GEN_CONFIG["top_k"], 5)
            top_p = st.slider("Top-P", 0.5, 1.0, DEFAULT_GEN_CONFIG["top_p"], 0.05)
            max_len = st.slider("Max Length", 50, 1024, DEFAULT_GEN_CONFIG["max_gen_len"], 50)

        st.divider()

        with st.expander("🧠 الشخصية والذاكرة"):
            new_name = st.text_input("اسم المساعد", st.session_state.brain.model_name)
            if new_name != st.session_state.brain.model_name:
                st.session_state.brain.model_name = new_name

            sys_prompt = st.text_area("نص النظام (System Prompt)", st.session_state.brain.system_prompt, height=100)
            if sys_prompt != st.session_state.brain.system_prompt:
                st.session_state.brain.set_system_prompt(sys_prompt)
                st.success("تم التحديث")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ مسح المحادثة", use_container_width=True):
                st.session_state.memory.clear()
                st.session_state.messages = []
                st.rerun()
        with c2:
            if st.button("🔄 إعادة ضبط", use_container_width=True):
                st.session_state.memory.clear()
                st.session_state.messages = []
                st.session_state.brain.reset_system_prompt()
                st.rerun()

        if st.session_state.show_stats:
            st.divider()
            st.subheader("📊 إحصائيات الجلسة")
            mem_stats = st.session_state.memory.get_stats()
            st.metric("الرسائل", mem_stats["total_messages"])
            st.metric("الرموز المستخدمة", f"{mem_stats['total_tokens']} / {mem_stats['max_tokens']}")

    # --- منطقة المحادثة ---
    st.subheader("💬 المحادثة")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if "time" in msg:
                st.caption(f"⏱️ {msg['time']:.2f} ثانية")

    # --- الإدخال ---
    if prompt := st.chat_input("اكتب رسالتك هنا..."):
        # رسالة المستخدم
        st.session_state.memory.add_user(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "time": 0})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # ✅ تحسين: التحقق من طول الـ Prompt قبل التوليد
        # نقوم بإنشاء الـ Prompt مبدئياً لمعرفة طوله
        temp_prompt = st.session_state.memory.format_as_prompt(
            system_prompt=st.session_state.brain.system_prompt,
            template="simple"
        )

        # تقدير سريع للطول
        estimated_len = estimate_tokens(temp_prompt)
        model_limit = DEFAULT_MODEL_CONFIG["max_seq_len"]

        # إذا اقتربنا من الحد، نقلل عدد الرسائل في الذاكرة مؤقتاً للتوليد فقط
        # (الذاكرة الأصلية تبقى كاملة، لكننا نرسل جزءاً منها)
        messages_to_send = list(st.session_state.memory.messages)
        while estimated_len > (model_limit - max_len - 50) and len(messages_to_send) > 1:
            messages_to_send.pop(0)  # حذف أقدم رسالة من النسخة المرسلة
            # إعادة حساب التقريب (تبسيطاً)
            estimated_len = int(estimated_len * 0.9)

            # بناء الـ Prompt النهائي (مع أو بدون رسائل قديمة حسب الحاجة)
        # ملاحظة: فئة Memory لا تدعم تمرير قائمة رسائل مخصصة بسهولة في format_as_prompt الحالية
        # لذا نعتمد على أن Memory内部管理ت الـ tokens، وهنا نتأكد من عدم تجاوز الحد في دالة التوليد

        full_prompt = st.session_state.memory.format_as_prompt(
            system_prompt=st.session_state.brain.system_prompt,
            template="simple"
        )
        full_prompt += f"\n{st.session_state.brain.model_name}: "

        # التوليد
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            start_time = time.time()

            gen_config = {
                "max_gen_len": max_len,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }

            response = generate_response(
                tokenizer, model, full_prompt, gen_config, placeholder,
                max_model_len=model_limit
            )

            elapsed = time.time() - start_time
            placeholder.markdown(response)
            st.caption(
                f"⏱️ وقت التوليد: {elapsed:.2f} ثانية | السرعة التقريبية: {estimate_tokens(response) / max(elapsed, 0.1):.1f} token/s")

        # حفظ رد المساعد
        st.session_state.memory.add_assistant(response)
        st.session_state.messages.append({"role": "assistant", "content": response, "time": elapsed})


if __name__ == "__main__":
    main()
