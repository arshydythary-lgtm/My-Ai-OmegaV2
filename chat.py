# chat.py - واجهة Streamlit احترافية للمحادثة مع model_optimized.py
import os
import torch
import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict
import time
import inspect  # <--- إضافة لفحص دوال النموذج

from model_optimized import OptimizedMiniLLM
from memory import Memory
from brain import Brain
from tokenizer import MyTokenizer

# ============================================================
# الإعدادات
# ============================================================
PAGE_CONFIG = {
    "page_title": "🤖 MiniLLM Chat",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "try"
TOKENIZER_PATH = "my_tokenizer"

DEFAULT_GEN_CONFIG = {
    "max_gen_len": 200,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}

# معاملات النموذج المحسّن
DEFAULT_MODEL_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "num_layers": 6,
    "max_seq_len": 512,
    "use_lora": True,
    "lora_r": 8,
    "use_gradient_checkpoint": True,
    "use_flash_attn": True,
    "dropout": 0.1,
}

# ============================================================
# تهيئة Streamlit
# ============================================================
st.set_page_config(**PAGE_CONFIG)

# CSS مخصص
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-stats {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .optimization-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# دوال التحميل والـ Cache
# ============================================================

@st.cache_resource
def load_tokenizer():
    """تحميل الـ tokenizer مع الـ cache"""
    try:
        tokenizer = MyTokenizer.load(TOKENIZER_PATH)
        st.success(f"✅ Tokenizer: vocab_size={tokenizer.vocab_size}")
        return tokenizer
    except FileNotFoundError:
        st.error(f"❌ خطأ: لم يتم العثور على tokenizer في {TOKENIZER_PATH}")
        st.stop()


@st.cache_resource
def load_model(_tokenizer):
    """تحميل النموذج المحسّن مع الـ cache"""
    vocab_size = _tokenizer.vocab_size

    # البحث عن checkpoint
    best_model_path = None
    model_config = None

    if os.path.exists(CHECKPOINT_DIR):
        # البحث عن أفضل checkpoint
        for name in ["best_model.pth", "final_model.pth"]:
            path = os.path.join(CHECKPOINT_DIR, name)
            if os.path.exists(path):
                best_model_path = path
                break

        # محاولة تحميل الـ config من latest_checkpoint
        latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
        if os.path.exists(latest_ckpt):
            try:
                ckpt = torch.load(latest_ckpt, map_location=DEVICE)
                model_config = ckpt.get('config', DEFAULT_MODEL_CONFIG)
            except:
                model_config = DEFAULT_MODEL_CONFIG
        else:
            model_config = DEFAULT_MODEL_CONFIG

    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG

    try:
        model = OptimizedMiniLLM(
            vocab_size=vocab_size,
            d_model=model_config.get("d_model", 512),
            n_heads=model_config.get("n_heads", 8),
            num_layers=model_config.get("num_layers", 6),
            max_seq_len=model_config.get("max_seq_len", 512),
            use_lora=model_config.get("use_lora", True),
            lora_r=model_config.get("lora_r", 8),
            use_gradient_checkpoint=model_config.get("use_gradient_checkpoint", True),
            use_flash_attn=model_config.get("use_flash_attn", True),
            dropout=model_config.get("dropout", 0.1),
        ).to(DEVICE)

        if best_model_path:
            state_dict = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model_name = os.path.basename(best_model_path)
            st.success(f"✅ Model: {model_name}")
        else:
            model_name = "random_init"
            st.warning("⚠️ لا يوجد checkpoint، استخدام نموذج عشوائي")

        model.eval()

        # عرض معلومات التحسينات
        optimizations = []
        if model_config.get("use_lora"):
            optimizations.append(f"LoRA (r={model_config.get('lora_r', 8)})")
        if model_config.get("use_gradient_checkpoint"):
            optimizations.append("Grad Checkpoint")
        if model_config.get("use_flash_attn"):
            optimizations.append("Flash Attention")

        total_params = model.get_total_params_count()
        trainable_params = model.get_trainable_params_count()

        st.session_state.model_info = {
            "name": model_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "optimizations": optimizations,
            "config": model_config,
        }

        return model, model_name

    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        st.stop()


# ============================================================
# دوال التوليد والمعالجة
# ============================================================

def generate_response(
        tokenizer: MyTokenizer,
        model: OptimizedMiniLLM,
        prompt: str,
        config: Dict,
        progress_placeholder
) -> str:
    """توليد رد من النموذج مع التوافق التلقائي للمعاملات"""
    try:
        # ترميز الـ prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

        progress_placeholder.info(" جاري التوليد...")

        # بناء معاملات التوليد الأساسية
        gen_kwargs = {
            "max_new_tokens": config["max_gen_len"],
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"],
            "device": DEVICE
        }

        # التحقق مما إذا كان النموذج يدعم repetition_penalty
        sig = inspect.signature(model.generate)
        if "repetition_penalty" in sig.parameters:
            gen_kwargs["repetition_penalty"] = config["repetition_penalty"]
        else:
            # تحذير لمرة واحدة فقط في الجلسة
            if not st.session_state.get("_warned_repetition_penalty", False):
                st.warning("⚠️ النموذج الحالي لا يدعم 'repetition_penalty' (سيتم تجاهل القيمة).")
                st.session_state._warned_repetition_penalty = True

        # التوليد
        with torch.no_grad():
            generated = model.generate(input_tensor, **gen_kwargs)

        # فك الترميز
        response_ids = generated[0].tolist()
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # تنظيف النص
        response_text = clean_response(response_text)

        progress_placeholder.empty()
        return response_text

    except Exception as e:
        progress_placeholder.error(f"❌ خطأ: {str(e)}")
        return f"عذراً، حدث خطأ في التوليد: {str(e)}"


def clean_response(text: str) -> str:
    """تنظيف النص المولد"""
    import re

    # إزالة البادئات غير المرغوبة
    prefixes = [
        "Assistant:", "AI:", "User:", "System:",
        "الإجابة:", "المساعد:", "السؤال:", "النظام:"
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    # إزالة الأسطر الفارغة المتعددة
    text = re.sub(r'\n{3,}', '\n\n', text)
    # إزالة المسافات الزائدة
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def estimate_tokens(text: str) -> int:
    """تقدير عدد الرموز"""
    return len(text.split()) + len(text) // 10


# ============================================================
# تهيئة الجلسة
# ============================================================

def initialize_session():
    """تهيئة متغيرات الجلسة"""
    if "memory" not in st.session_state:
        st.session_state.memory = Memory(max_history=50, max_tokens=2000)

    if "brain" not in st.session_state:
        st.session_state.brain = Brain(
            model_name="أوميقا",
            max_context_length=2000,
            max_history=20
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_stats" not in st.session_state:
        st.session_state.show_stats = False

    if "show_optimizations" not in st.session_state:
        st.session_state.show_optimizations = False

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if "model_info" not in st.session_state:
        st.session_state.model_info = None

    if "_warned_repetition_penalty" not in st.session_state:
        st.session_state._warned_repetition_penalty = False


# ============================================================
# الواجهة الرئيسية
# ============================================================

def main():
    initialize_session()

    # تحميل الموارد
    if not st.session_state.model_loaded:
        with st.spinner("⏳ جاري تحميل الموارد..."):
            tokenizer = load_tokenizer()
            model, model_name = load_model(tokenizer)
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_name = model_name
            st.session_state.model_loaded = True
    else:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model
        model_name = st.session_state.model_name

    # العنوان
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    with col1:
        st.title("🤖 MiniLLM - محادثة ذكية")
        st.caption(f"💻 الجهاز: {DEVICE.upper()} | 📦 النموذج: {model_name}")

    with col2:
        if st.button("📊 إحصائيات", use_container_width=True):
            st.session_state.show_stats = not st.session_state.show_stats

    with col3:
        if st.button("⚡ التحسينات", use_container_width=True):
            st.session_state.show_optimizations = not st.session_state.show_optimizations

    # عرض معلومات التحسينات
    if st.session_state.show_optimizations and st.session_state.model_info:
        model_info = st.session_state.model_info
        st.markdown("### ⚡ التحسينات المفعّلة")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("إجمالي المعاملات", f"{model_info['total_params']:,}")
        with col2:
            st.metric("المعاملات القابلة للتدريب", f"{model_info['trainable_params']:,}")
        with col3:
            reduction = (1 - model_info['trainable_params'] / model_info['total_params']) * 100
            st.metric("تقليل", f"{reduction:.1f}%")

        st.markdown("**التقنيات المستخدمة:**")
        optimization_html = " ".join([
            f'<span class="optimization-badge">{opt}</span>'
            for opt in model_info['optimizations']
        ])
        st.markdown(optimization_html, unsafe_allow_html=True)

        # إحصائيات الأداء المتوقعة
        col1, col2, col3 = st.columns(3)
        with col1:
            speedup = model_info['total_params'] / max(model_info['trainable_params'], 1)
            st.metric("تسريع متوقع", f"{speedup:.1f}x")
        with col2:
            st.metric("توفير ذاكرة", "~70%")
        with col3:
            st.metric("تقليل أوزان", f"{reduction:.1f}%")

        st.divider()

    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ الإعدادات")

        # إعدادات التوليد
        with st.expander("🎚️ إعدادات التوليد", expanded=True):
            temperature = st.slider(
                "Temperature",
                0.1, 2.0,
                DEFAULT_GEN_CONFIG["temperature"],
                0.1,
                help="كلما زادت القيمة، زادت العشوائية"
            )

            top_k = st.slider(
                "Top-K",
                0, 100,
                DEFAULT_GEN_CONFIG["top_k"],
                5,
                help="عدد الخيارات الأفضل"
            )

            top_p = st.slider(
                "Top-P (Nucleus)",
                0.5, 1.0,
                DEFAULT_GEN_CONFIG["top_p"],
                0.05,
                help="تصفية النوى الاحتمالية"
            )

            rep_penalty = st.slider(
                "Repetition Penalty",
                1.0, 2.0,
                DEFAULT_GEN_CONFIG["repetition_penalty"],
                0.05,
                help="منع التكرار"
            )

            max_len = st.slider(
                "Max Length",
                50, 500,
                DEFAULT_GEN_CONFIG["max_gen_len"],
                50,
                help="أقصى طول للرد"
            )

        st.divider()

        # إعدادات الدماغ/الشخصية
        with st.expander("🧠 إعدادات الشخصية", expanded=False):
            model_name_custom = st.text_input(
                "اسم المساعد",
                st.session_state.brain.model_name
            )
            if model_name_custom != st.session_state.brain.model_name:
                st.session_state.brain.model_name = model_name_custom

            system_prompt = st.text_area(
                "نص النظام",
                st.session_state.brain.system_prompt,
                height=150
            )
            if system_prompt != st.session_state.brain.system_prompt:
                st.session_state.brain.set_system_prompt(system_prompt)
                st.success("✅ تم تحديث نص النظام")

        st.divider()

        # أزرار التحكم
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ مسح", use_container_width=True):
                st.session_state.memory.clear()
                st.session_state.messages = []
                st.session_state.brain.clear_cache()
                st.rerun()

        with col2:
            if st.button("🔄 إعادة", use_container_width=True):
                st.session_state.memory.clear()
                st.session_state.messages = []
                st.session_state.brain.reset_system_prompt()
                st.session_state.brain.clear_cache()
                st.rerun()

        st.divider()

        # عرض الإحصائيات
        if st.session_state.show_stats:
            st.subheader("📊 الإحصائيات")

            memory_stats = st.session_state.memory.get_stats()
            brain_stats = st.session_state.brain.get_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("الرسائل", memory_stats["total_messages"])
                st.metric("الرموز", f"{memory_stats['total_tokens']}/{memory_stats['max_tokens']}")

            with col2:
                st.metric("Prompts", brain_stats.get("total_prompts_built", 0))
                hit_rate = brain_stats.get("hit_rate", 0)
                st.metric("Cache Hit", f"{hit_rate:.1f}%")

            # تفاصيل الذاكرة
            st.text(st.session_state.memory)

    # منطقة المحادثة
    st.subheader("💬 المحادثة")

    # عرض الرسائل السابقة
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            # عرض معلومات إضافية
            with st.expander("📋 التفاصيل", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"الرموز: {msg.get('tokens', 0)}")
                with col2:
                    st.caption(msg.get("timestamp", "")[-8:])  # الوقت فقط
                with col3:
                    if "generation_time" in msg:
                        st.caption(f"الوقت: {msg['generation_time']:.2f}s")

    # إدخال المستخدم
    st.divider()

    user_input = st.chat_input("اكتب رسالتك هنا...")

    if user_input:
        # إضافة رسالة المستخدم
        st.session_state.memory.add_user(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "tokens": estimate_tokens(user_input),
            "timestamp": datetime.now().isoformat()
        })

        # عرض رسالة المستخدم
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # بناء الـ prompt
        full_prompt = st.session_state.memory.format_as_prompt(
            system_prompt=st.session_state.brain.system_prompt,
            template="simple"
        )
        full_prompt += f"\n{st.session_state.brain.model_name}: "

        # إعدادات التوليد
        gen_config = {
            "max_gen_len": max_len,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": rep_penalty,
        }

        # التوليد
        with st.chat_message("assistant", avatar="🤖"):
            progress_placeholder = st.empty()
            response_placeholder = st.empty()

            # قياس الوقت
            start_time = time.time()

            response = generate_response(
                tokenizer,
                model,
                full_prompt,
                gen_config,
                progress_placeholder
            )

            elapsed_time = time.time() - start_time

            # عرض الرد
            response_placeholder.markdown(response)

            # عرض معلومات الأداء
            with st.expander("⏱️ معلومات الأداء", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("الوقت", f"{elapsed_time:.2f}s")
                with col2:
                    tokens = estimate_tokens(response)
                    st.metric("الرموز", tokens)
                with col3:
                    speed = tokens / max(elapsed_time, 0.01)
                    st.metric("السرعة", f"{speed:.1f} T/s")

        # حفظ الرد في الذاكرة
        st.session_state.memory.add_assistant(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "tokens": estimate_tokens(response),
            "timestamp": datetime.now().isoformat(),
            "generation_time": elapsed_time
        })


# ============================================================
# نقطة الدخول
# ============================================================

if __name__ == "__main__":
    main()