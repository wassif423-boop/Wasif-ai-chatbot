```python
import streamlit as st
import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS ----------
st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f8fafc;
}

.block-container {
    max-width: 1100px;
    padding-top: 0.5rem !important;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stDecoration"] {
    display: none;
}

[data-testid="stSidebar"] {
    background: #10172a;
    border-right: 1px solid #27334d;
}

[data-testid="stSidebar"] * {
    color: #eef2ff !important;
}

.ai-header {
    background: linear-gradient(120deg, #172554, #312e81, #164e63);
    border: 1px solid #3c4b82;
    padding: 28px 20px;
    border-radius: 20px;
    margin-bottom: 25px;
    text-align: center;
}

.ai-header h1 {
    color: white !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    margin: 0;
}

.ai-header p {
    color: #e0e7ff !important;
    margin-top: 10px;
}

[data-testid="stChatMessage"] {
    background: #151e33;
    border: 1px solid #303d59;
    border-radius: 16px;
    padding: 15px;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #f8fafc !important;
    line-height: 1.7 !important;
}

[data-testid="stChatInput"] {
    background: #151e33 !important;
    border: 1px solid #6366f1 !important;
    border-radius: 15px !important;
}

[data-testid="stChatInput"] textarea {
    color: white !important;
    -webkit-text-fill-color: white !important;
    caret-color: #a5b4fc !important;
    background: #151e33 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #b8c3d9 !important;
    -webkit-text-fill-color: #b8c3d9 !important;
    opacity: 1 !important;
}

.stButton > button {
    background: #1a2540 !important;
    color: white !important;
    border: 1px solid #475569 !important;
    border-radius: 11px !important;
    min-height: 42px;
}

.stButton > button:hover {
    background: #3730a3 !important;
    border-color: #818cf8 !important;
}

.stMarkdown, .stMarkdown p, label {
    color: #e5e7eb;
}

.ai-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding: 25px 0 8px;
}

@media(max-width: 640px) {
    .ai-header h1 {
        font-size: 29px !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------- SESSION STATE ----------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = "chat_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "regenerate" not in st.session_state:
    st.session_state.regenerate = False


# ---------- CHAT HELPERS ----------
def new_chat():
    chat_id = "chat_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }
    st.session_state.current_chat = chat_id
    st.session_state.pending_question = None
    st.session_state.regenerate = False


def current_messages():
    return st.session_state.chats[
        st.session_state.current_chat
    ]["messages"]


# ---------- LOAD AI MODEL ----------
@st.cache_resource
def load_model():
    name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(name)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype="auto"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ---------- AI RESPONSE ----------
def generate_answer(question):
    tokenizer, model, device = load_model()

    conversation = [
        {
            "role": "system",
            "content": (
                "You are AI Chatbot, a helpful assistant. "
                "Give clear, useful and simple answers. "
                "Use readable paragraphs and lists when helpful. "
                "Be honest if you do not know something."
            )
        }
    ]

    # Keep recent conversation context
    recent = current_messages()[-10:]

    for msg in recent:
        conversation.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Add current question if not already included
    if not recent or recent[-1]["content"] != question:
        conversation.append({
            "role": "user",
            "content": question
        })

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    answer_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    reply = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    ).strip()

    return reply or "Sorry, please try asking in a different way."


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("# 🤖 AI Chatbot")
    st.caption("YOUR PERSONAL AI ASSISTANT")
    st.divider()

    if st.button("＋ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("### 💬 Chat History")

    chat_ids = list(st.session_state.chats.keys())[::-1]

    for chat_id in chat_ids:
        chat = st.session_state.chats[chat_id]
        title = chat["title"]

        if st.button(
            "💬 " + title[:25],
            key="open_" + chat_id,
            use_container_width=True
        ):
            st.session_state.current_chat = chat_id
            st.session_state.pending_question = None
            st.rerun()

    st.divider()

    if st.button("🗑️ Delete Current Chat", use_container_width=True):
        old_id = st.session_state.current_chat
        st.session_state.chats.pop(old_id, None)
        new_chat()
        st.rerun()

    if st.button("🧹 Clear All History", use_container_width=True):
        st.session_state.chats = {}
        new_chat()
        st.rerun()

    st.divider()

    st.markdown("### ⚙️ Settings")
    st.caption("Model: Qwen 2.5 0.5B")
    st.caption("Device: Automatic CPU/GPU")
    st.caption("Built with Streamlit")


# ---------- HEADER ----------
st.markdown("""
<div class="ai-header">
    <h1>AI Chatbot 🤖</h1>
    <p>Ask anything. Learn anything. Explore new ideas.</p>
</div>
""", unsafe_allow_html=True)


# ---------- CURRENT CHAT ----------
messages = current_messages()

for index, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Regenerate button under the latest assistant response
        if (
            msg["role"] == "assistant"
            and index == len(messages) - 1
        ):
            if st.button("🔄 Regenerate Answer", key="regen_" + st.session_state.current_chat):
                st.session_state.regenerate = True
                st.rerun()


# ---------- WELCOME SCREEN ----------
if not messages:
    st.markdown("### ✨ What can I help you with today?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💻 Explain Python", use_container_width=True):
            st.session_state.pending_question = "Explain Python in simple words."
            st.rerun()

        if st.button("🧠 What is AI?", use_container_width=True):
            st.session_state.pending_question = "What is Artificial Intelligence?"
            st.rerun()

    with col2:
        if st.button("📚 Learn English", use_container_width=True):
            st.session_state.pending_question = "Give me tips to learn English."
            st.rerun()

        if st.button("🚀 Project Ideas", use_container_width=True):
            st.session_state.pending_question = "Suggest three beginner-friendly AI projects."
            st.rerun()


# ---------- CHAT INPUT ----------
question = st.chat_input("Message AI Chatbot...")

if question:
    st.session_state.pending_question = question


# ---------- REGENERATE ----------
if st.session_state.regenerate:
    st.session_state.regenerate = False

    msgs = current_messages()

    if msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()

    if msgs and msgs[-1]["role"] == "user":
        question = msgs[-1]["content"]
    else:
        question = None

    if question:
        try:
            with st.chat_message("assistant"):
                with st.spinner("AI is thinking..."):
                    reply = generate_answer(question)
                    st.markdown(reply)

            msgs.append({
                "role": "assistant",
                "content": reply
            })

        except Exception as e:
            st.error("AI model could not respond.")
            st.code(str(e))

        st.rerun()


# ---------- GENERATE NEW ANSWER ----------
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    msgs = current_messages()

    msgs.append({
        "role": "user",
        "content": question
    })

    if len(msgs) == 1:
        st.session_state.chats[
            st.session_state.current_chat
        ]["title"] = question[:30]

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                reply = generate_answer(question)
                st.markdown(reply)

        msgs.append({
            "role": "assistant",
            "content": reply
        })

    except Exception as e:
        st.error("AI model could not respond.")
        st.code(str(e))

    st.rerun()


# ---------- DOWNLOAD HISTORY ----------
if st.session_state.chats:
    history_json = json.dumps(
        st.session_state.chats,
        indent=2,
        ensure_ascii=False
    )

    st.sidebar.download_button(
        "📥 Download Chat History",
        data=history_json,
        file_name="ai_chatbot_history.json",
        mime="application/json",
        use_container_width=True
    )


# ---------- FOOTER ----------
st.markdown("""
<div class="ai-footer">
    Made with ❤️ | AI Chatbot
</div>
""", unsafe_allow_html=True)
```
