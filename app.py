import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Nova AI | Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
/* Main page */
.stApp {
    background: #0b1020;
    color: #f8fafc;
}

/* Hide default Streamlit top decoration */
[data-testid="stDecoration"] {
    display: none;
}

/* Main content spacing */
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #10172a;
    border-right: 1px solid #27334d;
}

[data-testid="stSidebar"] * {
    color: #eef2ff !important;
}

/* Header */
.nova-header {
    background: linear-gradient(120deg, #172554, #312e81, #164e63);
    border: 1px solid #3c4b82;
    padding: 30px 24px;
    border-radius: 22px;
    margin-bottom: 25px;
    text-align: center;
}

.nova-header h1 {
    color: #ffffff !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    margin: 0;
}

.nova-header p {
    color: #e0e7ff !important;
    font-size: 16px;
    margin-top: 10px;
}

.nova-label {
    display: inline-block;
    color: #ddd6fe !important;
    background: #312e81;
    border: 1px solid #6366f1;
    border-radius: 30px;
    padding: 5px 13px;
    margin-bottom: 12px;
    font-size: 12px;
}

/* General text */
.stApp p,
.stApp label,
.stMarkdown,
.stMarkdown p {
    color: #e5e7eb;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #151e33;
    border: 1px solid #303d59;
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Make chat response text readable */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    color: #f8fafc !important;
    font-size: 16px !important;
    line-height: 1.75 !important;
}

/* Chat input container */
[data-testid="stChatInput"] {
    background: #151e33 !important;
    border: 1px solid #6366f1 !important;
    border-radius: 16px !important;
}

/* Input and textarea text */
[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #a5b4fc !important;
    background: #151e33 !important;
    font-size: 16px !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #b8c3d9 !important;
    -webkit-text-fill-color: #b8c3d9 !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton > button {
    background: #1a2540 !important;
    color: #ffffff !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    min-height: 45px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #3730a3 !important;
    border-color: #818cf8 !important;
    color: #ffffff !important;
}

/* Status messages */
[data-testid="stStatusWidget"] {
    color: #ffffff !important;
}

/* Footer */
.nova-footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 13px;
    padding: 24px 0 8px 0;
}

/* Small screen adjustments */
@media (max-width: 640px) {
    .nova-header {
        padding: 22px 12px;
    }

    .nova-header h1 {
        font-size: 29px !important;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("# 🤖 Nova AI")
    st.caption("YOUR PERSONAL AI ASSISTANT")
    st.divider()

    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### ✨ Try asking")
    st.caption("💡 Explain Artificial Intelligence")
    st.caption("💻 Teach me Python")
    st.caption("📚 Help me learn English")
    st.caption("🚀 Give me project ideas")

    st.divider()
    st.markdown("### 🧠 About")
    st.caption("Model: Qwen 2.5 0.5B")
    st.caption("Built with Streamlit")


# ---------- AI MODEL ----------
@st.cache_resource
def load_model():
    name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    return tokenizer, model


# ---------- HEADER ----------
st.markdown("""
<div class="nova-header">
    <div class="nova-label">✦ AI POWERED ASSISTANT</div>
    <h1>Nova AI 🤖</h1>
    <p>Ask anything. Learn anything. Explore new ideas.</p>
</div>
""", unsafe_allow_html=True)


# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- WELCOME SCREEN ----------
if not st.session_state.messages:
    st.markdown("### ✨ What can I help you with today?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💻  Explain Python", use_container_width=True):
            st.session_state.pending_question = (
                "Explain Python in simple words."
            )
            st.rerun()

        if st.button("🧠  What is AI?", use_container_width=True):
            st.session_state.pending_question = (
                "What is Artificial Intelligence?"
            )
            st.rerun()

    with col2:
        if st.button("📚  Learn English", use_container_width=True):
            st.session_state.pending_question = (
                "Give me some tips to learn English."
            )
            st.rerun()

        if st.button("🚀  Project Ideas", use_container_width=True):
            st.session_state.pending_question = (
                "Suggest three beginner-friendly AI projects."
            )
            st.rerun()


# ---------- CHAT INPUT ----------
question = st.chat_input("Message Nova AI...")

if question:
    st.session_state.pending_question = question

if st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Loading AI model..."):
            tokenizer, model = load_model()

        conversation = [
            {
                "role": "system",
                "content": (
                    "You are Nova AI, a helpful assistant. "
                    "Give clear, useful, simple answers. "
                    "Use readable paragraphs and lists when helpful. "
                    "Be honest if you do not know something."
                )
            }
        ]

        conversation.extend(st.session_state.messages[-10:])

        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        with st.chat_message("assistant"):
            with st.spinner("Nova AI is thinking..."):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )

                answer = outputs[0][
                    inputs["input_ids"].shape[1]:
                ]

                reply = tokenizer.decode(
                    answer,
                    skip_special_tokens=True
                ).strip()

                if not reply:
                    reply = (
                        "I couldn't generate a response. "
                        "Please try asking in a different way."
                    )

                st.markdown(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

    except Exception as e:
        st.error("The AI model could not respond.")
        st.code(str(e))


# ---------- FOOTER ----------
st.markdown("""
<div class="nova-footer">
    Made with ❤️ | Nova AI · Your AI Assistant
</div>
""", unsafe_allow_html=True)
