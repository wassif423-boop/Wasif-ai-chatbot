import streamlit as st
import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Document readers
from pypdf import PdfReader
from docx import Document
from io import BytesIO

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
.stApp { background: #0b1020; color: #f8fafc; }
.block-container { max-width: 1100px; padding-top: 0.5rem !important; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] { background: #10172a; border-right: 1px solid #27334d; }
[data-testid="stSidebar"] * { color: #eef2ff !important; }
.ai-header {
    background: linear-gradient(120deg, #172554, #312e81, #164e63);
    border: 1px solid #3c4b82; padding: 28px 20px; border-radius: 20px;
    margin-bottom: 25px; text-align: center;
}
.ai-header h1 { color: white !important; font-size: 38px !important; font-weight: 800 !important; margin: 0; }
.ai-header p { color: #e0e7ff !important; margin-top: 10px; }
[data-testid="stChatMessage"] {
    background: #151e33; border: 1px solid #303d59; border-radius: 16px;
    padding: 15px; margin-bottom: 12px;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    color: #f8fafc !important; line-height: 1.7 !important;
}
[data-testid="stChatInput"] {
    background: #151e33 !important; border: 1px solid #6366f1 !important;
    border-radius: 15px !important;
}
[data-testid="stChatInput"] textarea {
    color: white !important; -webkit-text-fill-color: white !important;
    caret-color: #a5b4fc !important; background: #151e33 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #b8c3d9 !important; -webkit-text-fill-color: #b8c3d9 !important;
    opacity: 1 !important;
}
.stButton > button, [data-testid="stDownloadButton"] button {
    background: #1a2540 !important; color: white !important;
    border: 1px solid #475569 !important; border-radius: 11px !important;
    min-height: 42px;
}
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
    background: #3730a3 !important; border-color: #818cf8 !important;
}
.stMarkdown, .stMarkdown p, label { color: #e5e7eb; }
.ai-footer { text-align: center; color: #94a3b8; font-size: 13px; padding: 25px 0 8px; }
@media(max-width: 640px) { .ai-header h1 { font-size: 29px !important; } }
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
if "document_context" not in st.session_state:
    st.session_state.document_context = ""

# ---------- CHAT HELPERS ----------
def new_chat():
    chat_id = "chat_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }
# ---------- CURRENT CHAT ----------
messages = current_messages()
for index, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and index == len(messages) - 1:
            if st.button(
                "🔄 Regenerate Answer",
                key="regen_" + st.session_state.current_chat
            ):
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

    question_to_repeat = (
        msgs[-1]["content"]
        if msgs and msgs[-1]["role"] == "user"
        else None
    )

    if question_to_repeat:
        try:
            with st.chat_message("assistant"):
                with st.spinner("AI is thinking..."):
                    reply = generate_answer(question_to_repeat)
                    st.markdown(reply)
            msgs.append({"role": "assistant", "content": reply})
        except Exception as error:
            st.error("AI model could not respond.")
            st.code(str(error))
    st.rerun()

# ---------- GENERATE NEW ANSWER ----------
if st.session_state.pending_question:
    question_to_answer = st.session_state.pending_question
    st.session_state.pending_question = None
    msgs = current_messages()

    msgs.append({"role": "user", "content": question_to_answer})

    if len(msgs) == 1:
        st.session_state.chats[st.session_state.current_chat]["title"] = question_to_answer[:30]

    with st.chat_message("user"):
        st.markdown(question_to_answer)

    try:
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                reply = generate_answer(question_to_answer)
                st.markdown(reply)
        msgs.append({"role": "assistant", "content": reply})
    except Exception as error:
        st.error("AI model could not respond.")
        st.code(str(error))

    st.rerun()

# ---------- DOWNLOAD HISTORY ----------
if st.session_state.chats:
    history_json = json.dumps(
        st.session_state.chats, indent=2, ensure_ascii=False
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
