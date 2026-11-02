import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Nova AI | Your AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #11152b 55%, #17102b 100%);
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #0b1020;
    border-right: 1px solid #29304a;
}

.hero {
    padding: 28px 20px;
    border: 1px solid #34365f;
    border-radius: 22px;
    background: linear-gradient(120deg, #20264b, #302052, #172c4a);
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero p {
    color: #c4c9e8;
    font-size: 16px;
}

.badge {
    display: inline-block;
    background: #34305f;
    color: #c9c4ff;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 13px;
    margin-bottom: 10px;
}

div[data-testid="stChatMessage"] {
    background: rgba(31, 41, 70, 0.72);
    border: 1px solid #343b5a;
    border-radius: 18px;
    padding: 15px;
    margin-bottom: 12px;
}

div[data-testid="stChatInput"] {
    border: 1px solid #6264c7;
    border-radius: 16px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #454875;
    background: #202541;
    color: white;
    padding: 10px;
}

div.stButton > button:hover {
    border-color: #9991ff;
    color: #c7c3ff;
}

.footer {
    text-align: center;
    color: #9099b8;
    font-size: 12px;
    padding: 20px 0;
}

</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🤖 Nova AI")
    st.caption("Your personal AI assistant")
    st.markdown("---")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### ✨ Try asking")
    st.caption("💡 Explain Artificial Intelligence")
    st.caption("💻 Teach me Python")
    st.caption("📚 Help me learn English")
    st.caption("🚀 Give me project ideas")

    st.markdown("---")
    st.caption("🧠 Model: Qwen 2.5")
    st.caption("⚡ Powered by Transformers")

# ---------- LOAD AI MODEL ----------
@st.cache_resource
def load_model():
    name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    return tokenizer, model

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <div class="badge">✦ YOUR AI ASSISTANT</div>
    <h1>Meet Nova AI 🤖</h1>
    <p>Ask questions, explore ideas, and learn something new.</p>
</div>
""", unsafe_allow_html=True)

# ---------- CHAT HISTORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- EXAMPLE PROMPTS ----------
if not st.session_state.messages:
    st.markdown("### ✨ What can I help you with today?")

    col1, col2 = st.columns(2)

    with col1:
        ask_python = st.button("💻 Explain Python", use_container_width=True)
        ask_ai = st.button("🧠 What is AI?", use_container_width=True)

    with col2:
        ask_english = st.button("📚 Help me learn English", use_container_width=True)
        ask_ideas = st.button("🚀 Give me project ideas", use_container_width=True)

    if ask_python:
        st.session_state.pending_question = "Explain Python in simple words."
        st.rerun()
    elif ask_ai:
        st.session_state.pending_question = "What is Artificial Intelligence?"
        st.rerun()
    elif ask_english:
        st.session_state.pending_question = "Help me learn English."
        st.rerun()
    elif ask_ideas:
        st.session_state.pending_question = "Give me beginner AI project ideas."
        st.rerun()

# ---------- USER INPUT ----------
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
        with st.spinner("🔄 Loading AI model..."):
            tokenizer, model = load_model()

        conversation = [
            {
                "role": "system",
                "content": (
                    "You are Nova AI, a helpful and friendly assistant. "
                    "Answer clearly and simply. If you do not know something, "
                    "be honest. Do not claim to have searched the internet."
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
            with st.spinner("✨ Nova AI is thinking..."):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )

                answer = outputs[0][inputs["input_ids"].shape[1]:]

                reply = tokenizer.decode(
                    answer,
                    skip_special_tokens=True
                ).strip()

                if not reply:
                    reply = "Sorry, I couldn't generate an answer. Please try again."

                st.markdown(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

    except Exception as e:
        st.error("Sorry! The AI model could not generate a response.")
        st.code(str(e))

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit and Qwen AI
</div>
""", unsafe_allow_html=True)
