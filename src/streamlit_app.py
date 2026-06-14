# This is the main Streamlit app for the Redi restaurant.
# It provides a chat interface where users can ask questions about the menu.
# The app uses the RAG chain to generate menu-grounded answers.

from pathlib import Path

import streamlit as st

from chunking import split_documents
from data_loader import load_all_documents
from rag_chain import generate_llm_answer
from vector_store import create_vector_store


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ChromaDB folder where the vector database is stored.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def ensure_vector_store_exists():
    # If ChromaDB does not exist on Streamlit Cloud, build it automatically.
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        documents = load_all_documents()
        chunks = split_documents(documents)
        create_vector_store(chunks)


# Configure the Streamlit browser page.
st.set_page_config(
    page_title="Redi restaurant menu assistant",
    page_icon="🍛",
    layout="wide",
)


# Custom CSS for a website-style header.
st.markdown(
    """
    <style>
    .hero {
        padding: 34px 38px;
        border-radius: 18px;
        background: linear-gradient(135deg, #fff7ed 0%, #fef3c7 45%, #ecfccb 100%);
        border: 1px solid #fed7aa;
        margin-bottom: 26px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 850;
        color: #1f2937;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #4b5563;
        max-width: 820px;
        line-height: 1.5;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }

    .badge {
        background: white;
        color: #7c2d12;
        border: 1px solid #fdba74;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Create chat history when the app starts.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I can help you choose from the Redi menu. "
                "Tell me your budget, allergies, or food preference."
            ),
        }
    ]


# Store a selected example question from the sidebar.
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


def is_closing_message(text):
    # Detect short closing messages so we do not call the LLM unnecessarily.
    text = text.lower().strip()

    closing_phrases = [
        "thank you",
        "thanks",
        "thankyou",
        "thank u",
        "bye",
        "goodbye",
        "done",
        "stop",
        "exit",
        "quit",
    ]

    return any(phrase in text for phrase in closing_phrases)


# Sidebar controls.
with st.sidebar:
    st.header("Redi restaurant menu assistant")
    st.write("A restaurant RAG chatbot for menu recommendations and allergy-aware answers.")

    st.divider()

    examples = [
        "I want mild chicken under 12 euros and no nuts",
        "Show me vegetarian lunch menu items",
        "Do you have vegan drinks?",
        "I am allergic to milk. What can I eat?",
        "Which dishes are under 5 euros?",
    ]

    selected = st.selectbox("Example questions", [""] + examples)

    if st.button("Use this question", use_container_width=True):
        if selected:
            st.session_state.pending_question = selected
            st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = ""
        st.rerun()


# Hero section.
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Redi restaurant menu assistant</div>
        <div class="hero-subtitle">
            Ask natural questions about dishes, prices, allergens, vegetarian options,
            vegan possibilities, and lunch menu items.
        </div>
        <div class="badge-row">
            <div class="badge">Budget-aware</div>
            <div class="badge">Allergy-aware</div>
            <div class="badge">Vegetarian & vegan support</div>
            <div class="badge">English answers</div>
            <div class="badge">Menu-grounded RAG</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Chat section.
st.subheader("Chat With The AI Waiter")


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input box.
typed_question = st.chat_input(
    "Ask about dishes, prices, allergens, or recommendations..."
)


# Use typed question or selected example question.
question = typed_question or st.session_state.pending_question


# Process the question.
if question:
    # Clear pending question after using it.
    st.session_state.pending_question = ""

    # Save user question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Show user message.
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response.
    with st.chat_message("assistant"):
        if is_closing_message(question):
            answer = "You're welcome. Thank you for using Redi restaurant menu assistant."
        else:
            with st.spinner("Checking the menu..."):
                ensure_vector_store_exists()
                answer = generate_llm_answer(question)

        st.markdown(answer)

    # Save assistant answer.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # Rerun so the chat history refreshes cleanly.
    st.rerun()


# Footer.
st.markdown("---")
st.caption(
    "Redi restaurant menu assistant uses structured menu data, Chroma retrieval, Hugging Face embeddings, and Groq/Llama."
)