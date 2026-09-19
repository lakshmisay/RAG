import os
from pathlib import PurePath, PureWindowsPath

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# -----------------------------------------
# ENV SETUP
# -----------------------------------------
load_dotenv()

st.set_page_config(
    page_title="HR Support Chatbot",
    page_icon="HR",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------
# CONSTANTS
# -----------------------------------------
VECTOR_DB_PATH = "hr_faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_CHAT_MODEL", "")
OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "groq": "Groq",
}

if DEFAULT_OPENAI_MODEL not in OPENAI_MODELS:
    OPENAI_MODELS.insert(0, DEFAULT_OPENAI_MODEL)
if DEFAULT_GROQ_MODEL and DEFAULT_GROQ_MODEL not in GROQ_MODELS:
    GROQ_MODELS.insert(0, DEFAULT_GROQ_MODEL)

POLICY_AREAS = [
    "Onboarding",
    "Leave and time off",
    "Compensation",
    "Benefits",
    "IT security",
    "Code of conduct",
    "Workplace safety",
]

EXAMPLE_QUESTIONS = [
    "What is the onboarding process for a new employee?",
    "How many leave days am I entitled to?",
    "What is the policy for password security?",
    "What should I do during a workplace safety incident?",
]


# -----------------------------------------
# STYLES
# -----------------------------------------
st.markdown(
    """
    <style>
        :root {
            --hr-bg: #f7f9fc;
            --hr-panel: #ffffff;
            --hr-text: #182230;
            --hr-muted: #667085;
            --hr-line: #d9e2ec;
            --hr-accent: #1f7a8c;
            --hr-accent-strong: #145c6b;
            --hr-soft: #e8f4f7;
        }

        .stApp {
            background: var(--hr-bg);
            color: var(--hr-text);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--hr-line);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 1180px;
        }

        .app-header {
            border-bottom: 1px solid var(--hr-line);
            padding-bottom: 1.1rem;
            margin-bottom: 1.25rem;
        }

        .app-kicker {
            color: var(--hr-accent-strong);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .app-title {
            color: var(--hr-text);
            font-size: 2.35rem;
            line-height: 1.08;
            font-weight: 760;
            margin: 0;
        }

        .app-subtitle {
            color: var(--hr-muted);
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.6;
            margin-top: 0.65rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem;
        }

        .metric-tile {
            background: var(--hr-panel);
            border: 1px solid var(--hr-line);
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }

        .metric-label {
            color: var(--hr-muted);
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: var(--hr-text);
            font-size: 1.05rem;
            font-weight: 720;
        }

        .policy-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0 1rem;
        }

        .policy-pill {
            background: var(--hr-soft);
            border: 1px solid #c8e7ed;
            border-radius: 999px;
            color: var(--hr-accent-strong);
            font-size: 0.82rem;
            font-weight: 650;
            padding: 0.35rem 0.65rem;
        }

        .section-label {
            color: var(--hr-text);
            font-size: 0.96rem;
            font-weight: 720;
            margin: 1rem 0 0.4rem;
        }

        .empty-chat {
            background: #ffffff;
            border: 1px dashed #bdd5dc;
            border-radius: 8px;
            color: var(--hr-muted);
            padding: 1rem;
            line-height: 1.55;
        }

        div[data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid var(--hr-line);
            border-radius: 8px;
            padding: 0.45rem 0.65rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--hr-line);
            background: #ffffff;
            color: var(--hr-text);
            min-height: 2.45rem;
            white-space: normal;
            text-align: left;
        }

        .stButton > button:hover {
            border-color: var(--hr-accent);
            color: var(--hr-accent-strong);
            background: #f4fbfc;
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid var(--hr-line);
            background: rgba(247, 249, 252, 0.96);
        }

        @media (max-width: 760px) {
            .app-title {
                font-size: 1.8rem;
            }

            .metric-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------
# LOAD VECTOR STORE (CACHED)
# -----------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# -----------------------------------------
# LOAD LLM (CACHED)
# -----------------------------------------
@st.cache_resource
def load_llm(provider, model_name):
    if provider == "groq":
        return ChatGroq(model=model_name, temperature=0)

    return ChatOpenAI(model=model_name, temperature=0)


@st.cache_data(ttl=300)
def load_groq_models(api_key):
    if not api_key:
        return []

    models = Groq(api_key=api_key).models.list().data
    model_ids = sorted(model.id for model in models if getattr(model, "id", None))
    return model_ids


# -----------------------------------------
# PROMPT
# -----------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are an HR Support Assistant for company employees.

Answer questions using ONLY the information provided in the HR documents.
If the answer is not present, say:
"I'm not sure based on current HR policies."

Be clear, professional, and concise.

HR Context:
{context}

Employee Question:
{question}
"""
)


# -----------------------------------------
# SESSION STATE
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "provider" not in st.session_state:
    if os.getenv("OPENAI_API_KEY"):
        st.session_state.provider = "openai"
    elif os.getenv("GROQ_API_KEY"):
        st.session_state.provider = "groq"
    else:
        st.session_state.provider = "openai"


# -----------------------------------------
# UI HELPERS
# -----------------------------------------
def render_header():
    st.markdown(
        """
        <div class="app-header">
            <div class="app-kicker">Employee self-service</div>
            <h1 class="app-title">HR Support Chatbot</h1>
            <div class="app-subtitle">
                Ask policy questions and get grounded answers from the company HR knowledge base.
                The assistant only uses indexed HR documents and will say when the answer is not available.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(provider, model_name):
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-label">Knowledge base</div>
                <div class="metric-value">{VECTOR_DB_PATH}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Retrieved context</div>
                <div class="metric-value">Top 4 policy chunks</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Answer provider</div>
                <div class="metric-value">{PROVIDER_LABELS[provider]} / {model_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_policy_areas():
    pills = "".join(f'<span class="policy-pill">{area}</span>' for area in POLICY_AREAS)
    st.markdown(
        f"""
        <div class="section-label">Policy areas</div>
        <div class="policy-strip">{pills}</div>
        """,
        unsafe_allow_html=True,
    )


def queue_question(question):
    st.session_state.pending_question = question


def render_sidebar():
    with st.sidebar:
        st.header("Controls")
        st.caption("Tune the demo and manage the current chat session.")

        st.divider()
        st.subheader("Model")
        provider = st.selectbox(
            "Provider",
            options=["openai", "groq"],
            format_func=lambda value: PROVIDER_LABELS[value],
            key="provider",
        )

        if provider == "openai":
            model_name = st.selectbox(
                "Chat model",
                options=OPENAI_MODELS,
                index=OPENAI_MODELS.index(DEFAULT_OPENAI_MODEL),
                key="openai_model",
            )
        else:
            groq_api_key = os.getenv("GROQ_API_KEY")
            groq_models = []
            groq_models_error = None
            if groq_api_key:
                try:
                    groq_models = load_groq_models(groq_api_key)
                except Exception as exc:
                    groq_models_error = exc

            if groq_models:
                groq_default = (
                    DEFAULT_GROQ_MODEL
                    if DEFAULT_GROQ_MODEL in groq_models
                    else groq_models[0]
                )
                model_name = st.selectbox(
                    "Chat model",
                    options=groq_models,
                    index=groq_models.index(groq_default),
                    key="groq_model",
                )
            else:
                fallback_default = DEFAULT_GROQ_MODEL or GROQ_MODELS[0]
                model_name = st.text_input(
                    "Groq model",
                    value=fallback_default,
                    key="groq_model_manual",
                    help="Enter a model ID from your Groq console if model listing is unavailable.",
                )
                if groq_models_error:
                    st.warning(f"Could not load Groq model list: {groq_models_error}")

        if provider == "groq" and not os.getenv("GROQ_API_KEY"):
            st.warning("Set GROQ_API_KEY in your .env file to use Groq.")
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            st.warning("Set OPENAI_API_KEY in your .env file to use OpenAI.")
        if not os.getenv("OPENAI_API_KEY"):
            st.info("This FAISS index still needs OPENAI_API_KEY for embeddings.")

        st.divider()
        st.subheader("Session")
        question_count = sum(
            1 for message in st.session_state.messages if message["role"] == "user"
        )
        answer_count = sum(
            1 for message in st.session_state.messages if message["role"] == "assistant"
        )
        metric_cols = st.columns(2)
        metric_cols[0].metric("Questions", question_count)
        metric_cols[1].metric("Answers", answer_count)

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.pending_question = None
            st.rerun()

        st.divider()
        st.subheader("Try a question")
        for question in EXAMPLE_QUESTIONS:
            st.button(
                question,
                key=f"example-{question}",
                use_container_width=True,
                on_click=queue_question,
                args=(question,),
            )

        st.divider()
        st.subheader("Runtime")
        st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
        st.caption(f"Chat: `{PROVIDER_LABELS[provider]} / {model_name}`")
        st.caption("Source: local FAISS index")

    return provider, model_name


def render_sources():
    if not st.session_state.last_sources:
        return

    with st.expander("View sources from the last answer"):
        for index, doc in enumerate(st.session_state.last_sources, start=1):
            source_path = doc.metadata.get("source", "HR policy document")
            source_name = PureWindowsPath(str(source_path)).name
            source_name = PurePath(source_name).name
            st.markdown(f"**Source {index}:** `{source_name}`")
            st.write(doc.page_content[:700].strip())
            if index != len(st.session_state.last_sources):
                st.divider()


# -----------------------------------------
# APP
# -----------------------------------------
provider, model_name = render_sidebar()
render_header()
render_metrics(provider, model_name)
render_policy_areas()

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is required for retrieval because this FAISS index was "
        "created with OpenAI embeddings. You can still select Groq for answers "
        "after adding the OpenAI key for embeddings."
    )
    st.stop()

if provider == "groq" and not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to `.env` or select OpenAI in the sidebar.")
    st.stop()

try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = load_llm(provider, model_name)
except Exception as exc:
    st.error("The HR assistant could not start. Check your API key, selected model, and FAISS index.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-chat">
            Start with a specific HR question such as leave eligibility, onboarding steps,
            payroll timing, workplace conduct, or IT security rules.
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

render_sources()

typed_question = st.chat_input("Ask an HR-related question...")
user_question = st.session_state.pending_question or typed_question
st.session_state.pending_question = None

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Searching HR policies and drafting a grounded answer..."):
        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)
        st.session_state.last_sources = docs

        response = llm.invoke(
            prompt.format_messages(
                context=context,
                question=user_question,
            )
        )

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )
    st.rerun()
