import streamlit as st
import requests
import json
from pathlib import Path
from langchain_ollama.llms import OllamaLLM

# --- Configuration ---
st.set_page_config(
    page_title="Ollama Coding Playground",
    page_icon="🖥️",
    layout="wide"  # Use wide layout for a better chat UI
)

# Directory and file management
HISTORY_DIR = Path("conversation_histories")
HISTORY_DIR.mkdir(exist_ok=True)

# Ollama API base URL
OLLAMA_API_URL = "http://localhost:11434/api"

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def fetch_models():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}. Is Ollama running?")
        return []

def load_history(session_id):
    session_file = HISTORY_DIR / f"{session_id}.json"
    if not session_file.exists():
        return []

    try:
        with open(session_file, "r") as f:
            history_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    return history_data if history_data else []

def save_history(session_id, history):
    session_file = HISTORY_DIR / f"{session_id}.json"
    with open(session_file, "w") as f:
        json.dump(history, f, indent=2)

def get_session_ids():
    return [p.stem for p in HISTORY_DIR.glob("*.json")]

def delete_sessions(session_ids):
    for session_id in session_ids:
        session_file = HISTORY_DIR / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

# --- Sidebar UI ---
st.sidebar.title("Settings")
st.sidebar.markdown("---")

available_models = fetch_models()
if available_models:
    model_name = st.sidebar.selectbox("Select LLM Model", available_models)
else:
    model_name = None
    st.sidebar.warning("No models found. Ensure Ollama is running and models are installed.")

# Popular programming languages dropdown
programming_languages = ["Python", "JavaScript", "Java", "C++", "Go", "Bash", "HTML", "Markdown"]
selected_language = st.sidebar.selectbox("Select Programming Language", programming_languages)

# Session management
st.sidebar.subheader("Chat Sessions")
session_ids = get_session_ids()
if session_ids:
    selected_sessions = st.sidebar.multiselect("Select Sessions to Delete", session_ids)
    if st.sidebar.button("Delete Selected Sessions"):
        delete_sessions(selected_sessions)
        st.sidebar.success("Selected sessions deleted!")
        st.rerun()  # Refresh the sidebar
else:
    st.sidebar.info("No chat sessions available.")

# Single Save Session input
current_session_name = st.sidebar.text_input("Save this session as:", key="unique_save_session_input")
if st.sidebar.button("Save Current Chat"):
    if current_session_name:
        save_history(current_session_name, st.session_state.messages)
        st.sidebar.success(f"Session '{current_session_name}' saved!")
    else:
        st.sidebar.warning("Please enter a session name.")

# --- Main App UI ---
st.title("Ollama Coding Playground 🖥️")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load selected session if exists
selected_session_id = st.sidebar.selectbox("Load Chat Session", [""] + session_ids)
if selected_session_id:
    st.session_state.messages = load_history(selected_session_id)

# Display chat messages in a styled manner
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box for chat queries
if model_name:
    if prompt := st.chat_input("Enter your code or query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using the selected model
        try:
            llm = OllamaLLM(model=model_name)
            # Prepend the programming language to the prompt for context
            contextual_prompt = f"Please respond in the context of {selected_language}: {prompt}"
            with st.chat_message("assistant"):
                full_response = st.write_stream(llm.stream(contextual_prompt))

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please select a model from the sidebar to begin.")
