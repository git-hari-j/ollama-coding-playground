import streamlit as st
import requests
import json
from pathlib import Path
from langchain_ollama.llms import OllamaLLM

# --- Configuration ---
st.set_page_config(
    page_title="Ollama Coding Playground",
    page_icon="🖥️",
    layout="centered"
)

# Directory and file management
HISTORY_DIR = Path("conversation_histories")
HISTORY_DIR.mkdir(exist_ok=True)  # Ensure directory exists

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
with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    
    available_models = fetch_models()
    if available_models:
        model_name = st.selectbox("Select LLM Model", available_models)
    else:
        model_name = None
        st.warning("No models found. Please ensure Ollama is running and models are installed.")

    code_language = st.selectbox(
        "Highlight Language",
        ["python", "javascript", "java", "c++", "go", "bash", "html", "markdown", "auto"]
    )
    
    st.markdown("---")
    
    # Session management
    st.subheader("Chat Sessions")
    session_ids = get_session_ids()
    
    if session_ids:
        selected_sessions = st.multiselect("Select Sessions to Delete", session_ids)
        if st.button("Delete Selected Sessions", use_container_width=True):
            delete_sessions(selected_sessions)
            st.success("Selected sessions deleted!")
            st.rerun()
    else:
        st.info("No chat sessions available.")

    # Save current session
    current_session_name = st.text_input("Save this session as:", "")
    if st.button("Save Current Chat", use_container_width=True):
        if current_session_name:
            save_history(current_session_name, st.session_state.messages)
            st.success(f"Session '{current_session_name}' saved!")
        else:
            st.warning("Please enter a session name.")

# --- Main App UI ---
st.title("Ollama Coding Playground 🖥️")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load selected session if exists
selected_session_id = st.sidebar.selectbox("Load Chat Session", [""] + session_ids)
if selected_session_id:
    st.session_state.messages = load_history(selected_session_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if model_name:
    if prompt := st.chat_input("Enter your code or query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            llm = OllamaLLM(model=model_name)
            with st.chat_message("assistant"):
                full_response = st.write_stream(llm.stream(prompt))

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # Save history of current chat dynamically
            current_session_name = st.text_input("Save this session as:", "") 
            if current_session_name:  
                save_history(current_session_name, st.session_state.messages)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please select a model from the sidebar to begin.")