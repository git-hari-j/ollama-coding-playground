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

# File to store conversation history
HISTORY_FILE = Path("conversation_history.json")

# Ollama API base URL
OLLAMA_API_URL = "http://localhost:11434/api"

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def fetch_models():
    """Fetches the list of available models from the Ollama API."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}. Is Ollama running?")
        return []

# MODIFIED FUNCTION TO FIX THE ERROR
def load_history():
    """
    Loads conversation history, converting old format if necessary.
    Old format: [{"query": ..., "response": ...}]
    New format: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r") as f:
            history_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    if not history_data:
        return []

    # Check if migration is needed by inspecting the first record
    if "query" in history_data[0]:
        migrated_history = []
        for item in history_data:
            if "query" in item:
                migrated_history.append({"role": "user", "content": item["query"]})
            if "response" in item:
                migrated_history.append({"role": "assistant", "content": item["response"]})
        
        # Overwrite the old file with the new, corrected format
        save_history(migrated_history)
        return migrated_history
        
    return history_data # Already in the new format


def save_history(history):
    """Saves conversation history to the JSON file."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

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
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.success("Chat history cleared!")
        st.rerun()

# --- Main App UI ---
st.title("Ollama Coding Playground 🖥️")

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

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
            save_history(st.session_state.messages)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please select a model from the sidebar to begin.")