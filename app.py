import streamlit as st
import os
import logging
from dotenv import load_dotenv
from scripts.utils import load_yaml_config
from scripts.prompt_builder import build_prompt_from_config
from scripts.vector_db_rag import retrieve_relevant_documents, setup_logging, get_system_status
from scripts.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH
from langchain_groq import ChatGroq
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-size: 1.1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'config_loaded' not in st.session_state:
    st.session_state.config_loaded = False

# Load configurations
@st.cache_resource
def load_configurations():
    """Load app and prompt configurations"""
    try:
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        return app_config, prompt_config
    except Exception as e:
        st.error(f"Error loading configurations: {e}")
        return None, None

# Initialize RAG system
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components"""
    try:
        setup_logging()
        app_config, prompt_config = load_configurations()
        if app_config and prompt_config:
            return app_config, prompt_config
        return None, None
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None, None

# Main function for RAG query processing
def process_rag_query(query, app_config, prompt_config, n_results=5, threshold=0.3):
    """Process a query using the RAG system"""
    try:
        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_documents(
            query, n_results=n_results, threshold=threshold
        )
        
        if not relevant_docs:
            return "I couldn't find any relevant documents for your query. Please try rephrasing your question."
        
        # Build prompt with context
        input_data = f"Relevant documents:\n\n{relevant_docs}\n\nUser's question:\n\n{query}"
        rag_prompt = build_prompt_from_config(prompt_config["rag_assistant_prompt"], input_data=input_data)
        
        # Get LLM response
        llm = ChatGroq(model=app_config["llm"])
        response = llm.invoke(rag_prompt)
        
        return response.content, relevant_docs
        
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return f"An error occurred while processing your query: {e}", []

# Sidebar for configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Load configurations
    app_config, prompt_config = initialize_rag_system()
    
    if app_config and prompt_config:
        st.success("✅ RAG System Initialized")
        
        # Configuration parameters
        st.markdown("### 🔍 Retrieval Settings")
        n_results = st.slider("Number of documents to retrieve", 1, 10, 5)
        threshold = st.slider("Similarity threshold", 0.1, 1.0, 0.5, 0.1)
        
        st.markdown("### 🤖 LLM Settings")
        st.info(f"**Model:** {app_config['llm']}")
        
        # Display reasoning strategies
        if 'reasoning_strategies' in app_config:
            st.markdown("### 🧠 Reasoning Strategies")
            strategy = st.selectbox(
                "Select reasoning strategy",
                list(app_config['reasoning_strategies'].keys())
            )
            
            if strategy:
                st.text_area(
                    f"{strategy} Strategy",
                    value=app_config['reasoning_strategies'][strategy],
                    height=150,
                    disabled=True
                )
    else:
        st.error("❌ Failed to initialize RAG system")
        st.stop()

# Main application
def main():
    st.markdown('<h1 class="main-header">🤖 RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("🔍 Searching documents..."):
                # Process the query
                response, relevant_docs = process_rag_query(
                    prompt, app_config, prompt_config, n_results, threshold
                )
            
            # Display response
            message_placeholder.markdown(response)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show relevant documents in expander
            if relevant_docs:
                with st.expander("📚 Relevant Documents Found"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Document {i}:**")
                        st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                        st.divider()
            else:
                with st.expander("📚 Document Search Results"):
                    st.warning("No relevant documents found")
                    st.info("This could be due to:")
                    st.info("• ChromaDB running in fallback mode")
                    st.info("• No documents matching your query")
                    st.info("• System compatibility issues")
    
    # Sidebar actions
    with st.sidebar:
        # System Status
        st.markdown("### 🔧 System Status")
        try:
            system_status = get_system_status()
            if system_status["chromadb_available"]:
                st.success("✅ ChromaDB: Operational")
            else:
                st.warning("⚠️ ChromaDB: Fallback Mode")
                st.info("Vector search is limited due to SQLite3 compatibility")
            
            st.info(f"Status: {system_status['status']}")
        except Exception as e:
            st.error(f"❌ Status check failed: {e}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Export chat button
        if st.session_state.messages:
            chat_text = "\n\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in st.session_state.messages
            ])
            
            st.download_button(
                label="💾 Export Chat",
                data=chat_text,
                file_name=f"rag_chat_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
