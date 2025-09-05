# Fix for SQLite version compatibility on Streamlit Cloud
import sys
import subprocess
import importlib

# Install and setup pysqlite3 for Streamlit Cloud
try:
    # Try to import pysqlite3 and replace sqlite3 module
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3 is not available, try to install it
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pysqlite3-binary'])
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except:
        pass  # Fall back to system sqlite3

import streamlit as st
import os
import logging
from dotenv import load_dotenv
from scripts.utils import load_yaml_config
from scripts.prompt_builder import build_prompt_from_config
from scripts.vector_db_rag import retrieve_relevant_documents, setup_logging
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
    
    # Welcome message and description
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;">
            <h3>🔍 Intelligent AI/ML Research Assistant</h3>
            <p>Ask me anything about <strong>Computer Vision</strong>, <strong>Generative Models</strong>, <strong>Time Series Analysis</strong>, and <strong>Machine Learning</strong>!</p>
            <p>I have knowledge about CLIP, VAEs, Distance Profiles, Class Imbalance, and more. Check the sidebar for example questions and available documents.</p>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Sidebar actions
    with st.sidebar:
        st.markdown("---")
        
        # Information about the system
        st.markdown("## 📊 Knowledge Base")
        
        # Data information
        with st.expander("📚 Available Documents"):
            st.markdown("""
            **Current Publications:**
            
            🎨 **CLIP Implementation Guide**
            - Multi-modal learning with Vision Transformers
            - Contrastive learning techniques
            - Image-text alignment
            
            🔄 **VAE Comprehensive Study**  
            - Variational Autoencoders applications
            - Data compression and generation
            - Anomaly detection and denoising
            
            📈 **Distance Profile Analysis**
            - Time series pattern recognition
            - Similarity search algorithms
            - Time-step classification
            
            ⚖️ **Class Imbalance Study**
            - Binary classification techniques
            - SMOTE and class weight methods
            - Evaluation metrics for imbalanced data
            """)
        
        # Sample questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            **Computer Vision & Multi-Modal:**
            - "What is CLIP and how does it work?"
            - "How does contrastive learning align images and text?"
            - "Explain Vision Transformers in CLIP"
            
            **Generative Models:**
            - "What are VAEs and how do they differ from autoencoders?"
            - "How do VAEs handle data compression?"
            - "What is the reparameterization trick?"
            
            **Time Series Analysis:**
            - "What is distance profile in time series?"
            - "How does MASS algorithm work?"
            - "Explain time-step classification"
            
            **Machine Learning:**
            - "How to handle class imbalance?"
            - "What is SMOTE and when to use it?"
            - "Best metrics for imbalanced datasets?"
            """)
        
        # Quick tips
        with st.expander("🎯 Tips for Better Results"):
            st.markdown("""
            - **Be specific**: Ask about particular concepts or methods
            - **Follow up**: Build on previous questions for deeper insights
            - **Context matters**: Reference specific papers or topics
            - **Adjust settings**: Fine-tune retrieval parameters in the sidebar
            """)
        
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
