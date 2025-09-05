# RAGStream 🔍✨

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragstream-utrzbfzjw5vp5wtkgdc3et.streamlit.app/)

> **An intelligent RAG (Retrieval-Augmented Generation) system powered by Streamlit that answers questions about AI/ML research papers and tutorials.**

## 🎯 What is RAGStream?

RAGStream is a conversational AI assistant that specializes in answering questions about artificial intelligence and machine learning research. It uses advanced RAG (Retrieval-Augmented Generation) technology to provide accurate, contextual answers based on a curated collection of AI/ML publications.

## 🤖 What Kind of Questions Can RAGStream Answer?

Our RAG system can intelligently answer questions across multiple AI/ML domains:

### 🎨 **Computer Vision & Multi-Modal Learning**
- **CLIP (Contrastive Language-Image Pretraining)**
  - "What is CLIP and how does it work?"
  - "How does CLIP align image and text representations?"
  - "Explain the contrastive learning approach in CLIP"
  - "What are the practical applications of CLIP?"
  - "How does the Vision Transformer work in CLIP's image encoder?"

### 🔄 **Generative Models & Autoencoders**
- **Variational Autoencoders (VAEs)**
  - "What are variational autoencoders and how do they differ from regular autoencoders?"
  - "How can VAEs be used for data compression?"
  - "Explain how VAEs generate new data"
  - "How do VAEs handle anomaly detection?"
  - "What is the reparameterization trick in VAEs?"
  - "How do VAEs work for missing data imputation?"

### 📈 **Time Series Analysis**
- **Distance Profile & Pattern Recognition**
  - "What is distance profile in time series analysis?"
  - "How does MASS algorithm work for similarity search?"
  - "Explain time-step classification in time series"
  - "What are the applications of distance profile?"

### ⚖️ **Machine Learning Fundamentals**
- **Class Imbalance & Binary Classification**
  - "How do you handle class imbalance in binary classification?"
  - "What is SMOTE and how does it work?"
  - "Explain decision threshold calibration"
  - "How do class weights help with imbalanced datasets?"
  - "What metrics are best for evaluating imbalanced datasets?"

## 📊 Dataset Information

Our knowledge base contains high-quality AI/ML research papers and tutorials:

### 📋 **Current Publications**

| Document | Topic | Key Focus Areas |
|----------|-------|-----------------|
| **CLIP Implementation Guide** | Multi-Modal Learning | Contrastive learning, Vision Transformers, Image-text alignment |
| **VAE Comprehensive Study** | Generative Models | Data compression, generation, denoising, anomaly detection |
| **Distance Profile Analysis** | Time Series | Pattern recognition, similarity search, time-step classification |
| **Class Imbalance Study** | Classification | SMOTE, class weights, threshold calibration, evaluation metrics |

### 🎯 **Specialization Areas**
- **Deep Learning Architectures**: Vision Transformers, CNNs, Autoencoders
- **Generative AI**: VAEs, data generation, synthetic data creation
- **Computer Vision**: Image processing, multi-modal learning, CLIP
- **Time Series**: Pattern recognition, similarity search, classification
- **Machine Learning**: Classification, imbalanced learning, evaluation metrics

## 🚀 Features

- **💬 Interactive Chat Interface**: Natural conversation flow with context awareness
- **🔍 Intelligent Retrieval**: Advanced vector search using ChromaDB and HuggingFace embeddings
- **🧠 Context-Aware Responses**: Maintains conversation context for follow-up questions
- **📚 Comprehensive Knowledge Base**: Curated collection of AI/ML research papers
- **⚡ Fast Response Time**: Optimized vector database for quick information retrieval
- **🎨 User-Friendly UI**: Clean, intuitive Streamlit interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama models)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers
- **Framework**: LangChain
- **Language**: Python 3.9+

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.9 or higher
- Groq API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SosiSis/RAGStream.git
   cd RAGStream
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo 'GROQ_API_KEY="your_groq_api_key_here"' > .env
   echo 'VECTOR_DB_DIR="./outputs/vector_db"' >> .env
   ```

4. **Initialize the vector database**
   ```bash
   python scripts/vector_db_ingest.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🗂️ Project Structure

```
RAGStream/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables
├── data/                      # Knowledge base documents
│   ├── 57Nhu0gMyonV.md       # CLIP implementation guide
│   ├── yzN0OCQT7hUS.md       # VAE comprehensive study
│   ├── ljGAbBceZbpv.md       # Distance profile analysis
│   ├── tum5RnE4A5W8.md       # Class imbalance study
│   └── yzN0OCQT7hUS-sample-questions.yaml  # Sample questions
├── scripts/                   # Core functionality
│   ├── vector_db_ingest.py   # Document ingestion
│   ├── vector_db_rag.py      # RAG implementation
│   ├── prompt_builder.py     # Prompt engineering
│   ├── utils.py              # Utility functions
│   ├── paths.py              # Path configurations
│   └── config/               # Configuration files
│       ├── config.yaml       # App configuration
│       └── prompt_config.yaml # Prompt templates
└── outputs/                   # Generated outputs
    ├── vector_db/            # ChromaDB storage
    └── rag_assistant.log     # Application logs
```

## 🎨 Example Conversations

### Multi-turn Conversation Example:
```
User: "What are variational autoencoders?"
RAG: "Variational Autoencoders (VAEs) are powerful generative models that use a probabilistic approach to encode data into a distribution of latent variables..."

User: "How do they differ from regular autoencoders?"
RAG: "Unlike regular autoencoders that encode data into fixed points, VAEs encode data into probability distributions in the latent space..."

User: "Can you explain their applications in data compression?"
RAG: "VAEs can achieve significant data compression. For example, with MNIST images, they can compress 28x28 pixel images into much smaller latent representations..."
```

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Add new research papers to the knowledge base
- Improve the RAG implementation
- Enhance the user interface
- Fix bugs or optimize performance

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the authors of the research papers included in our knowledge base
- Streamlit community for the amazing framework
- HuggingFace for the embedding models
- ChromaDB for the vector database solution

---

**Ready to explore AI/ML research with intelligent conversations?** 
[![Try RAGStream Now](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragstream.streamlit.app/)
