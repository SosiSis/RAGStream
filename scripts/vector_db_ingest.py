import os
import shutil
from typing import Optional, Union, List, TYPE_CHECKING
from scripts.paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scripts.utils import load_all_publications

# Global flag to track ChromaDB availability
CHROMADB_AVAILABLE = False
CHROMADB_COLLECTION = None

# Try to import ChromaDB, but don't fail if it's not available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
    print("ChromaDB successfully imported")
except Exception as e:
    print(f"ChromaDB not available: {e}")
    print("Running in fallback mode - vector search will be limited")
    CHROMADB_AVAILABLE = False


def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
    delete_existing: bool = False,
) -> Optional[Union["chromadb.Collection", bool]]:
    """
    Initialize a ChromaDB instance and persist it to disk.
    Falls back gracefully if ChromaDB is not available.
    """
    if not CHROMADB_AVAILABLE:
        print("ChromaDB not available - running in fallback mode")
        return False
    
    try:
        if os.path.exists(persist_directory) and delete_existing:
            shutil.rmtree(persist_directory)

        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        client = chromadb.PersistentClient(path=persist_directory)

        # Create or get a collection
        try:
            # Try to get existing collection first
            collection = client.get_collection(name=collection_name)
            print(f"Retrieved existing collection: {collection_name}")
        except Exception:
            # If collection doesn't exist, create it
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:batch_size": 10000,
                },  # Use cosine distance for semantic search
            )
            print(f"Created new collection: {collection_name}")

        print(f"ChromaDB initialized with persistent storage at: {persist_directory}")
        global CHROMADB_COLLECTION
        CHROMADB_COLLECTION = collection
        return collection
        
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        print("Running in fallback mode")
        return False


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
) -> Optional[Union["chromadb.Collection", bool]]:
    """
    Get a ChromaDB client instance.
    Falls back gracefully if ChromaDB is not available.
    """
    if not CHROMADB_AVAILABLE:
        return False
    
    try:
        return chromadb.PersistentClient(path=persist_directory).get_collection(
            name=collection_name
        )
    except Exception as e:
        print(f"Failed to get ChromaDB collection: {e}")
        return False


def chunk_publication(
    publication: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Chunk the publication into smaller documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(publication)


def embed_documents(documents: List[str]) -> List[List[float]]:
    """
    Embed documents using a model.
    """
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    embeddings = model.embed_documents(documents)
    return embeddings


def insert_publications(collection: Union["chromadb.Collection", bool], publications: List[str]):
    """
    Insert documents into a ChromaDB collection.
    Falls back gracefully if ChromaDB is not available.
    """
    if not CHROMADB_AVAILABLE or collection is False:
        print("ChromaDB not available - skipping document insertion")
        return
    
    try:
        next_id = collection.count()

        for publication in publications:
            chunked_publication = chunk_publication(publication)
            embeddings = embed_documents(chunked_publication)
            ids = list(range(next_id, next_id + len(chunked_publication)))
            ids = [f"document_{id}" for id in ids]
            collection.add(
                embeddings=embeddings,
                ids=ids,
                documents=chunked_publication,
            )
            next_id += len(chunked_publication)
    except Exception as e:
        print(f"Failed to insert publications: {e}")


def is_chromadb_available() -> bool:
    """Check if ChromaDB is available and working."""
    return CHROMADB_AVAILABLE and CHROMADB_COLLECTION is not None


def main():
    """Main function for testing ChromaDB functionality."""
    if not CHROMADB_AVAILABLE:
        print("ChromaDB not available - cannot run main function")
        return
        
    collection = initialize_db(
        persist_directory=VECTOR_DB_DIR,
        collection_name="publications",
        delete_existing=True,
    )
    
    if collection:
        publications = load_all_publications()
        insert_publications(collection, publications)
        print(f"Total documents in collection: {collection.count()}")
    else:
        print("Failed to initialize ChromaDB")


if __name__ == "__main__":
    main()