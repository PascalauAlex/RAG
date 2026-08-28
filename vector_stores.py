from tempfile import tempdir

from huggingface_hub.utils import _cache_manager
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample docs
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def chroma_basics():
    #Create a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model, # Transforming the documents into embeddings
            persist_directory=tempdir
        )
        print(
            f"Vector store created {vectorstore._collection.count()} documents and persisted."
        )
        #Prepare the query
        query = "What is LangChain?"
        # Perform a similarity search, using top 2 results
        results = vectorstore.similarity_search(query,k=2)

        print(f"Top 2 results for query: `{query}`:")
        for i, doc in enumerate(results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )


def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tempdir,
        )

    query = "Explain vector stores."
    result_with_scores = vectorstore.similarity_search_with_score(query=query,k=3)
    print(f"Top 3 results with scores for query: `{query}`:")
    for i , (doc, score) in enumerate(result_with_scores):
        print(f"\n\nResult {i+1}: {doc.page_content}\n(Score: {score:.4f})\nSource: ({doc.metadata["source"]})")

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,  # Transforming the documents into embeddings
            persist_directory=tempdir
        )

        query = "What database are available?"
        results = vectorstore.similarity_search(query,k=5)
        for i, doc in enumerate(results):
            print(
                f"RESULT {i+1}\n{doc.page_content}\nSource: {doc.metadata['source']}"
            )
        filter_criteria = {"topic":"database"}
        filtered_results = vectorstore.similarity_search(query,k=5, filter=filter_criteria)
        print(f"\nResults with metadata filtering for query: {query}:")
        for i, doc in enumerate(filtered_results):
            print(
                f"RESULT {i+1}\n{doc.page_content}\nSource: {doc.metadata['source']}"
            )



if __name__ == "__main__":
    metadata_filtering()




























