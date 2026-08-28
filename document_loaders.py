from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
                                                    TextLoader,
                                                    PyPDFLoader
                                                )
load_dotenv()


def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello this is a sample text file.\nThis file is used to show the functionality of TextLoader")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} documents")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")
        for doc in documents:
            print("PAGE CONTENT:\n",doc.page_content)
    finally:
        os.remove(temp_file_path)


def pdf_loader(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i , doc in enumerate(documents):
        print(f"DOCUMENT {i+1}\n Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    pdf_loader("./documents/langchain_demo.pdf")

