from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma

import os
from constant import chroma_settings

persist_directory = "db"

def main():
    documents = []
    for root, dirs, files in os.walk("doc"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Loading: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                loaded_documents = loader.load()
                print(f"Loaded {len(loaded_documents)} documents from {file}.")
                documents.extend(loaded_documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Total text chunks created: {len(texts)}")

    if texts:
        # Create embeddings
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print(f"Preparing to create embeddings for {len(texts)} text chunks.")

        # Create and persist the vector store
        db = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
        print("Documents indexed successfully.")

        # Persist the database
        print("Persisting the database...")
        db.persist()
        print("Database persisted.")

        # Optionally check if files are created
        print(f"Checking contents of {persist_directory} directory:")
        if os.path.exists(persist_directory):
            print(os.listdir(persist_directory))
        else:
            print("No files in the persist directory.")
    else:
        print("No text chunks available for embedding.")

if __name__ == "__main__":
    main()
