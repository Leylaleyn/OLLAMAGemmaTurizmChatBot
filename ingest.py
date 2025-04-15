# ingest.py
# Doküman yükleme ve vektörleştirme

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import config


def prepare_vectorstore():
    """
    Vektör veritabanını hazırla - Bu fonksiyon orijinal kodunuzdaki
    mantığı birebir takip eder
    """
    # FAISS index dizinini kontrol et
    faiss_path = config.FAISS_INDEX_PATH

    # Eğer daha önce oluşturulmuşsa onu yükle, yoksa oluştur ve kaydet
    if os.path.exists(faiss_path):
        print("FAISS index bulundu, yükleniyor...")
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("FAISS index bulunamadı, oluşturuluyor...")
        # Doküman yükleme - orijinal kodunuzdaki PyPDFLoader
        loader = PyPDFLoader(config.PDF_PATH)
        documents = loader.load()

        # Metin bölme
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents=documents)

        # Embeddings oluşturma
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        # Vektörleri oluşturma
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Vektörleri kaydetme
        vectorstore.save_local(faiss_path)

    return vectorstore