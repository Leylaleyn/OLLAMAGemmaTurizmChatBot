# main.py
# Ana uygulama dosyası

from ingest import prepare_vectorstore
from chain import create_qa_chain


def main():
    # Vektör veritabanını hazırla
    vectorstore = prepare_vectorstore()

    # Retriever oluştur
    retriever = vectorstore.as_retriever()

    # QA zincirini oluştur
    qa = create_qa_chain(retriever)

    # Kullanıcı arabirimi
    run_qa_interface(qa)


def run_qa_interface(qa):
    """Kullanıcı arayüzünü çalıştır"""
    print("\nOtel bilgi sistemine hoş geldiniz!")
    print("Çıkmak için 'q' yazın\n")

    while True:
        # Kullanıcıdan soru al
        question = input("\nSoru (Çıkmak için 'q' yazın): ")
        if question.lower() == 'q':
            print("Görüşmek üzere :)")
            break

        # Cevabı al
        result = qa.invoke({"query": question})

        # Cevabı yazdır
        print("\nCevap:", result["result"])


if __name__ == "__main__":
    main()