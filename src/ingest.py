import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


def ingest_pdf():
    for var in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(var):
            raise SystemExit(f"Variável de ambiente {var} não definida. Confira seu arquivo .env")

    # 1. Carrega o PDF
    docs = PyPDFLoader(PDF_PATH).load()

    # 2. Divide em chunks de 1000 caracteres com overlap de 150
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise SystemExit("Nenhum conteúdo extraído do PDF.")

    enriched = [
        doc.__class__(
            page_content=doc.page_content,
            metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)},
        )
        for doc in chunks
    ]

    # 3. Gera embeddings e armazena no PostgreSQL + pgVector
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    ids = [f"doc-{i}" for i in range(len(enriched))]
    store.add_documents(documents=enriched, ids=ids)

    print(f"Ingestão concluída: {len(enriched)} chunks armazenados na coleção "
          f"'{os.getenv('PG_VECTOR_COLLECTION_NAME')}'.")


if __name__ == "__main__":
    ingest_pdf()
