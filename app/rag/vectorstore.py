
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

# Local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def get_embedding_dimension() -> int:
    return EMBEDDING_DIMENSION


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    names = pc.list_indexes().names()

    if index_name in names:
        index_info = pc.describe_index(index_name)
        current_dimension = index_info.dimension

        if current_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Index '{index_name}' has dimension "
                f"{current_dimension}, but MiniLM requires "
                f"{EMBEDDING_DIMENSION}. "
                "Use a new Pinecone index name."
            )

        print(f"Existing Pinecone index found: {index_name}")

    else:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print(f"Creating Pinecone index: {index_name}")

    # Wait for index readiness, with a timeout
    deadline = time.monotonic() + 120

    while True:
        info = pc.describe_index(index_name)

        if info.status["ready"]:
            break

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index '{index_name}' "
                "was not ready within 120 seconds."
            )

        time.sleep(2)

    return pc.Index(index_name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    if not chunks:
        raise ValueError("No document chunks provided")

    store = get_vectorstore()
    return store.add_documents(chunks)
