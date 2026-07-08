import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Query to test
query = "How much did Microsoft pay to acquire GitHub?"
# query = "How do you plant tomatoes in a garden?"
print(f"Query: {query}\n")

# ──────────────────────────────────────────────────────────────────
# METHOD 1: Basic Similarity Search
# ──────────────────────────────────────────────────────────────────
print("=== METHOD 1: Similarity Search (k=3) ===")
retriever = db.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:\n")
for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")
print("-" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 2: Similarity with Score Threshold
# ──────────────────────────────────────────────────────────────────
# print("\n=== METHOD 2: Similarity with Score Threshold ===")
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3
#     }
# )
# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")
# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}:")
#     print(f"{doc.page_content}\n")
# print("-" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 3: Maximum Marginal Relevance (MMR)
# ──────────────────────────────────────────────────────────────────
# print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,
#         "fetch_k": 10,
#         "lambda_mult": 0.5
#     }
# )
# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents (λ=0.5):\n")
# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}:")
#     print(f"{doc.page_content}\n")
# print("=" * 60)
# print("Done! Try different queries or parameters to see the differences.")