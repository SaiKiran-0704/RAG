**RAG**

A collection of standalone scripts working through core Retrieval-Augmented Generation techniques from the ground up — ingestion, chunking strategies, retrieval methods, and generation — using LangChain, ChromaDB, and OpenRouter/OpenAI as swappable LLM backends.

**What's in here**

**Ingestion**

ingestion_pipeline.py — loads .txt docs (currently 5 sample company profiles: Google, Microsoft, Nvidia, SpaceX, Tesla), splits them, embeds with sentence-transformers/all-MiniLM-L6-v2, and persists to a local ChromaDB store. Idempotent — reuses the existing store if one's already built instead of re-embedding every run.

**Chunking strategies (compared side by side)**

semantic_chunking.py — splits on semantic similarity rather than fixed character counts
agentic_chunking.py — uses an LLM itself to decide chunk boundaries at natural topic breaks (demonstrated on a Tesla earnings excerpt), with provider auto-detection between OpenRouter and OpenAI based on which valid API key is present in .env
recursive_character_textsplitter_ — fixed-size recursive splitting baseline for comparison

**Retrieval**

retrieval_methods.py — similarity search, similarity with score threshold, and Maximum Marginal Relevance (MMR), run against the same query so the differences in results are directly comparable
multi_query_retrieval — generates multiple reworded versions of a query to widen recall before merging results
retrieval_pipeline.py — the retrieval step wired up as a reusable pipeline

**Generation**

answer.generation.py — basic retrieve-then-generate QA
history_aware_generation.py — multi-turn chat that rewrites follow-up questions into standalone queries using conversation history before retrieving, so "what about their revenue?" resolves correctly after a prior question about a specific company

**Running it**
  pip install -r requirements.txt  # not present yet — see note below
  cp .env.example .env  # add OPENROUTER_API_KEY or OPENAI_API_KEY
  python ingestion_pipeline.py       # build the vector store first
  python history_aware_generation.py # then chat against it

**Notes for next cleanup pass**
No requirements.txt is currently checked in
multi_query_retrieval and recursive_character_textsplitter_ are missing their .py extensions (the latter also has a trailing space in the filename)
The pre-built db/chroma_db/chroma.sqlite3 is committed to the repo — worth .gitignore-ing since it's regeneratable from ingestion_pipeline.py

**Stack**

LangChain · ChromaDB · HuggingFace sentence-transformers · OpenRouter / OpenAI · python-dotenv
