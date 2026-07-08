from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


def is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    placeholder_markers = [
        "your_",
        "your-",
        "placeholder",
        "example",
        "replace_me",
        "changeme",
        "api_key_here",
    ]
    return any(marker in normalized for marker in placeholder_markers)


def is_malformed_api_key(value: str | None) -> bool:
    if not value:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if "=" in stripped or " " in stripped or "\n" in stripped:
        return True
    return False


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

provider = os.getenv("LLM_PROVIDER", "").strip().lower()
openrouter_key = os.getenv("OPENROUTER_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not provider:
    if openrouter_key and not is_placeholder(openrouter_key) and not is_malformed_api_key(openrouter_key):
        provider = "openrouter"
    elif openai_key and not is_placeholder(openai_key) and not is_malformed_api_key(openai_key):
        provider = "openai"
    else:
        raise ValueError(
            "No valid API key found. Replace the placeholder or malformed value in .env with your real OPENROUTER_API_KEY or OPENAI_API_KEY."
        )

if provider == "openrouter":
    if not openrouter_key or is_placeholder(openrouter_key) or is_malformed_api_key(openrouter_key):
        raise ValueError(
            "LLM_PROVIDER=openrouter was selected, but OPENROUTER_API_KEY is missing, still contains a placeholder, or is malformed. Replace it with your real OpenRouter key."
        )

    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "tesla-chunking-test",
        },
    )
elif provider == "openai":
    if not openai_key or is_placeholder(openai_key) or is_malformed_api_key(openai_key):
        raise ValueError(
            "LLM_PROVIDER=openai was selected, but OPENAI_API_KEY is missing, still contains a placeholder, or is malformed. Replace it with your real OpenAI key."
        )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_key,
    )
else:
    raise ValueError("Unsupported LLM_PROVIDER. Use 'openrouter' or 'openai'.")

# Tesla text to chunk
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.
Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.
Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Create the prompt
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.
Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks
Text:
{tesla_text}
Return the text with <<<SPLIT>>> markers where you want to split:
"""

print("🤖 Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

chunks = marked_text.split("<<<SPLIT>>>")

clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()