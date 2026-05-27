import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


def get_embeddings_endpoint():
    """Get the Azure OpenAI endpoint, removing /openai/v1 suffix if present."""
    endpoint = os.getenv("AI_ENDPOINT", "")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint.replace("/openai/v1", "")
    elif endpoint.endswith("/openai/v1/"):
        endpoint = endpoint.replace("/openai/v1/", "")
    return endpoint


BOOKS = [
    {
        "title": "The AI Revolution",
        "summary": "How artificial intelligence is transforming society and business",
    },
    {
        "title": "JavaScript Mastery",
        "summary": "Complete guide to modern web development with JavaScript",
    },
    {
        "title": "Data Science Handbook",
        "summary": "Statistical analysis and machine learning for beginners",
    },
    {
        "title": "The Startup Playbook",
        "summary": "Building and scaling technology companies from scratch",
    },
    {
        "title": "Mystery at Midnight",
        "summary": "A detective solves crimes in Victorian London",
    },
    {
        "title": "Space Odyssey",
        "summary": "Humans explore distant galaxies and alien civilizations",
    },
    {
        "title": "Cooking Basics",
        "summary": "Essential techniques for home chefs and food enthusiasts",
    },
    {
        "title": "Python for Data",
        "summary": "Using Python for data analysis and visualization",
    },
]


def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    documents = [
        Document(
            page_content=book["summary"],
            metadata={"title": book["title"]},
        )
        for book in BOOKS
    ]

    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)

    queries = [
        "books about programming",
        "stories set in space",
        "learning about AI and technology",
        "cooking and recipes",
    ]

    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=3)

        for index, (doc, score) in enumerate(results):
            print(f"\n{index + 1}. {doc.metadata.get('title')}")
            print(f"Relevance: {score * 100:.1f}%")
            print(f"Summary: {doc.page_content}")


if __name__ == "__main__":
    main()
