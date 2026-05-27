from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )
    messages = [{"name": "Pirate", "prompt": SystemMessage(content="You are a pirate AI assistant. Answer all questions in pirate speak with 'Arrr!' and nautical terms.")},
        {"name": "Analyst", "prompt": SystemMessage(content="You are a professional business analyst. Give precise, data-driven answers.")},
        {"name": "Teacher", "prompt": SystemMessage(content="You are a friendly teacher explaining concepts to 8-year-old children.")}
    ]
    question = HumanMessage(content="What is a Blockchain?")

    for message in messages:
        msg = [message["prompt"], question]
        response = model.invoke(msg)
        print(f"\n{message['name']} assistant : ", response.content)

if __name__ == "__main__":
    print("Testing different personas of LLLM.\n")
    main()
