import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY")
)

messages = [SystemMessage(content="You are an expert in Java programming.")]

while True:
    user = input("User: ").strip()
    if user == "quit" and "exit":
        break

    messages.append(HumanMessage(content=user))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("AI: ", response.content)