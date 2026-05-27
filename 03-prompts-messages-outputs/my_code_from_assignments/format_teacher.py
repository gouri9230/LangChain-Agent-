import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
    temperature=0.0
)

teaching_messages = [
    {"input": "Premium wireless headphones with noise cancellation, $199", "output": {"name": "wireless headphones","price": "$199.00","category": "Premium","highlight": "noise cancellation"}},
    {"input": "Organic cotton t-shirt in blue, comfortable fit, $29.99", "output": {"name": "t-shirts","price": "$29.99","category": "Organic cotton clothes","highlight": "blue color, comfortable fit"}},
    {"input": "Gaming laptop with RTX 4070, 32GB RAM, $1,499", "output": {"name": "Gaming Laptop","price": "$1,499","category": "Electronics","highlight": "RTX 4070, 32GB RAM"}}
]

example_template = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_examples = FewShotChatMessagePromptTemplate(
    example_prompt=example_template,
    examples=teaching_messages
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert product descriptions into a specific JSON format:"),
    few_shot_examples,
    ("human", "{input}")
])

chain = final_prompt | model
response = chain.invoke({"input": "Vegan chips with less oil, baked not fried for $1.99"})
print(response.content)