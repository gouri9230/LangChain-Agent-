import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Product price")
    category: list[str] = Field(description="List of products")
    in_stock: bool = Field(description="product is in stock or not")
    rating: float = Field(description="rating of the product (1 to 5)")
    features: list[str] = Field(description="Product features")

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY")
)

structured_output = model.with_structured_output(Product)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Extract the product information based on the given text. If any information is not available then dont make assumptions and mention 'NA"),
    ("human", "{text}") 
])

chain = prompt_template | structured_output

product_description = ["MacBook Pro 16-inch with M3 chip, $2,499. Currently in stock. Users rate it 4.8/5. Features: Liquid Retina display, 18-hour battery, 1TB SSD", 
                       "Cozy wool sweater, blue color, medium size. $89, available now! Customers love it - 4.5 stars. Hand-washable, made in Ireland", 
                       "The Great Gatsby by F. Scott Fitzgerald. Classic novel, paperback edition for $12.99. In stock. Rated 4.9 stars. 180 pages, published 1925"]

for description in product_description:
    response = chain.invoke(description)
    print(f"\nProduct {response.name} Details: ")
    print(f"Category: {', '.join(response.category)}")
    print(f"Price: {response.price}")
    print(f"Features: {', '.join(response.features)}")
    print(f"Product Rating: {response.rating}")
    print(f"Is in Stock?: {response.in_stock}")
    