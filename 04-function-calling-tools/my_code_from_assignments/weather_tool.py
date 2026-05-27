import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal

# Load environment variables
load_dotenv()

class Weather(BaseModel):
    city:str = Field(description="The city name eg. 'Tokoyo', 'New York'")
    units: Optional[Literal["celsius", "fahrenheit"]] = Field(
        default="fahrenheit",
        description="Temperature unit (default: fahrenheit)",
    )

@tool(args_schema=Weather)
def get_weather(city:str, units:str="fahrenheit") -> str:
    """ Get the weather of a given city and the temperature of the city in celsius or fahrenheit with fahrenheit as the default unit.
    Use this when the user asks about weather, temperature, or conditions in a specific location."""
    weather_data = {
        "Tokyo": {"temp_f": 75, "temp_c": 24, "condition": "partly cloudy"},
        "Paris": {"temp_f": 64, "temp_c": 18, "condition": "sunny"},
        "London": {"temp_f": 59, "temp_c": 15, "condition": "rainy"},
        "New York": {"temp_f": 72, "temp_c": 22, "condition": "clear"},
        "Seattle": {"temp_f": 62, "temp_c": 17, "condition": "cloudy"},
        "Sydney": {"temp_f": 79, "temp_c": 26, "condition": "sunny"},
        "Mumbai": {"temp_f": 88, "temp_c": 31, "condition": "humid and hot"},
    }
    city_data = weather_data.get(city)

    if not city_data:
        available_cities = ", ".join(weather_data.keys())
        return f"Weather data not available for city {city}. Available cities: {available_cities}"
    
    units = units or "fahrenheit"
    temp = city_data["temp_f"] if units == "fahrenheit" else city_data["temp_c"]
    unit_symbol = "°C" if units == "celsius" else "°F"

    return f"The weather in {city} is {temp}{unit_symbol}, {city_data['condition']}"

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )

    model_with_tools = model.bind_tools([get_weather])

    queries = ["What's the weather in Tokyo?", 
               "Tell me the temperature in Paris in celsius", 
               "Is it raining in London?"]

    for query in queries:
        # Step 1: Get tool call from LLM
        response = model_with_tools.invoke([HumanMessage(content=query)])

        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            print(f"Tool: {tool_call['name']}")
            print(f"Args: {tool_call['args']}")
        else:
            print(" No tool call generated")

        # Step 2: Execute the tool
        tool_result = get_weather.invoke(tool_call["args"])
        print(f" Tool execution Result: {tool_result}")

        # Step 3: Send result back to LLM
        messages = [
            HumanMessage(content=query),
            AIMessage(
                content=str(response.content),
                tool_calls=response.tool_calls,
            ),
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ),
        ]
        final_response = model.invoke(messages)
        print(f" Final answer: {final_response.content}\n")

if __name__ == "__main__":
    main()
