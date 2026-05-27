from langchain_openai import ChatOpenAI
import os, time
from dotenv import load_dotenv

load_dotenv()

def model_performance():
    models = ["gpt-5", "gpt-5-mini", "gpt-3.5-turbo"]

    system_prompt = "Provide 3 differences between machine learning and deep learning."
    for model_name in models:
        model = ChatOpenAI(
            model=model_name,
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY")
        )
        start_time = time.time()
        response = model.invoke(system_prompt)
        duration = (time.time() - start_time) * 1000

        print(f"\nModel {model_name}: ")
        print("Output: ", response.content)
        print("Output Length: ", len(response.content))
        print(f"Time taken: {duration}")

model_performance()


