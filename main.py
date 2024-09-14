from dotenv import load_dotenv
from graph.graph import app

_ = load_dotenv()

if __name__ == "__main__":
    print("Hello Advanced RAG")
    response = app.invoke({"question": "What is agent memory?"})
    print(response)
