from langchain_ollama import ChatOllama
import requests

OLLAMA_API_URL = "http://localhost:11434/api"
    
def get_installed_models():
    model_name_list = []
    response = requests.get(f"{OLLAMA_API_URL}/tags", timeout=10)
    if response.status_code == 200:
        models = response.json()
        for model in models['models']:
            model_name_list.append(model['name'])
            print(f"{model['name']} - Parameter Size: {model['details']['parameter_size']}")
    else:
        print("Failed to retrieve installed models.")
    return model_name_list

def create_model(model_name: str):
    ollama_client = ChatOllama(model=model_name, validate_model_on_init = True, temperature=0.7, reasoning=True)
    return ollama_client

if __name__ == "__main__":
    installed_models = get_installed_models()
    create_model(installed_models[0])