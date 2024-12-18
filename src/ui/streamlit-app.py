import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st

# Load environment variables
ENV_DIR = Path().absolute().parent / ".env"
DEV_ENV_FILE_PATH = ENV_DIR / "example.env"
load_dotenv(DEV_ENV_FILE_PATH, override=True)

# Initialize Azure OpenAI client
def initialize_client():
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_SLN"],
        api_key=os.environ["AZURE_OPENAI_API_KEY_SLN"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

# System message to define the AI's role
base_messages = [
    {"role": "system", "content": "You are an AI assistant. Help user with the query."},
]

# Chat with AI function
def chat_with_ai(client, query):
    messages = base_messages.copy()
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_SLN"],
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response_content = response.choices[0].message.content
        st.write("### Chat completion response:")
        st.write(response_content)

        embedding_response = client.embeddings.create(
            input=query,
            model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME_ADA_SLN"]
        )
        st.write("### Embedding response:")
        st.json(embedding_response.model_dump_json(indent=2))

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main Streamlit App
def main():
    st.title("Azure OpenAI CLI Chat Application")

    client = initialize_client()
    
    query = st.text_input("Enter your query (type 'exit' to quit):", "")
    
    if st.button("Submit"):
        if query.lower() == 'exit':
            st.stop()
        chat_with_ai(client, query)

if __name__ == "__main__":
    main()