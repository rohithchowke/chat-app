import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import argparse

# Load environment variables
ENV_DIR = Path().absolute() / ".env"
DEV_ENV_FILE_PATH = ENV_DIR / "dev.env"
x = load_dotenv(DEV_ENV_FILE_PATH, override=True)
print(x)

def initialize_client():
    try:
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT_SLN"]
        api_key = os.environ["AZURE_OPENAI_API_KEY_SLN"]
        api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    except KeyError as e:
        raise KeyError(f"Environment variable {e} not found. Please check your .env file.")

    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

base_messages = [
    {"role": "system", "content": "You are an AI assistant. Help user with the query."},
]

def chat_with_ai(client, query, messages):
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
        st.subheader("Generated Response")
        st.markdown(f"> {response_content}")

        # Add assistant's response to the message history
        messages.append({"role": "assistant", "content": response_content})

    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    # Set up the argument parser (still useful for running from command line)
    parser = argparse.ArgumentParser(description="Azure OpenAI CLI Chat Application")
    parser.add_argument('--query', type=str, help='User query for chat completion')
    args, unknown = parser.parse_known_args()

    client = initialize_client()

    # Use session state to store messages
    if 'messages' not in st.session_state:
        st.session_state.messages = base_messages.copy()

    st.title("Chat with AI")
    st.markdown("Enter your query and the AI will respond.")

    if args.query:
        chat_with_ai(client, args.query, st.session_state.messages)
    else:
        query = st.text_input("User query:", key="text_input_key")
        if st.button("Submit", key="submit_button_key"):
            if query.lower() == 'exit':
                st.stop()
            else:
                chat_with_ai(client, query, st.session_state.messages)
                # Keep only the last 5 exchanges (10 messages: 5 user + 5 assistant)
                st.session_state.messages = st.session_state.messages[-10:]

if __name__ == "__main__":
    main()