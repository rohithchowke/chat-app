# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# import argparse
# import sys

# # Load environment variables
# ENV_DIR = Path(__file__).absolute().parent.parent.parent / ".env"
# DEV_ENV_FILE_PATH = ENV_DIR / "dev.env"
# if not DEV_ENV_FILE_PATH.exists():
#     raise FileNotFoundError(f"{DEV_ENV_FILE_PATH} does not exist. Please check the path to your .env file.")
# load_dotenv(DEV_ENV_FILE_PATH, override=True)

# # Function to initialize the Azure OpenAI client
# def initialize_client():
#     try:
#         azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT_SLN"]
#         api_key = os.environ["AZURE_OPENAI_API_KEY_SLN"]
#         api_version = os.environ["AZURE_OPENAI_API_VERSION"]
#     except KeyError as e:
#         raise KeyError(f"Environment variable {e} not found. Please check your .env file.")

#     return AzureOpenAI(
#         azure_endpoint=azure_endpoint,
#         api_key=api_key,
#         api_version=api_version,
#     )

# base_messages = [
#     {"role": "system", "content": "You are an AI assistant. Help user with the query."},
# ]

# # Function to interact with the Azure OpenAI API
# def chat_with_ai(client, query):
#     messages = base_messages.copy()
#     messages.append({"role": "user", "content": query})

#     try:
#         response = client.chat.completions.create(
#             model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_SLN"],
#             messages=messages,
#             temperature=0.7,
#             max_tokens=1000,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0,
#         )
#         response_content = response.choices[0].message.content
#         print("Chat completion response:")
#         print(response_content)

#     except Exception as e:
#         print(f"An error occurred: {e}")

# def main():
#     parser = argparse.ArgumentParser(description="Azure OpenAI CLI Chat Application")
#     parser.add_argument('--query', type=str, help='User query for chat completion')
#     args, unknown = parser.parse_known_args()

#     client = initialize_client()

#     if args.query:
#         chat_with_ai(client, args.query)
#     else:
#         while True:
#             query = input("Enter your query (type 'exit' to quit): ")
#             if query.lower() == 'exit':
#                 break
#             chat_with_ai(client, query)

# if __name__ == "__main__":
#     main()

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import argparse

ENV_DIR = Path().absolute() / ".env"
DEV_ENV_FILE_PATH = ENV_DIR / "dev.env"
load_dotenv(DEV_ENV_FILE_PATH, override=True)


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
    prompt = f"User query: {query}"
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
        print("=== Input Prompt ===")
        print(prompt)
        print("=== Generated Response ===")
        print(response_content)

        # Add assistant's response to the message history
        messages.append({"role": "assistant", "content": response_content})

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI CLI Chat Application")
    parser.add_argument('--query', type=str, help='User query for chat completion')
    args = parser.parse_args()

    client = initialize_client()
    messages = base_messages.copy()

    if args.query:
        chat_with_ai(client, args.query, messages)  # Added 'messages' argument here
    else:
        while True:
            query = input("Enter your query (type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            chat_with_ai(client, query, messages)  # Added 'messages' argument here


if __name__ == "__main__":
    main()