import argparse
import sys
from chatapp.__about__ import __version__
from chatapp.simple_chat import initialize_client, chat_with_ai

def main():
    parser = argparse.ArgumentParser(description="AGTS CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--chat", action="store_true", help="Activate chat mode")

    args = parser.parse_args()
    
    if args.chat:
        client = initialize_client()
        messages = []  # Initialize an empty list to store the messages
        while True:
            user_input = input("Please provide your query (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            chat_with_ai(client, user_input, messages)  # Pass the messages list to chat_with_ai()

if __name__ == "__main__":
    main()