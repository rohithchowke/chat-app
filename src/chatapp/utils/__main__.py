import argparse
from .__about__ import __version__


def main():
    parser = argparse.ArgumentParser(description="AGTS CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--chat")

    args = parser.parse_args()
    # Add more commands and functionality here if needed
    if args.chat:
        while True:
            user_input = input("Please provide your query: ")
            print(f"Response:\n {user_input}")

if __name__ == "__main__":
    main()