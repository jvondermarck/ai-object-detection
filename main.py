import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print(os.getenv("PICSELLIA_API_TOKEN"))


if __name__ == "__main__":
    main()
