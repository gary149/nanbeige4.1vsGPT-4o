from pathlib import Path

from openai import OpenAI

from api_keys import resolve_openrouter_api_key


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=resolve_openrouter_api_key(repo_root),
    )

    completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "What is the meaning of life?",
            }
        ],
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
