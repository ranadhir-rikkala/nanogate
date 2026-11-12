import json
import os
import sys
from pathlib import Path

from openai import OpenAI

BASE_URL = os.environ.get("NANOGATE_URL", "http://127.0.0.1:8080/v1")
KEYS_FILE = Path(os.environ.get("NANOGATE_KEYS", "var/bootstrap_keys.json"))


def main() -> int:
    key = json.loads(KEYS_FILE.read_text())["app:acme/it"]["key"]
    client = OpenAI(base_url=BASE_URL, api_key=key)
    question = " ".join(sys.argv[1:]) or "How do I reset the VPN client?"

    raw = client.chat.completions.with_raw_response.create(
        model="nanogate-auto", messages=[{"role": "user", "content": question}]
    )
    print(raw.parse().choices[0].message.content)
    for header in ("route", "reason", "data-class", "cost-usd", "receipt-id"):
        print(f"  {header}: {raw.headers.get(f'x-nanogate-{header}')}")

    stream = client.chat.completions.create(
        model="nanogate-auto", messages=[{"role": "user", "content": question}], stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
