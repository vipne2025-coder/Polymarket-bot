import os
import time
import requests

def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID")

    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN")
    if not chat_id:
        raise RuntimeError("Missing CHAT_ID")

    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    print("Telegram response:", r.status_code, r.text, flush=True)
    r.raise_for_status()

def main() -> None:
    print("Booting...", flush=True)
    send_telegram("Bot started")
    while True:
        print("Tick", flush=True)
        time.sleep(15)

if __name__== "main":
    main()

