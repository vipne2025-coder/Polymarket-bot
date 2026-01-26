import os
import time
import requests

def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

BOT_TOKEN = env("TELEGRAM_TOKEN")
CHAT_ID = env("CHAT_ID")

def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=20)
    print("Telegram response:", r.status_code, r.text, flush=True)
    r.raise_for_status()

if __name__ == "main":
    print("Booting...", flush=True)
    send_telegram("Bot started")
    while True:
        print("Tick", flush=True)
        time.sleep(15)
