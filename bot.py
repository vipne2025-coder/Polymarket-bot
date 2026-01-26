import os
import time
import requests

TOKEN = os.environ.get("TELEGRAM_TOKEN")

if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set")

BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

def get_updates(offset=None):
    url = f"{BASE_URL}/getUpdates"
    params = {"timeout": 60}
    if offset:
        params["offset"] = offset
    response = requests.get(url, params=params, timeout=70)
    return response.json()

def send_message(chat_id, text):
    url = f"{BASE_URL}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    requests.post(url, data=data, timeout=10)

def main():
    print("BOT STARTED", flush=True)
    offset = None

    while True:
        updates = get_updates(offset)

        if updates.get("ok"):
            for update in updates["result"]:
                offset = update["update_id"] + 1

                if "message" in update:
                    chat_id = update["message"]["chat"]["id"]
                    text = update["message"].get("text", "")

                    print(f"Message from {chat_id}: {text}", flush=True)
                    send_message(chat_id, f"Ты написал: {text}")

        time.sleep(1)

if __name__ == "main":
    main()
