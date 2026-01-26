import os
import time
import requests

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not TOKEN or not CHAT_ID:
    print("Missing TELEGRAM_TOKEN or CHAT_ID", flush=True)
    raise SystemExit(1)

API = f"https://api.telegram.org/bot{TOKEN}"

def send_message(chat_id: str, text: str) -> None:
    r = requests.post(f"{API}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=20)
    print("sendMessage:", r.status_code, r.text, flush=True)
    r.raise_for_status()

def get_updates(offset: int | None) -> dict:
    params = {"timeout": 30}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(f"{API}/getUpdates", params=params, timeout=40)
    r.raise_for_status()
    return r.json()

if __name__ == "main":
    send_message(CHAT_ID, "Bot is running. Send me any text.")

    offset = None
    while True:
        try:
            data = get_updates(offset)
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1

                msg = upd.get("message") or upd.get("channel_post")
                if not msg:
                    continue

                chat_id = str(msg["chat"]["id"])
                text = msg.get("text", "")

                # отвечаем только в твой чат
                if chat_id != str(CHAT_ID):
                    continue

                if text.lower() in ("/start", "start"):
                    send_message(chat_id, "Ok. I can send alerts here.")
                elif text.lower() == "ping":
                    send_message(chat_id, "pong")
                else:
                    send_message(chat_id, f"Got: {text}")

        except Exception as e:
            print("Error:", repr(e), flush=True)
            time.sleep(5)
