import os
import time
import requests

TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    print("Missing TELEGRAM_TOKEN", flush=True)
    raise SystemExit(1)

API = f"https://api.telegram.org/bot{TOKEN}"

def call(method: str, params=None):
    r = requests.post(f"{API}/{method}", data=params or {}, timeout=30)
    print(method, r.status_code, r.text, flush=True)
    r.raise_for_status()
    return r.json()

def get_updates(offset=None):
    params = {"timeout": 30}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(f"{API}/getUpdates", params=params, timeout=40)
    print("getUpdates", r.status_code, r.text[:300], flush=True)
    r.raise_for_status()
    return r.json()

if __name__ == "main":
    # 1) выключаем webhook (иначе polling может молчать)
    call("deleteWebhook", {"drop_pending_updates": "true"})

    # 2) сообщение в логах, чтобы видеть что бот жив
    print("Bot is running and waiting for messages...", flush=True)

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

                # отвечаем ВСЕМ, чтобы не упереться в неверный CHAT_ID
                call("sendMessage", {"chat_id": chat_id, "text": f"Got: {text}\nYour chat_id: {chat_id}"})

        except Exception as e:
            print("Error:", repr(e), flush=True)
            time.sleep(5)
