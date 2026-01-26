import os
import time
import requests

BOT_TOKEN = os.environ.get("BOT_TOKEN")

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    requests.post(url, data=data)

if __name__ == "main":
    print("Bot started")
    while True:
        time.sleep(10)

