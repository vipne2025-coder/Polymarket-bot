import os
import time
import requests

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not TOKEN or not CHAT_ID:
    print("Missing TELEGRAM_TOKEN or CHAT_ID")
    exit(1)

def send_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": text
    }
    r = requests.post(url, data=data)
    print("Telegram response:", r.text)

# сообщение при старте
send_message("Bot started successfully")

# ВАЖНО: держим процесс живым
while True:
    time.sleep(60)
