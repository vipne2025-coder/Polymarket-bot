import requests
import time

import os

BOT_TOKEN = os.environ["8493651620:AAEMBlQkKHj4rjYBJEjBms_NlI8sbLPLNXk"]
CHAT_ID = os.environ["6195588874"]

GRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/matic-markets"

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": text})

def get_trades():
    query = """
    {
      trades(first: 20, orderBy: timestamp, orderDirection: desc) {
        trader
        collateralAmountUSD
        price
        market {
          question
        }
      }
    }
    """
    r = requests.post(GRAPH_URL, json={'query': query})
    return r.json()['data']['trades']

seen = set()

while True:
    try:
        trades = get_trades()
        for t in trades:
            trade_id = t['trader'] + t['market']['question']
            if trade_id in seen:
                continue

            amount = float(t['collateralAmountUSD'])
            price = float(t['price'])

            if amount > 5000 and price < 0.15:
                msg = f"""
🐋 КРУПНЫЙ ВХОД ЗА КОПЕЙКИ

Рынок: {t['market']['question']}
Сумма: ${amount:,.0f}
Цена: {price}
Кошелёк: {t['trader']}
"""
                send_telegram( Бот запустился и на связи.)
                seen.add(trade_id)

        time.sleep(60)

    except Exception as e:
        print("Error:", e)

        time.sleep(60)


