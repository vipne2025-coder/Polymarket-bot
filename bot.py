import os
import time
import requests

GRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/matic-markets"

def send_telegram(text: str) -> None:
    token = os.environ.get("8493651620:AAFM629s3a77HKOU3Za1hKk6W_1l7RKt5Vw")
    chat_id = os.environ.get("6195588874")

    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID in Railway Variables")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=20)
    r.raise_for_status()

def get_trades():
    query = """
    {
      trades(first: 30, orderBy: timestamp, orderDirection: desc) {
        id
        trader
        collateralAmountUSD
        price
        market { question }
        timestamp
      }
    }
    """
    r = requests.post(GRAPH_URL, json={"query": query}, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data["data"]["trades"]

def main():
    # Тест при запуске (без emoji)
    send_telegram("Bot started on Railway")

    seen = set()

    while True:
        try:
            trades = get_trades()

            for t in trades:
                trade_id = t.get("id")
                if not trade_id or trade_id in seen:
                    continue

                amount = float(t.get("collateralAmountUSD") or 0)
                price = float(t.get("price") or 0)
                question = (t.get("market") or {}).get("question", "Unknown market")
                trader = t.get("trader", "Unknown")

                # Настройки фильтра (можешь менять)
                if amount >= 5000 and price > 0 and price <= 0.15:
                    msg = (
                        "Whale buy detected\n\n"
                        f"Market: {question}\n"
                        f"Amount: ${amount:,.0f}\n"
                        f"Price: {price}\n"
                        f"Wallet: {trader}"
                    )
                    send_telegram(msg)

                seen.add(trade_id)

            # чтобы seen не рос бесконечно
            if len(seen) > 2000:
                seen = set(list(seen)[-500:])

            time.sleep(60)

        except Exception as e:
            # Пишем в логи, но не спамим телеграм
            print("Error:", repr(e))
            time.sleep(60)

if __name__ == "main":
    main()

