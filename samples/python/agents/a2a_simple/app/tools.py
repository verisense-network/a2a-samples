import requests
from langchain_core.tools import tool


@tool
def get_btc_price() -> dict:
    """Get the current Bitcoin (BTC) price in USD.
    
    Returns:
        dict: Contains the current BTC price and additional market data
    """
    try:
        # Using CoinGecko API (free, no auth required)
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true"
            }
        )
        response.raise_for_status()
        
        data = response.json()
        bitcoin_data = data.get("bitcoin", {})
        
        return {
            "price_usd": bitcoin_data.get("usd", 0),
            "24h_change_percent": bitcoin_data.get("usd_24h_change", 0),
            "market_cap_usd": bitcoin_data.get("usd_market_cap", 0),
            "24h_volume_usd": bitcoin_data.get("usd_24h_vol", 0),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "message": "Failed to fetch BTC price"
        }