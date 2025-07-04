#!/usr/bin/env python
"""Test the BTC price tool directly."""

from app.tools import get_btc_price

# Test the tool directly
print("Testing BTC price tool directly...")
result = get_btc_price.invoke({})
print(f"Result: {result}")

# Format the output
if result.get("status") == "success":
    print(f"\nBTC Price: ${result['price_usd']:,.2f}")
    print(f"24h Change: {result['24h_change_percent']:.2f}%")
    print(f"Market Cap: ${result['market_cap_usd']:,.0f}")
    print(f"24h Volume: ${result['24h_volume_usd']:,.0f}")
else:
    print(f"Error: {result.get('message', 'Unknown error')}")