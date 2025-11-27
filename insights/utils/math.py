from __future__ import annotations

from decimal import Decimal

SQRT_PRICE_SCALE = Decimal(2) ** 96
WETH_DECIMALS = 18
USDC_DECIMALS = 6


def sqrt_price_x96_to_price(sqrt_price: Decimal) -> Decimal:
    """Convert Uniswap sqrtPriceX96 to Token1 per Token0 price, adjusting for decimals."""

    base = (sqrt_price / SQRT_PRICE_SCALE) ** 2
    decimal_adjust = Decimal(10) ** (WETH_DECIMALS - USDC_DECIMALS)
    return base * decimal_adjust


def price_spread_pct(price_a: Decimal, price_b: Decimal) -> Decimal:
    """Immutable factor describing relative spread."""

    if price_a <= 0 or price_b <= 0:
        return Decimal(0)

    mean_price = (price_a + price_b) / 2
    return abs(price_a - price_b) / mean_price


def is_arb_opportunity(price_base: Decimal, price_bsc: Decimal, threshold: Decimal = Decimal("0.0005")) -> bool:
    """Threshold filter for cross-chain arbitrage windows."""

    pct = price_spread_pct(price_base, price_bsc)
    return pct > threshold
