from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from matplotlib import pyplot as plt

try:
    from insights.data_types import TransactionData
except ImportError:
    from data_types import TransactionData

CHAIN_BY_DEX = {
    "PancakeSwap": "bsc",
    "Aerodrome": "base",
}
KNOWN_CHAINS = set(CHAIN_BY_DEX.values())

CHAIN_TOKEN_DECIMALS = {
    "base": (18, 6),  # token0=WETH (18) and token1=USDC/USDT (6) on Base
    "bsc": (18, 18),  # Both token0/token1 use 18 decimals on BSC
}


@dataclass(frozen=True)
class ChainSpreadPoint:
    """Snapshot the spread between one chain and the other at a trade timestamp."""

    timestamp: int
    chain: str
    spread: Decimal


def chain_for_dex(dex_name: str) -> str:
    """Infer the chain from a DEX name, otherwise fall back to 'unknown'."""

    return CHAIN_BY_DEX.get(dex_name, "unknown")


def normalize_to_decimal(
    amount: str,
    *,
    chain: str | None = None,
    token_index: int = 0,
    decimals: int | None = None,
) -> Decimal:
    """Convert raw on-chain integer amounts into real units using chain/token decimals."""

    default_decimals = CHAIN_TOKEN_DECIMALS.get(chain, (18, 18))[token_index]
    scale = Decimal(10) ** (decimals if decimals is not None else default_decimals)
    return Decimal(amount) / scale


def group_transactions_by_chain(
    records: Iterable[TransactionData],
) -> dict[str, list[TransactionData]]:
    """Group transactions by the chain implied by their DEX for easier analysis."""

    grouped = {chain: [] for chain in KNOWN_CHAINS}
    grouped["unknown"] = []

    for record in records:
        chain = chain_for_dex(record.dex_name)
        grouped.setdefault(chain, []).append(record)

    return grouped


def compute_price_spread_points(
    records: Iterable[TransactionData],
) -> Sequence[ChainSpreadPoint]:
    """Read transactions chronologically, track latest prices, and emit spread snapshots."""

    sorted_records = sorted(records, key=lambda tx: tx.timestamp)
    last_price_by_chain: dict[str, Decimal | None] = {chain: None for chain in KNOWN_CHAINS}
    spread_points: list[ChainSpreadPoint] = []

    for record in sorted_records:
        chain = chain_for_dex(record.dex_name)
        if chain not in last_price_by_chain:
            continue

        price = Decimal(record.post_sqrt_price)
        last_price_by_chain[chain] = price
        other_chain = "base" if chain == "bsc" else "bsc"
        other_price = last_price_by_chain.get(other_chain)

        if other_price is None:
            continue

        spread_points.append(
            ChainSpreadPoint(timestamp=record.timestamp, chain=chain, spread=price - other_price)
        )

    return spread_points


def _timestamp_to_datetime(timestamp: int) -> datetime:
    return datetime.utcfromtimestamp(timestamp)


def plot_spread_trend(
    spread_points: Sequence[ChainSpreadPoint],
    *,
    file_path: Path | str | None = None,
    title: str = "Cross-chain spread",
) -> plt.Figure:
    """Plot the spread trajectories for each chain over time."""

    if not spread_points:
        raise ValueError("No spread points available for plotting")

    grouped: Mapping[str, list[ChainSpreadPoint]] = {}
    for point in spread_points:
        grouped.setdefault(point.chain, []).append(point)

    fig, ax = plt.subplots()
    for chain, points in grouped.items():
        sorted_points = sorted(points, key=lambda point: point.timestamp)
        timestamps = [_timestamp_to_datetime(point.timestamp) for point in sorted_points]
        spreads = [float(point.spread) for point in sorted_points]
        ax.plot(timestamps, spreads, label=f"{chain} spread")

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Spread (sqrt price)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    if file_path:
        fig.savefig(Path(file_path), bbox_inches="tight")

    return fig


def plot_chain_price_evolution(
    records: Iterable[TransactionData],
    *,
    file_path: Path | str | None = None,
    title: str = "On-chain price evolution",
) -> plt.Figure:
    """Plot each chain's `post_sqrt_price` over time."""

    grouped = group_transactions_by_chain(records)
    fig, ax = plt.subplots()
    for chain, txs in grouped.items():
        if not txs or chain == "unknown":
            continue

        sorted_txs = sorted(txs, key=lambda tx: tx.timestamp)
        timestamps = [_timestamp_to_datetime(tx.timestamp) for tx in sorted_txs]
        prices = [float(Decimal(tx.post_sqrt_price)) for tx in sorted_txs]
        ax.plot(timestamps, prices, label=f"{chain} price")

    ax.set_xlabel("UTC time")
    ax.set_ylabel("sqrt price")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()

    if file_path:
        fig.savefig(Path(file_path), bbox_inches="tight")

    return fig
