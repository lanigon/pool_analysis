from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from insights.utils.math import sqrt_price_x96_to_price
    from insights.utils.tx_group import normalize_to_decimal
except ImportError:
    from utils.math import sqrt_price_x96_to_price
    from utils.tx_group import normalize_to_decimal

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VIEW_DIR = Path(__file__).resolve().parent / "view"
DB_PATH = DATA_DIR / "transactions.db"
WINDOW = "12s"

CHAIN_NATIVE_PRICES = {"base": Decimal("1900"), "bsc": Decimal("310")}


def _ensure_view_dir() -> None:
    VIEW_DIR.mkdir(parents=True, exist_ok=True)


def _save_plot(fig: plt.Figure, name: str) -> Path:
    _ensure_view_dir()
    path = VIEW_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


@dataclass
class CompetitorSummary:
    sender: str
    total_volume: Decimal
    total_txs: int
    revenue: Decimal
    cost: Decimal
    net_profit: Decimal


CHAIN_MAP = {"PancakeSwap": "bsc", "Aerodrome": "base"}


def load_transactions(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    conn.execute(f"ATTACH DATABASE '{DATA_DIR / 'pools.db'}' AS pools_db")
    df = pd.read_sql_query(
        """
        SELECT t.pool_address,
               t.sender_address AS sender_address,
               t.tx_hash,
               t.amount0,
        t.amount1,
               t.post_sqrt_price,
               t.pre_tick,
               t.current_tick,
               t.pre_liquidity,
               t.gas_price,
        p.fee AS pool_fee,
               t.timestamp,
               t.block_number,
               p.dex_name
        FROM transactions AS t
        LEFT JOIN pools_db.pools AS p
        ON t.pool_address = p.pool_address
        """,
        conn,
    )
    conn.execute("DETACH DATABASE pools_db")
    df["chain"] = df["dex_name"].map(CHAIN_MAP).fillna("unknown")
    df["timestamp"] = df["timestamp"].astype(int)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("datetime")

    def _to_decimal(value: str) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal(0)

    for col in ["pre_liquidity", "gas_price", "pool_fee"]:
        df[col] = df[col].map(_to_decimal)

    df["amount1_decimal"] = df.apply(
        lambda row: normalize_to_decimal(
            row["amount1"], chain=row["chain"], token_index=1
        ),
        axis=1,
    )
    df["amount0_decimal"] = df.apply(
        lambda row: normalize_to_decimal(
            row["amount0"], chain=row["chain"], token_index=0
        ),
        axis=1,
    )
    df["sqrt_price"] = df["post_sqrt_price"].apply(lambda x: Decimal(str(x)))
    df["price"] = df["sqrt_price"].apply(sqrt_price_x96_to_price)

    # Case A: price is too small (WETH -> USDC), roughly 1e-9, so scale by 1e12 to ~2000
    mask_too_small = df["price"] < Decimal("0.1")
    df.loc[mask_too_small, "price"] = df.loc[mask_too_small, "price"] * Decimal("1e12")

    mask_too_large = df["price"] > Decimal("1000000")
    df.loc[mask_too_large, "price"] = df.loc[mask_too_large, "price"] / Decimal("1e12")

    df["volume_usd"] = df["amount1_decimal"].abs().astype(float)
    df["fee_rate"] = df["pool_fee"] / Decimal("10000")
    df["log_index"] = df.get("log_index", 0)
    return df


def identify_competitors(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("sender_address")
        .agg(
            total_volume=pd.NamedAgg(column="volume_usd", aggfunc="sum"),
            total_txs=pd.NamedAgg(column="tx_hash", aggfunc="count"),
            arb_txs=pd.NamedAgg(column="is_arb_trade", aggfunc="sum"),
            revenue=pd.NamedAgg(column="revenue", aggfunc="sum"),
            cost=pd.NamedAgg(column="gas_fee_usd", aggfunc="sum"),
            net_profit=pd.NamedAgg(column="net_profit", aggfunc="sum"),
        )
        .reset_index()
    )
    stats["arb_share"] = stats["arb_txs"] / stats["total_txs"].replace(0, 1)
    competitors = stats[
        (stats["total_txs"] > 10) & (stats["arb_share"] > 0.5)
    ].copy()
    competitors["total_volume"] = competitors["total_volume"]
    competitors["net_profit"] = competitors["net_profit"]
    return competitors


def print_leaderboard(df: pd.DataFrame) -> None:
    print("Top Competitor Leaderboard")
    if df.empty:
        print("No competitors detected.")
        return
    table = df.sort_values("total_txs", ascending=False).head(10)
    table_to_print = table[
        [
            "sender_address",
            "total_volume",
            "total_txs",
            "revenue",
            "net_profit",
        ]
    ].copy()
    table_to_print.caption = "USD units"
    print(table_to_print.to_string(index=False, float_format="{:,.2f}".format))


def describe_top_competitor(df: pd.DataFrame, trades: pd.DataFrame) -> None:
    if df.empty:
        return
    top_sender = df.sort_values("total_txs", ascending=False).iloc[0]["sender_address"]
    subset = trades[trades["sender_address"] == top_sender]
    print(f"\nProfitability Stats for top competitor {top_sender}:")
    numeric_cols = subset[["gas_fee_usd", "volume_usd", "net_profit"]]
    print(numeric_cols.describe())


def correlation_analysis(trades: pd.DataFrame) -> None:
    print("\nCorrelation analysis (volume vs gas price, net profit vs gas price):")
    volume_corr = trades["volume_usd"].corr(trades["gas_fee_usd"])
    profit_corr = trades["net_profit"].corr(trades["gas_fee_usd"])
    print(f"- Volume vs Gas Fee USD: {volume_corr:.4f}")
    print(f"- Net Profit vs Gas Fee USD: {profit_corr:.4f}")


def plot_daily_volume(df: pd.DataFrame) -> None:
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    pivot = (
        df.groupby(["date", "is_arb_trade"])["volume_usd"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot = pivot.rename(columns={False: "Other Trades", True: "Arb Trades"})
    pivot.plot(
        kind="bar",
        stacked=True,
        color={"Arb Trades": "darkorange", "Other Trades": "steelblue"},
        figsize=(12, 6),
    )
    fig = plt.gcf()
    plt.title("Daily Volume Breakdown (USD)")
    plt.ylabel("Volume (USD)")
    plt.tight_layout()
    plt.legend()
    _save_plot(fig, "daily_volume_breakdown")


def plot_spread_duration(persistence: pd.DataFrame) -> None:
    if persistence.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    point_count = len(persistence)
    print(f"\nSpread persistence scatter plot points: {point_count}")
    sns.scatterplot(
        data=persistence,
        x="duration",
        y="mean_abs_spread",
        size="mean_abs_spread",
        hue="duration",
        palette="viridis",
        sizes=(40, 120),
        ax=ax,
    )
    ax.set_title("Spread persistence vs duration")
    ax.set_xlabel("Duration (seconds until spread ≤ threshold)")
    ax.set_ylabel("Mean absolute spread")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    _save_plot(fig, "spread_persistence_duration")


def same_timestamp_gas_priority(df: pd.DataFrame) -> float:
    df_copy = df.reset_index().copy()
    df_copy["gas_price_float"] = (
        pd.to_numeric(df_copy["gas_price"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    df_copy.sort_values(["pool_address", "timestamp", "block_number"], inplace=True)
    groups = df_copy.groupby(["pool_address", "timestamp"])
    ratio_count = 0
    ratio_total = 0
    for _, group in groups:
        if len(group) < 2:
            continue
        ratio_total += 1
        first = group.iloc[0]
        if np.isclose(first["gas_price_float"], group["gas_price_float"].max()):
            ratio_count += 1
    return ratio_count / ratio_total if ratio_total else 0.0


def attach_chain_reference(
    source: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Match each source trade with the most recent reference-chain price."""
    ref_sorted = reference[["timestamp", "price"]].sort_values("timestamp")
    src_sorted = source.sort_values(
        ["timestamp", "block_number", "log_index"], ascending=True
    )
    merged = pd.merge_asof(
        src_sorted,
        ref_sorted.rename(columns={"price": "ref_price"}),
        on="timestamp",
        direction="backward",
    )
    return merged


def analyze_pool_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how the two pools drift apart before flagging arbitrage trades."""
    bsc_df = df[df["chain"] == "bsc"]
    base_df = df[df["chain"] == "base"]
    if bsc_df.empty or base_df.empty:
        print("\nInsufficient data to compare BSC and Base pool prices.")
        return pd.DataFrame()

    bsc_ref = attach_chain_reference(bsc_df, base_df)
    base_ref = attach_chain_reference(base_df, bsc_df)
    spread_df = (
        pd.concat([bsc_ref, base_ref], ignore_index=True)
        .sort_values(["timestamp", "block_number", "log_index"], ascending=True)
        .reset_index(drop=True)
    )
    spread_df["ref_price"] = spread_df["ref_price"].ffill()
    spread_df["price_float"] = spread_df["price"].astype(float)
    spread_df["ref_float"] = spread_df["ref_price"].astype(float)
    spread_df["spread_pct"] = (
        (spread_df["price_float"] - spread_df["ref_float"])
        / spread_df["ref_float"].replace(0, np.nan)
    )
    summary = (
        spread_df.groupby("chain")
        .agg(
            trades=("tx_hash", "count"),
            mean_spread=("spread_pct", "mean"),
            median_spread=("spread_pct", "median"),
            positive_spreads=("spread_pct", lambda x: (x > 0).sum()),
            negative_spreads=("spread_pct", lambda x: (x < 0).sum()),
        )
        .reset_index()
    )
    summary["positive_ratio"] = summary["positive_spreads"] / summary["trades"].replace(
        0, 1
    )
    print("\nPool price spread overview:")
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    high_spread = spread_df[spread_df["spread_pct"].abs() > 0.0005]
    if not high_spread.empty:
        chain_counts = high_spread.groupby("chain")["tx_hash"].count().rename("large_spread_trades")
        by_chain = chain_counts.reset_index()
        print(f"\nHigh-spread trades (>0.05%): {len(high_spread)} total")
        print(
            by_chain.to_string(
                index=False,
                float_format="{:,.0f}".format,
            )
        )
        print("\nSample high-spread trades:")
        print(
            high_spread[
                ["timestamp", "chain", "tx_hash", "price", "ref_price", "spread_pct"]
            ]
            .head(5)
            .to_string(index=False, float_format="{:,.4f}".format)
        )
    return summary


def analyze_arbitrage_trades(full_df: pd.DataFrame) -> pd.DataFrame:
    """Identify cross-chain arbitrage trades event-by-event and estimate revenue."""

    sorted_df = full_df.sort_values(
        ["timestamp", "block_number", "log_index"], ascending=True
    )
    bsc_df = sorted_df[sorted_df["chain"] == "bsc"]
    base_df = sorted_df[sorted_df["chain"] == "base"]

    bsc_ref = attach_chain_reference(bsc_df, base_df)
    base_ref = attach_chain_reference(base_df, bsc_df)

    combined = (
        pd.concat([bsc_ref, base_ref], ignore_index=True)
        .sort_values(["timestamp", "block_number", "log_index"], ascending=True)
        .reset_index(drop=True)
    )

    combined["ref_price"] = combined["ref_price"].ffill()
    combined["side"] = combined["amount1_decimal"].apply(lambda v: "sell" if v < 0 else "buy")
    combined["price"] = combined["price"].astype(float)
    combined["ref_price"] = combined["ref_price"].astype(float)
    combined["spread_pct"] = (
        (combined["price"] - combined["ref_price"])
        / combined["ref_price"].replace(0, np.nan)
    )
    threshold = 0.0005
    combined["is_arb_trade"] = (
        ((combined["side"] == "sell") & (combined["spread_pct"] > threshold))
        | ((combined["side"] == "buy") & (combined["spread_pct"] < -threshold))
    )
    combined["revenue"] = combined["volume_usd"].abs() * combined["spread_pct"].abs()
    combined["swap_fee_usd"] = (
        combined["volume_usd"].abs() * combined["fee_rate"].astype(float)
    )

    gas_fee = (
        pd.to_numeric(combined["gas_price"], errors="coerce").fillna(0.0).astype(float)
        / 1e18
    )
    native_price = combined["chain"].map(CHAIN_NATIVE_PRICES).fillna(Decimal(0)).apply(float)
    combined["gas_fee_usd"] = gas_fee * native_price
    combined["net_profit"] = combined["revenue"] - combined["gas_fee_usd"]
    combined["window"] = (combined["timestamp"] // 10) * 10
    return combined


def summarize_arbitrage_addresses(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    arb = df[df["is_arb_trade"]].copy()
    if arb.empty:
        return pd.DataFrame()
    summary = (
        arb.groupby("sender_address")
        .agg(
            arb_tx_count=pd.NamedAgg(column="tx_hash", aggfunc="count"),
            total_volume_usd=pd.NamedAgg(column="volume_usd", aggfunc="sum"),
            total_net_profit=pd.NamedAgg(column="net_profit", aggfunc="sum"),
            avg_spread_pct=pd.NamedAgg(column="spread_pct", aggfunc="mean"),
            avg_gas_fee_usd=pd.NamedAgg(column="gas_fee_usd", aggfunc="mean"),
            chains=pd.NamedAgg(column="chain", aggfunc=lambda s: ",".join(sorted(set(s)))),
        )
        .sort_values("arb_tx_count", ascending=False)
        .reset_index()
    )
    return summary.head(top_n)


def _aggregate_windows(df: pd.DataFrame) -> pd.DataFrame:
    arb = df[df["is_arb_trade"]].copy()
    if arb.empty:
        return pd.DataFrame()

    agg = (
        arb.groupby(["window", "chain", "side"])
        .agg(
            volume_usd=pd.NamedAgg(column="volume_usd", aggfunc="sum"),
            net_profit=pd.NamedAgg(column="net_profit", aggfunc="sum"),
            eth_amount=pd.NamedAgg(column="amount0_decimal", aggfunc=lambda x: x.abs().sum()),
            spread_pct=pd.NamedAgg(column="spread_pct", aggfunc="mean"),
        )
        .reset_index()
    )

    events = []
    for window, group in agg.groupby("window"):
        bsc_sell = group[(group["chain"] == "bsc") & (group["side"] == "sell")]
        base_buy = group[(group["chain"] == "base") & (group["side"] == "buy")]
        base_sell = group[(group["chain"] == "base") & (group["side"] == "sell")]
        bsc_buy = group[(group["chain"] == "bsc") & (group["side"] == "buy")]
        sell_buy_volume = (
            bsc_sell["volume_usd"].sum() + base_buy["volume_usd"].sum()
        )
        buy_sell_volume = (
            base_sell["volume_usd"].sum() + bsc_buy["volume_usd"].sum()
        )
        if sell_buy_volume <= 0 or buy_sell_volume <= 0:
            continue
        net_flow = sell_buy_volume - buy_sell_volume
        net_direction = (
            "sell>buy"
            if sell_buy_volume >= buy_sell_volume
            else "buy>sell"
        )
        sell_eth_volume = bsc_sell["eth_amount"].sum() + base_buy["eth_amount"].sum()
        buy_eth_volume = base_sell["eth_amount"].sum() + bsc_buy["eth_amount"].sum()
        eth_flow = sell_eth_volume - buy_eth_volume
        symmetry_ratio = abs(eth_flow) / (
            sell_eth_volume + buy_eth_volume
        )
        avg_spread = np.nanmean(
            [
                *bsc_sell["spread_pct"].dropna().tolist(),
                *base_buy["spread_pct"].dropna().tolist(),
                *base_sell["spread_pct"].dropna().tolist(),
                *bsc_buy["spread_pct"].dropna().tolist(),
            ]
        )
        events.append(
            {
                "window": window,
                "bsc_sell_volume": float(bsc_sell["volume_usd"].sum()),
                "base_buy_volume": float(base_buy["volume_usd"].sum()),
                "bsc_buy_volume": float(bsc_buy["volume_usd"].sum()),
                "base_sell_volume": float(base_sell["volume_usd"].sum()),
                "total_volume": float(sell_buy_volume + buy_sell_volume),
                "total_net_profit": float(group["net_profit"].sum()),
                "sell_buy_volume": float(sell_buy_volume),
                "buy_sell_volume": float(buy_sell_volume),
                "net_flow": float(net_flow),
                "net_direction": net_direction,
                "symmetry_ratio": float(symmetry_ratio),
                "sell_eth_volume": float(sell_eth_volume),
                "buy_eth_volume": float(buy_eth_volume),
                "avg_spread_pct": float(avg_spread) if not np.isnan(avg_spread) else 0.0,
            }
        )

    if not events:
        return pd.DataFrame()

    return pd.DataFrame(events)


def strict_cross_chain_trade_groups(df: pd.DataFrame, threshold: float = 0.0005) -> pd.DataFrame:
    """Return windows where both sides trade and each leg meets the opposite-price condition."""
    events = _aggregate_windows(df)
    if events.empty:
        return pd.DataFrame()

    arb = df[df["is_arb_trade"]].copy()
    strict_records = []
    event_index = events.set_index("window")

    for window in events["window"].unique():
        window_trades = arb[arb["window"] == window]
        if window_trades.empty:
            continue
        bsc_sell = window_trades[
            (window_trades["chain"] == "bsc") & (window_trades["side"] == "sell")
        ]
        base_buy = window_trades[
            (window_trades["chain"] == "base") & (window_trades["side"] == "buy")
        ]
        base_sell = window_trades[
            (window_trades["chain"] == "base") & (window_trades["side"] == "sell")
        ]
        bsc_buy = window_trades[
            (window_trades["chain"] == "bsc") & (window_trades["side"] == "buy")
        ]

        sell_buy_match = (
            not bsc_sell.empty
            and not base_buy.empty
            and (bsc_sell["spread_pct"] > threshold).any()
            and (base_buy["spread_pct"] < -threshold).any()
        )
        buy_sell_match = (
            not base_sell.empty
            and not bsc_buy.empty
            and (base_sell["spread_pct"] > threshold).any()
            and (bsc_buy["spread_pct"] < -threshold).any()
        )

        if not (sell_buy_match or buy_sell_match):
            continue

        direction = "sell>buy" if sell_buy_match else "buy>sell"
        summary = window_trades.groupby(["chain", "side"]).size()
        strict_records.append(
            {
                "window": window,
                "pair_type": direction,
                "tx_count": len(window_trades),
                "total_net_profit": float(window_trades["net_profit"].sum()),
                "symmetry_ratio": float(event_index.at[window, "symmetry_ratio"])
                if window in event_index.index
                else 0.0,
                "avg_spread_pct": float(window_trades["spread_pct"].mean()),
                "bsc_sell_count": int(summary.get(("bsc", "sell"), 0)),
                "base_buy_count": int(summary.get(("base", "buy"), 0)),
                "base_sell_count": int(summary.get(("base", "sell"), 0)),
                "bsc_buy_count": int(summary.get(("bsc", "buy"), 0)),
                "tx_hashes": window_trades["tx_hash"].tolist(),
            }
        )

    return pd.DataFrame(strict_records)


def print_strict_cross_chain_groups(df: pd.DataFrame, limit: int = 3) -> None:
    groups = strict_cross_chain_trade_groups(df)
    if groups.empty:
        return
    groups = groups.sort_values("total_net_profit", ascending=False).reset_index(drop=True)
    print("\nStrict cross-chain groups (opposite-sides that satisfy all price constraints):")
    print(
        groups[["window", "pair_type", "tx_count", "total_net_profit", "symmetry_ratio", "avg_spread_pct"]]
        .head(limit)
        .to_string(
            index=False,
            float_format="{:,.4f}".format,
        )
    )
    for _, row in groups.head(limit).iterrows():
        group_trades = df[
            (df["window"] == row["window"]) & df["is_arb_trade"]
        ].sort_values(["chain", "side", "timestamp"])
        if group_trades.empty:
            continue
        print(f"\nWindow {row['window']} trades (pair_type={row['pair_type']}):")
        preview = group_trades[
            ["timestamp", "chain", "side", "tx_hash", "price", "ref_price", "spread_pct", "net_profit"]
        ].head(10)
        print(
            preview.to_string(
                index=False,
                float_format=lambda x: f"{x:,.6f}",
            )
        )


def cross_chain_match_ratio(df: pd.DataFrame) -> float:
    events = _aggregate_windows(df)
    total_windows = df["window"].nunique()
    if total_windows == 0:
        return 0.0
    matched_windows = events["window"].nunique()
    return matched_windows / total_windows


def spread_persistence_stats(df: pd.DataFrame, threshold: float = 0.0005) -> pd.DataFrame:
    window_stats = (
        df.groupby("window")["spread_pct"]
        .mean()
        .abs()
        .reset_index(name="mean_abs_spread")
        .sort_values("window")
    )
    window_stats["duration"] = np.nan
    active = window_stats[window_stats["mean_abs_spread"] > threshold].copy()
    for idx, row in active.iterrows():
        future = window_stats[
            (window_stats["window"] > row["window"])
            & (window_stats["mean_abs_spread"] <= threshold)
        ]
        if not future.empty:
            active.at[idx, "duration"] = future["window"].iloc[0] - row["window"]
    return active


def trace_high_spread_recovery(df: pd.DataFrame, threshold: float = 0.005, lookahead: int = 5) -> pd.DataFrame:
    window_stats = (
        df.groupby("window")["spread_pct"]
        .mean()
        .abs()
        .reset_index(name="mean_abs_spread")
        .sort_values("window")
    )
    recovery_rows = []
    for _, row in window_stats[window_stats["mean_abs_spread"] > threshold].iterrows():
        window = row["window"]
        for offset in range(1, lookahead + 1):
            next_row = window_stats[window_stats["window"] == window + offset]
            if next_row.empty:
                continue
            recovery_rows.append(
                {
                    "origin_window": window,
                    "lookahead": offset,
                    "spread": next_row["mean_abs_spread"].values[0],
                    "origin_spread": row["mean_abs_spread"],
                }
            )
    return pd.DataFrame(recovery_rows)


def analyze_sender_history(df: pd.DataFrame, sender_address: str) -> pd.DataFrame:
    base_trades = df[
        (df["sender_address"].str.lower() == sender_address.lower())
        & (df["chain"] == "base")
    ].sort_values("timestamp").copy()
    if base_trades.empty:
        return pd.DataFrame()

    bsc_prices = (
        df[df["chain"] == "bsc"][["timestamp", "price"]]
        .sort_values("timestamp")
        .rename(columns={"price": "next_bsc_price"})
    )
    base_with_next = pd.merge_asof(
        base_trades.reset_index(),
        bsc_prices,
        on="timestamp",
        direction="forward",
        allow_exact_matches=False,
    )
    base_with_next["prev_bsc_price"] = base_trades["ref_price"]
    base_with_next["prev_bsc_price"] = base_with_next["prev_bsc_price"].fillna(
        base_with_next["next_bsc_price"]
    )

    base_with_next["side"] = np.where(
        base_with_next["amount1_decimal"] < 0, "sell", "buy"
    )
    base_with_next["signed_volume_usd"] = base_with_next["volume_usd"].where(
        base_with_next["side"] == "buy",
        -base_with_next["volume_usd"],
    )

    base_with_next["spread_to_prev"] = (
        base_with_next["price"] - base_with_next["prev_bsc_price"]
    ) / base_with_next["prev_bsc_price"].replace(0, np.nan)
    base_with_next["next_bsc_return"] = (
        base_with_next["next_bsc_price"] - base_with_next["price"]
    ) / base_with_next["price"].replace(0, np.nan)

    base_with_next["price_diff"] = base_with_next["price"] - base_with_next[
        "prev_bsc_price"
    ]
    base_with_next["price_gap_pct"] = base_with_next["price_diff"] / base_with_next[
        "prev_bsc_price"
    ]
    base_with_next["tick_delta"] = (
        base_with_next["current_tick"] - base_with_next["pre_tick"]
    )
    base_with_next["tick_movement"] = base_with_next["tick_delta"].abs()
    base_with_next["tick_direction"] = np.select(
        [
            base_with_next["tick_delta"] > 0,
            base_with_next["tick_delta"] < 0,
            base_with_next["tick_delta"] == 0,
        ],
        ["up", "down", "flat"],
        default="unknown",
    )

    return base_with_next[
        [
            "timestamp",
            "price",
            "prev_bsc_price",
            "price_diff",
            "price_gap_pct",
            "next_bsc_price",
            "next_bsc_return",
            "side",
            "signed_volume_usd",
            "volume_usd",
            "spread_pct",
            "net_profit",
            "pre_tick",
            "current_tick",
            "tick_delta",
            "tick_movement",
            "tick_direction",
        ]
    ]


def describe_sender_tick_movement(sender_history: pd.DataFrame, sender_address: str) -> None:
    if sender_history.empty or "tick_movement" not in sender_history.columns:
        return
    ticks = sender_history["tick_movement"].dropna()
    if ticks.empty:
        print(f"\nNo tick movement recorded for {sender_address}.")
        return

    stats = {
        "trade_count": len(ticks),
        "total_ticks": ticks.sum(),
        "avg_ticks": ticks.mean(),
        "median_ticks": ticks.median(),
        "max_ticks": ticks.max(),
    }
    print(
        f"\nTick movement distribution for {sender_address}:"
        f" {stats['trade_count']} trades, {stats['total_ticks']:.0f} ticks moved in total,"
        f" average {stats['avg_ticks']:.2f}, median {stats['median_ticks']:.2f},"
        f" max {stats['max_ticks']:.0f}."
    )


def summarize_sender_profitability(sender_history: pd.DataFrame) -> dict:
    if sender_history.empty:
        return {}
    hist = sender_history.copy()
    hist["profitable_if_reverse"] = (
        ((hist["side"] == "sell") & (hist["next_bsc_return"] > 0))
        | ((hist["side"] == "buy") & (hist["next_bsc_return"] < 0))
    )
    total = len(hist)
    win = int(hist["profitable_if_reverse"].sum())
    avg_flip = hist["next_bsc_return"].mean()
    return {
        "total_trades": total,
        "profitable_if_reverse": win,
        "profitable_ratio": win / total if total else 0.0,
        "avg_next_bsc_return": avg_flip,
    }


def plot_sender_gap_vs_volume(sender_history: pd.DataFrame, sender_address: str) -> None:
    if sender_history.empty or "price_gap_pct" not in sender_history.columns:
        return
    data = sender_history.dropna(subset=["price_gap_pct", "volume_usd"])
    if data.empty:
        return
    data = data.assign(abs_gap_pct=data["price_gap_pct"].abs())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x="price_gap_pct",
        y="volume_usd",
        hue="side",
        size="abs_gap_pct",
        sizes=(20, 200),
        palette="Set2",
        alpha=0.8,
        edgecolor="w",
        ax=ax,
    )
    ax.set_title(f"{sender_address} Price Gap vs Volume")
    ax.set_xlabel("Price gap vs BSC (pct)")
    ax.set_ylabel("Volume USD")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="side", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    sender_short = sender_address[:10]
    _save_plot(fig, f"gap_vs_volume_{sender_short}")


def summarize_spread_origins(df: pd.DataFrame, sender_address: str, threshold: float = 0.0005) -> dict:
    trades = analyze_sender_history(df, sender_address)
    if trades.empty:
        return {}
    trades = trades.dropna(subset=["spread_pct", "prev_bsc_price"])
    trades["prev_spread"] = trades["prev_bsc_price"] / trades["prev_bsc_price"].shift(1) - 1
    trades["bsc_prior_spread"] = trades["prev_spread"].abs() > threshold
    trades["post_bsc_spread"] = trades["next_bsc_return"].abs() > threshold
    before = trades["bsc_prior_spread"].sum()
    after = trades["post_bsc_spread"].sum()
    total = len(trades)
    return {
        "total_trades": total,
        "prior_large_spread": int(before),
        "post_large_spread": int(after),
        "prior_pct": before / total if total else 0.0,
        "post_pct": after / total if total else 0.0,
    }


def match_cross_chain_events(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    events = _aggregate_windows(df)
    if events.empty:
        return events
    return (
        events.sort_values("total_net_profit", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def detect_price_anomalies(
    df: pd.DataFrame, suspect_price: float = 2954.226369, tol: float = 1e-9
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag and drop trades whose price hits the known bad number.
    """
    mask = np.isclose(df["price"], suspect_price, atol=tol)
    anomalies = df[mask].copy()
    if not anomalies.empty:
        print(
            f"\nDetected {len(anomalies)} trades at price {suspect_price}:"
        )
        print(
            anomalies[
                ["timestamp", "chain", "price", "ref_price", "spread_pct", "tx_hash"]
            ]
            .to_string(index=False, float_format="{:,.6f}".format)
        )
    return df[~mask].copy(), anomalies




def main() -> None:
    df = load_transactions(DB_PATH)
    analyze_pool_spread(df)
    analyzed = analyze_arbitrage_trades(df)
    analyzed, _ = detect_price_anomalies(analyzed)
    competitors = identify_competitors(analyzed)
    print(f"Detected {len(competitors)} competitors.")
    print_leaderboard(competitors)
    describe_top_competitor(competitors, analyzed)
    correlation_analysis(analyzed)
    plot_daily_volume(analyzed)
    ratio = same_timestamp_gas_priority(analyzed)
    print(f"\nAnalysis E – Same-timestamp gas priority ratio: {ratio:.2%}")
    match_ratio = cross_chain_match_ratio(analyzed)
    print(f"Cross-chain match ratio (window with both sides): {match_ratio:.2%}")
    persistence = spread_persistence_stats(analyzed)
    if not persistence.empty:
        print(
            "\nSpread persistence (seconds until spread falls below threshold) – top 5 windows:"
        )
        print(
            persistence.sort_values("duration", ascending=False)
            .head(5)
            .to_string(
                index=False,
                formatters={
                    "window": str,
                    "mean_abs_spread": "{:.6f}".format,
                    "duration": "{:.0f}".format,
                },
            )
        )
        plot_spread_duration(persistence)
    recovery = trace_high_spread_recovery(analyzed)
    if not recovery.empty:
        print("\nHigh spread recovery (lookahead windows):")
        print(
            recovery.sort_values(["origin_window", "lookahead"])
            .head(10)
            .to_string(
                index=False,
                formatters={
                    "origin_window": str,
                    "lookahead": lambda x: f"{x}x10s",
                    "spread": "{:.6f}".format,
                    "origin_spread": "{:.6f}".format,
                },
            )
        )
    arb_addresses = summarize_arbitrage_addresses(analyzed)
    if not arb_addresses.empty:
        print("\nTop arbitrage senders:")
        print(
            arb_addresses.to_string(
                index=False,
                formatters={
                    "total_volume_usd": "{:,.2f}".format,
                    "total_net_profit": "{:,.2f}".format,
                    "avg_spread_pct": "{:.4f}".format,
                    "avg_gas_fee_usd": "{:.4f}".format,
                },
            )
        )
    events = match_cross_chain_events(analyzed)
    if not events.empty:
        print(
            f"\nMatched cross-chain windows: {len(events)} windows, "
            f"{events['window'].nunique()} unique 10s buckets."
        )
    balanced = events[events["symmetry_ratio"] < 0.3].sort_values(
        "total_net_profit", ascending=False
    )
    if not balanced.empty:
        print("\nTop cross-chain arbitrage events (symmetry ratio ≤ 5%):")
        print(
            balanced.to_string(
                index=False,
                formatters={
                    "bsc_sell_volume": "{:,.2f}".format,
                    "base_buy_volume": "{:,.2f}".format,
                    "bsc_buy_volume": "{:,.2f}".format,
                    "base_sell_volume": "{:,.2f}".format,
                    "total_volume": "{:,.2f}".format,
                    "total_net_profit": "{:,.2f}".format,
                    "net_flow": "{:,.2f}".format,
                    "net_direction": str,
                    "symmetry_ratio": "{:.3f}".format,
                    "sell_eth_volume": "{:,.2f}".format,
                    "buy_eth_volume": "{:,.2f}".format,
                },
            )
        )
    balanced = events[events["symmetry_ratio"] < 0.05].sort_values("total_net_profit", ascending=False)
    if not balanced.empty:
        print("\nTop balanced sell/buy events (≤5% imbalance):")
        print(
            balanced.to_string(
                index=False,
                formatters={
                    "bsc_sell_volume": "{:,.2f}".format,
                    "base_buy_volume": "{:,.2f}".format,
                    "total_volume": "{:,.2f}".format,
                    "total_net_profit": "{:,.2f}".format,
                        "symmetry_ratio": "{:.3f}".format,
                        "sell_eth_volume": "{:,.2f}".format,
                        "buy_eth_volume": "{:,.2f}".format,
                },
            )
        )

    print_strict_cross_chain_groups(analyzed)

    target_sender = "0x43f9a7aec2a683c4cd6016f92ff76d5f3e7b44d3"
    spread_origin = summarize_spread_origins(analyzed, target_sender)
    if spread_origin:
        print("\nSpread origin summary for 0x43f9a7a...:")
        print(
            f"prior large spread trades: {spread_origin['prior_large_spread']} "
            f"({spread_origin['prior_pct']:.1%}), "
            f"post spread hits: {spread_origin['post_large_spread']} "
            f"({spread_origin['post_pct']:.1%}) out of {spread_origin['total_trades']} eligible trades."
        )
    sender_history = analyze_sender_history(analyzed, target_sender)
    if not sender_history.empty:
        print("\nSender 0x43f9a7a... base trades with spread detail (price gap vs BSC):")
        print(
            sender_history.to_string(
                index=False,
                float_format="{:.6f}".format,
            )
        )
        describe_sender_tick_movement(sender_history, target_sender)
        plot_sender_gap_vs_volume(sender_history, target_sender)
        profit_stats = summarize_sender_profitability(sender_history)
        if profit_stats:
            print(
                f"\nPotentially profitable reverse BSC trades: {profit_stats['profitable_if_reverse']} / "
                f"{profit_stats['total_trades']} ({profit_stats['profitable_ratio']:.1%}), "
                f"avg next BSC return {profit_stats['avg_next_bsc_return']:.4f}"
            )


if __name__ == "__main__":
    sns.set(style="whitegrid")
    main()
