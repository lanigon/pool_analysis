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
    def _print_threshold_stats(threshold: float) -> None:
        matched = spread_df[spread_df["spread_pct"].abs() > threshold]
        if matched.empty:
            return
        counts = matched.groupby("chain")["tx_hash"].count().rename("large_spread_trades").reset_index()
        print(f"\nHigh-spread trades (>{threshold*100:.2f}%): {len(matched)} total")
        print(
            counts.to_string(
                index=False,
                float_format="{:,.0f}".format,
            )
        )
        if threshold == 0.0005:
            print("\nSample high-spread trades:")
            print(
                matched[
                    ["timestamp", "chain", "tx_hash", "price", "ref_price", "spread_pct"]
                ]
                .head(5)
                .to_string(index=False, float_format="{:,.4f}".format)
            )

    _print_threshold_stats(0.0005)
    _print_threshold_stats(0.001)
    _print_threshold_stats(0.002)
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


def build_cross_chain_pairs(
    df: pd.DataFrame, buffer_seconds: int = 60, threshold: float = 0.0005
) -> pd.DataFrame:
    arb = df[df["is_arb_trade"]].sort_values(
        ["timestamp", "block_number", "log_index"], ascending=True
    )
    if arb.empty:
        return pd.DataFrame()

    opposite = {
        ("bsc", "sell"): ("base", "buy"),
        ("base", "buy"): ("bsc", "sell"),
        ("base", "sell"): ("bsc", "buy"),
        ("bsc", "buy"): ("base", "sell"),
    }
    pending: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    streaks: defaultdict[tuple[str, str], int] = defaultdict(int)
    matches: list[dict[str, object]] = []

    for _, row in arb.iterrows():
        chain_side = (row["chain"], row["side"])
        if abs(row["spread_pct"]) <= threshold:
            continue
        now = row["timestamp"]

        for key in list(pending.keys()):
            pending[key] = [
                entry
                for entry in pending[key]
                if now - entry["row"]["timestamp"] <= buffer_seconds
            ]
        opp = opposite.get(chain_side)
        if not opp:
            continue

        if pending[opp]:
            vol_b = abs(float(row.get("volume_usd", 0.0) or 0.0))
            def _match_score(idx: int) -> tuple[float, float, float]:
                candidate = pending[opp][idx]["row"]
                cand_volume = abs(float(candidate.get("volume_usd", 0.0) or 0.0))
                cand_timestamp = candidate.get("timestamp", now)
                time_diff = abs(now - cand_timestamp)
                if cand_volume > 0 and vol_b > 0:
                    vol_ratio = min(cand_volume, vol_b) / max(cand_volume, vol_b)
                else:
                    vol_ratio = 0.0
                cand_spread = float(candidate.get("spread_pct", 0.0) or 0.0)
                spread_score = abs(cand_spread)
                # Higher tuple values mean better match: prioritize smaller time_diff, then closer volume ratio, then spread
                return (-time_diff, vol_ratio, spread_score)
            best_idx = max(range(len(pending[opp])), key=_match_score)
            candidate = pending[opp].pop(best_idx)
            candidate_row = candidate["row"]
            consecutive_a = candidate["consecutive"]
            consecutive_b = streaks[chain_side] + 1

            vol_a = abs(candidate_row["volume_usd"])
            vol_b = abs(row["volume_usd"])
            total_net_profit = candidate_row["net_profit"] + row["net_profit"]
            time_diff = abs(candidate_row["timestamp"] - now)
            max_vol = max(vol_a, vol_b)
            volume_ratio = min(vol_a, vol_b) / max_vol if max_vol > 0 else 0.0
            amount1_a = float(candidate_row.get("amount1_decimal", 0.0) or 0.0)
            amount1_b = float(row.get("amount1_decimal", 0.0) or 0.0)
            signed_stable_a = vol_a * float(np.sign(amount1_a))
            signed_stable_b = vol_b * float(np.sign(amount1_b))
            stable_delta_usd = signed_stable_a + signed_stable_b
            amount0_a = float(candidate_row.get("amount0_decimal", 0.0) or 0.0)
            amount0_b = float(row.get("amount0_decimal", 0.0) or 0.0)
            eth_delta = amount0_a + amount0_b
            price_a = float(candidate_row.get("price", 0.0) or 0.0)
            price_b = float(row.get("price", 0.0) or 0.0)
            ref_price_a = float(candidate_row.get("ref_price", price_a) or price_a)
            ref_price_b = float(row.get("ref_price", price_b) or price_b)
            market_price = float(
                np.nanmean(
                    [
                        ref_price_a if ref_price_a > 0 else price_a,
                        ref_price_b if ref_price_b > 0 else price_b,
                    ]
                )
            )
            if not np.isfinite(market_price) or market_price == 0.0:
                market_price = max(price_a, price_b, 0.0)
            eth_delta_usd = eth_delta * market_price
            gas_a = float(candidate_row.get("gas_fee_usd", 0.0) or 0.0)
            gas_b = float(row.get("gas_fee_usd", 0.0) or 0.0)
            gas_total = gas_a + gas_b
            total_net_profit = stable_delta_usd + eth_delta_usd - gas_total

            matches.append(
                {
                    "pair_id": len(matches) + 1,
                    "pair_type": f"{candidate_row['chain']}_{candidate_row['side']} -> "
                    f"{row['chain']}_{row['side']}",
                    "chain_a": candidate_row["chain"],
                    "side_a": candidate_row["side"],
                    "timestamp_a": candidate_row["timestamp"],
                    "tx_hash_a": candidate_row.get("tx_hash"),
                    "volume_a": float(vol_a),
                    "spread_a": float(candidate_row.get("spread_pct", 0.0) or 0.0),
                    "net_profit_a": float(candidate_row.get("net_profit", 0.0) or 0.0),
                    "chain_b": row["chain"],
                    "side_b": row["side"],
                    "timestamp_b": now,
                    "tx_hash_b": row.get("tx_hash"),
                    "volume_b": float(vol_b),
                    "spread_b": float(row.get("spread_pct", 0.0) or 0.0),
                    "net_profit_b": float(row.get("net_profit", 0.0) or 0.0),
                    "time_diff": float(time_diff),
                    "volume_ratio": float(volume_ratio),
                    "total_net_profit": float(total_net_profit),
                    "price_a": float(candidate_row.get("price", 0.0) or 0.0),
                    "price_b": float(row.get("price", 0.0) or 0.0),
                    "amount1_a": float(candidate_row.get("amount1_decimal", 0.0) or 0.0),
                    "amount1_b": float(row.get("amount1_decimal", 0.0) or 0.0),
                    "volume_usd_a": float(candidate_row.get("volume_usd", 0.0) or 0.0),
                    "volume_usd_b": float(row.get("volume_usd", 0.0) or 0.0),
                    "gas_fee_a": float(candidate_row.get("gas_fee_usd", 0.0) or 0.0),
                    "gas_fee_b": float(row.get("gas_fee_usd", 0.0) or 0.0),
                    "window": int(min(candidate_row["timestamp"], now) // 10 * 10),
                    "sender_a": candidate_row.get("sender_address"),
                    "sender_b": row.get("sender_address"),
                    "consecutive_a": int(consecutive_a),
                    "consecutive_b": int(consecutive_b),
                    "stable_delta_usd": stable_delta_usd,
                    "eth_delta": eth_delta,
                    "eth_delta_usd": eth_delta_usd,
                    "market_price": market_price,
                    "gas_fee_total": gas_total,
                    "pair_profit_usd": total_net_profit,
                }
            )
            streaks[opp] = len(pending[opp])
            streaks[chain_side] = 0
        else:
            streaks[chain_side] += 1
            pending[chain_side].append(
                {"row": row.to_dict(), "consecutive": streaks[chain_side]}
            )

    if not matches:
        return pd.DataFrame()

    return pd.DataFrame(matches).sort_values("time_diff").reset_index(drop=True)


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
    print("\n=== Single-trade arbitrage analysis ===")
    competitors = identify_competitors(analyzed)
    print(f"Detected {len(competitors)} competitors.")
    print_leaderboard(competitors)
    describe_top_competitor(competitors, analyzed)
    correlation_analysis(analyzed)
    plot_daily_volume(analyzed)
    ratio = same_timestamp_gas_priority(analyzed)
    print(f"\nAnalysis E – Same-timestamp gas priority ratio: {ratio:.2%}")
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
        snippet = sender_history.tail(50)
        print(
            snippet.to_string(
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

    print("\n=== Cross-chain trading group analysis ===")
    pair_df = build_cross_chain_pairs(analyzed)
    print(f"Buffered cross-chain pairs (60s buffer, spread ≥ 0.05%): {len(pair_df)} matches")
    filtered_pairs = pair_df[pair_df["volume_ratio"] >= 0.8]
    kept = len(filtered_pairs)
    print(f"Kept {kept} matches with volume_ratio >= 0.8.")
    if filtered_pairs.empty:
        print("No cross-chain pairs with volume_ratio >= 0.8.")
        return

    if not filtered_pairs.empty:
        print(
            "\nFiltered matches (ratio >= 0.8):"
            " pair_id time_diff volume_ratio total_net_profit"
        )
        print(
            filtered_pairs[
                ["pair_id", "time_diff", "volume_ratio", "total_net_profit"]
            ]
            .to_string(index=False, float_format="{:.4f}".format)
        )
        total_profit = filtered_pairs["total_net_profit"].sum()
        positive_pairs = filtered_pairs[filtered_pairs["total_net_profit"] > 0]
        negative_pairs = filtered_pairs[filtered_pairs["total_net_profit"] <= 0]
        positive_profit = positive_pairs["total_net_profit"].sum()
        print(f"Total profit across filtered matches: ${total_profit:,.2f}")
        print(f"Positive-match count: {len(positive_pairs)} (sum ${positive_profit:,.2f}); "
              f"negative-match count: {len(negative_pairs)}")
        summary_cols = [
            "pair_type",
            "chain_a",
            "side_a",
            "chain_b",
            "side_b",
            "time_diff",
            "volume_ratio",
            "total_net_profit",
            "stable_delta_usd",
            "eth_delta_usd",
        ]
        print(filtered_pairs[summary_cols].head(10).to_string(index=False, float_format="{:,.4f}".format))
        print("\nPaired time diff stats (seconds):")
        print(pair_df["time_diff"].describe().to_string(float_format="{:,.2f}".format))
        print("\nPaired volume ratio stats:")
        print(pair_df["volume_ratio"].describe().to_string(float_format="{:,.4f}".format))
        net_profit_stats = filtered_pairs["total_net_profit"]
        net_profit_desc = net_profit_stats.describe(percentiles=[0.25, 0.5, 0.75])
        print(
            "\nPair net profit stats:"
            f" mean {net_profit_desc['mean']:.2f}"
            f" min {net_profit_desc['min']:.2f}"
            f" 25% {net_profit_desc['25%']:.2f}"
            f" median {net_profit_desc['50%']:.2f}"
            f" 75% {net_profit_desc['75%']:.2f}"
            f" max {net_profit_desc['max']:.2f}"
            f" total {net_profit_stats.sum():,.2f}"
        )
        stable_stats = filtered_pairs["stable_delta_usd"]
        stable_desc = stable_stats.describe(percentiles=[0.25, 0.5, 0.75])
        print(
            "\nStable delta stats (USDT/USDC differences):"
            f" mean {stable_desc['mean']:.2f}"
            f" min {stable_desc['min']:.2f}"
            f" 25% {stable_desc['25%']:.2f}"
            f" median {stable_desc['50%']:.2f}"
            f" 75% {stable_desc['75%']:.2f}"
            f" max {stable_desc['max']:.2f}"
            f" total {stable_stats.sum():,.2f}"
        )
        eth_stats = filtered_pairs["eth_delta_usd"]
        eth_desc = eth_stats.describe(percentiles=[0.25, 0.5, 0.75])
        print(
            "\nNet ETH delta (USD) stats:"
            f" mean {eth_desc['mean']:.2f}"
            f" min {eth_desc['min']:.2f}"
            f" 25% {eth_desc['25%']:.2f}"
            f" median {eth_desc['50%']:.2f}"
            f" 75% {eth_desc['75%']:.2f}"
            f" max {eth_desc['max']:.2f}"
            f" total {eth_stats.sum():,.2f}"
        )
        top_pairs = (
            filtered_pairs.sort_values("total_net_profit", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        detail_cols = [
            "pair_id",
            "pair_type",
            "chain_a",
            "side_a",
            "chain_b",
            "side_b",
            "time_diff",
            "tx_hash_a",
            "tx_hash_b",
            "window",
            "price_a",
            "amount1_a",
            "price_b",
            "amount1_b",
            "total_net_profit",
            "stable_delta_usd",
            "eth_delta",
            "eth_delta_usd",
            "market_price",
        ]
        print("\nTop 20 cross-chain matches by total net profit:")
        print(
            top_pairs[detail_cols].to_string(
                index=False,
                float_format="{:,.6f}".format,
            )
        )
        price_stats = {
            "price_a": pair_df["price_a"],
            "price_b": pair_df["price_b"],
        }
        print("\nPrice distribution for both sides:")
        for label, series in price_stats.items():
            desc = series.describe(percentiles=[0.25, 0.5, 0.75])
            print(
                f"- {label}: mean {desc['mean']:.6f}, min {desc['min']:.6f}, "
                f"25% {desc['25%']:.6f}, median {desc['50%']:.6f}, "
                f"75% {desc['75%']:.6f}, max {desc['max']:.6f}"
            )
        pair_df["gas_fee_total"] = pair_df["gas_fee_a"] + pair_df["gas_fee_b"]
        pair_df["volume_usd_total"] = pair_df["volume_usd_a"] + pair_df["volume_usd_b"]
        print("\nTotal gas fee stats (sum of both sides):")
        print(
            pair_df["gas_fee_total"]
            .describe()
            .to_string(float_format="{:,.6f}".format)
        )
        print("\nTotal volume USD stats (sum of both sides):")
        print(
            pair_df["volume_usd_total"]
            .describe()
            .to_string(float_format="{:,.2f}".format)
        )
        corr = pair_df[["gas_fee_total", "volume_usd_total", "total_net_profit"]].corr()
        print("\nCorrelations (gas fee vs volume/profit):")
        print(f"- gas vs volume: {corr.at['gas_fee_total','volume_usd_total']:.4f}")
        print(f"- gas vs profit: {corr.at['gas_fee_total','total_net_profit']:.4f}")
        sender_pairs = (
            filtered_pairs.groupby(["sender_a", "sender_b"])
            .agg(
                matches=("pair_id", "count"),
                total_net_profit=("total_net_profit", "sum"),
            )
            .sort_values("total_net_profit", ascending=False)
            .head(3)
            .reset_index()
        )
        if not sender_pairs.empty:
            print("\nTop sender pairs by cumulative net profit:")
            print(
                sender_pairs.to_string(
                    index=False,
                    formatters={
                        "total_net_profit": "{:,.2f}".format,
                        "matches": "{:d}".format,
                    },
                )
            )


if __name__ == "__main__":
    sns.set(style="whitegrid")
    main()
