from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

try:
    from insights.data_types import TransactionData
except ImportError:
    from data_types import TransactionData

try:
    from insights.utils.db_conn import create_sqlite_session, read_all_transactions
except ImportError:
    from utils.db_conn import create_sqlite_session, read_all_transactions

try:
    from insights.utils.tx_group import chain_for_dex, normalize_to_decimal
except ImportError:
    from utils.tx_group import chain_for_dex, normalize_to_decimal

try:
    from insights.arbitrage_analysis import (
        same_timestamp_gas_priority,
        load_transactions,
        analyze_arbitrage_trades,
    )
except ImportError:
    from arbitrage_analysis import (
        same_timestamp_gas_priority,
        load_transactions,
        analyze_arbitrage_trades,
    )

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRANSACTIONS_DB_PATH = DATA_DIR / "transactions.db"


@dataclass(frozen=True)
class SenderStats:
    tx_count: int
    first_seen: int
    last_seen: int
    total_amount0: Decimal
    total_amount1: Decimal
    amount0_sell: Decimal
    amount0_buy: Decimal
    amount1_sell: Decimal
    amount1_buy: Decimal


def is_base_record(record: TransactionData) -> bool:
    return chain_for_dex(record.dex_name) == "base"


def summarize_senders(records: Iterable[TransactionData]) -> Mapping[str, SenderStats]:
    stats: dict[str, SenderStats] = {}
    temp: dict[str, dict[str, Decimal | int]] = defaultdict(
            lambda: {
                "tx_count": 0,
                "first_seen": None,
                "last_seen": None,
                "total_amount0": Decimal(0),
                "total_amount1": Decimal(0),
                "amount0_sell": Decimal(0),
                "amount0_buy": Decimal(0),
                "amount1_sell": Decimal(0),
                "amount1_buy": Decimal(0),
            }
        )

    for record in records:
        sender = record.sender_address
        entry = temp[sender]
        entry["tx_count"] += 1
        entry["first_seen"] = (
            record.timestamp
            if entry["first_seen"] is None or record.timestamp < entry["first_seen"]
            else entry["first_seen"]
        )
        entry["last_seen"] = (
            record.timestamp
            if entry["last_seen"] is None or record.timestamp > entry["last_seen"]
            else entry["last_seen"]
        )
        chain = chain_for_dex(record.dex_name)
        amt0 = normalize_to_decimal(record.amount0, chain=chain, token_index=0)
        amt1 = normalize_to_decimal(record.amount1, chain=chain, token_index=1)
        entry["total_amount0"] += amt0
        entry["total_amount1"] += amt1
        if amt0 >= 0:
            entry["amount0_sell"] += amt0
        else:
            entry["amount0_buy"] += amt0
        if amt1 >= 0:
            entry["amount1_sell"] += amt1
        else:
            entry["amount1_buy"] += amt1

    for sender, entry in temp.items():
        stats[sender] = SenderStats(
            tx_count=entry["tx_count"],
            first_seen=entry["first_seen"],
            last_seen=entry["last_seen"],
            total_amount0=entry["total_amount0"],
            total_amount1=entry["total_amount1"],
            amount0_sell=entry["amount0_sell"],
            amount0_buy=entry["amount0_buy"],
            amount1_sell=entry["amount1_sell"],
            amount1_buy=entry["amount1_buy"],
        )

    return stats


def active_senders(stats: Mapping[str, SenderStats], min_txs: int = 5) -> Mapping[str, SenderStats]:
    return {sender: stat for sender, stat in stats.items() if stat.tx_count >= min_txs}


def print_top_senders(
    stats: Mapping[str, SenderStats],
    *,
    limit: int = 10,
    chain_label: str = "base",
) -> None:
    ranked = sorted(stats.items(), key=lambda item: item[1].tx_count, reverse=True)[:limit]
    print(f"Top senders on {chain_label} by tx count:")
    for sender, stat in ranked:
        first_seen = datetime.fromtimestamp(stat.first_seen, timezone.utc).isoformat()
        last_seen = datetime.fromtimestamp(stat.last_seen, timezone.utc).isoformat()
        print(
            f"- {sender} txs={stat.tx_count} "
            f"amount0={stat.total_amount0:.3f} "
            f"(sell={stat.amount0_sell:.3f} buy={stat.amount0_buy:.3f}) "
            f"amount1={stat.total_amount1:.3f} "
            f"(sell={stat.amount1_sell:.3f} buy={stat.amount1_buy:.3f}) "
            f"first={first_seen} last={last_seen}"
        )


def pair_cross_chain_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match every arb trade on one chain with an opposite-side arb trade on the other chain,
    without relying on fixed time buckets.
    """
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
    pending: dict[tuple[str, str], list[dict[str, object]]] = {
        key: [] for key in opposite
    }
    consecutive_count: dict[tuple[str, str], int] = defaultdict(int)
    groups: list[dict[str, object]] = []

    for _, row in arb.iterrows():
        chain_side = (row["chain"], row["side"])
        if chain_side not in opposite:
            continue
        opp_side = opposite[chain_side]
        current_row = row.to_dict()

        if pending[opp_side]:
            best_idx = max(
                range(len(pending[opp_side])),
                key=lambda idx: abs(pending[opp_side][idx]["row"]["volume_usd"]),
            )
            candidate = pending[opp_side].pop(best_idx)
            candidate_row = candidate["row"]
            current_sequence = consecutive_count[chain_side] + 1

            time_diff = abs(current_row["timestamp"] - candidate_row["timestamp"])
            vol_a = abs(candidate_row["volume_usd"])
            vol_b = abs(current_row["volume_usd"])
            total_net_profit = candidate_row["net_profit"] + current_row["net_profit"]
            max_vol = max(vol_a, vol_b)
            volume_ratio = min(vol_a, vol_b) / max_vol if max_vol > 0 else 0.0

            groups.append(
                {
                    "pair_id": len(groups) + 1,
                    "pair_type": f"{candidate_row['chain']}_{candidate_row['side']} -> "
                    f"{current_row['chain']}_{current_row['side']}",
                    "chain_a": candidate_row["chain"],
                    "side_a": candidate_row["side"],
                    "timestamp_a": candidate_row["timestamp"],
                    "tx_hash_a": candidate_row.get("tx_hash"),
                    "volume_a": float(vol_a),
                    "spread_a": float(candidate_row.get("spread_pct", 0.0) or 0.0),
                    "net_profit_a": float(candidate_row.get("net_profit", 0.0) or 0.0),
                    "chain_b": current_row["chain"],
                    "side_b": current_row["side"],
                    "timestamp_b": current_row["timestamp"],
                    "tx_hash_b": current_row.get("tx_hash"),
                    "volume_b": float(vol_b),
                    "spread_b": float(current_row.get("spread_pct", 0.0) or 0.0),
                    "net_profit_b": float(current_row.get("net_profit", 0.0) or 0.0),
                    "time_diff": float(time_diff),
                    "volume_ratio": float(volume_ratio),
                    "total_net_profit": float(total_net_profit),
                    "consecutive_a": int(candidate["consecutive_count"]),
                    "consecutive_b": int(current_sequence),
                }
            )

            consecutive_count[opp_side] = len(pending[opp_side])
            consecutive_count[chain_side] = len(pending[chain_side])
        else:
            consecutive_count[chain_side] += 1
            pending[chain_side].append(
                {"row": current_row, "consecutive_count": consecutive_count[chain_side]}
            )

    if not groups:
        return pd.DataFrame()

    result = pd.DataFrame(groups)
    return result.sort_values("time_diff").reset_index(drop=True)


def buffered_cross_chain_pairs(
    df: pd.DataFrame, buffer_seconds: int = 60, threshold: float = 0.0005
) -> pd.DataFrame:
    """Match trades with opposite flow within a rolling 60s buffer."""
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

        # expire stale candidates
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
            best_idx = max(
                range(len(pending[opp])),
                key=lambda idx: abs(pending[opp][idx]["row"]["volume_usd"]),
            )
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

            matches.append(
                {
                    "pair_id": len(matches) + 1,
                    "chain_a": candidate_row["chain"],
                    "side_a": candidate_row["side"],
                    "timestamp_a": candidate_row["timestamp"],
                    "volume_a": float(vol_a),
                    "spread_a": float(candidate_row.get("spread_pct", 0.0) or 0.0),
                    "net_profit_a": float(candidate_row.get("net_profit", 0.0) or 0.0),
                    "chain_b": row["chain"],
                    "side_b": row["side"],
                    "timestamp_b": now,
                    "volume_b": float(vol_b),
                    "spread_b": float(row.get("spread_pct", 0.0) or 0.0),
                    "net_profit_b": float(row.get("net_profit", 0.0) or 0.0),
                    "time_diff": float(time_diff),
                    "volume_ratio": float(volume_ratio),
                    "total_net_profit": float(total_net_profit),
                    "consecutive_a": int(consecutive_a),
                    "consecutive_b": int(consecutive_b),
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


def main() -> None:
    session = create_sqlite_session(TRANSACTIONS_DB_PATH)
    all_records = read_all_transactions(session)
    base_records = [record for record in all_records if is_base_record(record)]
    summary = summarize_senders(base_records)
    active = active_senders(summary, min_txs=5)

    print(f"Base records loaded: {len(base_records)}")
    print(f"Unique senders: {len(summary)}; active ({len(active)} with ≥5 txs)")
    print_top_senders(active)
    records_df = pd.DataFrame([asdict(r) for r in base_records])
    ratio = same_timestamp_gas_priority(records_df)
    print(f"\nAnalysis E – Same-timestamp gas priority ratio: {ratio:.2%}")

    full_df = load_transactions(TRANSACTIONS_DB_PATH)
    arb_df = analyze_arbitrage_trades(full_df)
    pair_df = pair_cross_chain_actions(arb_df)
    print(f"\nCross-chain pair matches (no window): {len(pair_df)}")
    if not pair_df.empty:
        to_print = pair_df[
            [
                "pair_type",
                "chain_a",
                "side_a",
                "chain_b",
                "side_b",
                "time_diff",
                "volume_ratio",
                "total_net_profit",
            ]
        ]
        print(to_print.head(10).to_string(index=False, float_format="{:,.4f}".format))
        print("\nTime diff stats (seconds):")
        print(pair_df["time_diff"].describe().to_string(float_format="{:,.2f}".format))
        print("\nVolume ratio stats:")
        print(pair_df["volume_ratio"].describe().to_string(float_format="{:,.4f}".format))

    buffer_df = buffered_cross_chain_pairs(arb_df)
    print(f"\nBuffered cross-chain pairs (60s): {len(buffer_df)}")
    if not buffer_df.empty:
        cols = [
            "chain_a",
            "side_a",
            "chain_b",
            "side_b",
            "time_diff",
            "volume_ratio",
            "total_net_profit",
        ]
        print(buffer_df[cols].head(10).to_string(index=False, float_format="{:,.4f}".format))
        print("\nBuffered time diff stats (seconds):")
        print(buffer_df["time_diff"].describe().to_string(float_format="{:,.2f}".format))
        print("\nBuffered volume ratio stats:")
        print(buffer_df["volume_ratio"].describe().to_string(float_format="{:,.4f}".format))


if __name__ == "__main__":
    main()
