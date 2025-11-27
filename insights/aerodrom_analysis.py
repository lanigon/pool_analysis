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
    from insights.arbitrage_analysis import same_timestamp_gas_priority
except ImportError:
    from arbitrage_analysis import same_timestamp_gas_priority

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


if __name__ == "__main__":
    main()
