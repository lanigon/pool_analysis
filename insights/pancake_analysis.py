from __future__ import annotations

from decimal import Decimal
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import Column, Integer, String, select
from sqlalchemy.orm import Session

try:
    from insights.data_types import TransactionData
except ImportError:
    from data_types import TransactionData

try:
    from insights.utils.db_conn import Base, create_sqlite_session, read_all_transactions
except ImportError:
    from utils.db_conn import Base, create_sqlite_session, read_all_transactions

try:
    from insights.utils.tx_group import chain_for_dex
except ImportError:
    from utils.tx_group import chain_for_dex

try:
    from insights.aerodrom_analysis import (
        summarize_senders,
        active_senders,
        print_top_senders,
    )
except ImportError:
    from aerodrom_analysis import (
        summarize_senders,
        active_senders,
        print_top_senders,
    )

try:
    from insights.arbitrage_analysis import same_timestamp_gas_priority
except ImportError:
    from arbitrage_analysis import same_timestamp_gas_priority

DATA_TEST_DIR = Path(__file__).resolve().parents[1] / "data"
TRANSACTIONS_DB_PATH = DATA_TEST_DIR / "transactions.db"


class Transaction(Base):  # pragma: no cover - map to existing SQLite table
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    pool_address = Column(String, nullable=False)
    tx_hash = Column(String, nullable=False, unique=True)
    sender_address = Column(String, nullable=False)
    pre_sqrt_price = Column(String, nullable=False)
    post_sqrt_price = Column(String, nullable=False)
    amount0 = Column(String, nullable=False)
    amount1 = Column(String, nullable=False)
    pre_liquidity = Column(String, nullable=True)
    fee_ratio = Column(String, nullable=True)
    pre_tick = Column(Integer, nullable=False)
    current_tick = Column(Integer, nullable=False)
    current_liquidity = Column(String, nullable=True)
    gas_price = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False, index=True)
    block_number = Column(Integer, nullable=False, index=True)

    def to_transaction_data(self) -> TransactionData:
        return TransactionData(
            pool_address=self.pool_address,
            tx_hash=self.tx_hash,
            sender_address=self.sender_address,
            pre_sqrt_price=self.pre_sqrt_price,
            post_sqrt_price=self.post_sqrt_price,
            amount0=self.amount0,
            amount1=self.amount1,
            pre_liquidity=self.pre_liquidity,
            fee_ratio=self.fee_ratio,
            pre_tick=self.pre_tick,
            current_tick=self.current_tick,
            current_liquidity=self.current_liquidity,
            gas_price=self.gas_price,
            direction=self.direction,
            timestamp=self.timestamp,
            block_number=self.block_number,
        )


def read_transactions(session: Session) -> list[TransactionData]:
    return [
        record
        for record in read_all_transactions(session)
        if chain_for_dex(record.dex_name) == "bsc"
    ]


def _as_decimal(value: str) -> Decimal:
    return Decimal(value)


def _optional_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    return _as_decimal(value)


def summarize_transactions(records: Sequence[TransactionData]) -> dict[str, str | int]:
    if not records:
        return {
            "total_transactions": 0,
            "unique_pools": 0,
            "average_amount0": "0",
            "average_amount1": "0",
            "average_fee_ratio": "0",
            "average_pre_liquidity": "0",
            "average_tick_move": "0",
            "earliest_block": "unknown",
            "latest_block": "unknown",
        }

    total = len(records)
    pool_addresses = {record.pool_address for record in records}
    amount0_values = [_as_decimal(record.amount0).copy_abs() for record in records]
    amount1_values = [_as_decimal(record.amount1).copy_abs() for record in records]
    fee_ratios = [
        value
        for value in (_optional_decimal(record.fee_ratio) for record in records)
        if value is not None
    ]
    pre_liquidity_values = [
        value
        for value in (_optional_decimal(record.pre_liquidity) for record in records)
        if value is not None
    ]
    ticks = [record.current_tick - record.pre_tick for record in records]

    return {
        "total_transactions": total,
        "unique_pools": len(pool_addresses),
        "average_amount0": str(mean(amount0_values)),
        "average_amount1": str(mean(amount1_values)),
        "average_fee_ratio": str(mean(fee_ratios)) if fee_ratios else "0",
        "average_pre_liquidity": str(mean(pre_liquidity_values)) if pre_liquidity_values else "0",
        "average_tick_move": str(mean(ticks)),
        "earliest_block": str(min(record.block_number for record in records)),
        "latest_block": str(max(record.block_number for record in records)),
    }


def display_summary(summary: dict[str, str | int]) -> None:
    print("Transaction overview:")
    for key, value in summary.items():
        print(f" - {key}: {value}")


def main() -> None:
    session = create_sqlite_session(TRANSACTIONS_DB_PATH)
    records = read_transactions(session)
    stats = summarize_senders(records)
    active = active_senders(stats, min_txs=5)
    print(f"BSC records loaded: {len(records)}")
    print(f"Unique senders: {len(stats)}; active ({len(active)} with ≥5 txs)")
    print_top_senders(active, chain_label="bsc")
    records_df = pd.DataFrame([asdict(r) for r in records])
    ratio = same_timestamp_gas_priority(records_df)
    print(f"\nAnalysis E – Same-timestamp gas priority ratio: {ratio:.2%}")


if __name__ == "__main__":
    main()
