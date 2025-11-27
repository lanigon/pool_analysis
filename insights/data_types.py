
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TransactionData:
    """Model the transaction record fields written by `substreams/src/bin/initial_db.rs`."""

    pool_address: str
    tx_hash: str
    sender_address: str
    pre_sqrt_price: str
    post_sqrt_price: str
    amount0: str
    amount1: str
    pre_liquidity: Optional[str]
    fee_ratio: Optional[str]
    pre_tick: int
    current_tick: int
    current_liquidity: Optional[str]
    gas_price: str
    direction: str
    timestamp: int
    block_number: int
    dex_name: str
