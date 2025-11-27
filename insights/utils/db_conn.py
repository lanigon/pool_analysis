from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

try:
    from insights.data_types import TransactionData
except ImportError:
    from data_types import TransactionData

Base = declarative_base()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
POOLS_DB_PATH = DATA_DIR / "pools.db"


def create_sqlite_engine(db_path: Path | str) -> Engine:
    target = Path(db_path).expanduser().resolve()
    return create_engine(
        f"sqlite:///{target}",
        connect_args={"check_same_thread": False},
        future=True,
    )


def create_sqlite_session(db_path: Path | str) -> Session:
    engine = create_sqlite_engine(db_path)
    SessionLocal = sessionmaker(bind=engine, future=True)
    return SessionLocal()


def _transaction_from_row(row: dict[str, str | int | None]) -> TransactionData:
    return TransactionData(
        pool_address=row["pool_address"],
        tx_hash=row["tx_hash"],
        sender_address=row["sender_address"],
        pre_sqrt_price=row["pre_sqrt_price"],
        post_sqrt_price=row["post_sqrt_price"],
        amount0=row["amount0"],
        amount1=row["amount1"],
        pre_liquidity=row["pre_liquidity"],
        fee_ratio=row["fee_ratio"],
        pre_tick=row["pre_tick"],
        current_tick=row["current_tick"],
        current_liquidity=row["current_liquidity"],
        gas_price=row["gas_price"],
        direction=row["direction"],
        timestamp=row["timestamp"],
        block_number=row["block_number"],
        dex_name=row["dex_name"],
    )


def read_transactions_by_dex(session: Session, dex_name: str) -> list[TransactionData]:
    stmt = text(
        """
        SELECT t.pool_address,
               t.tx_hash,
               t.sender_address,
               t.pre_sqrt_price,
               t.post_sqrt_price,
               t.amount0,
               t.amount1,
               t.pre_liquidity,
               t.fee_ratio,
               t.pre_tick,
               t.current_tick,
               t.current_liquidity,
               t.gas_price,
               t.direction,
               t.timestamp,
               t.block_number,
               p.dex_name AS dex_name
        FROM transactions AS t
        INNER JOIN pools AS p
        ON t.pool_address = p.pool_address
        WHERE p.dex_name = :dex_name
        ORDER BY t.timestamp
        """
    )
    rows = session.execute(stmt, {"dex_name": dex_name}).mappings().all()
    return [_transaction_from_row(row) for row in rows]


def read_pancakeswap_transactions(session: Session) -> list[TransactionData]:
    return read_transactions_by_dex(session, "PancakeSwap")


def read_aerodrome_transactions(session: Session) -> list[TransactionData]:
    return read_transactions_by_dex(session, "Aerodrome")


def read_all_transactions(session: Session) -> list[TransactionData]:
    attach_stmt = text(f"ATTACH DATABASE '{POOLS_DB_PATH}' AS pools_db")
    detach_stmt = text("DETACH DATABASE pools_db")
    query = text(
        """
        SELECT t.pool_address,
               t.tx_hash,
               t.sender_address,
               t.pre_sqrt_price,
               t.post_sqrt_price,
               t.amount0,
               t.amount1,
               t.pre_liquidity,
               t.fee_ratio,
               t.pre_tick,
               t.current_tick,
               t.current_liquidity,
               t.gas_price,
               t.direction,
               t.timestamp,
               t.block_number,
               p.dex_name AS dex_name
        FROM transactions AS t
        INNER JOIN pools_db.pools AS p
        ON t.pool_address = p.pool_address
        ORDER BY t.timestamp
        """
    )

    session.execute(attach_stmt)
    try:
        rows = session.execute(query).mappings().all()
    finally:
        session.execute(detach_stmt)

    return [_transaction_from_row(row) for row in rows]
