use num_bigint::BigUint;
use substreams::errors::Error;
use substreams_ethereum::pb::eth::v2 as eth;

use crate::clmm::{ModuleContext, extract_swaps};
use crate::pb::dex::{RawSwap, RawSwaps, SwapRecord, SwapRecords};

const DEX_NAME: &str = "pancakeswap_v3";
const CHAIN: &str = "bsc";

#[substreams::handlers::map]
pub fn map_pancake_swaps(params: String, block: eth::Block) -> Result<RawSwaps, Error> {
    extract_swaps(
        &params,
        block,
        ModuleContext {
            dex: DEX_NAME,
            chain: CHAIN,
        },
    )
}

#[substreams::handlers::map]
pub fn map_pancake_swaps_enriched(
    swaps: RawSwaps,
    block: eth::Block,
) -> Result<SwapRecords, Error> {
    let mut records = SwapRecords::default();

    for swap in swaps.swaps.into_iter() {
        let (pre_sqrt, pre_liq, pre_tick) = extract_pre_swap_state(&block, &swap);

        let record = SwapRecord {
            dex: swap.dex.clone(),
            chain: swap.chain.clone(),
            pool_address: swap.pool_address.clone(),
            tx_hash: swap.tx_hash.clone(),
            sender: swap.sender.clone(),
            recipient: swap.recipient.clone(),
            amount0: swap.amount0.clone(),
            amount1: swap.amount1.clone(),
            pre_sqrt_price_x96: pre_sqrt,
            post_sqrt_price_x96: swap.sqrt_price_x96.clone(),
            pre_liquidity: pre_liq,
            post_liquidity: swap.liquidity.clone(),
            pre_tick,
            post_tick: swap.tick,
            block_number: swap.block_number,
            block_hash: swap.block_hash.clone(),
            block_timestamp: swap.block_timestamp,
            log_index: swap.log_index,
            fee_ratio: swap.fee_ratio.clone(),
            gas_price: swap.gas_price.clone(),
            gas_used: swap.gas_used,
            gas_fee: swap.gas_fee.clone(),
        };
        // println!("swap record: {:?}", record);

        // log::info!(
        //     "swap record pool={} tx={} amount0={} amount1={} pre_tick={} post_tick={} pre_liq={} post_liq={} gas_price={} gas_used={} gas_fee={}",
        //     record.pool_address,
        //     record.tx_hash,
        //     record.amount0,
        //     record.amount1,
        //     record.pre_tick,
        //     record.post_tick,
        //     record.pre_liquidity,
        //     record.post_liquidity,
        //     record.gas_price,
        //     record.gas_used,
        //     record.gas_fee,
        // );

        records.records.push(record);
    }

    Ok(records)
}

const SLOT0_KEY: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const SLOT5_KEY: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
];

fn extract_pre_swap_state(block: &eth::Block, swap: &RawSwap) -> (String, String, i64) {
    let default = || (String::new(), String::new(), 0i64);
    let Some(pool_address) = decode_hex(&swap.pool_address) else {
        return default();
    };
    let Some(tx_hash) = decode_hex(&swap.tx_hash) else {
        return default();
    };
    let Some(trx) = block
        .transaction_traces
        .iter()
        .find(|trx| trx.hash.as_slice() == tx_hash.as_slice())
    else {
        return default();
    };

    let slot0_change = find_storage_change(trx, pool_address.as_slice(), &SLOT0_KEY, swap.ordinal);
    let slot5_change = find_storage_change(trx, pool_address.as_slice(), &SLOT5_KEY, swap.ordinal);

    let mut pre_tick = 0;
    let mut pre_sqrt = String::new();
    if let Some(change) = slot0_change {
        let state = decode_slot0_state(&change.old_value);
        pre_tick = state.tick;
        pre_sqrt = state.sqrt_price_x96;
    }

    let mut pre_liq = swap.liquidity.clone();
    if let Some(change) = slot5_change {
        pre_liq = decode_uint(&change.old_value);
    }

    (pre_sqrt, pre_liq, pre_tick)
}

fn find_storage_change<'a>(
    tx: &'a eth::TransactionTrace,
    address: &[u8],
    key: &[u8; 32],
    max_ordinal: u64,
) -> Option<&'a eth::StorageChange> {
    tx.calls
        .iter()
        .flat_map(|call| call.storage_changes.iter())
        .filter(|change| {
            change.address.as_slice() == address
                && change.key.as_slice() == key
                && (max_ordinal == 0 || change.ordinal <= max_ordinal)
        })
        .max_by_key(|change| change.ordinal)
}

struct Slot0State {
    sqrt_price_x96: String,
    tick: i64,
}

fn decode_slot0_state(value: &[u8]) -> Slot0State {
    let padded = pad_to_32_bytes(value);
    let sqrt = BigUint::from_bytes_be(&padded[12..32]).to_string();
    let tick = decode_signed_24(&padded[9..12]);
    Slot0State {
        sqrt_price_x96: sqrt,
        tick,
    }
}

fn decode_uint(value: &[u8]) -> String {
    let padded = pad_to_32_bytes(value);
    BigUint::from_bytes_be(&padded).to_string()
}

fn pad_to_32_bytes(value: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    if value.len() >= 32 {
        let start = value.len() - 32;
        out.copy_from_slice(&value[start..]);
    } else {
        out[32 - value.len()..].copy_from_slice(value);
    }
    out
}

fn decode_signed_24(bytes: &[u8]) -> i64 {
    let raw = ((bytes[0] as i32) << 16) | ((bytes[1] as i32) << 8) | bytes[2] as i32;
    let value = if raw & 0x800000 != 0 {
        raw - 0x1000000
    } else {
        raw
    };
    value as i64
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    let sanitized = value.strip_prefix("0x").unwrap_or(value);
    if sanitized.len() % 2 != 0 {
        return None;
    }
    hex::decode(sanitized).ok()
}
