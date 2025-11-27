use std::collections::HashMap;

use num_bigint::BigUint;
use substreams::errors::Error;
use substreams_ethereum::pb::eth::v2 as eth;
#[cfg(target_arch = "wasm32")]
use tiny_keccak::{Hasher, Keccak};

use crate::clmm::{ModuleContext, extract_swaps};
use crate::pb::dex::{RawSwap, RawSwaps, SwapRecord, SwapRecords};

const DEX_NAME: &str = "aerodrome_slipstream";
const CHAIN: &str = "base";

#[substreams::handlers::map]
pub fn map_aerodrome_swaps(params: String, block: eth::Block) -> Result<SwapRecords, Error> {
    let raw_block = block.clone();
    let raw_swaps = extract_swaps(
        &params,
        raw_block,
        ModuleContext {
            dex: DEX_NAME,
            chain: CHAIN,
        },
    )?;

    enrich_aerodrome_swaps(block, raw_swaps)
}

fn enrich_aerodrome_swaps(block: eth::Block, swaps: RawSwaps) -> Result<SwapRecords, Error> {
    let mut records = SwapRecords::default();
    let mut fee_cache: HashMap<String, String> = HashMap::new();

    for swap in swaps.swaps.into_iter() {
        let (pre_sqrt, pre_liq, pre_tick) = extract_pre_swap_state(&block, &swap);
        let fee_ratio = fee_cache
            .entry(swap.pool_address.clone())
            .or_insert_with(|| fetch_pool_fee_ratio(&swap.pool_address).unwrap_or_default())
            .clone();

        records.records.push(SwapRecord {
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
            gas_used: swap.gas_used,
            gas_price: swap.gas_price.clone(),
            gas_fee: swap.gas_fee.clone(),
            fee_ratio,
        });
    }

    Ok(records)
}

const SLOT_PRICE_TICK_KEY: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,
];
const SLOT_LIQUIDITY_KEY: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16,
];

fn extract_pre_swap_state(block: &eth::Block, swap: &RawSwap) -> (String, String, i64) {
    let default = || (String::new(), String::new(), 0i64);

    let Some(pool_address) = decode_hex(&swap.pool_address) else {
        substreams::log::info!(
            "failed to decode pool address {} for tx {}",
            swap.pool_address,
            swap.tx_hash
        );
        return default();
    };

    let Some(tx_hash) = decode_hex(&swap.tx_hash) else {
        substreams::log::info!("failed to decode tx hash {}", swap.tx_hash);
        return default();
    };

    let Some(trx) = block
        .transaction_traces
        .iter()
        .find(|trx| trx.hash.as_slice() == tx_hash.as_slice())
    else {
        return default();
    };

    let slot_price_tick =
        find_storage_change(trx, pool_address.as_slice(), &SLOT_PRICE_TICK_KEY, swap.ordinal);
    let slot_liquidity =
        find_storage_change(trx, pool_address.as_slice(), &SLOT_LIQUIDITY_KEY, swap.ordinal);

    let mut pre_tick = swap.tick;
    let mut pre_sqrt = swap.sqrt_price_x96.clone();
    if let Some(change) = slot_price_tick {
        let state = decode_slot0_state(&change.old_value);
        pre_tick = state.tick;
        pre_sqrt = state.sqrt_price_x96;
    }

    let mut pre_liquidity = swap.liquidity.clone();
    if let Some(change) = slot_liquidity {
        pre_liquidity = decode_packed_uint128(&change.old_value);
    }

    (pre_sqrt, pre_liquidity, pre_tick)
}

fn decode_packed_uint128(value: &[u8]) -> String {
    let padded = pad_to_32_bytes(value);
    let lower = &padded[16..32];
    BigUint::from_bytes_be(lower).to_string()
}

fn find_storage_change<'a>(
    tx: &'a eth::TransactionTrace,
    address: &[u8],
    key: &[u8; 32],
    max_ordinal: u64,
) -> Option<&'a eth::StorageChange> {
    // DEBUG helper: log the keys we see when we cannot find a match.
    let target_hex = hex::encode(address);
    let key_hex = hex::encode(key);
    let debug_mode =
        key_hex == "0000000000000000000000000000000000000000000000000000000000000000";

    let mut found = None;
    let mut best_ordinal = 0;

    for call in &tx.calls {
        for change in &call.storage_changes {
            if change.address.as_slice() != address {
                continue;
            }

            if debug_mode {
                substreams::log::info!(
                    "[DEBUG] Found key for {}: {} (ordinal: {}, max: {})",
                    target_hex,
                    hex::encode(&change.key),
                    change.ordinal,
                    max_ordinal
                );
            }

            if change.key.as_slice() != key {
                continue;
            }
            if max_ordinal != 0 && change.ordinal > max_ordinal {
                continue;
            }

            if change.ordinal >= best_ordinal {
                best_ordinal = change.ordinal;
                found = Some(change);
            }
        }
    }

    if debug_mode && found.is_none() {
        substreams::log::info!(
            "[DEBUG] slot not found for {} (max ordinal: {})",
            target_hex,
            max_ordinal
        );
    }

    found
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

fn fetch_pool_fee_ratio(pool_address: &str) -> Option<String> {
    let address = decode_hex(pool_address)?;
    let fee_ppm = fetch_pool_fee_ppm(&address)?;
    Some(format_fee_ratio_from_ppm(fee_ppm))
}

#[cfg(target_arch = "wasm32")]
fn fetch_pool_fee_ppm(address: &[u8]) -> Option<u32> {
    use substreams_ethereum::pb::eth::rpc::{RpcCall, RpcCalls};
    use substreams_ethereum::rpc;

    let selector = fee_selector();
    let call = RpcCall {
        to_addr: address.to_vec(),
        data: selector.to_vec(),
    };
    let responses = rpc::eth_call(&RpcCalls { calls: vec![call] });
    let response = responses.responses.first()?;
    if response.failed {
        return None;
    }
    decode_u32_return(&response.raw)
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_pool_fee_ppm(_address: &[u8]) -> Option<u32> {
    None
}

#[cfg(target_arch = "wasm32")]
fn decode_u32_return(data: &[u8]) -> Option<u32> {
    if data.len() < 4 {
        return None;
    }
    let slice = &data[data.len() - 4..];
    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    Some(u32::from_be_bytes(buf))
}

#[cfg(target_arch = "wasm32")]
fn fee_selector() -> [u8; 4] {
    let mut output = [0u8; 32];
    let mut keccak = Keccak::v256();
    keccak.update(b"fee()");
    keccak.finalize(&mut output);
    [output[0], output[1], output[2], output[3]]
}

fn format_fee_ratio_from_ppm(fee: u32) -> String {
    const SCALE: u32 = 1_000_000;
    let int_part = fee / SCALE;
    let remainder = fee % SCALE;
    if remainder == 0 {
        return int_part.to_string();
    }
    let mut frac = format!("{:06}", remainder);
    while frac.ends_with('0') {
        frac.pop();
    }
    if frac.is_empty() {
        int_part.to_string()
    } else {
        format!("{int_part}.{frac}")
    }
}
