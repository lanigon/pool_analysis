use std::collections::HashSet;

use anyhow::{Context, anyhow};
use ethabi::{ParamType, Token, decode};
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use once_cell::sync::Lazy;
use substreams::log;
use tiny_keccak::{Hasher, Keccak};

use crate::pb::dex::{RawSwap, RawSwaps};
use substreams_ethereum::pb::eth::v2 as eth;

type HandlerResult<T> = Result<T, substreams::errors::Error>;

const STANDARD_SWAP_SIGNATURE: &str = "Swap(address,address,int256,int256,uint160,uint128,int24)";
const PANCAKE_SWAP_SIGNATURE: &str =
    "Swap(address,address,int256,int256,uint160,uint128,int24,uint128,uint128)";

static STANDARD_SWAP_TOPIC: Lazy<[u8; 32]> = Lazy::new(|| {
    let mut output = [0u8; 32];
    let mut keccak = Keccak::v256();
    keccak.update(STANDARD_SWAP_SIGNATURE.as_bytes());
    keccak.finalize(&mut output);
    output
});

static PANCAKE_SWAP_TOPIC: Lazy<[u8; 32]> = Lazy::new(|| {
    let mut output = [0u8; 32];
    let mut keccak = Keccak::v256();
    keccak.update(PANCAKE_SWAP_SIGNATURE.as_bytes());
    keccak.finalize(&mut output);
    output
});

#[derive(Clone, Copy)]
pub struct ModuleContext {
    pub dex: &'static str,
    pub chain: &'static str,
}

pub fn extract_swaps(
    params: &str,
    block: eth::Block,
    ctx: ModuleContext,
) -> HandlerResult<RawSwaps> {
    let filter = PoolFilter::from_params(params)?;
    let timestamp = block
        .header
        .as_ref()
        .and_then(|header| header.timestamp.as_ref())
        .map(|timestamp| timestamp.seconds.max(0) as u64)
        .unwrap_or_default();
    let block_hash = to_hex(&block.hash);
    let block_number = block.number;

    let mut response = RawSwaps::default();

    for trx in block.transaction_traces {
        let tx_hash = to_hex(&trx.hash);
        let (gas_price, gas_used, gas_fee) = extract_gas_metrics(&trx);
        let Some(receipt) = trx.receipt else { continue };

        for log in receipt.logs.into_iter() {
            if !filter.matches(&log.address) {
                continue;
            }

            let Some(kind) = detect_swap_kind(&log) else {
                continue;
            };

            if log.topics.len() < 3 {
                continue;
            }

            match decode_swap(&log, kind) {
                Ok(decoded) => {
                    let ordinal = log.ordinal;
                    response.swaps.push(RawSwap {
                        dex: ctx.dex.to_string(),
                        chain: ctx.chain.to_string(),
                        pool_address: to_hex(&log.address),
                        tx_hash: tx_hash.clone(),
                        sender: decoded.sender,
                        recipient: decoded.recipient,
                        amount0: decoded.amount0,
                        amount1: decoded.amount1,
                        sqrt_price_x96: decoded.sqrt_price_x96,
                        liquidity: decoded.liquidity,
                        tick: decoded.tick,
                        block_number,
                        block_hash: block_hash.clone(),
                        block_timestamp: timestamp,
                        log_index: log.index,
                        ordinal,
                        fee_ratio: decoded.fee_ratio.clone(),
                        gas_price: gas_price.clone(),
                        gas_used,
                        gas_fee: gas_fee.clone(),
                    });
                }
                Err(err) => {
                    log::debug!("Skip log decode error: {:?}", err);
                }
            }
        }
    }
    // println!("response: {:?}", response);
    Ok(response)
}

struct PoolFilter {
    addresses: HashSet<[u8; 20]>,
}

impl PoolFilter {
    fn from_params(params: &str) -> HandlerResult<Self> {
        let mut addresses = HashSet::new();
        let sanitized = params.replace('\n', ",").replace(';', ",");

        for entry in sanitized.split(',') {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                continue;
            }

            let address_str = trimmed
                .rsplit_once('=')
                .map(|(_, value)| value)
                .unwrap_or(trimmed);

            let decoded = decode_address(address_str)?;
            addresses.insert(decoded);
        }

        if addresses.is_empty() {
            return Err(anyhow!("no pool addresses passed via --params"));
        }

        Ok(Self { addresses })
    }

    fn matches(&self, address: &[u8]) -> bool {
        if address.len() != 20 {
            return false;
        }

        let mut key = [0u8; 20];
        key.copy_from_slice(address);
        self.addresses.contains(&key)
    }
}

fn decode_address(value: &str) -> HandlerResult<[u8; 20]> {
    let cleaned = value
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
    anyhow::ensure!(
        cleaned.len() == 40,
        "address {value} must have 40 hex chars"
    );
    let bytes = hex::decode(cleaned).with_context(|| format!("invalid address {value}"))?;
    let len = bytes.len();
    let raw: [u8; 20] = bytes
        .try_into()
        .map_err(|_| anyhow!("address {value} decoded into {len} bytes"))?;
    Ok(raw)
}

fn to_hex(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(bytes))
}

#[derive(Clone, Copy)]
enum SwapEventKind {
    Standard,
    Pancake,
}

struct DecodedSwap {
    sender: String,
    recipient: String,
    amount0: String,
    amount1: String,
    sqrt_price_x96: String,
    liquidity: String,
    tick: i64,
    fee_ratio: String,
}

fn detect_swap_kind(log: &eth::Log) -> Option<SwapEventKind> {
    let topic = log.topics.first()?;
    let bytes = topic.as_slice();
    if bytes == STANDARD_SWAP_TOPIC.as_slice() {
        Some(SwapEventKind::Standard)
    } else if bytes == PANCAKE_SWAP_TOPIC.as_slice() {
        Some(SwapEventKind::Pancake)
    } else {
        None
    }
}

fn decode_swap(log: &eth::Log, kind: SwapEventKind) -> HandlerResult<DecodedSwap> {
    let sender = topic_to_address(log.topics.get(1).context("missing sender topic in swap")?)?;
    let recipient = topic_to_address(
        log.topics
            .get(2)
            .context("missing recipient topic in swap")?,
    )?;

    let tokens = match kind {
        SwapEventKind::Standard => decode(
            &[
                ParamType::Int(256),
                ParamType::Int(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Int(256),
            ],
            log.data.as_ref(),
        )?,
        SwapEventKind::Pancake => decode(
            &[
                ParamType::Int(256),
                ParamType::Int(256),
                ParamType::Uint(160),
                ParamType::Uint(128),
                ParamType::Int(24),
                ParamType::Uint(128),
                ParamType::Uint(128),
            ],
            log.data.as_ref(),
        )?,
    };

    let amount0 = token_to_bigint(&tokens[0])?;
    let amount1 = token_to_bigint(&tokens[1])?;
    let sqrt_price_x96 = token_to_bigint(&tokens[2])?;
    let liquidity = token_to_bigint(&tokens[3])?;
    let tick_value = token_to_bigint(&tokens[4])?;
    let tick = tick_value
        .to_i64()
        .ok_or_else(|| anyhow!("tick value does not fit i64"))?;
    let fee_ratio = match kind {
        SwapEventKind::Standard => String::new(),
        SwapEventKind::Pancake => {
            let protocol_fee0 = token_to_bigint(&tokens[5])?;
            let protocol_fee1 = token_to_bigint(&tokens[6])?;
            derive_fee_ratio(&protocol_fee0, &protocol_fee1, &amount0, &amount1).unwrap_or_default()
        }
    };

    Ok(DecodedSwap {
        sender,
        recipient,
        amount0: amount0.to_string(),
        amount1: amount1.to_string(),
        sqrt_price_x96: sqrt_price_x96.to_string(),
        liquidity: liquidity.to_string(),
        tick,
        fee_ratio,
    })
}

fn derive_fee_ratio(
    protocol_fee0: &BigInt,
    protocol_fee1: &BigInt,
    amount0: &BigInt,
    amount1: &BigInt,
) -> Option<String> {
    compute_fee_ratio(protocol_fee0, amount0).or_else(|| compute_fee_ratio(protocol_fee1, amount1))
}

const FEE_RATIO_DECIMALS: usize = 10;

fn compute_fee_ratio(fee: &BigInt, amount: &BigInt) -> Option<String> {
    if fee.is_zero() || amount.is_zero() {
        return None;
    }
    let scale = BigInt::from(10u32).pow(FEE_RATIO_DECIMALS as u32);
    let scaled_fee = fee.abs() * &scale;
    let amount_abs = amount.abs();
    if amount_abs.is_zero() {
        return None;
    }
    let quotient = scaled_fee / amount_abs;
    Some(format_scaled_decimal(
        quotient.to_string(),
        FEE_RATIO_DECIMALS,
    ))
}

fn format_scaled_decimal(mut digits: String, scale: usize) -> String {
    if scale == 0 {
        return digits;
    }
    if digits.len() <= scale {
        let padding = scale + 1 - digits.len();
        let mut buf = String::with_capacity(scale + 1);
        for _ in 0..padding {
            buf.push('0');
        }
        buf.push_str(&digits);
        digits = buf;
    }
    let split = digits.len() - scale;
    let (int_part, frac_part) = digits.split_at(split);
    let int_part = if int_part.is_empty() { "0" } else { int_part };
    let frac = frac_part.trim_end_matches('0');
    if frac.is_empty() {
        int_part.to_string()
    } else {
        format!("{int_part}.{frac}")
    }
}

fn topic_to_address(topic: &[u8]) -> HandlerResult<String> {
    anyhow::ensure!(topic.len() == 32, "topic must be 32 bytes");
    let address = &topic[12..];
    Ok(to_hex(address))
}

fn token_to_bigint(token: &Token) -> HandlerResult<BigInt> {
    match token {
        Token::Int(value) => {
            let mut bytes = [0u8; 32];
            value.to_big_endian(&mut bytes);
            Ok(BigInt::from_signed_bytes_be(&bytes))
        }
        Token::Uint(value) => {
            let mut bytes = [0u8; 32];
            value.to_big_endian(&mut bytes);
            Ok(BigInt::from_bytes_be(Sign::Plus, &bytes))
        }
        _ => Err(anyhow!("unexpected token type")),
    }
}

fn extract_gas_metrics(trx: &eth::TransactionTrace) -> (String, u64, String) {
    let gas_price_bigint = trx.gas_price.as_ref().map(proto_bigint_to_bigint);
    let gas_price = gas_price_bigint
        .as_ref()
        .map(|value| value.to_string())
        .unwrap_or_default();
    let gas_used = trx.gas_used;
    let gas_fee = gas_price_bigint
        .as_ref()
        .map(|price| (price * BigInt::from(gas_used)).to_string())
        .unwrap_or_default();

    (gas_price, gas_used, gas_fee)
}

fn proto_bigint_to_bigint(value: &eth::BigInt) -> BigInt {
    BigInt::from_signed_bytes_be(value.bytes.as_slice())
}
