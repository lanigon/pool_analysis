use std::str::FromStr;

use alloy::primitives::{Address, Bytes, keccak256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::rpc::types::eth::{TransactionInput, TransactionRequest};
use alloy::transports::http::{Http, reqwest::Client as ReqwestClient};
use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use url::Url;

#[derive(Parser, Debug)]
#[command(
    name = "fetch_pool",
    about = "Call pool contracts via Alloy to read tick spacing/fee settings"
)]
struct Args {
    /// Chain name, e.g. bsc / base
    #[arg(long)]
    chain: String,
    /// Pool addresses (repeatable or comma separated)
    #[arg(long = "pool", value_delimiter = ',', required = true)]
    pools: Vec<String>,
    /// Optional custom RPC endpoint (defaults to public RPC)
    #[arg(long)]
    rpc_endpoint: Option<String>,
}

struct ChainConfig {
    name: &'static str,
    default_endpoint: &'static str,
}

impl ChainConfig {
    fn resolve(chain: &str) -> Result<Self> {
        match chain {
            "bsc" => Ok(Self {
                name: "BNB Smart Chain",
                default_endpoint: "https://bsc-dataseed.binance.org",
            }),
            "base" => Ok(Self {
                name: "Base",
                default_endpoint: "https://mainnet.base.org",
            }),
            other => bail!("Only bsc/base are supported, received {other}"),
        }
    }
}

#[derive(Debug)]
struct PoolState {
    tick_spacing: i32,
    fee: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let chain_key = args.chain.to_lowercase();
    let chain_cfg = ChainConfig::resolve(&chain_key)?;
    let endpoint = args
        .rpc_endpoint
        .as_deref()
        .unwrap_or(chain_cfg.default_endpoint);

    let url = Url::parse(endpoint).context("invalid RPC endpoint")?;
    let provider = ProviderBuilder::new().on_http(url);

    println!("Chain: {} ({})", chain_cfg.name, chain_key);
    println!("RPC: {}", endpoint);

    for pool in args.pools.iter().filter(|addr| !addr.trim().is_empty()) {
        match Address::from_str(pool) {
            Ok(address) => match fetch_pool_state(&provider, address).await {
                Ok(state) => {
                    println!("\n=== {} ===", pool);
                    println!("Tick spacing: {}", state.tick_spacing);
                    println!("Fee (ppm): {}", state.fee);
                }
                Err(err) => {
                    eprintln!("\nPool {} call failed: {err}", pool);
                }
            },
            Err(err) => {
                eprintln!("Failed to parse address {}: {err}", pool);
            }
        }
    }

    Ok(())
}

async fn fetch_pool_state<P>(provider: &P, address: Address) -> Result<PoolState>
where
    P: Provider<Http<ReqwestClient>> + Sync,
{
    let tick_spacing = {
        let call_data = encode_selector("tickSpacing()");
        let output = eth_call(provider, address, call_data).await?;
        decode_i32(output.as_ref())?
    };

    let fee = {
        let call_data = encode_selector("fee()");
        let output = eth_call(provider, address, call_data).await?;
        decode_u32(output.as_ref())?
    };

    Ok(PoolState { tick_spacing, fee })
}

async fn eth_call<P>(provider: &P, to: Address, data: Vec<u8>) -> Result<Bytes>
where
    P: Provider<Http<ReqwestClient>> + Sync,
{
    let request = TransactionRequest::default()
        .to(to)
        .input(TransactionInput::new(Bytes::from(data)));

    provider.call(&request).await.context("eth_call failed")
}

fn decode_i32(data: &[u8]) -> Result<i32> {
    if data.len() < 4 {
        bail!("tickSpacing response length is invalid: {}", data.len());
    }
    let slice = &data[data.len() - 4..];
    let bytes: [u8; 4] = slice
        .try_into()
        .map_err(|_| anyhow!("Failed to parse tickSpacing bytes"))?;
    let value = i32::from_be_bytes(bytes);
    Ok(value)
}

fn decode_u32(data: &[u8]) -> Result<u32> {
    if data.len() < 4 {
        bail!("fee response length is invalid: {}", data.len());
    }
    let slice = &data[data.len() - 4..];
    let bytes: [u8; 4] = slice
        .try_into()
        .map_err(|_| anyhow!("Failed to parse fee bytes"))?;
    let value = u32::from_be_bytes(bytes);
    Ok(value)
}

fn encode_selector(signature: &str) -> Vec<u8> {
    let hash = keccak256(signature.as_bytes());
    hash[..4].to_vec()
}
