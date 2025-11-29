use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use clap::{Args, Parser, Subcommand, ValueEnum};
use prost::Message;

use crate::db_conn::{DbConnManager, TransactionInput};
use crate::pb::dex::{SwapRecord, SwapRecords};

#[derive(Parser, Debug)]
#[command(
    name = "fetch_data",
    about = "Inspect local SQLite data or trigger Substreams runs"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CommandKind>,
}

#[derive(Subcommand, Debug)]
enum CommandKind {
    /// Print dex/chain/pool/transaction tables
    ShowDb,
    /// Execute Substreams with the provided chain/DEX/pool parameters
    Run(RunArgs),
}

#[derive(Args, Debug, Clone)]
struct RunArgs {
    /// Preset target (auto-configures dex/chain)
    #[arg(long, value_enum)]
    target: Option<RunTarget>,
    /// DEX name (required when --target is omitted)
    #[arg(long)]
    dex: Option<String>,
    /// Chain (required when --target is omitted)
    #[arg(long)]
    chain: Option<String>,
    /// Pool addresses (repeatable/comma separated); falls back to data/lindenshore.db when omitted
    #[arg(long = "pool", alias = "pools", value_delimiter = ',')]
    pools: Vec<String>,
    /// Path to the substreams CLI binary
    #[arg(long, default_value = "substreams")]
    substreams_bin: String,
    /// Path to the manifest or SPKG
    #[arg(long, default_value = "substream.yaml")]
    manifest: String,
    /// Endpoint, e.g. mainnet.eth.streamingfast.io:443
    #[arg(long)]
    endpoint: Option<String>,
    /// Start block
    #[arg(long)]
    start_block: Option<i64>,
    /// Stop block
    #[arg(long)]
    stop_block: Option<i64>,
    /// Only print the resolved params/command
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

const PANCAKE_DEFAULT_ENDPOINT: &str = "bsc.streamingfast.io:443";
// const PANCAKE_DEFAULT_START_BLOCK: i64 = 66_940_828;
const PANCAKE_DEFAULT_START_BLOCK: i64 = 69_489_653;
const PANCAKE_DEFAULT_STOP_BLOCK: i64 = 69_659_653;
const AERODROME_DEFAULT_ENDPOINT: &str = "base-mainnet.streamingfast.io:443";
const AERODROME_DEFAULT_START_BLOCK: i64 = 38_658_472;
const AERODROME_DEFAULT_STOP_BLOCK: i64 = 38_728_472;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RunTarget {
    Pancakeswap,
    Aerodrome,
}

impl RunTarget {
    fn dex_and_chain(&self) -> (&'static str, &'static str) {
        match self {
            RunTarget::Pancakeswap => ("PancakeSwap", "bsc"),
            RunTarget::Aerodrome => ("Aerodrome", "base"),
        }
    }
}
pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(CommandKind::ShowDb) => show_db().await,
        Some(CommandKind::Run(args)) => run_substreams(&args).await,
        None => auto_run().await,
    }
}

fn pancake_demo_args() -> RunArgs {
    RunArgs {
        target: Some(RunTarget::Pancakeswap),
        dex: None,
        chain: None,
        pools: Vec::new(),
        substreams_bin: "substreams".to_string(),
        manifest: "substream.yaml".to_string(),
        endpoint: Some(PANCAKE_DEFAULT_ENDPOINT.to_string()),
        start_block: Some(PANCAKE_DEFAULT_START_BLOCK),
        stop_block: Some(PANCAKE_DEFAULT_STOP_BLOCK),
        dry_run: false,
    }
}

fn aerodrome_demo_args() -> RunArgs {
    RunArgs {
        target: Some(RunTarget::Aerodrome),
        dex: None,
        chain: None,
        pools: Vec::new(),
        substreams_bin: "substreams".to_string(),
        manifest: "substream.yaml".to_string(),
        endpoint: Some(AERODROME_DEFAULT_ENDPOINT.to_string()),
        start_block: Some(AERODROME_DEFAULT_START_BLOCK),
        stop_block: Some(AERODROME_DEFAULT_STOP_BLOCK),
        dry_run: false,
    }
}

async fn show_db() -> Result<()> {
    let manager = DbConnManager::new()
        .await
        .context("failed to open SQLite databases")?;

    println!("==== DEX Contracts ====");
    for dex in manager.list_dex_contracts().await? {
        println!(
            "{} | {} | {:?} | {:?}",
            dex.dex_name, dex.contract_address, dex.contract_type, dex.description
        );
    }

    println!("\n==== Chains ====");
    for chain in manager.list_chains().await? {
        println!(
            "{} | {} | {:?}",
            chain.name, chain.chain_id, chain.rpc_endpoint
        );
    }

    println!("\n==== Pools ====");
    for pool in manager.list_pools().await? {
        println!(
            "{} | dex: {} | fee: {} | tick_spacing: {} | tokens: {} / {}",
            pool.pool_address, pool.dex_name, pool.fee, pool.tick_spacing, pool.token1, pool.token2
        );
    }

    println!("\n==== Transactions ====");
    for tx in manager.list_transactions().await? {
        println!(
            "{} | pool {} | sender {} | pre {} -> post {} | ticks {} -> {} | liquidity {:?} | amt0 {} | amt1 {} | pre_liq {:?} | fee_ratio {:?} | gas_total {} | dir {} | ts {} | block {}",
            tx.tx_hash,
            tx.pool_address,
            tx.sender_address,
            tx.pre_sqrt_price,
            tx.post_sqrt_price,
            tx.pre_tick,
            tx.current_tick,
            tx.current_liquidity,
            tx.amount0,
            tx.amount1,
            tx.pre_liquidity,
            tx.fee_ratio,
            tx.gas_price,
            tx.direction,
            tx.timestamp,
            tx.block_number
        );
    }

    Ok(())
}

async fn auto_run() -> Result<()> {
    let manager = DbConnManager::new()
        .await
        .context("failed to open SQLite databases")?;

    let has_pancake = !manager.list_pools_by_dex("PancakeSwap").await?.is_empty();
    let has_aerodrome = !manager.list_pools_by_dex("Aerodrome").await?.is_empty();

    drop(manager);

    let mut ran_any = false;
    if has_pancake {
        println!("Detected PancakeSwap pools; starting PancakeSwap fetch");
        let args = pancake_demo_args();
        run_substreams(&args).await?;
        ran_any = true;
    }
    if has_aerodrome {
        println!("Detected Aerodrome pools; starting Aerodrome fetch");
        let args = aerodrome_demo_args();
        run_substreams(&args).await?;
        ran_any = true;
    }

    if !ran_any {
        println!("No PancakeSwap or Aerodrome pools found locally; running both by default");
        let pancake_args = pancake_demo_args();
        run_substreams(&pancake_args).await?;
        let aerodrome_args = aerodrome_demo_args();
        run_substreams(&aerodrome_args).await?;
    }

    Ok(())
}

async fn run_substreams(args: &RunArgs) -> Result<()> {
    let (dex_name, chain_name) = match args.target {
        Some(target) => target.dex_and_chain(),
        None => {
            let dex = args
                .dex
                .as_deref()
                .ok_or_else(|| anyhow!("Specify a DEX via --target or --dex/--chain"))?;
            let chain = args
                .chain
                .as_deref()
                .ok_or_else(|| anyhow!("Specify a chain via --target or --dex/--chain"))?;
            (dex, chain)
        }
    };

    let spec = resolve_module(dex_name, chain_name)?;
    let manager = DbConnManager::new()
        .await
        .context("failed to open SQLite databases")?;

    let mut pool_records = Vec::new();
    if args.pools.is_empty() {
        pool_records = manager.list_pools_by_dex(spec.db_name).await?;
    }

    let mut pools = if pool_records.is_empty() {
        args.pools.clone()
    } else {
        pool_records
            .iter()
            .map(|pool| pool.pool_address.clone())
            .collect::<Vec<_>>()
    };

    pools.retain(|addr| !addr.trim().is_empty());
    if pools.is_empty() {
        return Err(anyhow!(
            "No pool addresses provided and no {} pools found in SQLite",
            spec.db_name
        ));
    }

    println!("Target DEX: {} | Chain: {}", spec.db_name, chain_name);
    println!(
        "Pool count: {} | addresses: {}",
        pools.len(),
        pools.join(",")
    );

    let params = format!("{}={}", spec.param_module, pools.join(","));

    println!("Substreams params: {params}");

    if args.dry_run {
        println!("--dry-run specified, skipping execution");
        return Ok(());
    }

    let defaults = args.target.and_then(target_defaults);
    let endpoint = args
        .endpoint
        .clone()
        .or_else(|| defaults.map(|cfg| cfg.endpoint.to_string()));
    let start_block = args
        .start_block
        .or_else(|| defaults.map(|cfg| cfg.start_block));
    let stop_block = args
        .stop_block
        .or_else(|| defaults.map(|cfg| cfg.stop_block));

    let mut command = Command::new(&args.substreams_bin);
    command.arg("run");
    // command.arg("--production-mode");
    if let Some(endpoint) = &endpoint {
        command.arg("--endpoint").arg(endpoint);
    }
    if let Some(start) = start_block {
        command.arg("--start-block").arg(start.to_string());
    }
    if let Some(stop) = stop_block {
        command.arg("--stop-block").arg(stop.to_string());
    }
    command.arg("--limit-processed-blocks").arg("200000");

    command.arg(&args.manifest);
    command.arg(spec.module);
    command.arg("--params").arg(&params);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::inherit());

    println!("Executing command: {:?}", command);

    let mut child = command.spawn().context("failed to run substreams CLI")?;
    let mut captured = Vec::new();
    if let Some(mut stdout) = child.stdout.take() {
        let mut handle = io::stdout();
        let mut buffer = [0u8; 8 * 1024];
        loop {
            let read = stdout
                .read(&mut buffer)
                .context("failed to read substreams stdout")?;
            if read == 0 {
                break;
            }
            handle
                .write_all(&buffer[..read])
                .context("failed to write substreams stdout")?;
            handle.flush().ok();
            captured.extend_from_slice(&buffer[..read]);
        }
    }

    let status = child
        .wait()
        .context("failed to wait for substreams CLI completion")?;

    if !status.success() {
        return Err(anyhow!("substreams CLI exited with status {status}"));
    }

    let mut decoded_records = decode_swap_records_from_stdout(&captured)?;
    let pool_fee_map = if spec.db_name == "PancakeSwap" {
        Some(
            fetch_pool_fee_ratios(&manager, &pools)
                .await
                .context("failed to load pool metadata")?,
        )
    } else {
        None
    };
    if let Some(fees) = &pool_fee_map {
        for record in decoded_records.iter_mut() {
            let key = normalize_address(&record.pool_address);
            if let Some(ratio) = fees.get(&key) {
                record.fee_ratio = ratio.clone();
            }
        }
    }
    if decoded_records.is_empty() {
        println!("No SwapRecord data decoded");
    } else {
        println!("Decoded {} SwapRecord entries:", decoded_records.len());
        persist_swap_records(&manager, &decoded_records).await?;
        // print_swap_records(&decoded_records);
    }

    Ok(())
}

struct ModuleSpec {
    module: &'static str,
    param_module: &'static str,
    db_name: &'static str,
}

#[derive(Clone, Copy)]
struct TargetDefaults {
    endpoint: &'static str,
    start_block: i64,
    stop_block: i64,
}

fn resolve_module(dex: &str, chain: &str) -> Result<ModuleSpec> {
    let dex_lower = dex.to_lowercase();
    let chain_lower = chain.to_lowercase();

    let spec = match (dex_lower.as_str(), chain_lower.as_str()) {
        ("pancakeswap", "bsc") | ("pancakeswap v3", "bsc") | ("pancakeswap_v3", "bsc") => {
            ModuleSpec {
                module: "map_pancake_swaps_enriched",
                param_module: "map_pancake_swaps",
                db_name: "PancakeSwap",
            }
        }
        ("aerodrome", "base") | ("aerodrome v2", "base") | ("aerodrome_clmm", "base") => {
            ModuleSpec {
                module: "map_aerodrome_swaps",
                param_module: "map_aerodrome_swaps",
                db_name: "Aerodrome",
            }
        }
        _ => {
            return Err(anyhow!("Unsupported dex={dex} chain={chain} combination"));
        }
    };

    Ok(spec)
}

fn target_defaults(target: RunTarget) -> Option<TargetDefaults> {
    match target {
        RunTarget::Pancakeswap => Some(TargetDefaults {
            endpoint: PANCAKE_DEFAULT_ENDPOINT,
            start_block: PANCAKE_DEFAULT_START_BLOCK,
            stop_block: PANCAKE_DEFAULT_STOP_BLOCK,
        }),
        RunTarget::Aerodrome => Some(TargetDefaults {
            endpoint: AERODROME_DEFAULT_ENDPOINT,
            start_block: AERODROME_DEFAULT_START_BLOCK,
            stop_block: AERODROME_DEFAULT_STOP_BLOCK,
        }),
    }
}

fn decode_swap_records_from_stdout(stdout: &[u8]) -> Result<Vec<SwapRecord>> {
    let text = std::str::from_utf8(stdout).context("substreams CLI output is not valid UTF-8")?;
    let mut records = Vec::new();
    let marker = "\"@bytes\": \"";

    for line in text.lines() {
        let trimmed = line.trim();
        let Some(pos) = trimmed.find(marker) else {
            continue;
        };
        let start = pos + marker.len();
        let remaining = &trimmed[start..];
        let Some(end_rel) = remaining.find('"') else {
            continue;
        };
        let encoded = &remaining[..end_rel];
        if encoded.is_empty() {
            continue;
        }
        let payload = general_purpose::STANDARD
            .decode(encoded)
            .with_context(|| format!("failed to decode @bytes payload: {encoded}"))?;
        let message = SwapRecords::decode(payload.as_slice())
            .context("failed to decode SwapRecords protobuf message")?;
        records.extend(message.records);
    }

    Ok(records)
}

fn print_swap_records(records: &[SwapRecord]) {
    for record in records {
        println!(
            " - pool {} | tx {} | sender {} -> {} | amount0 {} | amount1 {}",
            record.pool_address,
            record.tx_hash,
            record.sender,
            record.recipient,
            record.amount0,
            record.amount1,
        );
        println!(
            "   sqrt {} -> {} | ticks {} -> {} | liquidity {} -> {} | fee_ratio {} | block {} | ts {}",
            record.pre_sqrt_price_x96,
            record.post_sqrt_price_x96,
            record.pre_tick,
            record.post_tick,
            record.pre_liquidity,
            record.post_liquidity,
            record.fee_ratio,
            record.block_number,
            record.block_timestamp,
        );
        println!(
            "   gas_price {} | gas_used {} | gas_fee {} | log_index {}",
            record.gas_price, record.gas_used, record.gas_fee, record.log_index,
        );
    }
}

async fn fetch_pool_fee_ratios(
    manager: &DbConnManager,
    pools: &[String],
) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    for addr in pools.iter() {
        let normalized = normalize_address(addr);
        if map.contains_key(&normalized) {
            continue;
        }
        if let Some(pool) = manager.find_pool_by_address(addr).await? {
            map.insert(normalized, format_fee_ratio_from_bps(pool.fee));
        }
    }
    Ok(map)
}

fn normalize_address(value: &str) -> String {
    value.to_lowercase()
}

fn format_fee_ratio_from_bps(fee: i64) -> String {
    const SCALE: i64 = 1_000_000;
    let sign = if fee < 0 { "-" } else { "" };
    let abs_fee = fee.abs();
    let int_part = abs_fee / SCALE;
    let remainder = abs_fee % SCALE;
    if remainder == 0 {
        format!("{sign}{int_part}")
    } else {
        let mut frac = format!("{:06}", remainder);
        while frac.ends_with('0') {
            frac.pop();
        }
        format!("{sign}{int_part}.{frac}")
    }
}

async fn persist_swap_records(manager: &DbConnManager, records: &[SwapRecord]) -> Result<()> {
    for record in records {
        if manager
            .find_transaction_by_hash(&record.tx_hash)
            .await?
            .is_some()
        {
            continue;
        }

        let input = swap_record_to_transaction(record);
        if let Err(err) = manager.add_transaction(input).await {
            eprintln!(
                "failed to insert transaction {} into SQLite: {err}",
                record.tx_hash
            );
        }
    }
    Ok(())
}

fn swap_record_to_transaction(record: &SwapRecord) -> TransactionInput {
    TransactionInput {
        pool_address: record.pool_address.clone(),
        tx_hash: record.tx_hash.clone(),
        sender_address: record.sender.clone(),
        pre_sqrt_price: record.pre_sqrt_price_x96.clone(),
        post_sqrt_price: record.post_sqrt_price_x96.clone(),
        amount0: record.amount0.clone(),
        amount1: record.amount1.clone(),
        pre_liquidity: optional_string(&record.pre_liquidity),
        fee_ratio: optional_string(&record.fee_ratio),
        pre_tick: record.pre_tick,
        current_tick: record.post_tick,
        current_liquidity: optional_string(&record.post_liquidity),
        gas_price: record.gas_fee.clone(),
        direction: detect_direction(&record.amount0, &record.amount1),
        timestamp: record.block_timestamp as i64,
        block_number: record.block_number as i64,
    }
}

fn optional_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn detect_direction(amount0: &str, amount1: &str) -> String {
    let amt0_neg = amount0.trim_start().starts_with('-');
    let amt1_neg = amount1.trim_start().starts_with('-');
    match (amt0_neg, amt1_neg) {
        (true, false) => "token0_to_token1".into(),
        (false, true) => "token1_to_token0".into(),
        _ => "unknown".into(),
    }
}
