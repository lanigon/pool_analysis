use anyhow::{Context, Result};
use clap::Parser;
use lindenshore_substreams::db_conn::{
    ChainInput, DbConnManager, DexContractInput, PoolInput, TransactionInput,
};

#[derive(Parser, Debug)]
#[command(name = "initial_db", about = "Seed or reset the data/*.db SQLite files")]
struct Args {
    /// Insert sample data
    #[arg(long, default_value_t = false)]
    seed: bool,
    /// Drop all tables before seeding
    #[arg(long, default_value_t = false)]
    reset: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let manager = DbConnManager::new()
        .await
        .context("failed to initialize SQLite databases")?;

    println!("SQLite databases initialized under data/*.db");

    if args.reset {
        clear_all_tables(&manager).await?;
        println!("All tables cleared");
    }

    if args.seed {
        seed_example_data(&manager).await?;
        println!("Sample data inserted");
    } else if !args.reset {
        println!("Use --seed to insert sample data; combine with --reset to wipe first");
    }

    Ok(())
}

async fn clear_all_tables(manager: &DbConnManager) -> Result<()> {
    manager
        .drop_all_transactions()
        .await
        .context("failed to clear transactions")?;
    manager
        .drop_all_pools()
        .await
        .context("failed to clear pools")?;
    manager
        .drop_all_dex_contracts()
        .await
        .context("failed to clear dex metadata")?;
    manager
        .drop_all_chains()
        .await
        .context("failed to clear chains")?;
    Ok(())
}

async fn seed_example_data(manager: &DbConnManager) -> Result<()> {
    let chains = vec![
        ChainInput {
            chain_id: "56".into(),
            name: "BNB Smart Chain".into(),
            rpc_endpoint: Some("https://bsc-dataseed.binance.org".into()),
        },
        ChainInput {
            chain_id: "8453".into(),
            name: "Base".into(),
            rpc_endpoint: Some("https://mainnet.base.org".into()),
        },
    ];
    for chain in chains {
        insert_chain_if_missing(manager, chain).await?;
    }

    let dexes = vec![
        DexContractInput {
            dex_name: "PancakeSwap".into(),
            contract_address: "0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865".into(),
            contract_type: Some("factory".into()),
            description: Some("PancakeSwap v3 factory on BSC".into()),
        },
        DexContractInput {
            dex_name: "Aerodrome".into(),
            contract_address: "0xb2cc224c1c9feE385f8ad6a55b4d94E92359DC59".into(),
            contract_type: Some("factory".into()),
            description: Some("Aerodrome factory on Base".into()),
        },
    ];
    for dex in dexes {
        insert_dex_if_missing(manager, dex).await?;
    }

    let pools = vec![
        PoolInput {
            pool_address: "0x539e0EBfffd39e54A0f7E5F8FEc40ade7933A664".into(),
            dex_name: "PancakeSwap".into(),
            fee: 500,
            tick_spacing: 10,
            token1: "ETH".into(),
            token2: "USDC".into(),
        },
        PoolInput {
            pool_address: "0x9f599f3d64a9d99ea21e68127bb6ce99f893da61".into(),
            dex_name: "PancakeSwap".into(),
            fee: 100,
            tick_spacing: 1,
            token1: "ETH".into(),
            token2: "USDT".into(),
        },
        PoolInput {
            pool_address: "0xbe141893e4c6ad9272e8c04bab7e6a10604501a5".into(),
            dex_name: "PancakeSwap".into(),
            fee: 500,
            tick_spacing: 10,
            token1: "ETH".into(),
            token2: "USDT".into(),
        },
        PoolInput {
            pool_address: "0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59".into(),
            dex_name: "Aerodrome".into(),
            fee: 500,
            tick_spacing: 100,
            token1: "WETH".into(),
            token2: "USDC".into(),
        },
        PoolInput {
            pool_address: "0x9785ef59e2b499fb741674ecf6faf912df7b3c1b".into(),
            dex_name: "Aerodrome".into(),
            fee: 500,
            tick_spacing: 100,
            token1: "WETH".into(),
            token2: "USDT".into(),
        },
    ];
    for pool in pools {
        insert_pool_if_missing(manager, pool).await?;
    }

    // let txs = vec![TransactionInput {
    //     pool_address: "0xPancakePoolCAKEBNB".into(),
    //     tx_hash: "0xPancakeSwapTx".into(),
    //     sender_address: "0xSenderPancake".into(),
    //     pre_sqrt_price: "179228162514264337593543950336".into(),
    //     post_sqrt_price: "179228162514264337593543950512".into(),
    //     amount0: "-250000000000000000".into(),
    //     amount1: "625000000000000000".into(),
    //     pre_liquidity: Some("100000000000000000000".into()),
    //     fee_ratio: Some("0.0025".into()),
    //     pre_tick: -88000,
    //     current_tick: -87940,
    //     current_liquidity: Some("100000000000000000000".into()),
    //     gas_price: "5100000000".into(),
    //     direction: "token0_to_token1".into(),
    //     timestamp: 1_705_000_000,
    //     block_number: 32_000_000,
    // }];
    // for tx in txs {
    //     insert_transaction_if_missing(manager, tx).await?;
    // }

    Ok(())
}

async fn insert_chain_if_missing(manager: &DbConnManager, chain: ChainInput) -> Result<()> {
    if manager.find_chain_by_id(&chain.chain_id).await?.is_none() {
        manager.add_chain(chain).await?;
    }
    Ok(())
}

async fn insert_dex_if_missing(manager: &DbConnManager, dex: DexContractInput) -> Result<()> {
    if manager
        .find_dex_contract_by_address(&dex.contract_address)
        .await?
        .is_none()
    {
        manager.add_dex_contract(dex).await?;
    }
    Ok(())
}

async fn insert_pool_if_missing(manager: &DbConnManager, pool: PoolInput) -> Result<()> {
    if manager
        .find_pool_by_address(&pool.pool_address)
        .await?
        .is_none()
    {
        manager.add_pool(pool).await?;
    }
    Ok(())
}

async fn insert_transaction_if_missing(
    manager: &DbConnManager,
    tx: TransactionInput,
) -> Result<()> {
    if manager
        .find_transaction_by_hash(&tx.tx_hash)
        .await?
        .is_none()
    {
        manager.add_transaction(tx).await?;
    }
    Ok(())
}
