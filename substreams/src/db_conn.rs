use anyhow::{Context, Result};
use sea_orm::NotSet;
use sea_orm::entity::prelude::*;
use sea_orm::{
    ColumnTrait, ConnectionTrait, Database, DatabaseConnection, DbBackend, QueryFilter, QueryOrder,
    Schema, Set, Statement,
};

pub const MAIN_DB_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/lindenshore.db");

#[derive(Clone)]
pub struct DbConnManager {
    db: DatabaseConnection,
}

impl DbConnManager {
    pub async fn new() -> Result<Self> {
        let db = connect_sqlite(MAIN_DB_PATH).await?;

        init_dex_schema(&db).await?;
        init_chain_schema(&db).await?;
        init_pools_schema(&db).await?;
        init_transactions_schema(&db).await?;

        Ok(Self { db })
    }

    pub async fn add_dex_contract(&self, input: DexContractInput) -> Result<dex_metadata::Model> {
        let active_model = dex_metadata::ActiveModel {
            id: NotSet,
            dex_name: Set(input.dex_name),
            contract_address: Set(input.contract_address),
            contract_type: Set(input.contract_type),
            description: Set(input.description),
        };

        active_model
            .insert(&self.db)
            .await
            .context("failed to add dex contract")
    }

    pub async fn add_chain(&self, input: ChainInput) -> Result<chains::Model> {
        let active_model = chains::ActiveModel {
            id: NotSet,
            chain_id: Set(input.chain_id),
            name: Set(input.name),
            rpc_endpoint: Set(input.rpc_endpoint),
        };

        active_model
            .insert(&self.db)
            .await
            .context("failed to add chain metadata")
    }

    pub async fn add_pool(&self, input: PoolInput) -> Result<pools::Model> {
        let active_model = pools::ActiveModel {
            id: NotSet,
            pool_address: Set(input.pool_address),
            dex_name: Set(input.dex_name),
            fee: Set(input.fee),
            tick_spacing: Set(input.tick_spacing),
            token1: Set(input.token1),
            token2: Set(input.token2),
        };

        active_model
            .insert(&self.db)
            .await
            .context("failed to add pool metadata")
    }

    pub async fn add_transaction(&self, input: TransactionInput) -> Result<transactions::Model> {
        let active_model = transactions::ActiveModel {
            id: NotSet,
            pool_address: Set(input.pool_address),
            tx_hash: Set(input.tx_hash),
            sender_address: Set(input.sender_address),
            pre_sqrt_price: Set(input.pre_sqrt_price),
            post_sqrt_price: Set(input.post_sqrt_price),
            amount0: Set(input.amount0.clone()),
            amount1: Set(input.amount1.clone()),
            pre_liquidity: Set(input.pre_liquidity.clone()),
            fee_ratio: Set(input.fee_ratio.clone()),
            pre_tick: Set(input.pre_tick),
            current_tick: Set(input.current_tick),
            current_liquidity: Set(input.current_liquidity),
            gas_price: Set(input.gas_price),
            direction: Set(input.direction),
            timestamp: Set(input.timestamp),
            block_number: Set(input.block_number),
        };

        active_model
            .insert(&self.db)
            .await
            .context("failed to add transaction record")
    }

    pub async fn list_dex_contracts(&self) -> Result<Vec<dex_metadata::Model>> {
        dex_metadata::Entity::find()
            .all(&self.db)
            .await
            .context("failed to list dex contracts")
    }

    pub async fn list_chains(&self) -> Result<Vec<chains::Model>> {
        chains::Entity::find()
            .all(&self.db)
            .await
            .context("failed to list chains")
    }

    pub async fn list_pools(&self) -> Result<Vec<pools::Model>> {
        pools::Entity::find()
            .all(&self.db)
            .await
            .context("failed to list pools")
    }

    pub async fn list_pools_by_dex(&self, dex_name: &str) -> Result<Vec<pools::Model>> {
        pools::Entity::find()
            .filter(pools::Column::DexName.eq(dex_name))
            .all(&self.db)
            .await
            .context("failed to list pools for dex")
    }

    pub async fn list_transactions(&self) -> Result<Vec<transactions::Model>> {
        transactions::Entity::find()
            .order_by_asc(transactions::Column::Timestamp)
            .all(&self.db)
            .await
            .context("failed to list transactions")
    }

    pub async fn find_dex_contract_by_address(
        &self,
        address: &str,
    ) -> Result<Option<dex_metadata::Model>> {
        dex_metadata::Entity::find()
            .filter(dex_metadata::Column::ContractAddress.eq(address))
            .one(&self.db)
            .await
            .context("failed to fetch dex contract")
    }

    pub async fn find_chain_by_id(&self, chain_id: &str) -> Result<Option<chains::Model>> {
        chains::Entity::find()
            .filter(chains::Column::ChainId.eq(chain_id))
            .one(&self.db)
            .await
            .context("failed to fetch chain metadata")
    }

    pub async fn find_pool_by_address(&self, address: &str) -> Result<Option<pools::Model>> {
        pools::Entity::find()
            .filter(pools::Column::PoolAddress.eq(address))
            .one(&self.db)
            .await
            .context("failed to fetch pool metadata")
    }

    pub async fn find_transaction_by_hash(
        &self,
        tx_hash: &str,
    ) -> Result<Option<transactions::Model>> {
        transactions::Entity::find()
            .filter(transactions::Column::TxHash.eq(tx_hash))
            .one(&self.db)
            .await
            .context("failed to fetch transaction")
    }

    pub async fn drop_all_transactions(&self) -> Result<()> {
        transactions::Entity::delete_many()
            .exec(&self.db)
            .await
            .context("failed to truncate transactions")?;
        Ok(())
    }

    pub async fn drop_all_pools(&self) -> Result<()> {
        pools::Entity::delete_many()
            .exec(&self.db)
            .await
            .context("failed to truncate pools")?;
        Ok(())
    }

    pub async fn drop_all_dex_contracts(&self) -> Result<()> {
        dex_metadata::Entity::delete_many()
            .exec(&self.db)
            .await
            .context("failed to truncate dex metadata")?;
        Ok(())
    }

    pub async fn drop_all_chains(&self) -> Result<()> {
        chains::Entity::delete_many()
            .exec(&self.db)
            .await
            .context("failed to truncate chains")?;
        Ok(())
    }
}

pub struct DexContractInput {
    pub dex_name: String,
    pub contract_address: String,
    pub contract_type: Option<String>,
    pub description: Option<String>,
}

pub struct ChainInput {
    pub chain_id: String,
    pub name: String,
    pub rpc_endpoint: Option<String>,
}

pub struct PoolInput {
    pub pool_address: String,
    pub dex_name: String,
    pub fee: i64,
    pub tick_spacing: i64,
    pub token1: String,
    pub token2: String,
}

pub struct TransactionInput {
    pub pool_address: String,
    pub tx_hash: String,
    pub sender_address: String,
    pub pre_sqrt_price: String,
    pub post_sqrt_price: String,
    pub amount0: String,
    pub amount1: String,
    pub pre_liquidity: Option<String>,
    pub fee_ratio: Option<String>,
    pub pre_tick: i64,
    pub current_tick: i64,
    pub current_liquidity: Option<String>,
    pub gas_price: String,
    pub direction: String,
    pub timestamp: i64,
    pub block_number: i64,
}

async fn connect_sqlite(path: &str) -> Result<DatabaseConnection> {
    let url = format!("sqlite:{}?mode=rwc", path);
    Database::connect(&url)
        .await
        .with_context(|| format!("failed to connect to {}", path))
}

async fn init_dex_schema(conn: &DatabaseConnection) -> Result<()> {
    create_table_if_needed(conn, dex_metadata::Entity).await
}

async fn init_chain_schema(conn: &DatabaseConnection) -> Result<()> {
    create_table_if_needed(conn, chains::Entity).await
}

async fn init_pools_schema(conn: &DatabaseConnection) -> Result<()> {
    create_table_if_needed(conn, pools::Entity).await
}

async fn init_transactions_schema(conn: &DatabaseConnection) -> Result<()> {
    create_table_if_needed(conn, transactions::Entity).await?;
    ensure_transaction_columns(conn).await?;
    drop_legacy_transaction_columns(conn).await
}

async fn ensure_transaction_columns(conn: &DatabaseConnection) -> Result<()> {
    add_column_if_missing(conn, "fee_ratio", "TEXT").await?;
    add_column_if_missing(conn, "sender_address", "TEXT").await?;
    add_column_if_missing(conn, "current_liquidity", "TEXT").await?;
    add_column_if_missing(conn, "amount0", "TEXT").await?;
    add_column_if_missing(conn, "amount1", "TEXT").await?;
    add_column_if_missing(conn, "pre_liquidity", "TEXT").await?;
    fill_missing_amount_columns(conn).await?;
    Ok(())
}

async fn drop_legacy_transaction_columns(conn: &DatabaseConnection) -> Result<()> {
    let has_amount = table_has_column(conn, "amount").await?;
    let has_fee_amount = table_has_column(conn, "fee_amount").await?;
    if !has_amount && !has_fee_amount {
        return Ok(());
    }

    let builder = DbBackend::Sqlite;
    conn.execute(Statement::from_string(
        builder,
        "ALTER TABLE transactions RENAME TO transactions_backup",
    ))
    .await
    .context("failed to rename legacy transactions table")?;

    create_table_if_needed(conn, transactions::Entity)
        .await
        .context("failed to recreate transactions table")?;

    let copy_sql = r#"
        INSERT INTO transactions (
            pool_address,
            tx_hash,
            sender_address,
            pre_sqrt_price,
            post_sqrt_price,
            amount0,
            amount1,
            pre_liquidity,
            fee_ratio,
            pre_tick,
            current_tick,
            current_liquidity,
            gas_price,
            direction,
            timestamp,
            block_number
        )
        SELECT
            pool_address,
            tx_hash,
            sender_address,
            pre_sqrt_price,
            post_sqrt_price,
            COALESCE(amount0, amount),
            COALESCE(amount1, amount),
            pre_liquidity,
            fee_ratio,
            pre_tick,
            current_tick,
            current_liquidity,
            gas_price,
            direction,
            timestamp,
            block_number
        FROM transactions_backup
    "#;
    conn.execute(Statement::from_string(builder, copy_sql))
        .await
        .context("failed to copy legacy transactions data")?;

    conn.execute(Statement::from_string(
        builder,
        "DROP TABLE transactions_backup",
    ))
    .await
    .context("failed to drop legacy transactions_backup table")?;

    Ok(())
}

async fn table_has_column(conn: &DatabaseConnection, column: &str) -> Result<bool> {
    let builder = DbBackend::Sqlite;
    let stmt = Statement::from_string(builder, "PRAGMA table_info('transactions')");
    let rows = conn
        .query_all(stmt)
        .await
        .context("failed to query transaction columns")?;
    for row in rows {
        let name: String = row.try_get("", "name")?;
        if name.eq_ignore_ascii_case(column) {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn fill_missing_amount_columns(conn: &DatabaseConnection) -> Result<()> {
    let builder = DbBackend::Sqlite;
    for column in ["amount0", "amount1", "pre_liquidity"] {
        let sql = format!("UPDATE transactions SET {column} = '' WHERE {column} IS NULL");
        let statement = Statement::from_string(builder, sql);
        conn.execute(statement)
            .await
            .context("failed to backfill transaction amount columns")?;
    }
    Ok(())
}

async fn add_column_if_missing(
    conn: &DatabaseConnection,
    column: &str,
    column_type: &str,
) -> Result<()> {
    let builder = DbBackend::Sqlite;
    let sql = format!("ALTER TABLE transactions ADD COLUMN {column} {column_type}");
    let statement = Statement::from_string(builder, sql);
    match conn.execute(statement).await {
        Ok(_) => Ok(()),
        Err(err) => {
            if err.to_string().contains("duplicate column name") {
                Ok(())
            } else {
                Err(err.into())
            }
        }
    }
}

async fn create_table_if_needed<E>(conn: &DatabaseConnection, entity: E) -> Result<()>
where
    E: EntityTrait,
{
    let builder = conn.get_database_backend();
    let schema = Schema::new(builder);
    let mut stmt = schema.create_table_from_entity(entity);
    stmt.if_not_exists();
    conn.execute(builder.build(&stmt))
        .await
        .context("failed to create table")?;
    Ok(())
}

mod dex_metadata {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "dex_metadata")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub dex_name: String,
        #[sea_orm(unique)]
        pub contract_address: String,
        pub contract_type: Option<String>,
        pub description: Option<String>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

mod chains {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "chains")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub chain_id: String,
        pub name: String,
        pub rpc_endpoint: Option<String>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

mod pools {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "pools")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        #[sea_orm(unique)]
        pub pool_address: String,
        pub dex_name: String,
        pub fee: i64,
        pub tick_spacing: i64,
        pub token1: String,
        pub token2: String,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}

mod transactions {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "transactions")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub id: i32,
        pub pool_address: String,
        #[sea_orm(unique)]
        pub tx_hash: String,
        pub sender_address: String,
        pub pre_sqrt_price: String,
        pub post_sqrt_price: String,
        pub amount0: String,
        pub amount1: String,
        pub pre_liquidity: Option<String>,
        pub fee_ratio: Option<String>,
        pub pre_tick: i64,
        pub current_tick: i64,
        pub current_liquidity: Option<String>,
        pub gas_price: String,
        pub direction: String,
        pub timestamp: i64,
        pub block_number: i64,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}

    impl ActiveModelBehavior for ActiveModel {}
}
