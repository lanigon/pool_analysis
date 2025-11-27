use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    lindenshore_substreams::cli::run().await
}
