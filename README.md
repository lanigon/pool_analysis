# Cross-chain Arbitrage Assessment

This project studies real-world arbitrage activity between PancakeSwap V3 on BSC and Aerodrome V2 Slipstream on Base. We collect every swap for the ETH/USDC and ETH/USDT pools, load them into SQLite, and run detailed analytics inside `insights/` to understand cross-chain price gaps, trader behavior, and MEV opportunities.

## Runbook

1. **Install the Substreams CLI** following the official guide: <https://docs.substreams.dev/how-to-guides/installing-the-cli>.
2. **Initialize the SQLite databases** by running the Rust helper:
   ```bash
   cargo run --bin initial_db
   ```
   (or trigger the equivalent Substreams CLI pipeline) to seed `data/dex.db`, `data/chains.db`, `data/pools.db`, and `data/transactions.db`.
3. **Stream live data**: use the Substreams CLI to capture Aerodrome (Base) and PancakeSwap (BSC) ETH/USDC|USDT pool events, writing swaps, ticks, gas, and metadata into the SQLite files above.
4. **Set up Python tooling**: activate the repo’s virtualenv and install `insights/requirements.txt` (includes pandas, SQLAlchemy, matplotlib, seaborn, etc.).
5. **Run the analytics**:
   ```bash
   python -m insights.arbitrage_analysis
   ```
   The script now first compares the two pools’ spread trajectories (reporting ≥0.05% deviations), then marks individual trend-following arbitrage trades, aggregates them into competitor/arb-sender leaderboards, matches dual-chain windows, and prints address-level summaries before persisting charts under `insights/view/`.

## Data Pipeline

- Substreams simultaneously ingests swaps from PancakeSwap V3 (BSC) and Aerodrome V2 (Base) for ETH/USDC and ETH/USDT pools.
- Transactions land in `data/transactions.db`; pool metadata stays in `data/pools.db`. `read_transactions` joins these tables and annotates each row with `dex_name` and `chain`.

## Analysis Highlights

1. Analyze the BSC vs. Base pool spread paths first, flagging any trace where `|spread_pct| > 0.05%` along with counts and sample trades.
2. Sort swaps chronologically across chains, continuously align each trade with the latest price from the opposite chain via `pd.merge_asof(direction="backward")`, and mark trades whose price vs. reference exceeds the 0.05% threshold in the arbitrage direction.
3. Build competitor/arb-sender leaderboards from the flagged `is_arb_trade` rows (volume, `net_profit`, spread statistics, etc.).
4. Bucket arbitrage trades into 10-second windows, require both chains to contribute opposite-sided volume, and compute symmetry, high-spread groups, and net-flow metrics.
5. Deep-dive on a specific wallet (`0x43f9a7aec2a683c4cd6016f92ff76d5f3e7b44d3`): enumerate Base trades, compare BSC reference prices, track tick movements, and evaluate whether flipping the signal on the opposite chain would have been profitable.

### Generated Visuals (`insights/view/`)

![Daily volume comparison](insights/view/daily_volume_breakdown.png)
![Gap vs volume for 0x43f9a7ae…](insights/view/gap_vs_volume_0x43f9a7ae.png)
![Spread persistence vs duration](insights/view/spread_persistence_duration.png)

## What We Learned

- BSC/Base spreads exist but decay rapidly—typically ~10 seconds before arbitrage collapses the gap.
- Some wallets act like market makers, firing continuously when spreads appear; even with tiny tick impacts per trade, they maintain balanced signed volume as gaps emerge.
- Cross-chain match windows are rare (~2% of 10-second buckets), so most opportunities either disappear before both chains can trade or are too slim to cover gas.

## Why This Matters

- These traces offer first-hand evidence of cross-chain price formation, reversion speed, arbitrage costs (swap + gas), and market-making behavior.
- The data underpins real-time arbitrage monitors or MEV bot research (price prediction, liquidity readiness).
- The same workflow can be extended to new assets/chains or combined with MEV relay/Flashbots quotes to study priority and profitability.

