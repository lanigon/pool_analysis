# Cross-chain Arbitrage Assessment

This project studies real-world arbitrage activity between PancakeSwap V3 on BSC and Aerodrome V2 on Base. We collect every swap for the ETH/USDC and ETH/USDT pools, load them into SQLite, and run detailed analytics inside `insights/` to understand cross-chain price gaps, trader behavior, and MEV opportunities.

## Runbook

1. **Install and compile Substreams**
   ```bash
   cargo install substreams
   ```
   (or follow the upstream Substreams README if you need additional prerequisites.)
2. **Initialize the SQLite databases** by running `substreams/src/bin/initial_db.rs`, which seeds `data/dex.db`, `data/chains.db`, `data/pools.db`, and `data/transactions.db`.
3. **Stream live data**: use the Substreams CLI to capture Aerodrome (Base) and PancakeSwap (BSC) ETH/USDC|USDT pool events, writing swaps, ticks, gas, and metadata into the SQLite files above.
4. **Set up Python tooling**: activate a virtualenv at the repo root and install `insights/requirements.txt` (includes pandas, SQLAlchemy, matplotlib, seaborn, etc.).
5. **Run the analytics**:
   ```bash
   python -m insights.arbitrage_analysis
   # or
   python insights/arbitrage_analysis.py
   ```
   The script loads every recorded transaction, performs cross-chain spread/arbitrage analysis, and saves plots under `insights/view/`.

## Data Pipeline

- A Substreams sink simultaneously ingests swaps from PancakeSwap V3 (BSC) and Aerodrome V2 (Base) for ETH/USDC and ETH/USDT pools.
- Transactions land in `data/transactions.db`; pool metadata stays in `data/pools.db`. `read_transactions` joins these tables and annotates each row with `dex_name` and `chain`.

## Analysis Highlights

- Sort swaps chronologically across chains, continuously align each trade with the latest price from the opposite chain via `pd.merge_asof(direction="backward")`.
- Apply a 0.05% spread threshold; classify buys/sells as real cross-chain arbitrage by computing `spread_pct`, `revenue`, `gas_fee_usd`, and `net_profit`.
- Aggregate 10-second windows to find periods where both chains show volume; compute symmetry and duration metrics for these cross-chain bursts.
- Deep-dive on frequent arbitrageurs (e.g., address `0x43f9a7aec2a683c4cd6016f92ff76d5f3e7b44d3`): report gap distributions, tick movements, signed volume, scatter plots, and descriptive stats.

### Generated Visuals (`insights/view/`)

- `daily_volume_breakdown.png` – daily-side-by-side volume comparison for Base vs BSC flows.
- `gap_vs_volume_0x43f9a7ae.png` – gap vs. trade-size scatter for address `0x43f9a7aec2a...`.
- `spread_persistence_duration.png` – relationship between spread size and how long it persists before reverting.

## What We Learned

- BSC/Base spreads exist but decay rapidly—typically ~10 seconds before arbitrage collapses the gap.
- Some wallets act like market makers, firing continuously when spreads appear; even with tiny tick impacts per trade, they maintain balanced signed volume as gaps emerge.
- Spread persistence, cross-chain match ratios, and gas-priority analysis signal which flows are genuine multi-chain arbitrage vs. single-chain liquidity mining.

## Why This Matters

- These traces offer first-hand evidence of cross-chain price formation, reversion speed, arbitrage costs (swap + gas), and market-making behavior.
- The data underpins real-time arbitrage monitors (alerting when spreads breach thresholds) or MEV bot research (price prediction, liquidity readiness).
- The same workflow can be extended to new assets/chains or combined with MEV relay/Flashbots quotes to study priority and profitability.

## Next Steps

- Expand coverage to more pools or chains, especially where altcoin spreads may persist longer.
- Feed the spread metrics into alerting dashboards or automated strategies, then back-test against subsequent market moves.
- Cross-reference Substreams-derived events with mempool data to evaluate latency advantages for top-performing bots.
