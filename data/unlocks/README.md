# Token Unlock Events Dataset

Generated: 2026-09-09

## Files

### 1. `defillama_unlocks.csv` (primary, 4.5 MB)
**Source:** DefiLlama `/unlocks` page — Next.js SSG JSON fetched from
`https://defillama.com/_next/data/{buildId}/unlocks.json` (public, no auth required).

**Coverage:**
- 39,579 total rows across 370 protocols
- Date range: 2011-10-07 to 2026-10-09
- In-target window (2021-01 to 2026-08): 38,203 rows, 348 distinct tokens
- Tokens with USDT perp futures on Binance/Bybit (manually mapped): 87 tokens, 16,237 rows

**Fields:**

| Field | Description |
|---|---|
| `date` | Event date (UTC, YYYY-MM-DD) derived from Unix timestamp |
| `token` | Protocol/token name as labelled by DefiLlama |
| `slug` | DefiLlama protocol slug |
| `gecko_id` | CoinGecko ID when available |
| `category` | Unlock category: `insiders`, `privateSale`, `ecosystem`, `airdrop`, `farming`, `liquidity`, `staking`, `publicSale`, `community`, `noncirculating`, `burned`, `Uncategorized` |
| `unlock_type` | `cliff` (single lump-sum) or `linear` (rate change event) |
| `unlock_amount` | For cliff: tokens released. For linear: new weekly/monthly rate after change |
| `rate_duration_days` | For linear events: the rate period in days (e.g. 30 = monthly rate) |
| `circ_supply` | Circulating supply at time of data fetch (current, not historical) |
| `max_supply` | Maximum/total supply |
| `pct_of_circ` | unlock_amount / circ_supply * 100 (approximate; circ_supply is current, not at event date) |
| `ticker` | Manually mapped ticker symbol (NULL where mapping is unknown) |
| `source` | Always `defillama` |
| `usd_value` | NULL (not provided by this source) |

**Known gaps / caveats:**
- `circ_supply` and `pct_of_circ` reflect current supply, not supply at event date, so pct_of_circ is approximate for historical events.
- `linear` events represent rate-change points, not individual daily unlock amounts; to get daily amounts divide `unlock_amount` by `rate_duration_days`.
- DefiLlama's unlock adapter coverage is uneven: some major tokens (e.g. SOL, ETH, BTC) have few events because their emissions are protocol-level inflation rather than discrete vesting.
- USD value is not provided by this endpoint.
- `ticker` column populated for ~87 perp-futures tokens via manual slug-to-ticker mapping; ~250 tokens have no ticker assigned.

---

### 2. `6mv_token_vesting.csv` (supplementary observed daily, 1.2 MB)
**Source:** 6th-Man-Ventures token-vesting GitHub repo
(`https://github.com/6th-Man-Ventures/token-vesting`), Excel files
`data/daily_unlocks/daily_tables_internal.xlsx` and `data/daily_unlocks/daily_tables_external.xlsx`.

**Coverage:**
- 11,363 rows
- 18 distinct tokens: NYM, UNI (Uniswap), EUL (Euler), GAL (Project Galaxy), BIT (BitDAO),
  TORN (Tornado Cash), FORT (Forta), POOL (PoolTogether), MC (Merit Circle), MANA (Decentraland),
  LOOKS (LooksRare), SWEAT (Sweatcoin), CRV (Curve), SWISE (StakeWise), GMT (STEPN),
  GEL (Gelato), LDO (Lido), FIL (Filecoin), APT (Aptos)
- Date range: 2020-08-13 to 2023-04-27
- "internal" = team/investor allocations; "external" = public/community allocations

**Fields:**

| Field | Description |
|---|---|
| `date` | Date of unlock event (UTC, YYYY-MM-DD) |
| `token` | Token symbol (upper-case) |
| `group` | Vesting recipient group (e.g. "Team and Investors", "Backers", "Community") |
| `amount` | Tokens unlocked on this date |
| `circulating_supply` | Historical circulating supply at event date |
| `pct_of_circulating` | amount / circulating_supply * 100 (historical, accurate) |
| `price_usd` | Token price (USD) on that date |
| `usd_value` | amount * price_usd |
| `source_file` | `internal` (team/investor) or `external` (public/community) |

**Known gaps:** Only 18 tokens, study ended 2023-04. Covers only protocols studied in the 6MV research piece (2022-2023 focus).

---

### 3. `6mv_schedules_derived.csv` (supplementary schedule-based, 2.5 MB)
**Source:** Same 6th-Man-Ventures repo, `data/vesting_data.py` — vesting schedule metadata used to compute unlock events algorithmically.

**Coverage:**
- 25,792 rows
- 19 distinct tokens (same set as above plus Aptos from public allocations)
- Date range: 2017-08-10 to 2032-11-11 (includes Filecoin historical and Aptos future)
- Split into "private" (team/investor) and "public" (community/treasury) categories

**Fields:**

| Field | Description |
|---|---|
| `date` | Computed unlock date |
| `token` | Token symbol |
| `group` | Vesting recipient group |
| `category` | `private` or `public` |
| `event_type` | `cliff`, `linear_daily`, `linear_weekly`, `linear_monthly`, `linear_quarterly` |
| `unlock_amount` | Computed tokens per period |
| `total_supply` | Token total supply |
| `pct_of_total` | unlock_amount / total_supply * 100 |
| `vesting_frequency` | Original vesting frequency from schedule |

**Known gaps:** Prices and USD values not available. Circulating supply not available. Computed from schedule metadata, so may deviate from actual on-chain events if projects modified schedules.

---

### 4. `unlocks_events.csv` (merged, 6.1 MB)
Combines `defillama_unlocks.csv` (filtered to 2021-01..2026-08) and `6mv_token_vesting.csv` (filtered to same window). Rows from 6mv supplement the 18 tokens not well-covered by DefiLlama's cliff/linear model.

**Coverage:**
- 49,081 rows
- Date range: 2021-01-01 to 2026-08-31
- 366 distinct tokens, 128 with ticker symbol assigned
- Sources: `defillama` (38,203 rows), `6mv_observed` (10,878 rows)

**Year distribution:**

| Year | Rows |
|---|---|
| 2021 | 5,985 |
| 2022 | 9,966 |
| 2023 | 7,376 |
| 2024 | 7,675 |
| 2025 | 10,775 |
| 2026 (Jan-Aug) | 7,304 |

**Schema:** Superset of DefiLlama fields plus `source` and `usd_value` columns.

---

## Sources Tried — Status Summary

| Source | Status | Notes |
|---|---|---|
| `api.llama.fi/emissions` | PAYWALLED (HTTP 402) | Requires Pro subscription |
| `api.llama.fi/emission/<protocol>` | PAYWALLED (HTTP 402) | Per-protocol endpoint also 402 |
| DefiLlama frontend SSG JSON | **OK** | 8.8 MB JSON fetched from `/_next/data/…/unlocks.json`; 370 protocols, 39,579 events |
| 6th-Man-Ventures GitHub repo | **OK** | Excel daily tables + vesting_data.py; 18 tokens, 2020-2023 |
| tokenomist.ai CSV export | PAYWALLED | Requires Pro plan; CSV download is paid feature |
| tokenomist.ai API endpoints | NOT FOUND (404/403) | No public API found |
| cryptorank.io/token-unlock | BLOCKED (403/401) | Cloudflare-protected, no public API |
| Keyrock 16,000-event study | NO DATA LINK | Blog post only; raw dataset not published |
| Dune Analytics | REQUIRES API KEY (401) | No public unauthenticated export |
| CoinMarketCap unlock API | NOT FOUND (404) | Endpoint does not exist |
| Nansen research | BLOCKED (308/redirect) | Login required |
| DefiLlama GitHub (emissions-adapters) | NOT FOUND (404) | No such repo in the DefiLlama org |
| Binance/Bybit REST APIs | BLOCKED (451/403) | Geo/IP restriction in sandbox |

## How to Use

```python
import pandas as pd

df = pd.read_csv('unlocks_events.csv')

# Filter to tokens with USDT perp on Binance/Bybit
perp_df = df[df['ticker'].notna()]

# Get cliff events only (discrete large unlocks)
cliffs = df[df['unlock_type'] == 'cliff']

# Get events with known USD size
with_usd = df[df['usd_value'].notna()]  # only 6mv_observed rows
```

## Known Gaps

1. **Pct-of-circulating is approximate for DefiLlama rows** — supply figures are current, not historical.
2. **USD value** — only available for the 18 tokens in 6mv_token_vesting.csv (2020-2023).
3. **Linear events** — DefiLlama `linear` rows are rate-change events, not daily amounts; convert with `unlock_amount / rate_duration_days`.
4. **Token coverage** — DefiLlama covers ~370 protocols but ticker-to-perp mapping is manual and covers ~87 tokens; many DeFi protocols listed do not have USDT perp futures.
5. **Pre-2022 data is sparse** — most DeFi Llama adapters started tracking from late 2021/2022; 2021 coverage reflects only tokens in the 6MV dataset.
6. **No data for** BTC, SOL staking inflation, ETH supply changes before Merge — these are monetary policy events, not vesting unlocks.
