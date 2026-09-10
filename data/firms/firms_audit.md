# Crypto Prop Firm Audit — Belarus API Bot Trader
**Audit date:** 2026-09-09
**Use case:** Market-neutral bot, 100–160 simultaneous positions, rebalanced every 8 h with ~100 limit orders (PostOnly then market), holding 1–7 days. Bybit API preferred. Scale across up to 10 firms in parallel. Trader resident in Belarus (BY).

**Critical filter:** Only firms with a genuine exchange API (Bybit/OKX/Binance demo sub-account) allow the trader to run their own code. Firms using MT4/MT5, Match-Trader, cTrader, TradeLocker, or proprietary platforms are marked NOT EXCHANGE API.

---

## 1. HyroTrader

**Website:** https://hyrotrader.com  
**Sources seen:** hyrotrader.com (2026-09-09), FAQ crawl (2026-09-09)

### Checklist

**1. Account type / bots allowed**
Real exchange integration: Traders connect their own Bybit account via API keys (real Bybit USDT perpetuals, 700+ pairs). Also: Tealstreet terminal with real Bybit order book, and Cleo platform with Binance market data via a provisioned account. HyroTrader monitors the sub-account through the API connection.
- Challenge, Verification, and Funded stage all operate in a **simulated demo environment** (quote: *"The HyroTrader challenge, verification, and funded accounts operate in a simulated demo environment. After proving consistent performance, traders may be offered the opportunity to trade real capital."*)
- Bots/EAs: UNKNOWN from public pages — the FAQ lists a "Trading Restrictions" section with 5 articles but the content could not be extracted. Given the API-connection model, the firm has technical ability to monitor for automation. No explicit EA-allowed or EA-banned statement was found on the main site.
- No explicit mention of "HFT / latency arbitrage / exploitation of simulated environment" clauses found, but a Trading Restrictions section exists in FAQ.

**2. Restricted countries / KYC**
- Restricted countries: UNKNOWN — no country list visible on public pages. Bybit itself requires KYC which may exclude Belarus depending on Bybit's own policies.
- KYC: **Required before funded account activation** (quote from main FAQ schema: *"complete KYC and sign the funded trader agreement to activate your funded demo account"*). Payout provider: USDT/USDC crypto direct to wallet. No mention of Rise/Deel.
- Russia: UNKNOWN.

**3. Challenge structure — $10k account**
- Fee: UNKNOWN (refundable deposit model — the deposit is returned with the first payout, unlike a one-time fee). Exact amount for $10k not extracted.
- Phases: 1-step (minimum 5 trading days, 10% profit target) or 2-step (Phase 1: 5 days + 10%; Phase 2: Verification, 5% target).
- Daily drawdown: UNKNOWN — a "Swing Daily Drawdown Upgrade" is mentioned, implying a standard mode exists.
- Max drawdown: UNKNOWN.
- Min trading days: 5 (1-step), 5 Phase 1 + 5 Phase 2 (2-step).
- Time limit: None stated.
- Consistency rule: UNKNOWN.
- Per-trade loss limit: UNKNOWN.
- Min holding time: UNKNOWN (no 50-second rule mentioned, unlike Klein).
- News trading: UNKNOWN.
- Weekend holding: Implied allowed (crypto 24/7).

**4. Funded stage**
- Profit split: Start 80%, scale to 90% for sustained performance.
- Payout: On-demand, processed within 12 hours. First payout 1 day after first funded trade with min $100 profit. In USDT or USDC.
- Payout min: $100 profit threshold before first payout.
- Scaling: UNKNOWN specific plan.
- Max allocation: Up to $200,000.
- Max open positions: UNKNOWN.
- Funded account: Still simulated demo environment ("simulated demo environment" confirmed).

**5. Multi-account policy**
- UNKNOWN — no explicit multi-account rules published on public pages. Given API monitoring, running identical positions across multiple HyroTrader accounts would likely be detectable and at risk.

**6. Funding fees / flat-for-payout**
- Funding fees on perpetuals: UNKNOWN explicitly. Since the connection is to a real Bybit account (even if in demo mode), Bybit funding rates technically apply. Whether HyroTrader's P&L tracking includes them is unconfirmed. CRITICAL — must verify with support.
- Flat for payout: UNKNOWN.

**7. Reputation**
- Trustpilot: 4.7/5, 210 reviews (seen on hyrotrader.com, 2026-09-09).
- Founded: 2022 (claims "first direct exchange integration in crypto prop trading").
- Company registration: UNKNOWN from public pages.
- $5M+ paid to funded traders claimed.
- 1,700+ funded traders.
- No major Reddit/Twitter complaints identified in search results.

---

## 2. Mubite

**Website:** https://mubite.com  
**Sources:** mubite.com (2026-09-09), mubite.com/en/challengeRules (2026-09-09), allproptradingfirms.com/list/mubite-review (2026-06-27)

### Checklist

**1. Account type / bots allowed**
- Account type: Trader connects their **own Bybit account via API keys** (700+ USDT perpetuals, real market data, real order book, up to 1:100 leverage). Also: Cleo platform with Binance market data (360+ USDT futures), provisioned automatically.
- Funded stage: "simulated funded account" — all accounts are simulated but prices are live from Bybit/Binance.
- Bots/EAs: **Allowed** (quote from challengeRules page: *"Automated trading bots and EAs are allowed"*).
- Prohibited (exact text from challengeRules, 2026-09-09):
  - *"Latency arbitrage or HFT exploitation"*
  - *"Cross-account hedging between multiple accounts"*
  - *"Group trading or account mirroring"*
  - *"Over-leveraging beyond position limits"*
  - *"Tick scalping with manipulative intent"*
  - *"Traders may not copy or mirror trades between accounts unless approved under an official scaling plan."*
- No explicit "exploitation of simulated environment / non-market fills" clause found beyond the above.

**2. Restricted countries / KYC**
- Belarus: **UNKNOWN explicitly**. FAQ states: *"Due to regulatory requirements, there may be restrictions for traders from certain countries. Please check our Terms of Service or contact customer support for up-to-date information on country restrictions."* No specific country list found publicly. A trader with handle "Nik Russia" appears in testimonials, suggesting CIS traders may be accepted — but this is not confirmation.
- Russia: Testimonial "Nik Russia" present, suggesting Russia may be accepted. Not confirmed.
- KYC: **Required before any payout** (quote: *"Identity verification is required before any payout is processed."*). Payout method: crypto to wallet (no Rise/Deel mentioned; Mubite pays directly in crypto).

**3. Challenge structure — $10k account**
- Fee: **$110** (two-step, $10k account; refundable on first payout from funded account). One-step or Instant Funding also available at different fees.
- Phases (Two-Step): Phase 1 = 10% profit target; Phase 2 = 5% profit target.
- Daily drawdown: **5%** of starting balance (static, not trailing — example: $500 on $10k). Resets each day.
- Max drawdown: **8%** of starting balance (static). Add-on available to increase to 10% (+20% on fee).
- Min trading days: **10 days** (a trading day counts when at least one position is opened AND closed P&L exceeds 0.25% of starting balance). Can be removed with add-on.
- Time limit: **None** (unlimited).
- Consistency rule: **None** stated for standard challenges.
- Per-trade loss limit: **Max 3% of equity at trade open** (realized losses; no single partial close may realize more than 3%; cumulative net realized loss of a trade must never exceed 3% at its deepest point).
- Min holding time: None stated.
- News trading: Allowed (quote from challengeRules: *"News trading is permitted"*).
- Weekend holding: Allowed (quote: *"You can hold positions overnight and through weekends"*).

**4. Funded stage**
- Profit split: 80% standard; increases to 90% after 10% profit within 3 months, or 90% from start via add-on (+10% fee).
- Payout: 1st payout on-demand; subsequent payouts bi-weekly. No waiting period after verification.
- Payout min: UNKNOWN (on-demand implies no minimum other than having profit).
- Scaling plan: Cited to $1,000,000 (review source). Start at $10k, scale up to $200k standard max.
- Max allocation: $200,000 per account.
- Max open positions: **Cumulative exposure ≤ 3x initial account balance**; max position size per trade 2x initial balance. With 100–160 simultaneous positions this could bind — on a $10k account, 3x = $30k total notional, requiring small sizes per position.
- Funded account: Still simulated (demo environment, Bybit prices live).

**5. Multi-account policy**
- CRITICAL: *"Cross-account hedging between multiple accounts"* and *"Group trading or account mirroring"* are **explicitly prohibited** and result in account termination.
- *"Traders may not copy or mirror trades between accounts unless approved under an official scaling plan."*
- Running the **same bot on multiple Mubite accounts would directly violate this rule** as positions would mirror each other.
- Running the same bot on Mubite AND another firm: Mubite cannot technically detect this (different firms), but if the same Bybit sub-account is used for multiple evaluations, they would see it. Different Bybit accounts at different firms would be harder to detect.
- Number of accounts: UNKNOWN maximum stated.
- Account merging: Not mentioned.

**6. Funding fees / flat-for-payout**
- Funding fees: Since the trader connects their own Bybit account via API, and Bybit perpetuals charge/pay funding every 8 hours, **funding fees almost certainly apply** on positions held through 0:00/8:00/16:00 UTC. Not explicitly confirmed in Mubite's rules. CRITICAL — verify with support.
- Flat for payout: UNKNOWN — on-demand payout suggests positions need not be closed, but unconfirmed.

**7. Reputation**
- Trustpilot: **4.8/5, 126+ reviews** (mubite.com, 2026-09-09). Google: 4.7. Payout testimonials present and named.
- Founded: 2025 (per allproptradingfirms review, 2026-06-27). Very new.
- Company: **Mubite s.r.o., Školská 660/3, Nové Město, Praha 1, 110 00, Czech Republic. IČO: 23221551**.
- Verified payouts shown on website (named traders, amounts). Largest: $41,320.
- Red flags: Very young firm (founded 2025); limited long-term track record; Czech company with CIS-facing product.

---

## 3. Klein Funding

**Website:** https://kleinfunding.com  
**Sources:** kleinfunding.com (2026-09-09), kleinfunding.com/faqs/50-seconds-rule-bybit-only (2026-09-09), propvator.com/klein-funding/rules (checked 2026-07-29), kleinfunding.com/faqs/which-countries-are-restricted (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: Bybit sub-account connected via API (USDT perpetuals, up to 1:100 leverage) OR Cleo platform (Binance market data, up to 1:5 leverage — far lower). Standard/Flex/Cleo plans use Cleo. Instant Pro and some One-Step plans use Bybit.
- **Bots/EAs: STRICTLY PROHIBITED** (exact quote from propvator.com citing Klein rules, verified 2026-07-29): *"all trades must be initiated and managed by the trader manually and accounts using automated trading will be terminated. Expert advisors, bots, scripts and algorithms that execute trades without manual oversight are all covered."*
- **This makes Klein Funding completely unsuitable for an automated bot.**
- Prohibited: arbitrage, latency, front-running, tick scalping, group trading, exploiting demo environment. Copy trading between own accounts also prohibited.
- Additional Bybit-specific rule: **50-second minimum holding time** (exact quote from kleinfunding.com/faqs/50-seconds-rule-bybit-only, 2026-09-09): *"All trades on the Bybit platform must remain open for a minimum of 50 seconds before being closed. Any trade closed before the 50-second mark will be considered a rule violation. This rule applies exclusively to Bybit accounts and does not affect other platforms."*
- One account per household, device, IP, or VPS — sharing terminates all accounts.

**2. Restricted countries / KYC**
- Belarus: UNKNOWN. FAQ page states: *"Restricted countries may include those subject to international sanctions or where financial services regulations prevent us from operating. The specific list of restricted jurisdictions is maintained and updated regularly on our website."* No explicit list published in extracted content.
- Russia: UNKNOWN.
- KYC: UNKNOWN from sources found.

**3. Challenge structure — $10k account**
- Fee: Standard (Cleo) $10k: **$137.75**; Instant Pro (Bybit) $10k: **$255.55**.
- Phases: Standard/One-Step (1-phase, 6% target); Two-Step (2-phase); Flex (1-phase, 9% target, fixed trailing DD); Instant Pro (no challenge, 3 min days, scale to $2M).
- Daily drawdown: **3%** of starting balance (static). Resets at 12:05 AM UTC.
- Max drawdown: **6%** of starting balance (static equity floor). Some plans offer 6–10% range.
- Min trading days: 3 days with 0.5% profit each before first payout (Instant Pro). Standard requires 3 profitable days.
- Time limit: None on funded account.
- Consistency/Stability rule: No single day ≥ 30% of total profits (1-step) or ≥ 45% (2-step). Cleo plans have no stability rule.
- Per-trade loss: Not published.
- Min holding time: **50 seconds on Bybit** (violates rebalancing with fast limit orders if bot rebalances positions that could be open <50s).
- News trading: Allowed.
- Weekend holding: Allowed.

**4. Funded stage**
- Split: 70% standard (FAQ claims up to 90–100% possible).
- Payout: On-demand, 4–24 hours processing.
- Scaling: Double account at 10% profit (forfeiting half that profit). Instant Pro: scale to $2,000,000.
- Max allocation: Up to $300,000 virtual.
- Max open positions: Bounded by 1:100 leverage on balance.
- Funded: Still simulated.

**5. Multi-account policy**
- Copy trading prohibited, including between own accounts.
- One account per household/device/IP/VPS.
- **Running the same bot on multiple Klein accounts or sharing infrastructure would terminate all accounts.**

**6. Funding fees / flat-for-payout**
- Funding fees: If Bybit sub-account, likely applies. UNKNOWN confirmed.
- Flat for payout: UNKNOWN.

**7. Reputation**
- Trustpilot: **4.9/5** (stated on kleinfunding.com, 2026-09-09). Review count UNKNOWN.
- Founded: UNKNOWN.
- Red flags: Bots banned (fatal for this trader). 50-second rule. One-IP-per-account rule makes multi-firm strategy risky.

---

## 4. Breakout Prop

**Website:** https://breakoutprop.com  
**Sources:** breakoutprop.com (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **Proprietary platform** — NOT a Bybit/OKX/Binance API connection. Operated by Payward Oceanic Ltd. (POL), which is affiliated with Kraken. Traders receive a position in POL's book; they do not own or connect a real exchange account.
- Quote: *"you do not own any trading account...position, and hold no beneficial...proprietary interest in POL's assets or trades"*
- Bots: Not explicitly mentioned in FAQ. No EA-allowed or banned statement found. The platform appears to route orders internally: *"either (i) book entry and calculate... (ii) route the transaction... a market maker or exchange."*
- HFT/latency clause: Not explicitly stated but prop platform model limits this.
- API: No external exchange API for traders. The platform has its own mobile app.
- 60+ tradeable markets, up to 10x leverage on major assets (BTC, SP500, Silver, Oil).

**2. Restricted countries / KYC**
- Belarus: UNKNOWN. Quote: *"Regional restrictions apply."* No country list on public page.
- Russia: UNKNOWN.
- KYC: UNKNOWN from sources; payouts in USDC on Ethereum to wallet (no KYC provider mentioned).

**3. Challenge structure — $10k account**
- Fee: Products listed as Turbo from $20, Pro from $33, Classic from $45. Exact $10k fee: UNKNOWN (pricing table not fully extracted).
- Phases: **One step** (no minimum trading days, no time limit).
- Profit targets: 9–12% depending on product.
- Daily drawdown: **3%** of balance (resets at 00:30 UTC).
- Max drawdown: **3–6% static** (equity floor set at account start).
- Min trading days: **0** (can get funded in a single trade).
- Consistency rule: **None** (quote: *"No. There is no consistency rule on any Breakout account. Your best day can be 10x your average day."*).
- Per-trade loss: Not stated.
- Min holding time: Not stated.
- News trading: Allowed (quote: *"Yes. There are no news restrictions of any kind."*).
- Weekend holding: Allowed (quote: *"Hold overnight, over weekends, over holidays. There are no forced close requirements."*).

**4. Funded stage**
- Split: **80% standard; 90/10 available as upgrade at checkout**.
- Payout: On-demand 24/7, processed within hours. Min $50 after split. USDC on Ethereum.
- First payout: Same day as funding possible.
- Scaling: Max $200K across all funded accounts combined; multiple accounts allowed as long as total ≤ $200K.
- Max allocation: **$200K total per trader across all accounts**.
- Funded: Simulated (POL book entry).

**5. Multi-account policy**
- Multiple accounts and evaluations allowed simultaneously as long as combined funded capital ≤ $200K.
- Quote: *"avoid obvious issues like hedging across separate accounts."*
- No explicit copy-trading prohibition found on the public page.
- Running the same strategy on multiple Breakout accounts seems permitted as long as total capital ≤ $200K and no cross-account hedging.

**6. Funding fees / flat-for-payout**
- Funding fees: Since it is a proprietary platform (NOT a real Bybit account), funding rates are likely NOT charged/paid. The platform books P&L internally. UNKNOWN confirmed — need to verify.
- Flat for payout: No forced close requirement mentioned.

**7. Reputation**
- Trustpilot: **4.7/5, 1,000+ reviews** (breakoutprop.com, 2026-09-09).
- $60M+ paid to traders since launch. $50M also cited at another point on the page (likely different date of data).
- Operated by Payward Oceanic Ltd. — Kraken-affiliated structure.
- 160+ countries.
- Leaderboard public; largest lifetime payout $678,949.
- No major red flags found, but conflict of interest disclosed: *"because Breakout earns fees each time an evaluation trader fails and then re-purchases an evaluation, conflicts of interest..."* — standard prop firm disclosure.

---

## 5. Crypto Fund Trader (CFT)

**Website:** https://cryptofundtrader.com  
**Sources:** cryptofundtrader.com (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **Proprietary demo platform** — NOT a real Bybit API connection. CFT is a "Proud partner of BYBIT" and uses Bybit branding, but all trading is in their own simulated environment (quote: *"It's all demo, no real capital involved."*).
- Bots: UNKNOWN explicitly from extracted content.
- 900+ instruments including 715+ crypto pairs, forex, indices, stocks, commodities.
- Leverage: up to 1:100.

**2. Restricted countries / KYC**
- Belarus: UNKNOWN.
- Russia: UNKNOWN.
- KYC: UNKNOWN.

**3. Challenge structure — $10k account**
- Fee: UNKNOWN for $10k specifically (pricing table shows options but exact $10k fee not extracted).
- Options: Instant, 1-Phase, 2-Phase.
- 1-Phase: 10% profit target, 4% daily DD, 6% max trailing DD, no min days, indefinite time.
- 2-Phase: Phase 1 = 8% target, Phase 2 = 5% target; 5% daily DD, 10% max DD, no min days, indefinite time.
- Consistency rule: **40% rule** on BREAK funded accounts — no single day's profit may exceed 40% of total profits at payout time.
- Scale: Up to $1,280,000 (11 scaling levels starting at $10k → $20k → ... → $1.28M).
- Starting profit split: **50%** at $10k level, scaling to 90% at $160K+.

**4. Funded stage**
- Split: 50% → 90% (depends on account size level via scaling).
- Payout: UNKNOWN frequency; payouts of $486K/month claimed.
- Scaling: 11 levels from $10k to $1.28M; scale by hitting 10% target at each level (profit target doubles each step).
- Max allocation: $1,280,000.
- Funded: Simulated (demo capital).

**5. Multi-account policy**
- UNKNOWN.

**6. Funding fees / flat-for-payout**
- Funding fees: Proprietary platform, likely NOT real Bybit funding rates. UNKNOWN.
- Flat for payout: UNKNOWN.

**7. Reputation**
- $20,820,492 total paid to traders (cryptofundtrader.com, 2026-09-09).
- 56,000 traders educated.
- Trustpilot: UNKNOWN from extracted content.
- Partner with Bybit and C.A. Osasuna (Spanish football club). Founded: UNKNOWN.
- Red flags: 50% starting split is low; proprietary platform (not real Bybit API).

---

## 6. Alpha Capital Group

**Website:** https://alphacapitalgroup.uk  
**Sources:** alphacapitalgroup.uk (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **NOT exchange API** — uses MetaTrader 5, cTrader, DXTrade, TradeLocker (and upcoming Alpha Trader). Simulated institutional execution environment. NOT Bybit perpetuals.
- Bots: MT5 Expert Advisors explicitly supported (listed as platform feature).
- Simulated environment (quote: *"Evaluation accounts use simulated funds. After you pass, you stay in a simulated market environment as a Qualified Analyst."*).

**2. Restricted countries / KYC**
- Available in 140+ countries; claims 100+ countries on another section.
- Belarus: UNKNOWN — no country list extracted. Must check their published list.
- KYC: UNKNOWN from sources.

**3. Challenge structure — $10k account**
- Fee: UNKNOWN (pricing not extracted for $10k specifically).
- Phases: 1-step, multi-step options.
- Drawdown: UNKNOWN specific % from extracted content.
- Consistency: UNKNOWN.
- Platforms: MT5/cTrader/DXTrade.

**4. Funded stage**
- Split: 80% standard; 90% add-on (+10% on price).
- Payout: Bi-weekly or on-demand (different price tiers).
- Swap-free add-on: Available (adds ~10% to price) — relevant if funding fees are an issue.
- Max allocation: $200,000.
- $100M in performance fees paid.

**5. Multi-account policy**
- UNKNOWN.

**6. Funding fees / flat-for-payout**
- Swap-free add-on available, implying overnight swaps/rollovers ARE charged by default on standard accounts. MT5 swap model applies.
- Flat for payout: UNKNOWN.

**7. Reputation**
- 1.2 million traders claim.
- 100K+ Qualified Analysts.
- Trustpilot: UNKNOWN from extracted content.
- UK registered (alphacapitalgroup.uk). Connected to ACG Markets (broker), Alpha Futures, Alpha Prime.
- Red flags: NOT suitable for crypto perpetuals bot on Bybit API.

---

## 7. FXIFY

**Website:** https://fxify.com  
**Sources:** fxify.com (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **NOT exchange API** — uses MetaTrader 4/5, DXTrade, and other platforms. NOT Bybit API.
- Bots: **EAs Allowed** (explicitly listed as a feature). Martingale and Grid also allowed.
- Not crypto perpetuals on exchange.

**2. Restricted countries / KYC**
- 200 countries served.
- Belarus: UNKNOWN — no country list extracted.
- KYC: UNKNOWN from sources.

**3. Challenge structure — $10k account**
- Fee: $250 for $5k mentioned; $10k fee UNKNOWN.
- Phases: 1-Phase, 2-Phase, 3-Phase, Lightning (1-step, 5% target), Instant Funding.
- Leverage: 30:1 FX/Gold, 10:1 Indices, 5:1 Oil, 2:1 Stocks (low for crypto).
- Static and trailing drawdown options.
- Unlimited trading days.
- Consistency: UNKNOWN.

**4. Funded stage**
- Split: Up to 90%.
- Payout: On-demand from first funded trade. Min $50.
- Max allocation: $400,000.
- $40M+ paid.

**5. Multi-account policy**
- UNKNOWN.

**6. Funding fees / flat-for-payout**
- MT4/MT5 swap model applies. Overnight swaps charged by default.
- Flat for payout: UNKNOWN.

**7. Reputation**
- Trustpilot: **4.3/5, 6,201 reviews** (fxify.com, 2026-09-09).
- 4 global offices, 20+ years leadership experience.
- 250K+ active traders.
- Red flags: NOT Bybit API; low leverage for crypto vs. Bybit.

---

## 8. Hola Prime

**Website:** https://holaprime.com  
**Sources:** holaprime.com (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **NOT exchange API** — Forex and Futures prop firm. No Bybit API.
- Bots: UNKNOWN from sources.
- Platforms: UNKNOWN from extracted content (likely MT5 or proprietary).

**2. Restricted countries / KYC**
- Global (Hong Kong to London offices).
- Belarus: UNKNOWN.
- KYC: UNKNOWN.

**3. Challenge structure — $10k account**
- Fee: UNKNOWN.
- Pro Accounts: 100:1 leverage; Prime Accounts: 30:1 leverage (weekend holding and news trading allowed on Prime).
- Targets, DD: UNKNOWN (not extracted).

**4. Funded stage**
- Split options: Bi-Weekly 80%; Monthly 95%; Direct Plan bi-weekly up to 90%; On-Demand 80%.
- Payout: 1-hour processing, average 33 minutes.
- Scaling: Available.
- Payouts via Payout Junction or Blockchain.

**5. Multi-account policy**
- Quote: *"Yes, but only between your own Hola Prime (Sim. Funded) Account"* — context suggests hedging is permitted only within your own accounts (partial quote, context unclear).

**6. Funding fees / flat-for-payout**
- Proprietary/MT5 model — likely swap-based, not Bybit funding rates. UNKNOWN.
- Flat for payout: UNKNOWN.

**7. Reputation**
- Fastest payout prop firm award claimed (avg 33 min).
- 25K+ community members.
- Trustpilot: UNKNOWN from sources.
- Red flags: Forex/Futures focus, not crypto perpetuals via exchange API.

---

## 9. BitFunded

**Website:** https://bitfunded.io  
**Sources:** bitfunded.io (2026-09-09)

### Checklist

**Status: NOT LIVE — "Coming Soon"**
- BTC-only scalping prop firm, built in Dubai.
- Quote: *"Bitcoin-Only Prop Firm — Coming Soon... Launch Incoming — Join The Waitlist."*
- All details UNKNOWN (no rules, fees, or policies published).
- Not usable for the trader at this time.

---

## 10. Blue Guardian

**Website:** https://blueguardian.com  
**Sources:** blueguardian.com (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **NOT exchange API** — uses MT5, Match-Trader, NinjaTrader, TradeLocker, TradingView, Tradovate, DeepCharts. NOT Bybit API. Offers Crypto and Futures/Forex.
- Bots: EAs listed as a funded account attribute (implied allowed). Exact policy text not extracted.
- 170+ countries.

**2. Restricted countries / KYC**
- 170+ countries.
- Belarus: UNKNOWN.
- KYC: UNKNOWN.
- Payouts via Rise or Crypto.

**3. Challenge structure — $10k account**
- Fee: 1-Step Standard $10k: **$75**; 1-Step Nano $10k: (lower).
- Trailing drawdown model (loss floor follows highest balance).
- Max daily loss from day's starting balance.
- Plans: 1-Step Standard, 1-Step Nano, 2-Step Standard, 2-Step Nano, Instant.
- Specific DD% and targets: UNKNOWN from extracted content.

**4. Funded stage**
- Split: Up to 90%; some plans 100%.
- Payout: 7 days; Instant payouts on some plans. Payouts guaranteed within 24 hours (extra 10% if missed).
- Max allocation: 400K (seen in pricing).
- Scaling: UNKNOWN.
- Funded: Simulated.

**5. Multi-account policy**
- UNKNOWN.

**6. Funding fees / flat-for-payout**
- MT5/proprietary model; likely swap-based. UNKNOWN.
- Flat for payout: UNKNOWN.

**7. Reputation**
- PropFirmMatch: 4.6/5. Google: 4.9/5.
- Trustpilot: UNKNOWN from sources.
- Red flags: NOT Bybit API; trailing drawdown hurts long holding with unrealized gains.

---

## 11. The5ers

**Website:** https://the5ers.com  
**Sources:** the5ers.com (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **NOT exchange API** — Forex, metals, indices only. No crypto perpetuals on Bybit.
- Bots: UNKNOWN from extracted content (MT5 implied, EAs likely).
- Not suitable for the trader (wrong asset class, no exchange API).

**2. Restricted countries / KYC**
- **Belarus: EXPLICITLY RESTRICTED** — footer text found (2026-09-09): *"certain jurisdictions, including but not limited to: Afghanistan, Belarus, Burundi, Central African Republic..."*
- Russia: Likely also restricted (sanctions list context).

**3. Challenge structure — $10k account**
- Fee: $249 (1-step $100k plan shown; $10k fee UNKNOWN but likely lower).
- 1-Step: 10% target; max loss 6%; daily loss 3%; leverage 1:100.
- Consistency rule: **50% per day** — no single day may represent ≥50% of total profit.
- Payout cap: $2,000 per payout.
- Min withdrawal: $250.

**4. Funded stage**
- Split: 75%.
- Scaling: Up to $4,000,000.
- Max 2 active accounts.

**5. Multi-account policy**
- Max 2 active accounts per person.
- Copy trading/mirroring: UNKNOWN explicitly.

**6. Funding fees / flat-for-payout**
- High swap fees mentioned in a user review (*"My only major complaint is the high swap fees"*).
- Not applicable (no crypto perpetuals).

**7. Reputation**
- 262K funded traders. 10 years active (~2015). 171 employees in 24 countries.
- Trustpilot: UNKNOWN from extracted content (established firm with many reviews).
- Strong reputation in forex prop space.
- **DISQUALIFIED: Belarus explicitly banned; no crypto exchange API.**

---

## 12. FundedNext

**Website:** https://fundednext.com  
**Sources:** fundednext.com (2026-09-09)

### Checklist

**1. Account type / bots allowed**
- Account type: **NOT exchange API** — MT4, MT5, cTrader, Match-Trader. NOT Bybit API.
- Bots: UNKNOWN explicitly; MT5 EAs likely allowed.
- CFDs (not exchange-native perpetuals).
- Not suitable for Bybit API bot.

**2. Restricted countries / KYC**
- Belarus: UNKNOWN.
- KYC: UNKNOWN.

**3. Challenge structure — $10k account (Stellar 2-Step)**
- Targets: UNKNOWN specifically for $10k (15% phase-1 reward from challenge is mentioned).
- Plans: Stellar 2-Step, 1-Step, Lite, Instant.
- No time limit.

**4. Funded stage**
- Split: Up to 95%.
- Payout: Guaranteed within 24 hours; avg 5 hours; $1,000 extra if missed.
- Max: $300K.

**5. Multi-account policy**
- UNKNOWN.

**6. Funding fees / flat-for-payout**
- CFD overnight swaps apply (not Bybit funding rates).
- Flat for payout: UNKNOWN.

**7. Reputation**
- Strong presence; 25% new-user discount.
- Trustpilot: UNKNOWN from extracted content (well-known firm).
- Payment options: Skrill, PayPal, Mastercard, Neteller, etc.
- Red flags: NOT Bybit API; CFDs not exchange perpetuals.

---

## Additional Firms Searched: Bybit API Crypto Prop

Based on searches for "crypto prop firm API trading Bybit demo bots allowed" the only two established, live firms clearly offering a real Bybit sub-account via API for the challenge AND funded stage are **HyroTrader** and **Mubite**. Other firms either use proprietary simulators (CFT, Breakout) or MT4/MT5 (FXIFY, Alpha Capital, Blue Guardian, FundedNext, The5ers, Hola Prime).

---

## Summary: Belarus Eligibility

| Firm | Belarus Status |
|------|---------------|
| HyroTrader | UNKNOWN — no public list found; must contact support |
| Mubite | UNKNOWN — says "contact support"; CIS traders appear in reviews |
| Klein Funding | UNKNOWN — maintains restricted list, must check |
| Breakout Prop | UNKNOWN — "regional restrictions apply" |
| CFT | UNKNOWN |
| Alpha Capital | UNKNOWN — 140+ countries claimed |
| FXIFY | UNKNOWN — 200 countries claimed |
| Hola Prime | UNKNOWN |
| Blue Guardian | UNKNOWN — 170+ countries claimed |
| The5ers | **EXPLICITLY BANNED** |
| FundedNext | UNKNOWN |
| BitFunded | NOT LIVE |

---

## Summary: Multi-Account / Copy Trading Risk

| Firm | Same-bot multi-account risk |
|------|-----------------------------|
| HyroTrader | UNKNOWN rules — likely detectable via API monitoring |
| Mubite | HIGH RISK — explicit prohibition on "group trading or account mirroring"; running same bot across Mubite accounts would violate this |
| Klein Funding | FATAL — bots banned; one device/IP per account |
| Breakout Prop | LOWER RISK — multiple accounts allowed up to $200K total; no explicit copy-trading ban found |
| CFT | UNKNOWN |
| Alpha Capital | UNKNOWN |
| FXIFY | UNKNOWN |
| The5ers | Max 2 accounts; Belarus banned anyway |
| FundedNext | UNKNOWN |

---

## Ranked Shortlist (Top 5 for This Trader)

### #1 — HyroTrader
**Why:** Only firm with confirmed Bybit API integration (own Bybit account), real exchange data, founded 2022 with track record. Bot policy UNKNOWN but API model is most compatible. Crypto-native. Must confirm: bot allowance, Belarus, funding fees, multi-account rules before purchasing.
**Top 3 red flags:** (a) Bot policy not publicly confirmed; (b) Belarus eligibility unknown; (c) Funded stage still simulated with limited public rule detail.

### #2 — Mubite
**Why:** Bybit API + Cleo, bots explicitly allowed, transparent rules, Czech company, 4.8 Trustpilot. Only 2025 founding (newer), but detailed public rules. Fee refundable.
**Top 3 red flags:** (a) "Group trading/account mirroring" ban — same bot across multiple Mubite accounts violates this; (b) Belarus eligibility unknown; (c) Very new firm (2025), limited long-term payout track record.

### #3 — Breakout Prop
**Why:** No consistency rule, no min days, 80–90% split, 4.7 Trustpilot with $60M+ paid, multi-account allowed up to $200K total, 24/7 on-demand payouts. However, NOT a Bybit API account — uses proprietary platform. If trader can deploy bot to Breakout's API (if they have one), this is worth exploring. Otherwise unsuitable for the specific bot setup.
**Top 3 red flags:** (a) Proprietary platform, not Bybit API — bot must be adapted; (b) Belarus eligibility unknown; (c) Kraken-affiliated model introduces intermediary counterparty risk.

### #4 — FXIFY
**Why:** 200 countries, EAs allowed (Martingale/Grid too), 4.3 Trustpilot 6,200+ reviews, $40M paid, established. Only if trader can adapt bot to MT5. NOT Bybit API.
**Top 3 red flags:** (a) NOT Bybit API — requires MT5 EA rewrite; (b) Low crypto leverage (30:1 FX rate, not 100:1 crypto); (c) Belarus unknown.

### #5 — Blue Guardian
**Why:** EAs appear allowed, 170+ countries, $75 fee for $10k, crypto markets included, Rise or Crypto payouts. NOT Bybit API but broad platform support.
**Top 3 red flags:** (a) NOT Bybit API; (b) Trailing drawdown penalizes unrealized gains in multi-position portfolios; (c) Belarus unknown.

---

## Key Answers

**Q: Which firms explicitly accept Belarus?**
A: None confirmed. The5ers **explicitly bans** Belarus. All others are UNKNOWN — trader must contact each firm's support before purchasing.

**Q: Which firms' rules could be triggered by running the same market-neutral bot on several accounts/firms?**
- **Mubite:** Most likely — explicitly prohibits "group trading or account mirroring" and "cross-account hedging between multiple accounts." Running the same bot on two Mubite accounts would likely trigger this. Running it across different firms is harder to detect (different Bybit API keys), but if patterns are correlated, it could still be flagged.
- **Klein Funding:** Prohibits copy trading "including between accounts the same trader owns" and requires one account per IP/VPS. Running same bot from same VPS across Klein accounts would terminate all accounts.
- **HyroTrader:** Rules not fully public — likely has similar restrictions. Multiple Bybit API connections from same VPS/IP might be detectable.
- **Breakout Prop:** Multiple accounts allowed up to $200K total; no explicit copy-trading ban found — lowest risk for multi-account.
- **All firms:** Using the same bot across different firms (not the same firm) is generally harder for any single firm to detect, but behavioral analysis (identical position timing, identical entry prices) could flag it on platforms that share risk management infrastructure.
