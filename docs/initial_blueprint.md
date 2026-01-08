Deep Learning Bitcoin Trading Strategy Blueprint
1. Goal Description
The objective is to develop a Modular, Config-Driven, File-Centric automated trading research system for Bitcoin. The objective is to develop a Modular, Config-Driven, File-Centric automated trading research system for Bitcoin. Strategy Focus: Mid-Frequency Trend Following (Long & Short). Goal: Overcome high retail fees (0.1% round-trip) by capturing large structural trends using advanced Transformer models (PatchTST). Core Constraint: Avg Profit per Trade > 0.15% (after fees/slippage).

2. Design Philosophy
Test-Driven Development (TDD):
Rule: No code is written without a failing test first.
Go: Use standard testing package with table-driven tests.
Python: Use pytest with synthetic data fixtures for all logic verification.
Coverage: Critical paths (ETL, Money Management) require > 90% coverage.
File-Centric: Every step (ETL, Feature Engineering, Labeling) inputs a file and outputs a file. No hidden in-memory state between stages. This enables "resume from anywhere" capability.
Config-Driven: All parameters (Thresholds, Window sizes) are defined in config.yaml. No hardcoding.
Modular: Stages are independent scripts involving Reader -> Processor -> Writer.e parameters, model hyperparameters, walk-forward splits). No hardcoded magic numbers.
Plug-and-Play: New features or models are added as independent modules. The pipeline runner simply executes the sequence defined in the config.
3. Modular Pipeline Architecture
The pipeline consists of 5 independent stages.

Stage 1: Ingestion & ETL (Go)
Data Sources (Binance Vision Historical Data):
- Spot BTCUSDT: 2017-08 ~ Present (89+ months)
- Futures USDT-M BTCUSDT: 2019-09 ~ Present (64+ months)
- Format: Monthly ZIP archives containing RAW Trades (Not aggTrade)
- Orderbook/BBO Data: Not used (unavailable from Binance Vision)
Input: Binance RAW Trades (Not aggTrade).
Constraint: 1TB limit. Hot Data on SSD, Cold Data offloaded to HDD.
Process:
Performance Engineering (Go):
Streaming: Process line-by-line (Stream -> Decode -> Accumulate -> Write). Zero huge slices.
Primitives: Use float64/int64 only. Avoid complex structs to reduce GC pressure.
Precision Safety: Round all Prices/Volumes to 8 decimals before Parquet write.
Why? Floating point drift ($10^{-12}$) between Go/Python causes drift in sensitive features (FracDiff). Rounding unifies the "Truth".
Tick Imbalance Bar (TIB) Generation (Information-Driven Sampling):
Concept: Sample bars when "Informed Traders" create significant order flow imbalance.
Step 1 (Tick Rule): Determine sign $b_t$ for each raw tick.
Step 2 (Imbalance): Accumulate $\theta_T = \sum b_t$.
Step 3 (Threshold): Close bar when $|\theta_T| \geq E_0[T] \times |2P[b_t=1] - 1|$.
Adaptive Logic: Update Expectations using Dual EMA (Fast/Slow mix).
Flash Crash Safety: Carry-over logic for excess imbalance.
Output: tib_{year-month}.parquet (Columnar compression essential).
Why Go?: Python loops choke on billions of raw ticks. Go + Streaming is mandatory.
Stage 2: Feature Engineering (Python)
Crucial Rule: No Look-Ahead.
Stitching Logic:
Buffer: Load Last_K rows.
Adaptation: Use Adaptive Half-life stitching. If Volatility Regimes shift (e.g., Parkinson Vol spike), accelerate decay of past stats to prevent "Zombie Z-scores".
Input: Tick Imbalance Bar (TIB) Parquet files.
Process:
Structure (L1):
Log-Normal Fix: Apply $\log(x+1)$ to Volume & Taker Ratio.
Time Awareness: $\log(\text{Time_Duration})$ per bar (Crucial for TIB as time is irregular).
Taker Buy/Sell Ratio: Rolling Z-score.
Volume Imbalance: $(BuyVol - SellVol) / TotalVol$.
VWAP Deviation: $(Close - VWAP) / Close$.
Dual Input Strategy (Trend Context Fix):
Input A (Dynamics): FracDiff Prices (Stationary, for Short-term Pattern).
Input B (Context): Log-Price Detrended (e.g., $Price / EMA_{100d}$). Preserves "Long-Term Level" info.
Why? Strict ADF kills trend info. Input B re-injects Global Trend context.
Regime:
Shannon Entropy: Detects noise vs trend.
Parkinson Volatility: High-efficiency volatility estimator.
Stationarity: Fractional Differentiation (FracDiff).
Priority: ADF p-value < 0.05. (Used for Input A only).
Output: features_{hash_of_config}.parquet
Stage 3: Labeling (Dynamic Volatility-Adjusted)
Goal: Capture trends proportional to current volatility.
Process:
barriers:
Base: Realized Volatility $\sigma$ (e.g., Hourly Std Dev).
Top (PT): $n_{pt} \times \sigma$.
Bottom (SL): $n_{sl} \times \sigma$.
Vertical: Hybrid Barrier (Min_Bars OR Max_Time).
Logic: Close if Bars since Entry > 20 OR Time since Entry > 72h.
Why? Prevents premature closure in dead markets (TIB limitation) while strictly enforcing time limits.
Class Balancing: Removed. Rely solely on Focal Loss to avoid Double Penalty.
Output: labeled_dataset_{hash_of_config}.parquet
Stage 4: Network Training (Python / PyTorch)
Input: Dual Inputs (A: Dynamics 512, B: Context 512).
Model: PatchTST Classifier.
Backbone: Channel Independent Patching.
Head (Interaction Fix): Residual Bottleneck.
x = Flatten(Backbone)
residual = x
x = Dense(1024->256)(x) -> ReLU -> Dense(256->1024)(x)
out = Dropout(x + residual) -> Softmax.
Why? Skip connection preserves raw signal while Bottleneck forces feature compression.
Context: 512 Bars.
Optimization:
Mixed Precision: FP16 training enabled (Crucial for VRAM).
Memory Safety: Run each Walk-Forward fold in a Separate Subprocess.
Why? Python GC often fails to reclaim GPU memory after loops. multiprocessing ensures 100% RAM release after each fold to prevent OOM.
Loss: Focal Loss Only + Label Smoothing (e.g., 0.1).
Prediction: Do not trade if $\max(P_{long}, P_{short}) < Threshold$.
Validation: Walk-Forward (Train -> Test -> Slide).
Stage 5: Backtest & Money Management (Python)
Metrics: Equity Curve Analysis (MDD, Volatility based on Unrealized PnL).
Execution Logic:
Position Sizing: Fixed Risk ($n%$ of equity) or Half-Kelly.
Pyramiding: Scale-in (0.5x unit) if profit > $n \times \sigma$.
Buffered BEP: ON SCALE-IN, move SL to $AvgEntry - (0.5 \times \sigma)$.
Why? Explicit "Risk-Free" (at exactly BEP) triggers stop-outs on common volatility wicks. Buffer gives breathing room.
Dynamic Exit:
Asymmetric Floating Barrier:
TP: Expand if Volatility increases (Let profits run).
SL: NEVER WIDEN. Only tighten (Trailing) or stay fixed.
Trailing Stop: $AvgPrice - n_{trail} \times \sigma$.
Success Condition: Avg Profit per Trade > 0.150% (Net of Fees).
4. Configuration Schema (config.yaml)
data:
  sources:
    spot:
      symbol: "BTCUSDT"
      start: "2017-08"
      end: "present"
      base_url: "https://data.binance.vision/data/spot/monthly/trades"
    futures:
      symbol: "BTCUSDT"
      start: "2019-09"
      end: "present"
      base_url: "https://data.binance.vision/data/futures/um/monthly/trades"
  tib:
    initial_expected_tick_count: 2000
    initial_prob_buy: 0.5
    ema_alpha_fast: 0.1
    ema_alpha_slow: 0.01
features:
  l1_factors:
    - "taker_buy_sell_ratio"
    - "volume_imbalance"
    - "vwap_deviation"
  regime:
    - "shannon_entropy"
    - "parkinson_volatility"
labeling:
  volatility_barrier:
    base_vol_window: 24 # Hourly if bars are ~40min
    pt_multiplier: 2.0
    sl_multiplier: 1.0
    vertical_barrier_bars: 120
model:
  architecture: "PatchTST"
  context_window: 256
  
execution:
  fee_rate: 0.0005
  slippage: 0.0001
  risk_management:
    method: "fixed_risk" # or "half_kelly"
    risk_per_trade: 0.02
    pyramiding:
      enabled: true
      step_multiplier: 1.5
    trailing_stop:
      enabled: true
      multiplier: 2.5
5. Directory Structure
/home/kimhoyeon/dev/dl_rl_btc/
├── config/
│   └── experiments/        # YAML configs for different runs
├── data/
│   ├── raw/
│   │   ├── spot/           # Spot trade ZIPs (2017-08~)
│   │   └── futures/        # Futures USDT-M trade ZIPs (2019-09~)
│   ├── tib/
│   │   ├── spot/           # Spot TIB Parquet
│   │   └── futures/        # Futures TIB Parquet
│   ├── features/           # Feature Sets (Parquet)
│   └── labels/             # Labeled Datasets (Parquet)
├── src/
│   ├── go/
│   │   ├── etl/            # Tick -> Bar logic
│   │   └── trade/          # Execution Engine
│   └── python/
│       ├── pipeline/       # Runner logic (reads config, calls modules)
│       ├── features/       # FracDiff, Indicators
│       └── models/         # PyTorch Lightning Modules
└── artifacts/              # Model checkpoints, Logs
6. Verification Plan
ETL correctness: Compare Go-generated Dollar Bars against a Python reference implementation on a small subset of data.
Stationarity Check: File-centric ADF test report for all generated features.
Leakage Check: Verify that Test set timestamps are strictly > Train set timestamps in the Walk-Forward splitter.