"""
Fine-tune Deep LPPLS model on real BTC data.

Uses known bubble periods and pseudo-labels from traditional LPPLS fitting.

Known BTC bubble peaks:
- 2021-04-14: $64,863 -> dropped to $30,000 (May crash)
- 2021-11-10: $68,789 -> dropped to $33,000 (2022 bear market)
- 2024-03-14: $73,750 -> correction

Strategy:
1. Extract windows leading up to known crash dates
2. Fit traditional LPPLS to get pseudo-labels
3. Fine-tune neural network on this real data
4. Combine with synthetic data to prevent forgetting
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

from .strategy import load_btc_data
from .lppls import LPPLSModel, LPPLSResult
from .deep_lppls import (
    DeepLPPLSConfig,
    DeepLPPLSPredictor,
    DeepLPPLSTrainer,
    PLNNWithConv,
    LPPLSDataGenerator,
    LPPLSDataset,
)


# Known BTC bubble crash dates (peak dates)
KNOWN_CRASHES = [
    {'date': '2021-04-14', 'price': 64863, 'description': 'April 2021 peak before May crash'},
    {'date': '2021-11-10', 'price': 68789, 'description': 'November 2021 ATH before bear market'},
    {'date': '2024-03-14', 'price': 73750, 'description': 'March 2024 peak'},
]


def load_daily_btc_data() -> pd.DataFrame:
    """Load BTC data and resample to daily."""
    df = load_btc_data()
    df['date'] = df['datetime'].dt.date

    daily = df.groupby('date').agg({
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'open': 'first',
        'volume': 'sum',
    }).reset_index()

    daily['datetime'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('datetime').reset_index(drop=True)

    return daily


def extract_bubble_windows(
    df: pd.DataFrame,
    crash_dates: List[Dict],
    window_sizes: List[int] = [120, 180, 252],
) -> List[Tuple[np.ndarray, Dict]]:
    """
    Extract price windows leading up to crash dates.

    Returns list of (log_prices, metadata) tuples.
    """
    windows = []

    for crash in crash_dates:
        crash_date = pd.to_datetime(crash['date'])

        # Find index of crash date
        mask = df['datetime'] <= crash_date
        if not mask.any():
            continue

        crash_idx = mask.sum() - 1

        for window_size in window_sizes:
            start_idx = max(0, crash_idx - window_size)
            if crash_idx - start_idx < 60:  # Need at least 60 days
                continue

            prices = df['close'].iloc[start_idx:crash_idx+1].values
            log_prices = np.log(prices)

            # Days until crash from end of window
            days_to_crash = 0  # We're at the crash

            metadata = {
                'crash_date': crash_date,
                'window_size': crash_idx - start_idx + 1,
                'start_date': df['datetime'].iloc[start_idx],
                'end_date': df['datetime'].iloc[crash_idx],
                'days_to_crash': days_to_crash,
                'description': crash['description'],
            }

            windows.append((log_prices, metadata))

            # Also create windows ending before the crash (10, 20, 30 days before)
            for days_before in [10, 20, 30, 45]:
                end_idx = crash_idx - days_before
                if end_idx - start_idx < 60:
                    continue

                prices = df['close'].iloc[start_idx:end_idx+1].values
                log_prices = np.log(prices)

                metadata = {
                    'crash_date': crash_date,
                    'window_size': end_idx - start_idx + 1,
                    'start_date': df['datetime'].iloc[start_idx],
                    'end_date': df['datetime'].iloc[end_idx],
                    'days_to_crash': days_before,
                    'description': crash['description'],
                }

                windows.append((log_prices, metadata))

    return windows


def fit_traditional_lppls(
    log_prices: np.ndarray,
    n_starts: int = 20,
) -> Optional[LPPLSResult]:
    """Fit traditional LPPLS to get pseudo-labels."""
    model = LPPLSModel(tc_min_days=1, tc_max_days=120)
    result = model.fit(log_prices, method='nelder-mead', n_starts=n_starts)
    return result


def create_training_data(
    windows: List[Tuple[np.ndarray, Dict]],
    target_length: int = 252,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Create training data from bubble windows with pseudo-labels.

    Returns (X, y, metadata) where:
    - X: normalized log prices (n_samples, target_length)
    - y: [tc_norm, m, omega] labels
    - metadata: fitting info for each sample
    """
    X_list = []
    y_list = []
    meta_list = []

    for i, (log_prices, meta) in enumerate(windows):
        if verbose:
            print(f"Processing window {i+1}/{len(windows)}: {meta['description'][:30]}...")

        # Fit traditional LPPLS
        result = fit_traditional_lppls(log_prices)

        if result is None:
            if verbose:
                print(f"  Failed to fit LPPLS")
            continue

        # Check if result is valid
        if not (0.1 < result.params.m < 0.9):
            if verbose:
                print(f"  Invalid m: {result.params.m:.3f}")
            continue

        # Prepare input (pad/truncate to target_length)
        if len(log_prices) > target_length:
            log_prices_input = log_prices[-target_length:]
        else:
            pad_size = target_length - len(log_prices)
            log_prices_input = np.concatenate([
                np.full(pad_size, log_prices[0]),
                log_prices
            ])

        # Normalize to [0, 1]
        log_prices_norm = (log_prices_input - log_prices_input.min()) / \
                         (log_prices_input.max() - log_prices_input.min() + 1e-8)

        # Calculate tc_norm (relative to window end)
        tc_days_ahead = result.params.tc - len(log_prices)
        tc_norm = tc_days_ahead / target_length

        # Clip tc_norm to valid range
        tc_norm = np.clip(tc_norm, 0.01, 1.0)

        X_list.append(log_prices_norm.astype(np.float32))
        y_list.append(np.array([
            tc_norm,
            result.params.m,
            result.params.omega
        ], dtype=np.float32))

        meta['fitted_tc'] = result.params.tc
        meta['fitted_m'] = result.params.m
        meta['fitted_omega'] = result.params.omega
        meta['fitted_r2'] = result.r_squared
        meta_list.append(meta)

        if verbose:
            print(f"  tc_norm={tc_norm:.3f}, m={result.params.m:.3f}, "
                  f"omega={result.params.omega:.2f}, R²={result.r_squared:.3f}")

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(y_list), meta_list


def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    n_augments: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with noise and scaling variations.
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(n_augments):
        # Add noise
        noise = np.random.normal(0, 0.02, X.shape)
        X_noisy = X + noise
        X_noisy = np.clip(X_noisy, 0, 1)
        X_aug.append(X_noisy.astype(np.float32))
        y_aug.append(y)

        # Scale variation
        scale = np.random.uniform(0.95, 1.05, (X.shape[0], 1))
        X_scaled = X * scale
        X_scaled = (X_scaled - X_scaled.min(axis=1, keepdims=True)) / \
                   (X_scaled.max(axis=1, keepdims=True) - X_scaled.min(axis=1, keepdims=True) + 1e-8)
        X_aug.append(X_scaled.astype(np.float32))
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)


class FineTuner:
    """Fine-tune Deep LPPLS on real BTC data."""

    def __init__(
        self,
        config: Optional[DeepLPPLSConfig] = None,
        pretrained_path: Optional[str] = None,
    ):
        self.config = config or DeepLPPLSConfig(input_length=252)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = PLNNWithConv(self.config)
        self.model.to(self.device)

        # Load pretrained weights if available
        if pretrained_path and Path(pretrained_path).exists():
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            print(f"Loaded pretrained weights from {pretrained_path}")

    def prepare_data(
        self,
        crash_dates: Optional[List[Dict]] = None,
        n_synthetic: int = 10000,
        verbose: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data."""
        if crash_dates is None:
            crash_dates = KNOWN_CRASHES

        if verbose:
            print("Loading BTC data...")
        df = load_daily_btc_data()

        if verbose:
            print(f"Extracting bubble windows from {len(crash_dates)} known crashes...")
        windows = extract_bubble_windows(df, crash_dates)
        print(f"Extracted {len(windows)} windows")

        if verbose:
            print("Creating training data with LPPLS pseudo-labels...")
        X_real, y_real, meta = create_training_data(
            windows,
            target_length=self.config.input_length,
            verbose=verbose,
        )

        if len(X_real) == 0:
            raise ValueError("No valid training samples created from real data")

        if verbose:
            print(f"\nReal data samples: {len(X_real)}")

        # Augment real data
        if verbose:
            print("Augmenting real data...")
        X_real_aug, y_real_aug = augment_data(X_real, y_real, n_augments=10)
        if verbose:
            print(f"Augmented to {len(X_real_aug)} samples")

        # Generate synthetic data
        if verbose:
            print(f"Generating {n_synthetic} synthetic samples...")
        generator = LPPLSDataGenerator(length=self.config.input_length)
        X_syn, y_syn = generator.generate_batch(n_synthetic)

        # Combine datasets (weight real data more)
        X_train = np.concatenate([X_real_aug, X_syn])
        y_train = np.concatenate([y_real_aug, y_syn])

        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Split
        split_idx = int(0.9 * len(X_train))
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        if verbose:
            print(f"Training set: {len(X_tr)}, Validation set: {len(X_val)}")

        train_dataset = LPPLSDataset(X_tr, y_tr)
        val_dataset = LPPLSDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-4,
        verbose: bool = True,
    ) -> Dict:
        """Run fine-tuning."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10,
        )

        loss_weights = torch.tensor([1.0, 1.5, 2.0], device=self.device)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X)
                loss = (loss_weights * (pred - y) ** 2).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    loss = (loss_weights * (pred - y) ** 2).mean()
                    val_loss += loss.item()

                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(y.cpu().numpy())

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                all_preds = np.concatenate(all_preds)
                all_targets = np.concatenate(all_targets)
                mae_tc = np.mean(np.abs(all_preds[:, 0] - all_targets[:, 0]))
                mae_m = np.mean(np.abs(all_preds[:, 1] - all_targets[:, 1]))
                mae_omega = np.mean(np.abs(all_preds[:, 2] - all_targets[:, 2]))

                print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}, "
                      f"mae_tc={mae_tc:.3f}, mae_m={mae_m:.3f}, mae_omega={mae_omega:.2f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose:
            print(f"\nFine-tuning complete. Best val_loss: {best_val_loss:.4f}")

        return history

    def save(self, path: str):
        """Save fine-tuned model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}")

    def get_predictor(self) -> DeepLPPLSPredictor:
        """Get predictor with fine-tuned weights."""
        predictor = DeepLPPLSPredictor(config=self.config, use_conv=True)
        predictor.model.load_state_dict(self.model.state_dict())
        predictor._is_trained = True
        return predictor


def run_finetuning(
    save_path: str = 'strategies/lppls/finetuned_model.pt',
    epochs: int = 100,
    n_synthetic: int = 15000,
):
    """
    Run full fine-tuning pipeline.
    """
    print("="*60)
    print("Deep LPPLS Fine-tuning on BTC Data")
    print("="*60)

    config = DeepLPPLSConfig(
        input_length=252,
        batch_size=128,
    )

    finetuner = FineTuner(config=config)

    # Prepare data
    train_loader, val_loader = finetuner.prepare_data(
        n_synthetic=n_synthetic,
        verbose=True,
    )

    # Fine-tune
    print("\n" + "="*60)
    print("Starting fine-tuning...")
    print("="*60)

    history = finetuner.finetune(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=5e-4,
        verbose=True,
    )

    # Save model
    finetuner.save(save_path)

    # Test on recent data
    print("\n" + "="*60)
    print("Testing on recent BTC data...")
    print("="*60)

    predictor = finetuner.get_predictor()

    df = load_daily_btc_data()
    recent = df[df['datetime'] >= '2024-01-01'].reset_index(drop=True)

    if len(recent) >= 252:
        log_prices = np.log(recent['close'].values[-252:])
        pred = predictor.predict(log_prices)

        print(f"\nPrediction for recent data (ending {recent['datetime'].iloc[-1].strftime('%Y-%m-%d')}):")
        print(f"  tc_norm: {pred['tc_norm']:.3f} ({pred['tc_norm']*252:.0f} days ahead)")
        print(f"  m: {pred['m']:.3f}")
        print(f"  omega: {pred['omega']:.2f}")

    return finetuner, history


if __name__ == '__main__':
    finetuner, history = run_finetuning(
        epochs=100,
        n_synthetic=15000,
    )
