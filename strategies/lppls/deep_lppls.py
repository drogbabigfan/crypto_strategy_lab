"""
Deep LPPLS: Neural Network-based Parameter Estimation.

Based on Nielsen, Sornette, Raissi (2024) Deep LPPLS framework.

Two approaches:
1. M-LNN (Mono-LPPLS NN): Trained on single time series (PINN-style)
2. P-LNN (Poly-LPPLS NN): Pre-trained on synthetic data for fast inference

This implementation focuses on P-LNN for real-time screening capability.

Key advantages:
- ~1000x faster inference than traditional optimization
- No local minima issues
- Can screen thousands of assets in real-time
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
import warnings


@dataclass
class DeepLPPLSConfig:
    """Configuration for Deep LPPLS model."""
    # Input configuration
    input_length: int = 252          # Number of days (1 trading year)

    # Network architecture
    hidden_dims: List[int] = None    # Hidden layer dimensions
    dropout: float = 0.1

    # Training configuration
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 100

    # Output: predict (tc_normalized, m, omega)
    # tc_normalized = (tc - t_end) / input_length

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class LPPLSDataGenerator:
    """
    Generate synthetic LPPLS time series for training.

    Creates diverse training samples with varying parameters and noise levels.
    Includes realistic market conditions (trends, volatility clustering).
    """

    def __init__(
        self,
        length: int = 252,
        tc_range: Tuple[float, float] = (0.8, 0.05),  # (max, min) as fraction of length - closer to end
        m_range: Tuple[float, float] = (0.1, 0.9),
        omega_range: Tuple[float, float] = (4.0, 25.0),
        noise_std_range: Tuple[float, float] = (0.01, 0.05),  # Higher noise like real markets
        include_non_bubble: float = 0.3,  # 30% non-bubble samples
    ):
        self.length = length
        self.tc_range = tc_range
        self.m_range = m_range
        self.omega_range = omega_range
        self.noise_std_range = noise_std_range
        self.include_non_bubble = include_non_bubble

    def generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single LPPLS sample.

        Returns:
            (log_prices, params) where params = [tc_norm, m, omega]
            tc_norm = (tc - length) / length (normalized distance to crash)
        """
        t = np.arange(self.length, dtype=np.float64)

        # Randomly generate non-bubble sample
        if np.random.random() < self.include_non_bubble:
            return self._generate_non_bubble_sample()

        # Random parameters with bias toward imminent crashes
        # Use beta distribution to favor smaller tc_norm values
        tc_norm = np.random.beta(1.5, 3) * (self.tc_range[0] - self.tc_range[1]) + self.tc_range[1]
        tc = self.length + tc_norm * self.length

        m = np.random.uniform(*self.m_range)
        omega = np.random.uniform(*self.omega_range)

        # Random linear params
        A = np.random.uniform(8, 12)
        B = np.random.uniform(-1.0, -0.1)
        C = np.random.uniform(0.01, 0.15)
        phi = np.random.uniform(0, 2 * np.pi)

        # Generate LPPLS signal
        dt = tc - t
        dt_m = np.power(dt, m)
        omega_log_dt = omega * np.log(dt)

        log_prices = A + B * dt_m + C * dt_m * np.cos(omega_log_dt - phi)

        # Add realistic market noise
        noise_std = np.random.uniform(*self.noise_std_range)
        white_noise = np.random.normal(0, noise_std, self.length)

        # AR(1) component for autocorrelation
        ar_coef = np.random.uniform(0.4, 0.8)
        ar_noise = np.zeros(self.length)
        for i in range(1, self.length):
            ar_noise[i] = ar_coef * ar_noise[i-1] + white_noise[i]

        # GARCH-like volatility clustering
        vol_cluster = np.random.uniform(0.5, 1.5, self.length)
        vol_cluster = np.convolve(vol_cluster, np.ones(20)/20, mode='same')

        log_prices += ar_noise * vol_cluster

        # Normalize log_prices to [0, 1] for network input
        log_prices_norm = (log_prices - log_prices.min()) / (log_prices.max() - log_prices.min() + 1e-8)

        params = np.array([tc_norm, m, omega], dtype=np.float32)

        return log_prices_norm.astype(np.float32), params

    def _generate_non_bubble_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a non-bubble (random walk) sample."""
        t = np.arange(self.length, dtype=np.float64)

        # Simple random walk with drift
        drift = np.random.uniform(-0.0005, 0.001)
        noise_std = np.random.uniform(0.01, 0.03)

        returns = drift + np.random.normal(0, noise_std, self.length)
        log_prices = np.cumsum(returns)
        log_prices = log_prices - log_prices[0] + np.random.uniform(8, 12)

        # Normalize
        log_prices_norm = (log_prices - log_prices.min()) / (log_prices.max() - log_prices.min() + 1e-8)

        # Non-bubble params: large tc (far away), random m/omega
        tc_norm = np.random.uniform(0.8, 1.5)  # Far in future
        m = np.random.uniform(*self.m_range)
        omega = np.random.uniform(*self.omega_range)

        params = np.array([tc_norm, m, omega], dtype=np.float32)

        return log_prices_norm.astype(np.float32), params

    def generate_batch(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate batch of samples."""
        X = np.zeros((n_samples, self.length), dtype=np.float32)
        y = np.zeros((n_samples, 3), dtype=np.float32)

        for i in range(n_samples):
            X[i], y[i] = self.generate_sample()

        return X, y


class LPPLSDataset(Dataset):
    """PyTorch Dataset for LPPLS data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PLNN(nn.Module):
    """
    Poly-LPPLS Neural Network (P-LNN).

    A feed-forward network that predicts LPPLS parameters
    from normalized log price series.

    Input: (batch, length) - normalized log prices
    Output: (batch, 3) - [tc_norm, m, omega]
    """

    def __init__(self, config: DeepLPPLSConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.input_length

        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = hidden_dim

        # Output layer with constrained outputs
        layers.append(nn.Linear(in_dim, 3))

        self.network = nn.Sequential(*layers)

        # Output activation to constrain ranges
        self.tc_activation = nn.Sigmoid()  # [0, 1] -> scale later
        self.m_activation = nn.Sigmoid()   # [0, 1] -> [0.1, 0.9]
        self.omega_activation = nn.Sigmoid()  # [0, 1] -> [4, 25]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, length) normalized log prices

        Returns:
            (batch, 3) predictions [tc_norm, m, omega]
        """
        out = self.network(x)

        # Apply activations and scale to valid ranges
        tc_norm = self.tc_activation(out[:, 0])  # [0, 1]
        m = 0.1 + 0.8 * self.m_activation(out[:, 1])  # [0.1, 0.9]
        omega = 4.0 + 21.0 * self.omega_activation(out[:, 2])  # [4, 25]

        return torch.stack([tc_norm, m, omega], dim=1)


class PLNNWithConv(nn.Module):
    """
    P-LNN with 1D Convolutional layers for better feature extraction.

    Captures local patterns in price series before MLP.
    """

    def __init__(self, config: DeepLPPLSConfig):
        super().__init__()
        self.config = config

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )

        # MLP head
        conv_out_dim = 128 * 16
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(64, 3),
        )

        self.tc_activation = nn.Sigmoid()
        self.m_activation = nn.Sigmoid()
        self.omega_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension: (batch, length) -> (batch, 1, length)
        x = x.unsqueeze(1)

        # Conv features
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        # MLP prediction
        out = self.mlp(features)

        # Scale outputs
        tc_norm = self.tc_activation(out[:, 0])
        m = 0.1 + 0.8 * self.m_activation(out[:, 1])
        omega = 4.0 + 21.0 * self.omega_activation(out[:, 2])

        return torch.stack([tc_norm, m, omega], dim=1)


class DeepLPPLSTrainer:
    """Trainer for Deep LPPLS models."""

    def __init__(
        self,
        model: nn.Module,
        config: DeepLPPLSConfig,
        device: str = 'auto',
    ):
        self.model = model
        self.config = config

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
        )

        # Weighted loss (omega is harder to predict)
        self.loss_weights = torch.tensor([1.0, 1.5, 2.0], device=self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X)

            # Weighted MSE loss
            loss = (self.loss_weights * (pred - y) ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> Tuple[float, dict]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)

                loss = (self.loss_weights * (pred - y) ** 2).mean()
                total_loss += loss.item()

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Per-parameter metrics
        metrics = {}
        param_names = ['tc_norm', 'm', 'omega']
        for i, name in enumerate(param_names):
            mae = np.mean(np.abs(all_preds[:, i] - all_targets[:, i]))
            metrics[f'{name}_mae'] = mae

        return total_loss / len(dataloader), metrics

    def train(
        self,
        n_train_samples: int = 50000,
        n_val_samples: int = 5000,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop with synthetic data generation.

        Args:
            n_train_samples: Number of training samples to generate
            n_val_samples: Number of validation samples
            verbose: Print progress

        Returns:
            Training history
        """
        if verbose:
            print(f"Generating {n_train_samples} training samples...")

        generator = LPPLSDataGenerator(length=self.config.input_length)
        X_train, y_train = generator.generate_batch(n_train_samples)
        X_val, y_val = generator.generate_batch(n_val_samples)

        train_dataset = LPPLSDataset(X_train, y_train)
        val_dataset = LPPLSDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        best_val_loss = float('inf')

        if verbose:
            print(f"Training on {self.device}...")

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            self.scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model state
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"m_mae={val_metrics['m_mae']:.3f}, omega_mae={val_metrics['omega_mae']:.3f}")

        # Restore best model
        self.model.load_state_dict(self.best_state)

        if verbose:
            print(f"Training complete. Best val_loss: {best_val_loss:.4f}")

        return history

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])


class DeepLPPLSPredictor:
    """
    Fast LPPLS parameter predictor using pre-trained neural network.

    ~1000x faster than traditional optimization.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[DeepLPPLSConfig] = None,
        use_conv: bool = True,
    ):
        self.config = config or DeepLPPLSConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        if use_conv:
            self.model = PLNNWithConv(self.config)
        else:
            self.model = PLNN(self.config)

        self.model.to(self.device)
        self.model.eval()

        # Load weights if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state'])

        self._is_trained = model_path is not None

    def train(self, n_samples: int = 50000, epochs: int = 50, verbose: bool = True):
        """Train the model on synthetic data."""
        self.config.epochs = epochs
        trainer = DeepLPPLSTrainer(self.model, self.config, device=str(self.device))
        history = trainer.train(n_train_samples=n_samples, verbose=verbose)
        self._is_trained = True
        return history

    def predict(
        self,
        log_prices: np.ndarray,
        return_raw: bool = False,
    ) -> dict:
        """
        Predict LPPLS parameters from log price series.

        Args:
            log_prices: Log price array (will be truncated/padded to input_length)
            return_raw: Return raw network outputs

        Returns:
            Dict with predicted parameters: tc, m, omega
        """
        if not self._is_trained:
            warnings.warn("Model not trained. Predictions will be random.")

        # Prepare input
        length = self.config.input_length
        if len(log_prices) > length:
            log_prices = log_prices[-length:]
        elif len(log_prices) < length:
            # Pad with first value
            pad = np.full(length - len(log_prices), log_prices[0])
            log_prices = np.concatenate([pad, log_prices])

        # Normalize
        log_prices_norm = (log_prices - log_prices.min()) / (log_prices.max() - log_prices.min() + 1e-8)

        # Predict
        x = torch.from_numpy(log_prices_norm.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x).cpu().numpy()[0]

        tc_norm, m, omega = pred

        # Convert tc_norm to actual tc
        tc = len(log_prices) + tc_norm * length

        result = {
            'tc': tc,
            'm': m,
            'omega': omega,
            'tc_norm': tc_norm,
        }

        if return_raw:
            result['raw'] = pred

        return result

    def predict_batch(
        self,
        log_prices_batch: List[np.ndarray],
    ) -> List[dict]:
        """
        Batch prediction for multiple series.

        Much faster than calling predict() multiple times.
        """
        if not self._is_trained:
            warnings.warn("Model not trained. Predictions will be random.")

        length = self.config.input_length
        batch = []

        for log_prices in log_prices_batch:
            if len(log_prices) > length:
                log_prices = log_prices[-length:]
            elif len(log_prices) < length:
                pad = np.full(length - len(log_prices), log_prices[0])
                log_prices = np.concatenate([pad, log_prices])

            log_prices_norm = (log_prices - log_prices.min()) / (log_prices.max() - log_prices.min() + 1e-8)
            batch.append(log_prices_norm)

        x = torch.from_numpy(np.array(batch, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            preds = self.model(x).cpu().numpy()

        results = []
        for i, pred in enumerate(preds):
            tc_norm, m, omega = pred
            tc = len(log_prices_batch[i]) + tc_norm * length
            results.append({
                'tc': tc,
                'm': m,
                'omega': omega,
                'tc_norm': tc_norm,
            })

        return results

    def save(self, path: str):
        """Save model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
        }, path)


def compare_speed(n_samples: int = 100):
    """
    Compare inference speed: Deep LPPLS vs Traditional.
    """
    import time
    from .lppls import LPPLSModel

    print("="*60)
    print("Speed Comparison: Deep LPPLS vs Traditional Optimization")
    print("="*60)

    # Generate test data
    generator = LPPLSDataGenerator(length=252)
    test_samples = [generator.generate_sample()[0] for _ in range(n_samples)]

    # Traditional method
    print(f"\nTraditional (Nelder-Mead, {n_samples} samples)...")
    model = LPPLSModel(tc_max_days=60)

    start = time.time()
    for i, sample in enumerate(test_samples[:10]):  # Only 10 for traditional (slow)
        # Denormalize for traditional method
        log_prices = sample * 2 + 9  # Rough denorm
        model.fit(log_prices, n_starts=3)
    traditional_time = (time.time() - start) / 10
    print(f"  Average time per sample: {traditional_time*1000:.1f} ms")

    # Deep LPPLS
    print(f"\nDeep LPPLS (Neural Network, {n_samples} samples)...")
    predictor = DeepLPPLSPredictor(use_conv=True)

    print("  Training model (30 epochs)...")
    predictor.train(n_samples=20000, epochs=30, verbose=False)

    # Single predictions
    start = time.time()
    for sample in test_samples:
        predictor.predict(sample)
    deep_single_time = (time.time() - start) / n_samples
    print(f"  Single prediction time: {deep_single_time*1000:.2f} ms")

    # Batch prediction
    start = time.time()
    predictor.predict_batch(test_samples)
    deep_batch_time = (time.time() - start) / n_samples
    print(f"  Batch prediction time: {deep_batch_time*1000:.3f} ms")

    # Summary
    print(f"\n{'='*60}")
    print("Speed Improvement:")
    print(f"  Single: {traditional_time/deep_single_time:.0f}x faster")
    print(f"  Batch:  {traditional_time/deep_batch_time:.0f}x faster")
    print("="*60)

    return {
        'traditional_ms': traditional_time * 1000,
        'deep_single_ms': deep_single_time * 1000,
        'deep_batch_ms': deep_batch_time * 1000,
    }


if __name__ == '__main__':
    # Quick test
    print("Testing Deep LPPLS...")

    config = DeepLPPLSConfig(
        input_length=252,
        epochs=30,
    )

    predictor = DeepLPPLSPredictor(config=config, use_conv=True)
    history = predictor.train(n_samples=20000, epochs=30)

    # Test prediction
    generator = LPPLSDataGenerator(length=252)
    test_x, test_y = generator.generate_sample()

    pred = predictor.predict(test_x)
    print(f"\nTest prediction:")
    print(f"  True:  tc_norm={test_y[0]:.3f}, m={test_y[1]:.3f}, omega={test_y[2]:.2f}")
    print(f"  Pred:  tc_norm={pred['tc_norm']:.3f}, m={pred['m']:.3f}, omega={pred['omega']:.2f}")

    # Speed comparison
    print("\n")
    compare_speed(n_samples=50)
