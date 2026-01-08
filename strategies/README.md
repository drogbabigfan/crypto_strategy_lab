# Strategies

개별 전략 백테스트 파이프라인을 폴더 단위로 관리합니다.

## 폴더 구조 컨벤션

```
strategies/
├── README.md              # 이 문서
├── {strategy_name}/       # 전략별 폴더
│   ├── run.py             # 실행 스크립트 (필수)
│   ├── config.yaml        # 설정 파일 (선택)
│   ├── results/           # 결과 저장 (gitignore)
│   └── README.md          # 전략 설명 (선택)
```

## 새 전략 만들기

### 1. 폴더 생성

```bash
mkdir strategies/my_strategy
```

### 2. run.py 작성

기존 스크립트 참고해서 작성:
- `scripts/run_btc_wfa.py` - SSL + Fine-tuning + WFA
- `scripts/run_sniper_wfa.py` - Sniper 모델 기반

```python
# strategies/my_strategy/run.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 필요한 모듈만 import
from research.wfa.engine import WFAEngine, WFAConfig
from research.label_optimizer.tbm import TripleBarrierLabeler
# ...

def main():
    # 전략 로직 구현
    pass

if __name__ == "__main__":
    main()
```

### 3. 실행

```bash
python strategies/my_strategy/run.py
```

## 사용 가능한 모듈

### 핵심 모듈 (research/)

| 모듈 | 용도 |
|------|------|
| `research.ssl.*` | SSL 사전학습 |
| `research.trainer.*` | Fine-tuning |
| `research.sniper.*` | Sniper 분류기 |
| `research.wfa.engine` | WFA 오케스트레이션 |
| `research.wfa.go_bridge` | Go 백테스터 연동 |
| `research.wfa.signal_generator` | 시그널 생성 |
| `research.label_optimizer.*` | 라벨 최적화 |
| `research.backtest.backtester` | Python 백테스터 |
| `research.features.*` | 피처 엔지니어링 |

### Import 예시

```python
# SSL + Fine-tuning
from research.ssl.train import SSLTrainer, SSLTrainingConfig
from research.ssl.model import SSLModel
from research.trainer.model import PatchTSTClassifier
from research.trainer.train import FineTuningTrainer

# WFA
from research.wfa.engine import WFAEngine, WFAConfig
from research.wfa.go_bridge import GoBridge, BacktestConfig
from research.wfa.signal_generator import SignalGenerator

# Labels
from research.label_optimizer.tbm import TripleBarrierLabeler, TBMConfig
from research.label_optimizer.optimizer import LabelOptimizer
```

## config.yaml 예시 (선택)

```yaml
strategy:
  name: my_strategy
  version: v1.0

data:
  symbol: BTCUSDT
  path: etl/data/features-24/futures/BTCUSDT

wfa:
  train_months: 6
  test_months: 1

model:
  context_len: 512
  batch_size: 64

labels:
  sl_mult: 2.0
  pt_mult: 2.5
  max_hold_bars: 100
```

## 참고 스크립트

| 스크립트 | 설명 |
|----------|------|
| `scripts/run_btc_wfa.py` | SSL 인코더 + WFA 전체 파이프라인 |
| `scripts/run_sniper_wfa.py` | Sniper 분류기 기반 WFA |
