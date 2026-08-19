import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

_FEATURES = ["open", "high", "low", "close", "volume",
             "ema9", "ema21", "rsi14", "atr14", "vwap", "session_band", "ret1", "vol20",
             "boll_pos", "macd_hist", "body_pct", "upper_wick_pct",
             "lower_wick_pct", "dist_to_high_pct"]


class TradingEnv(gym.Env):
    def __init__(self, symbol, bars, initial_cash=100_000.0, cost_pct=0.001,
                 window=120, seed=42, holding_penalty=0.0):
        super().__init__()
        self.symbol = symbol
        self.bars = bars.fill_null(strategy="forward").fill_null(0.0)
        self.initial_cash = float(initial_cash)
        self.cost_pct = cost_pct
        self.window = window
        self.holding_penalty = holding_penalty
        self.n_features = len(_FEATURES)
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(3)  # 0=flat, 1=long, 2=short
        flat_dim = window * self.n_features + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32)
        self.reset(seed=seed)

    def _obs(self) -> np.ndarray:
        arr = self.bars.select(_FEATURES).to_numpy()
        start = max(0, self._idx - self.window + 1)
        chunk = arr[start:self._idx + 1]
        if len(chunk) < self.window:
            pad = np.zeros((self.window - len(chunk), self.n_features))
            chunk = np.vstack([pad, chunk])
        flat = chunk.reshape(-1)
        pos = np.array([self.position], dtype=np.float32)
        cash_ratio = np.array([self.cash / self.initial_cash], dtype=np.float32)
        return np.concatenate([flat, pos, cash_ratio]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._idx = self.window
        self.cash = self.initial_cash
        self.position = 0.0
        self.equity = self.initial_cash
        self._peak = self.initial_cash
        return self._obs(), {}

    def _price(self):
        return float(self.bars["close"][self._idx])

    def _equity(self):
        return self.cash + self.position * self._price()

    def step(self, action):
        target = {0: 0.0, 1: 1.0, 2: -1.0}[int(action)]
        turnover = 0.0
        if target != self.position:
            price = self._price()
            delta = target - self.position
            turnover = abs(delta) * self.equity * self.cost_pct
            if target > 0:
                self.cash -= target * price
            elif target < 0:
                self.cash += abs(target) * price
            else:
                self.cash += self.position * price
            self.position = target
        self._idx += 1
        prev = self.equity
        self.equity = self._equity()
        self._peak = max(self._peak, self.equity)
        dd = (self._peak - self.equity) / self._peak
        equity_delta = self.equity - prev
        cost_term = turnover
        dd_term = 0.1 * dd
        hold_term = self.holding_penalty * abs(self.position)
        reward = equity_delta - cost_term - dd_term - hold_term
        terminated = self._idx >= len(self.bars) - 1
        info = {"equity": self.equity,
                "reward_terms": {"equity_delta": equity_delta, "cost": cost_term,
                                 "drawdown": dd_term, "holding": hold_term}}
        return self._obs(), float(reward), terminated, False, info