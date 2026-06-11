import warnings
warnings.filterwarnings("ignore")

import math
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from arch import arch_model


class MultiAssetTargetVolatility:
    """
    Object-Oriented implementation of the Target Volatility Strategy.
    Includes RMSE evaluation, Bootstrap Confidence Intervals for Sharpe ratios,
    adaptive landscape grid plotting, and Pandas DataFrame reporting.
    """

    def __init__(self, tickers, target_vol=0.12, start_date="2015-01-01",
                 end_date="2026-06-01", seq_length=20, random_seed=42):
        self.tickers = tickers
        self.target_vol = target_vol
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_length
        self.results = {}
        self.summary_data = []
        self.rng = np.random.default_rng(random_seed)

    def _create_sequences(self, X, y):
        """Formats 3D tensors for LSTM input."""
        Xs, ys = [], []
        for i in range(len(X) - self.seq_len):
            Xs.append(X.iloc[i:(i + self.seq_len)].values)
            ys.append(y.iloc[i + self.seq_len])
        return np.array(Xs), np.array(ys)

    def _get_metrics(self, equity_curve):
        """Computes Annualized Return, Volatility, Sharpe Ratio, and Maximum Drawdown."""
        eq = np.array(equity_curve)
        rets = np.diff(eq) / eq[:-1]

        total_ret = (eq[-1] / eq[0]) - 1
        years = len(rets) / 252
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

        vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.001
        sharpe = (np.mean(rets) * 252) / vol if vol > 0 else 0

        peak = np.maximum.accumulate(eq)
        max_dd = np.max((peak - eq) / peak) if len(eq) > 0 else 0

        return ann_ret, vol, sharpe, max_dd

    def block_bootstrap_ci(self, stat_func, *args, block_size=20, n_boot=5000,
                           alpha=0.05) -> tuple[float, float]:
        """
        Generic block bootstrap method for computing confidence intervals.
        Uses the instance's local random generator (self.rng) for reproducibility.
        """
        n = len(args[0])
        num_blocks = int(np.ceil(n / block_size))
        boot_stats = np.zeros(n_boot)

        for b in range(n_boot):
            start_idx = self.rng.integers(0, n - block_size + 1, size=num_blocks)
            boot_idx = np.concatenate([np.arange(i, i + block_size)
                                       for i in start_idx])[:n]
            boot_stats[b] = stat_func(boot_idx, *args)

        lower = float(np.percentile(boot_stats, (alpha / 2) * 100))
        upper = float(np.percentile(boot_stats, (1 - alpha / 2) * 100))
        return lower, upper

    def bootstrap_sharpe_diff(self, eq_lstm, eq_garch, block_size=20,
                              n_boot=5000, alpha=0.05) -> tuple[float, float]:
        """Bootstrap CI for the difference in Sharpe Ratios (LSTM - GARCH)."""
        rets_lstm = np.diff(eq_lstm) / eq_lstm[:-1]
        rets_garch = np.diff(eq_garch) / eq_garch[:-1]

        def sharpe_diff_stat(idx, r_lstm, r_garch):
            b_lstm = r_lstm[idx]
            b_garch = r_garch[idx]

            v_lstm = np.std(b_lstm) * np.sqrt(252)
            v_garch = np.std(b_garch) * np.sqrt(252)

            s_lstm = (np.mean(b_lstm) * 252) / v_lstm if v_lstm > 0 else 0
            s_garch = (np.mean(b_garch) * 252) / v_garch if v_garch > 0 else 0

            return s_lstm - s_garch

        return self.block_bootstrap_ci(sharpe_diff_stat, rets_lstm, rets_garch,
                                       block_size=block_size, n_boot=n_boot,
                                       alpha=alpha)

    def bootstrap_rmse_diff(self, garch_preds, lstm_preds, actual_vol,
                            block_size=20, n_boot=5000, alpha=0.05) -> tuple[float, float]:
        """Bootstrap CI for the difference in RMSE (GARCH - LSTM)."""
        def rmse_diff_stat(idx, p_garch, p_lstm, actual):
            b_garch = p_garch[idx]
            b_lstm = p_lstm[idx]
            b_act = actual[idx]

            rmse_g = np.sqrt(np.mean((b_garch - b_act) ** 2))
            rmse_l = np.sqrt(np.mean((b_lstm - b_act) ** 2))

            return rmse_g - rmse_l

        return self.block_bootstrap_ci(rmse_diff_stat, garch_preds, lstm_preds,
                                       actual_vol, block_size=block_size,
                                       n_boot=n_boot, alpha=alpha)

    def process_ticker(self, ticker):
        print(f"--- PROCESSING ASSET: {ticker} ---")

        raw = yf.download([ticker, "^VIX"], start=self.start_date,
                          end=self.end_date, progress=False)['Close']
        df = pd.DataFrame()
        df['close'] = raw[ticker]
        df['vix'] = raw['^VIX']

        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['vol_hist'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252)

        # Target volatility definition: 5-day realized volatility, annualized
        df['future_vol'] = df['log_ret'].rolling(window=5).std() * np.sqrt(252)
        df['future_vol'] = df['future_vol'].shift(-5)
        df = df.dropna()

        split_idx = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

        # GARCH(1,1) with 5-day ahead forecast
        print("Training GARCH(1,1) model and generating 5-day forecasts...")
        am = arch_model(df['log_ret'] * 100, vol='Garch', p=1, q=1)
        res = am.fit(last_obs=train_df.index[-1], disp='off')
        forecasts = res.forecast(start=test_df.index[0], horizon=5, align='target')

        variance_5d = forecasts.variance[['h.1', 'h.2', 'h.3', 'h.4', 'h.5']].sum(axis=1)
        garch_vol_full = np.sqrt((variance_5d / 5) / 10000 * 252)
        garch_preds_test = garch_vol_full.loc[test_df.index].values

        # LSTM training
        print("Training LSTM neural network...")
        features = ['log_ret', 'vol_hist', 'vix']
        scaler_X, scaler_y = StandardScaler(), StandardScaler()

        X_train_scaled = pd.DataFrame(scaler_X.fit_transform(train_df[features]),
                                      columns=features)
        X_test_scaled = pd.DataFrame(scaler_X.transform(test_df[features]),
                                     columns=features)
        y_train_scaled = pd.Series(scaler_y.fit_transform(train_df[['future_vol']]).flatten())
        y_test_scaled = pd.Series(scaler_y.transform(test_df[['future_vol']]).flatten())

        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled)
        X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test_scaled)

        model = Sequential([
            LSTM(50, activation='tanh',
                 input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32,
                  validation_split=0.1, callbacks=[es], verbose=0)

        lstm_preds_scaled = model.predict(X_test_seq, verbose=0).flatten()
        lstm_preds_raw = scaler_y.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()

        lstm_preds = np.nan_to_num(lstm_preds_raw, nan=self.target_vol)
        garch_preds = np.nan_to_num(garch_preds_test[self.seq_len:], nan=self.target_vol)

        actual_returns = (np.exp(test_df['log_ret'].iloc[self.seq_len:]) - 1).values
        actual_vol = test_df['future_vol'].iloc[self.seq_len:].values
        dates_test = test_df.index[self.seq_len:]

        cap_garch, cap_lstm, cap_bh = [10000], [10000], [10000]
        for i in range(len(actual_returns) - 1):
            w_g = min(1.0, self.target_vol / max(0.001, garch_preds[i]))
            w_l = min(1.0, self.target_vol / max(0.001, lstm_preds[i]))

            cap_garch.append(cap_garch[-1] * (1 + w_g * actual_returns[i+1]))
            cap_lstm.append(cap_lstm[-1] * (1 + w_l * actual_returns[i+1]))
            cap_bh.append(cap_bh[-1] * (1 + actual_returns[i+1]))

        self.results[ticker] = {
            'dates': dates_test,
            'eq_garch': cap_garch,
            'eq_lstm': cap_lstm,
            'eq_bh': cap_bh,
            'vol_actual': actual_vol,
            'vol_garch': garch_preds,
            'vol_lstm': lstm_preds
        }

        mg = self._get_metrics(cap_garch)
        ml = self._get_metrics(cap_lstm)
        mbh = self._get_metrics(cap_bh)

        rmse_garch = np.sqrt(np.mean((garch_preds - actual_vol) ** 2))
        rmse_lstm = np.sqrt(np.mean((lstm_preds - actual_vol) ** 2))

        ci_lower_sharpe, ci_upper_sharpe = self.bootstrap_sharpe_diff(
            cap_lstm, cap_garch, block_size=20, n_boot=5000)
        ci_lower_rmse, ci_upper_rmse = self.bootstrap_rmse_diff(
            garch_preds, lstm_preds, actual_vol, block_size=20, n_boot=5000)

        self.summary_data.append({
            'Asset': ticker,
            'Strategy': 'Buy & Hold',
            'Ann. Return (%)': round(mbh[0] * 100, 2),
            'Sharpe Ratio': round(mbh[2], 2),
            'Max DD (%)': round(mbh[3] * 100, 2),
            'RMSE (%)': None,
            'RMSE Diff CI (GARCH-LSTM)': '-',
            'CI 95% (LSTM-GARCH)': '-'
        })
        self.summary_data.append({
            'Asset': ticker,
            'Strategy': 'Target Vol GARCH',
            'Ann. Return (%)': round(mg[0] * 100, 2),
            'Sharpe Ratio': round(mg[2], 2),
            'Max DD (%)': round(mg[3] * 100, 2),
            'RMSE (%)': round(rmse_garch * 100, 2),
            'RMSE Diff CI (GARCH-LSTM)': '-',
            'CI 95% (LSTM-GARCH)': '-'
        })
        self.summary_data.append({
            'Asset': ticker,
            'Strategy': 'Target Vol LSTM',
            'Ann. Return (%)': round(ml[0] * 100, 2),
            'Sharpe Ratio': round(ml[2], 2),
            'Max DD (%)': round(ml[3] * 100, 2),
            'RMSE (%)': round(rmse_lstm * 100, 2),
            'RMSE Diff CI (GARCH-LSTM)': f"[{ci_lower_rmse * 100:.2f}, {ci_upper_rmse * 100:.2f}]",
            'CI 95% (LSTM-GARCH)': f"[{ci_lower_sharpe:.2f}, {ci_upper_sharpe:.2f}]"
        })

    def run_all(self):
        """Iterates through all provided tickers."""
        print(f"Starting execution for Target Volatility: {self.target_vol:.0%}")
        for ticker in self.tickers:
            self.process_ticker(ticker)

    def display_summary_table(self):
        """Creates and displays a professional Pandas DataFrame of all metrics."""
        if not self.summary_data:
            print("No data to display. Please run the backtest first.")
            return None

        df = pd.DataFrame(self.summary_data)

        print("\n" + "=" * 135)
        print("STRATEGY PERFORMANCE SUMMARY".center(135))
        print("=" * 135)
        print(df.to_string(index=False, na_rep="N/A"))
        print("=" * 135 + "\n")

        return df

    def _setup_grid(self):
        """Generates an adaptive grid with landscape proportions."""
        n = len(self.tickers)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(24, 4.5 * rows))

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        return fig, axes

    def plot_equity_grid(self):
        """Plots the equity curve comparison for all assets."""
        if not self.results:
            return

        fig, axes = self._setup_grid()
        fig.suptitle(f"Performance Comparison: Target Volatility Strategy "
                     f"({self.target_vol:.0%})", fontsize=16, fontweight='bold', y=0.94)

        for i, ticker in enumerate(self.tickers):
            ax = axes[i]
            res = self.results[ticker]
            dates = res['dates']

            ax.plot(dates, res['eq_bh'], label="Buy & Hold", color='gray',
                    linestyle='--', linewidth=1.5, alpha=0.7)
            ax.plot(dates, res['eq_garch'], label="GARCH Portfolio", color='red',
                    linewidth=1.5)
            ax.plot(dates, res['eq_lstm'], label="LSTM Portfolio", color='blue',
                    linewidth=2.0)

            ax.set_title(f"{ticker}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Portfolio Value (USD)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim(dates[0], dates[-1])
            ax.tick_params(axis='x', rotation=45)

            if i == 0:
                ax.legend(loc="upper left")

        for j in range(len(self.tickers), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0.03, 0.08, 0.98, 0.92], h_pad=5.0, w_pad=6.0)
        plt.show()

    def plot_volatility_grid(self):
        """Plots the predicted vs actual volatility for all assets."""
        if not self.results:
            return

        fig, axes = self._setup_grid()
        fig.suptitle("Volatility Forecasting: Actual vs GARCH vs LSTM",
                     fontsize=16, fontweight='bold', y=0.94)

        for i, ticker in enumerate(self.tickers):
            ax = axes[i]
            res = self.results[ticker]
            dates = res['dates']

            ax.plot(dates, res['vol_actual'] * 100, label="Actual Volatility",
                    color='gray', linestyle='-', alpha=0.3, linewidth=2.0)
            ax.plot(dates, res['vol_garch'] * 100, label="GARCH Pred",
                    color='red', linewidth=1.2, alpha=0.8)
            ax.plot(dates, res['vol_lstm'] * 100, label="LSTM Pred",
                    color='blue', linewidth=1.5)

            ax.axhline(self.target_vol * 100, color='green', linestyle=':',
                       linewidth=2, label=f"Target ({self.target_vol:.0%})")

            ax.set_title(f"{ticker} Volatility Profiling", fontsize=12, fontweight='bold')
            ax.set_ylabel("Annualized Volatility (%)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim(dates[0], dates[-1])
            ax.tick_params(axis='x', rotation=45)

            if i == 0:
                ax.legend(loc="upper left")

        for j in range(len(self.tickers), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0.03, 0.08, 0.98, 0.92], h_pad=5.0, w_pad=6.0)
        plt.show()


if __name__ == "__main__":
    SEED =42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    ASSET_LIST = ["SPY", "NVDA", "PG"]

    strategy = MultiAssetTargetVolatility(tickers=ASSET_LIST, target_vol=0.12, random_seed=42)
    strategy.run_all()

    df_results = strategy.display_summary_table()

    strategy.plot_equity_grid()
    strategy.plot_volatility_grid()
