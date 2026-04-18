import copy
import sys
from pathlib import Path

import pandas as pd

# чтобы работали импорты из src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_backtest import run_backtest_df
from trend_pullback.config import load_config
from trend_pullback.datafeed import load_ohlcv

CONFIG_PATH = "configs/sample_backtest.yaml"

IS_BARS = 20_000
OOS_BARS = 5_000
STEP_BARS = 5_000

PARAM_SETS = [
    {"name": "main",      "fast_len": 50, "pull_len": 13, "atr_mul": 2.0, "rr": 2.25},
    {"name": "reserve_1", "fast_len": 45, "pull_len": 13, "atr_mul": 2.0, "rr": 2.25},
    {"name": "reserve_2", "fast_len": 34, "pull_len": 13, "atr_mul": 2.0, "rr": 2.25},
]


def generate_windows(df: pd.DataFrame, is_bars: int, oos_bars: int, step_bars: int):
    windows = []
    start = 0
    window_id = 1

    while start + is_bars + oos_bars <= len(df):
        windows.append({
            "window_id": window_id,
            "is_start": start,
            "is_end": start + is_bars,
            "oos_start": start + is_bars,
            "oos_end": start + is_bars + oos_bars,
        })
        start += step_bars
        window_id += 1

    return windows


def extract_metrics(result: dict) -> dict:
    summary = result["summary"]
    return {
        "net_profit_pct": summary.get("net_profit_pct"),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "total_trades": summary.get("total_trades"),
        "win_rate_pct": summary.get("win_rate_pct"),
    }


def run_oos_for_fixed_params(df: pd.DataFrame, windows: list[dict], base_cfg, param_sets: list[dict]) -> pd.DataFrame:
    rows = []

    for w in windows:
        is_df = df.iloc[w["is_start"]:w["is_end"]].copy()
        oos_df = df.iloc[w["oos_start"]:w["oos_end"]].copy()

        for params in param_sets:
            sp = copy.deepcopy(base_cfg.strategy)
            bp = copy.deepcopy(base_cfg.backtest)

            sp.fast_len = params["fast_len"]
            sp.pull_len = params["pull_len"]
            sp.atr_mul = params["atr_mul"]
            sp.rr = params["rr"]

            is_result = run_backtest_df(
                df=is_df,
                strategy_params=sp,
                backtest_params=bp,
                output_dir=None,
                save=False,
                print_report=False,
            )

            oos_result = run_backtest_df(
                df=oos_df,
                strategy_params=sp,
                backtest_params=bp,
                output_dir=None,
                save=False,
                print_report=False,
            )

            is_metrics = extract_metrics(is_result)
            oos_metrics = extract_metrics(oos_result)

            rows.append({
                "window_id": w["window_id"],

                "is_start_dt": is_df.index[0],
                "is_end_dt": is_df.index[-1],
                "oos_start_dt": oos_df.index[0],
                "oos_end_dt": oos_df.index[-1],

                "set_name": params["name"],
                "fast_len": params["fast_len"],
                "pull_len": params["pull_len"],
                "atr_mul": params["atr_mul"],
                "rr": params["rr"],

                "is_net_profit_pct": is_metrics["net_profit_pct"],
                "is_max_drawdown_pct": is_metrics["max_drawdown_pct"],
                "is_total_trades": is_metrics["total_trades"],
                "is_win_rate_pct": is_metrics["win_rate_pct"],

                "oos_net_profit_pct": oos_metrics["net_profit_pct"],
                "oos_max_drawdown_pct": oos_metrics["max_drawdown_pct"],
                "oos_total_trades": oos_metrics["total_trades"],
                "oos_win_rate_pct": oos_metrics["win_rate_pct"],
            })

    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("set_name", as_index=False)
        .agg(
            windows=("window_id", "count"),
            avg_oos_net_profit_pct=("oos_net_profit_pct", "mean"),
            median_oos_net_profit_pct=("oos_net_profit_pct", "median"),
            total_oos_net_profit_pct=("oos_net_profit_pct", "sum"),
            worst_oos_net_profit_pct=("oos_net_profit_pct", "min"),
            avg_oos_max_drawdown_pct=("oos_max_drawdown_pct", "mean"),
            worst_oos_max_drawdown_pct=("oos_max_drawdown_pct", "min"),
            avg_oos_win_rate_pct=("oos_win_rate_pct", "mean"),
            avg_oos_total_trades=("oos_total_trades", "mean"),
            positive_oos_windows=("oos_net_profit_pct", lambda s: int((s > 0).sum())),
        )
        .sort_values(
            ["positive_oos_windows", "total_oos_net_profit_pct", "avg_oos_max_drawdown_pct"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    return summary


def main():
    cfg = load_config(CONFIG_PATH)
    df = load_ohlcv(cfg.backtest.data_path)

    windows = generate_windows(df, IS_BARS, OOS_BARS, STEP_BARS)
    if not windows:
        raise RuntimeError("Not enough bars for the selected walk-forward setup.")

    results = run_oos_for_fixed_params(df, windows, cfg, PARAM_SETS)
    summary = build_summary(results)

    out_dir = Path("output/walk_forward")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "walk_forward_oos_results.csv"
    summary_path = out_dir / "walk_forward_oos_summary.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== WALK-FORWARD OOS SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results -> {results_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()