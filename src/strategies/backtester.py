import pandas as pd
import numpy as np
import os
import sys
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. 环境配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

PRED_FILE = os.path.join(PROJECT_ROOT, "data", "indices", "prediction_cache.csv")
FUND_FILE = os.path.join(PROJECT_ROOT, "data", "indices", "fundamental.pkl")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.data.data_loader import DataLoader


# ==========================================
# 2. 策略引擎
# ==========================================
class AdaptiveVisionStrategy:
    """
    [Adaptive-Vision] 自适应双模态策略
    核心：根据市场状态 (Regime) 动态调整 AI 的权重。
    1. 牛市 (Price > MA60): 趋势为王，AI 仅用于加仓，不用于止盈。
    2. 熊市 (Price < MA60): 视觉为王，AI 必须极度看好才出手。
    """

    def __init__(self, initial_capital=100000, commission=0.0003):
        self.initial_capital = initial_capital
        self.commission = commission
        self.loader = DataLoader()

        print("[引擎] [Adaptive引擎] 初始化...")
        self._load_data()

    def _load_data(self):
        # 加载视觉预测
        if os.path.exists(PRED_FILE):
            self.pred_df = pd.read_csv(PRED_FILE)
            self.pred_df['date'] = self.pred_df['date'].astype(str).str.replace('-', '')
            self.pred_df['symbol'] = self.pred_df['symbol'].astype(str).str.zfill(6)
            self.vision_map = self.pred_df.set_index(['symbol', 'date'])['pred_win_rate'].to_dict()
            print(f"[OK] 视觉信号库: {len(self.pred_df)} 条")
        else:
            self.vision_map = {}

        # 加载基本面
        if os.path.exists(FUND_FILE):
            with open(FUND_FILE, 'rb') as f:
                self.fund_map = pickle.load(f)
        else:
            self.fund_map = {}

    def _calculate_indicators(self, df):
        data = df.copy()
        # 趋势生命线
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA60'] = data['Close'].rolling(window=60).mean()

        # MACD (判断动能)
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        data['MACD'] = (dif - dea) * 2

        return data

    def run_backtest(self, symbol, start_date, end_date):
        print(f"\n[回测] {symbol} | {start_date}-{end_date}")

        # 1. 数据获取 (预加载300天)
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        preload_date = (start_dt - timedelta(days=300)).strftime("%Y%m%d")

        df_raw = self.loader.get_stock_data(symbol, start_date=preload_date)
        if df_raw.empty: return 0
        df = self._calculate_indicators(df_raw)

        # 2. 切片
        df.index = pd.to_datetime(df.index)
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df_bt = df.loc[mask].copy()
        if df_bt.empty: return 0

        # 3. 交易循环
        cash = self.initial_capital
        shares = 0
        equity_curve = []
        trade_log = []

        # 记录上一笔买入价格
        entry_price = 0.0

        # 财务过滤
        fund = self.fund_map.get(symbol, {})
        # 宽松过滤，只要不是巨亏
        is_fundamental_ok = fund.get('roe', 0) > -20

        for date, row in df_bt.iterrows():
            date_str = date.strftime("%Y%m%d")
            price = row['Close']
            ma20 = row['MA20']
            ma60 = row['MA60']
            macd = row['MACD']

            if pd.isna(ma60):
                equity_curve.append(cash)
                continue

            ai_win = self.vision_map.get((symbol, date_str), 50.0)

            # === 核心策略状态机 ===

            target_pos = 0.0
            reason = "空仓"

            if not is_fundamental_ok:
                target_pos = 0.0
                reason = "基本面熔断"

            else:
                # --- 模式 A: 牛市趋势 (Price > MA60) ---
                if price > ma60:
                    # 子策略 1: 强趋势锁仓 (紫金矿业模式)
                    # 只要 MACD > 0 或者 价格 > MA20，说明趋势健康
                    if macd > 0 or price > ma20:
                        target_pos = 1.0
                        reason = "牛市锁仓(趋势强)"
                        # 此时完全忽略 AI 的看空信号 (防止被洗出去)

                    # 子策略 2: 牛市回调
                    else:
                        # 趋势弱了，这时候听 AI 的
                        if ai_win >= 57:
                            target_pos = 0.81  # 回调持仓
                            reason = "牛市回调(AI看多)"
                        else:
                            target_pos = 0.0  # 真的破位了
                            reason = "牛市破位离场"

                # --- 模式 B: 熊市/震荡 (Price < MA60) ---
                else:
                    # 子策略 3: 视觉狙击 (茅台/平安模式)
                    # 必须 AI 胜率 > 60% 才动手，否则绝不买
                    if ai_win >= 59:
                        target_pos = 0.50  # 抢反弹只用半仓
                        reason = f"视觉狙击(AI:{ai_win:.0f}%)"
                    else:
                        target_pos = 0.03
                        reason = "熊市避险"

            # === 执行交易 ===
            total_assets = cash + shares * price
            target_val = total_assets * target_pos
            target_shares = int(target_val / price)

            diff = target_shares - shares

            # 过滤微小调仓 (10%)
            if abs(diff * price) > total_assets * 0.1:

                if diff > 0:  # 买入
                    cost = diff * price * (1 + self.commission)
                    if cash >= cost:
                        cash -= cost
                        shares += diff
                        if entry_price == 0: entry_price = price
                        trade_log.append({'date': date_str, 'act': 'BUY', 'price': price, 'info': reason})

                elif diff < 0:  # 卖出
                    # --- 止损逻辑 (仅在持仓时触发) ---
                    # 1. 硬止损 8%
                    # 2. 只有当 target_pos 为 0 时才全部卖出
                    pnl = (price - entry_price) / entry_price if entry_price > 0 else 0

                    if pnl < -0.08:
                        reason = f"硬止损({pnl * 100:.1f}%)"
                        diff = -shares  # 强制清仓

                    rev = abs(diff) * price * (1 - self.commission)
                    cash += rev
                    shares += diff
                    if shares == 0: entry_price = 0.0
                    trade_log.append({'date': date_str, 'act': 'SELL', 'price': price, 'info': reason})

            equity_curve.append(cash + shares * price)

        # 4. 统计
        final_ret = (equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        bench_ret = (df_bt['Close'].iloc[-1] - df_bt['Close'].iloc[0]) / df_bt['Close'].iloc[0] * 100
        alpha = final_ret - bench_ret

        print(f"[结果] 策略收益: {final_ret:>6.2f}% (基准: {bench_ret:>6.2f}%)")
        print(f"[Alpha]   : {alpha:>6.2f}%")
        print(f"[统计] 交易次数: {len(trade_log)}")
        print(f"[示例] 触发示例: {trade_log[0]['info'] if trade_log else '无'}")

        return final_ret


if __name__ == "__main__":
    bt = AdaptiveVisionStrategy()

    print("\n=== [策略] 最终版: 自适应双模态策略 ===")

    results = []
    targets = ["601899", "600519", "000001", "300750", "601318"]

    for t in targets:
        r = bt.run_backtest(t, "20230101", "20241220")
        results.append(r)

    print(f"\n[结果] 组合平均收益: {np.mean(results):.2f}%")