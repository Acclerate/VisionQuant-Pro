"""
分层回测可视化
Stratified Backtesting Visualization

可视化分层回测结果

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os

from src.strategies.backtest_engine import BacktestResult


class StratifiedVisualizer:
    """
    分层回测可视化器
    
    功能：
    1. 绘制分层对比图
    2. 绘制市值/行业热力图
    3. 绘制收益分布图
    """
    
    def __init__(self):
        """初始化可视化器"""
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_stratified_comparison(
        self,
        stratified_results: Dict[str, BacktestResult],
        output_path: str,
        metric: str = 'sharpe_ratio'
    ):
        """
        绘制分层对比图
        
        Args:
            stratified_results: 分层回测结果
            output_path: 输出路径
            metric: 对比指标 ('sharpe_ratio', 'total_return', 'win_rate')
        """
        # 提取数据
        strata = []
        values = []
        
        for stratum, result in stratified_results.items():
            strata.append(stratum)
            if metric == 'sharpe_ratio':
                values.append(result.sharpe_ratio)
            elif metric == 'total_return':
                values.append(result.total_return * 100)
            elif metric == 'win_rate':
                values.append(result.win_rate * 100)
            else:
                values.append(0.0)
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(strata, values, color='steelblue')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f' {val:.2f}', va='center')
        
        # 设置标签
        metric_labels = {
            'sharpe_ratio': 'Sharpe Ratio',
            'total_return': 'Total Return (%)',
            'win_rate': 'Win Rate (%)'
        }
        ax.set_xlabel(metric_labels.get(metric, metric))
        ax.set_title(f'Stratified Backtest Comparison - {metric_labels.get(metric, metric)}')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_heatmap(
        self,
        stratified_results: Dict[str, BacktestResult],
        output_path: str,
        metric: str = 'sharpe_ratio'
    ):
        """
        绘制热力图（市值×行业）
        
        Args:
            stratified_results: 分层回测结果
            output_path: 输出路径
            metric: 指标
        """
        # 构建矩阵
        market_caps = ['small', 'medium', 'large']
        industries = set()
        
        for stratum in stratified_results.keys():
            parts = stratum.split('_', 1)
            if len(parts) == 2:
                industries.add(parts[1])
        
        industries = sorted(list(industries))
        
        # 创建矩阵
        matrix = np.zeros((len(market_caps), len(industries)))
        
        for stratum, result in stratified_results.items():
            parts = stratum.split('_', 1)
            if len(parts) == 2:
                market_cap, industry = parts
                if market_cap in market_caps and industry in industries:
                    i = market_caps.index(market_cap)
                    j = industries.index(industry)
                    
                    if metric == 'sharpe_ratio':
                        matrix[i, j] = result.sharpe_ratio
                    elif metric == 'total_return':
                        matrix[i, j] = result.total_return * 100
                    elif metric == 'win_rate':
                        matrix[i, j] = result.win_rate * 100
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(
            matrix,
            xticklabels=industries,
            yticklabels=market_caps,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': metric_labels.get(metric, metric)}
        )
        
        ax.set_title(f'Stratified Backtest Heatmap - {metric_labels.get(metric, metric)}')
        ax.set_xlabel('Industry')
        ax.set_ylabel('Market Cap')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_return_distribution(
        self,
        stratified_results: Dict[str, BacktestResult],
        output_path: str
    ):
        """
        绘制收益分布图
        
        Args:
            stratified_results: 分层回测结果
            output_path: 输出路径
        """
        returns = [r.total_return * 100 for r in stratified_results.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(returns, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        ax.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.2f}%')
        
        ax.set_xlabel('Total Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Stratified Backtest Returns')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# 添加metric_labels（在类外部定义，供所有方法使用）
metric_labels = {
    'sharpe_ratio': 'Sharpe Ratio',
    'total_return': 'Total Return (%)',
    'win_rate': 'Win Rate (%)'
}


if __name__ == "__main__":
    print("=== 分层可视化器测试 ===")
    
    from src.strategies.backtest_engine import BacktestResult
    
    # 模拟结果
    results = {
        'small_银行': BacktestResult(
            total_return=0.15, sharpe_ratio=1.2, max_drawdown=0.10,
            win_rate=0.55, profit_factor=1.5, total_trades=100,
            annualized_return=0.15, avg_holding_period=5.0,
            equity_curve=pd.Series(), trade_log=[], fold_results=[]
        ),
        'large_科技': BacktestResult(
            total_return=0.25, sharpe_ratio=1.8, max_drawdown=0.12,
            win_rate=0.60, profit_factor=1.8, total_trades=80,
            annualized_return=0.25, avg_holding_period=6.0,
            equity_curve=pd.Series(), trade_log=[], fold_results=[]
        )
    }
    
    visualizer = StratifiedVisualizer()
    visualizer.plot_stratified_comparison(results, 'test_comparison.png')
    print("[OK] 可视化图表已生成")
