"""
Top10 形态分析增强模块
Enhanced Top-10 Pattern Analysis

增强Top10对比的信息量:
1. 统计信息 - 平均收益、胜率分布、风险指标
2. 时间分布 - 年份、月份、市场周期
3. 行业分布 - 匹配形态来自哪些行业
4. 收益轨迹 - 更详细的未来走势对比

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Top10Analyzer:
    """
    Top10形态深度分析器
    
    用法:
    ```python
    analyzer = Top10Analyzer(data_loader)
    stats = analyzer.analyze_matches(matches)
    chart = analyzer.create_enhanced_chart(matches, query_img_path)
    ```
    """
    
    def __init__(self, data_loader=None):
        """
        初始化分析器
        
        Args:
            data_loader: 数据加载器，用于获取历史数据
        """
        self.data_loader = data_loader
        
        # 行业映射（简化版）
        self.industry_map = {
            '60': '上海主板',
            '00': '深圳主板',
            '30': '创业板',
            '68': '科创板',
        }
        
    def analyze_matches(
        self,
        matches: List[Dict],
        future_days: int = 20
    ) -> Dict:
        """
        分析Top10匹配结果
        
        Args:
            matches: 匹配结果列表
            future_days: 计算未来收益的天数
            
        Returns:
            统计分析结果
        """
        if not matches:
            return {'valid': False, 'message': '无匹配数据'}
        
        # 收集统计数据
        returns = []
        max_returns = []
        max_drawdowns = []
        hit_days = []  # 达到止盈/止损的天数
        years = []
        months = []
        boards = []  # 板块
        
        for match in matches:
            symbol = str(match.get('symbol', '')).zfill(6)
            date_str = str(match.get('date', ''))
            
            # 解析日期
            try:
                if '-' in date_str:
                    match_date = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    match_date = datetime.strptime(date_str, '%Y%m%d')
            except:
                continue
            
            years.append(match_date.year)
            months.append(match_date.month)
            
            # 板块分类
            prefix = symbol[:2]
            boards.append(self.industry_map.get(prefix, '其他'))
            
            # 获取未来收益数据
            if self.data_loader:
                try:
                    df = self.data_loader.get_stock_data(symbol)
                    if df is not None and not df.empty:
                        df.index = pd.to_datetime(df.index)
                        
                        if match_date in df.index:
                            loc = df.index.get_loc(match_date)
                            
                            if loc + future_days < len(df):
                                entry_price = df.iloc[loc]['Close']
                                future_prices = df.iloc[loc+1:loc+1+future_days]['Close']
                                
                                # 计算收益
                                future_returns = (future_prices - entry_price) / entry_price * 100
                                
                                returns.append(future_returns.iloc[-1])
                                max_returns.append(future_returns.max())
                                max_drawdowns.append(future_returns.min())
                                
                                # 计算首次达到5%的天数
                                above_5 = future_returns >= 5
                                if above_5.any():
                                    hit_days.append(above_5.idxmax())
                except:
                    pass
        
        # 计算统计指标
        result = {
            'valid': True,
            'matches_count': len(matches),
            
            # 收益统计
            'avg_return': np.mean(returns) if returns else 0,
            'median_return': np.median(returns) if returns else 0,
            'std_return': np.std(returns) if returns else 0,
            'min_return': np.min(returns) if returns else 0,
            'max_return': np.max(returns) if returns else 0,
            
            # 胜率统计
            'positive_count': sum(1 for r in returns if r > 0),
            'negative_count': sum(1 for r in returns if r < 0),
            'neutral_count': sum(1 for r in returns if r == 0),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0,
            
            # 风险统计
            'avg_max_return': np.mean(max_returns) if max_returns else 0,
            'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            
            # 时间分布
            'year_distribution': dict(Counter(years)),
            'month_distribution': dict(Counter(months)),
            
            # 板块分布
            'board_distribution': dict(Counter(boards)),
            
            # 相似度统计
            'avg_similarity': np.mean([m.get('score', 0) for m in matches]),
            'max_similarity': max([m.get('score', 0) for m in matches]),
            'min_similarity': min([m.get('score', 0) for m in matches]),
        }
        
        # 计算风险调整收益
        if result['std_return'] > 0:
            result['sharpe_like'] = result['avg_return'] / result['std_return']
        else:
            result['sharpe_like'] = 0
        
        return result

    def analyze_multi_horizon(
        self,
        matches: List[Dict],
        horizons: List[int] = None
    ) -> Dict:
        """
        多期收益统计（5/10/20天）
        """
        if horizons is None:
            horizons = [5, 10, 20]
        result = {"valid": False}
        if not matches or not self.data_loader:
            return result

        horizon_stats = {}
        for h in horizons:
            stats = self.analyze_matches(matches, future_days=h)
            horizon_stats[h] = {
                "avg_return": stats.get("avg_return", 0),
                "median_return": stats.get("median_return", 0),
                "win_rate": stats.get("win_rate", 0),
                "max_drawdown": stats.get("avg_max_drawdown", 0)
            }
        result["valid"] = True
        result["horizon_stats"] = horizon_stats
        return result

    def return_distribution(
        self,
        matches: List[Dict],
        future_days: int = 20
    ) -> Dict:
        """
        收益分布估计：均值/中位数/分位数/CVaR
        """
        stats = self.analyze_matches(matches, future_days=future_days)
        if not stats.get("valid"):
            return {"valid": False}
        returns = stats.get("returns_raw") if "returns_raw" in stats else None
        if returns is None:
            # 兼容旧结构：重新计算
            returns = []
            for m in matches:
                symbol = str(m.get("symbol", "")).zfill(6)
                date_str = str(m.get("date", ""))
                try:
                    match_date = datetime.strptime(date_str, "%Y-%m-%d") if "-" in date_str else datetime.strptime(date_str, "%Y%m%d")
                    df = self.data_loader.get_stock_data(symbol)
                    if df is None or df.empty:
                        continue
                    df.index = pd.to_datetime(df.index)
                    if match_date in df.index:
                        loc = df.index.get_loc(match_date)
                        if loc + future_days < len(df):
                            entry = df.iloc[loc]["Close"]
                            future = df.iloc[loc + future_days]["Close"]
                            returns.append((future - entry) / entry * 100)
                except Exception:
                    continue
        if not returns:
            return {"valid": False}
        s = pd.Series(returns)
        q05 = float(s.quantile(0.05))
        cvar = float(s[s <= q05].mean()) if (s <= q05).any() else q05
        return {
            "valid": True,
            "mean": float(s.mean()),
            "median": float(s.median()),
            "q05": q05,
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "cvar": cvar,
            "count": int(len(s))
        }
    
    def get_return_trajectories(
        self,
        matches: List[Dict],
        future_days: int = 20
    ) -> pd.DataFrame:
        """
        获取所有匹配形态的未来收益轨迹
        
        Args:
            matches: 匹配结果
            future_days: 未来天数
            
        Returns:
            DataFrame，每列是一个匹配形态的收益轨迹
        """
        if not self.data_loader:
            return pd.DataFrame()
        
        trajectories = {}
        
        for i, match in enumerate(matches):
            symbol = str(match.get('symbol', '')).zfill(6)
            date_str = str(match.get('date', ''))
            
            try:
                if '-' in date_str:
                    match_date = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    match_date = datetime.strptime(date_str, '%Y%m%d')
                    
                df = self.data_loader.get_stock_data(symbol)
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index)
                    
                    if match_date in df.index:
                        loc = df.index.get_loc(match_date)
                        
                        if loc + future_days < len(df):
                            entry_price = df.iloc[loc]['Close']
                            future_prices = df.iloc[loc:loc+1+future_days]['Close']
                            
                            returns = (future_prices - entry_price) / entry_price * 100
                            returns = returns.reset_index(drop=True)
                            
                            trajectories[f'Top{i+1}_{symbol}'] = returns
            except:
                continue
        
        return pd.DataFrame(trajectories)
    
    def create_stats_summary(self, stats: Dict) -> str:
        """
        生成统计摘要文本
        
        Args:
            stats: analyze_matches返回的统计结果
            
        Returns:
            格式化的摘要文本
        """
        if not stats.get('valid'):
            return "[警告] 无有效统计数据"
        
        summary = []
        summary.append("[统计] **Top10 形态统计分析**")
        summary.append("")
        
        # 收益统计
        summary.append("[收益] **收益统计**")
        summary.append(f"- 平均收益: {stats['avg_return']:.2f}%")
        summary.append(f"- 中位数收益: {stats['median_return']:.2f}%")
        summary.append(f"- 收益区间: [{stats['min_return']:.2f}%, {stats['max_return']:.2f}%]")
        summary.append(f"- 波动率: {stats['std_return']:.2f}%")
        summary.append("")
        
        # 胜率
        summary.append("[胜率] **胜率分析**")
        summary.append(f"- 上涨数量: {stats['positive_count']}")
        summary.append(f"- 下跌数量: {stats['negative_count']}")
        summary.append(f"- 历史胜率: {stats['win_rate']:.1f}%")
        summary.append("")
        
        # 风险
        summary.append("[风险] **风险指标**")
        summary.append(f"- 平均最大涨幅: {stats['avg_max_return']:.2f}%")
        summary.append(f"- 平均最大回撤: {stats['avg_max_drawdown']:.2f}%")
        summary.append(f"- 风险调整收益: {stats['sharpe_like']:.2f}")
        summary.append("")
        
        # 分布
        if stats.get('board_distribution'):
            summary.append("[板块] **板块分布**")
            for board, count in stats['board_distribution'].items():
                summary.append(f"- {board}: {count}个")
        
        return "\n".join(summary)


def create_enhanced_top10_chart(
    query_image_path: str,
    matches: List[Dict],
    stats: Dict,
    trajectories: pd.DataFrame,
    output_path: str
):
    """
    创建增强版Top10对比图
    
    包含:
    - 查询图像和Top10匹配图
    - 收益轨迹对比图
    - 统计信息面板
    
    Args:
        query_image_path: 查询图像路径
        matches: 匹配结果
        stats: 统计结果
        trajectories: 收益轨迹
        output_path: 输出路径
    """
    import os
    from PIL import Image
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(3, 8, figure=fig, height_ratios=[1.2, 1, 0.8])
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    IMG_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
    
    # === 第一行: 查询图 + Top5匹配 ===
    # 查询图（大）
    ax_query = fig.add_subplot(gs[0, :2])
    if os.path.exists(query_image_path):
        img = Image.open(query_image_path)
        ax_query.imshow(img)
        ax_query.set_title("[形态] 当前形态 (Query)", fontsize=14, fontweight='bold', color='blue')
    ax_query.axis('off')
    
    # Top1-5
    for i in range(min(5, len(matches))):
        ax = fig.add_subplot(gs[0, 2+i])
        match = matches[i]
        
        img_name = f"{match['symbol']}_{match['date']}.png"
        img_path = os.path.join(IMG_BASE_DIR, img_name)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            
        title = f"Top {i+1}\n{match['symbol']}\n{match['date']}\nSim: {match['score']:.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # 统计面板
    ax_stats = fig.add_subplot(gs[0, 7])
    ax_stats.axis('off')
    stats_text = f"""[统计] 统计摘要
    
胜率: {stats.get('win_rate', 0):.1f}%
平均收益: {stats.get('avg_return', 0):.2f}%
最大收益: {stats.get('max_return', 0):.2f}%
最大回撤: {stats.get('avg_max_drawdown', 0):.2f}%
相似度: {stats.get('avg_similarity', 0):.3f}"""
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # === 第二行: Top6-10 + 收益轨迹图 ===
    for i in range(5, min(10, len(matches))):
        ax = fig.add_subplot(gs[1, i-5])
        match = matches[i]
        
        img_name = f"{match['symbol']}_{match['date']}.png"
        img_path = os.path.join(IMG_BASE_DIR, img_name)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            
        title = f"Top {i+1}\n{match['symbol']}\nSim: {match['score']:.3f}"
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # 收益轨迹图
    ax_traj = fig.add_subplot(gs[1, 5:])
    if not trajectories.empty:
        for col in trajectories.columns:
            ax_traj.plot(trajectories[col], alpha=0.5, linewidth=1)
        
        # 平均轨迹
        mean_traj = trajectories.mean(axis=1)
        ax_traj.plot(mean_traj, color='red', linewidth=3, label='平均轨迹')
        
        ax_traj.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_traj.axhline(y=5, color='green', linestyle='--', alpha=0.3, label='止盈线(+5%)')
        ax_traj.axhline(y=-3, color='red', linestyle='--', alpha=0.3, label='止损线(-3%)')
        
        ax_traj.set_xlabel('持有天数')
        ax_traj.set_ylabel('收益率 (%)')
        ax_traj.set_title('[收益] 未来20天收益轨迹对比', fontsize=12, fontweight='bold')
        ax_traj.legend(loc='upper right')
        ax_traj.grid(True, alpha=0.3)
    else:
        ax_traj.text(0.5, 0.5, '暂无收益数据', ha='center', va='center')
        ax_traj.axis('off')
    
    # === 第三行: 分布图 ===
    # 年份分布
    ax_year = fig.add_subplot(gs[2, :3])
    if stats.get('year_distribution'):
        years = list(stats['year_distribution'].keys())
        counts = list(stats['year_distribution'].values())
        ax_year.bar(years, counts, color='steelblue')
        ax_year.set_xlabel('年份')
        ax_year.set_ylabel('数量')
        ax_year.set_title('[年份] 年份分布', fontsize=11)
    
    # 板块分布
    ax_board = fig.add_subplot(gs[2, 3:6])
    if stats.get('board_distribution'):
        boards = list(stats['board_distribution'].keys())
        counts = list(stats['board_distribution'].values())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(boards)))
        ax_board.pie(counts, labels=boards, autopct='%1.0f%%', colors=colors)
        ax_board.set_title('[板块] 板块分布', fontsize=11)
    
    # 收益分布
    ax_ret = fig.add_subplot(gs[2, 6:])
    if not trajectories.empty:
        final_returns = trajectories.iloc[-1].dropna()
        ax_ret.hist(final_returns, bins=10, color='green', alpha=0.7, edgecolor='black')
        ax_ret.axvline(x=0, color='red', linestyle='--')
        ax_ret.set_xlabel('收益率 (%)')
        ax_ret.set_ylabel('数量')
        ax_ret.set_title('[收益] 收益分布', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"[OK] 增强版Top10对比图已保存: {output_path}")


if __name__ == "__main__":
    print("=== Top10分析器测试 ===")
    
    # 模拟匹配数据
    matches = [
        {'symbol': '600519', 'date': '20231015', 'score': 0.95},
        {'symbol': '000858', 'date': '20230820', 'score': 0.92},
        {'symbol': '601318', 'date': '20231105', 'score': 0.89},
        {'symbol': '300750', 'date': '20230615', 'score': 0.87},
        {'symbol': '600036', 'date': '20230310', 'score': 0.85},
    ]
    
    analyzer = Top10Analyzer()
    stats = analyzer.analyze_matches(matches)
    
    print("\n统计分析结果:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + analyzer.create_stats_summary(stats))
