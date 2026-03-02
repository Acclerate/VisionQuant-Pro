import akshare as ak
import pandas as pd
import os
import time
import logging
from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Optional

# === 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DEFAULT_START_DATE = "20100101"

# 日志（不强行覆盖全局 logging 配置，交由入口处统一配置）
logger = logging.getLogger(__name__)

# 导入数据源适配器
from .data_source import DataSource, DiggoldDataSource, EfinanceDataSource, AkshareDataSource
from .jqdata_adapter import JQDataAdapter
from .rqdata_adapter import RQDataAdapter
from .quality_checker import DataQualityChecker


class DataLoader:
    """
    数据加载器（支持多数据源切换）

    支持的数据源：
    - 'diggold': 东财掘金SDK数据源（默认，推荐，最稳定）
    - 'efinance': 东财掘金数据源（备用）
    - 'akshare': 免费数据源（备用）
    - 'jqdata': 聚宽数据源（需要认证）
    - 'rqdata': 米筐数据源（需要认证）
    """

    def __init__(self, data_source: str = 'diggold', **kwargs):
        """
        初始化数据加载器
        
        Args:
            data_source: 数据源名称 ('akshare', 'jqdata', 'rqdata')
            **kwargs: 数据源特定参数
                - 对于jqdata: username, password
                - 对于rqdata: username, password
        """
        if not os.path.exists(DATA_RAW_DIR):
            os.makedirs(DATA_RAW_DIR)
        self.data_dir = DATA_RAW_DIR
        
        # 初始化数据源
        self.data_source_name = data_source
        self.data_source = self._init_data_source(data_source, **kwargs)
        
        # 初始化数据质量检查器
        self.quality_checker = DataQualityChecker()
        self.enable_quality_check = kwargs.get('enable_quality_check', True)

        # 内存级缓存（减少重复磁盘读）
        self._mem_cache_enabled = kwargs.get("mem_cache", True)
        self._mem_cache_max = int(kwargs.get("mem_cache_max", 32))
        self._mem_cache = OrderedDict()
    
    def _init_data_source(self, source_name: str, **kwargs) -> DataSource:
        """
        初始化数据源

        Args:
            source_name: 数据源名称
            **kwargs: 数据源参数

        Returns:
            DataSource实例
        """
        if source_name == 'diggold':
            return DiggoldDataSource()
        elif source_name == 'efinance':
            return EfinanceDataSource()
        elif source_name == 'akshare':
            return AkshareDataSource()
        elif source_name == 'jqdata':
            username = kwargs.get('username') or kwargs.get('jq_username')
            password = kwargs.get('password') or kwargs.get('jq_password')
            return JQDataAdapter(username=username, password=password)
        elif source_name == 'rqdata':
            username = kwargs.get('username') or kwargs.get('rq_username')
            password = kwargs.get('password') or kwargs.get('rq_password')
            return RQDataAdapter(username=username, password=password)
        else:
            logger.warning("未知数据源: %s，使用 diggold 作为默认", source_name)
            return DiggoldDataSource()
    
    def switch_data_source(self, source_name: str, **kwargs):
        """
        切换数据源
        
        Args:
            source_name: 新数据源名称
            **kwargs: 数据源参数
        """
        self.data_source_name = source_name
        self.data_source = self._init_data_source(source_name, **kwargs)
        logger.info("已切换到数据源: %s", source_name)
    
    def get_current_data_source(self) -> str:
        """获取当前数据源名称"""
        return self.data_source_name

    def get_stock_data(self, symbol, start_date=DEFAULT_START_DATE, end_date=None, adjust="qfq", use_cache=True):
        """
        [智能更新版] 获取股票数据（支持多数据源）
        
        逻辑：
        1. 如果use_cache=True，先检查本地缓存
        2. 如果数据滞后或不存在，从当前数据源下载
        3. 如果当前数据源不可用，回退到akshare
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            use_cache: 是否使用本地缓存
        """
        def _to_dt(value, default_dt):
            if value is None or str(value).strip() == "":
                return default_dt
            try:
                dt = pd.to_datetime(value, errors="coerce")
                return default_dt if pd.isna(dt) else dt
            except Exception:
                return default_dt

        req_start_dt = _to_dt(start_date, pd.to_datetime(DEFAULT_START_DATE))
        req_end_dt = _to_dt(end_date, pd.to_datetime(datetime.now().strftime("%Y%m%d")))
        if req_end_dt < req_start_dt:
            req_end_dt = req_start_dt
        start_date = req_start_dt.strftime("%Y%m%d")
        end_date = req_end_dt.strftime("%Y%m%d")

        symbol = str(symbol).strip().zfill(6)
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        def _validate_df(df_new):
            if df_new is None or df_new.empty:
                return None
            if self.enable_quality_check:
                quality_result = self.quality_checker.check_data_quality(df_new, symbol)
                if not quality_result['is_valid']:
                    print(f"[警告] [{symbol}] 数据质量检查未通过 (得分: {quality_result['score']}/100)")
                    if quality_result['score'] < 50:
                        print(f"  错误: {quality_result['errors']}")
                        return None
            return df_new

        def _fetch_with_fallback(start_dt, end_dt):
            if start_dt > end_dt:
                return pd.DataFrame()
            start_str = start_dt.strftime("%Y%m%d")
            end_str = end_dt.strftime("%Y%m%d")
            df_new = None
            if self.data_source and self.data_source.is_available():
                print(f"[{self.data_source_name}] 拉取 {symbol} 行情 {start_str}-{end_str}...")
                df_new = self.data_source.get_stock_data(
                    symbol=symbol,
                    start_date=start_str,
                    end_date=end_str,
                    adjust=adjust
                )
                df_new = _validate_df(df_new)
            return df_new if df_new is not None else pd.DataFrame()

        def _detect_gaps(idx, gap_days: int = 45, max_gaps: int = 10):
            if idx is None or len(idx) < 2:
                return []
            idx = pd.to_datetime(idx)
            idx = idx.sort_values()
            gaps = []
            diffs = idx.to_series().diff().dt.days
            for i, gap in enumerate(diffs):
                if pd.isna(gap):
                    continue
                if gap > gap_days:
                    prev_dt = idx[i - 1]
                    next_dt = idx[i]
                    gap_start = prev_dt + timedelta(days=1)
                    gap_end = next_dt - timedelta(days=1)
                    gaps.append((gap_start, gap_end))
                    if len(gaps) >= max_gaps:
                        break
            return gaps

        # 内存缓存命中（优先）
        cache_key = (symbol, start_date, end_date, adjust)
        if use_cache and self._mem_cache_enabled:
            cached = self._mem_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        need_download = False
        df_cache_all = pd.DataFrame()

        # === 1. 检查本地缓存（如果启用） ===
        if use_cache and os.path.exists(file_path):
            try:
                df_cache_all = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df_cache_all = self._normalize_columns(df_cache_all)
                df_cache_all = self._ensure_datetime_index(df_cache_all)
                if not df_cache_all.empty:
                    df_cache_all.sort_index(inplace=True)
                    first_date_in_file = pd.to_datetime(df_cache_all.index.min()).normalize()
                    last_date_in_file = pd.to_datetime(df_cache_all.index.max()).normalize()
                    need_earlier = req_start_dt < first_date_in_file
                    need_later = req_end_dt > last_date_in_file
                    need_download = need_earlier or need_later
                    if not need_download:
                        # 检测并修复内部大段缺口（避免中途年份空窗）
                        gaps = _detect_gaps(df_cache_all.index)
                        if gaps:
                            for g_start, g_end in gaps:
                                if g_end < req_start_dt or g_start > req_end_dt:
                                    continue
                                patch = _fetch_with_fallback(g_start, g_end)
                                if patch is not None and not patch.empty:
                                    patch = self._normalize_columns(patch)
                                    patch = self._ensure_datetime_index(patch)
                                    df_cache_all = pd.concat([df_cache_all, patch], axis=0)
                            df_cache_all = self._ensure_datetime_index(df_cache_all)
                            df_cache_all.sort_index(inplace=True)
                            df_cache_all = df_cache_all[~df_cache_all.index.duplicated(keep='last')]
                            if use_cache:
                                df_cache_all.to_csv(file_path)
                        df_out = df_cache_all.loc[req_start_dt:req_end_dt]
                        if use_cache and self._mem_cache_enabled:
                            self._mem_cache[cache_key] = df_out
                            if len(self._mem_cache) > self._mem_cache_max:
                                self._mem_cache.popitem(last=False)
                        return df_out.copy()
                else:
                    need_download = True
            except Exception as e:
                logger.warning("读取本地缓存失败 %s (%s): %s", symbol, file_path, e)
                need_download = True
        else:
            need_download = True

        # === 2. 从数据源下载（如果需要） ===
        if need_download:
            if not df_cache_all.empty:
                df_cache_all.sort_index(inplace=True)
                first_date_in_file = pd.to_datetime(df_cache_all.index.min()).normalize()
                last_date_in_file = pd.to_datetime(df_cache_all.index.max()).normalize()
                need_earlier = req_start_dt < first_date_in_file
                need_later = req_end_dt > last_date_in_file

                new_parts = []
                if need_earlier:
                    new_parts.append(_fetch_with_fallback(req_start_dt, first_date_in_file - timedelta(days=1)))
                if need_later:
                    new_parts.append(_fetch_with_fallback(last_date_in_file + timedelta(days=1), req_end_dt))

                for part in new_parts:
                    if part is not None and not part.empty:
                        new_parts_df = self._normalize_columns(part)
                        df_cache_all = pd.concat([df_cache_all, new_parts_df], axis=0)

                if not df_cache_all.empty:
                    df_cache_all = self._ensure_datetime_index(df_cache_all)
                    df_cache_all.sort_index(inplace=True)
                    df_cache_all = df_cache_all[~df_cache_all.index.duplicated(keep='last')]

                    # 补齐中间缺口
                    gaps = _detect_gaps(df_cache_all.index)
                    if gaps:
                        for g_start, g_end in gaps:
                            if g_end < req_start_dt or g_start > req_end_dt:
                                continue
                            patch = _fetch_with_fallback(g_start, g_end)
                            if patch is not None and not patch.empty:
                                patch = self._normalize_columns(patch)
                                patch = self._ensure_datetime_index(patch)
                                df_cache_all = pd.concat([df_cache_all, patch], axis=0)
                        df_cache_all = self._ensure_datetime_index(df_cache_all)
                        df_cache_all.sort_index(inplace=True)
                        df_cache_all = df_cache_all[~df_cache_all.index.duplicated(keep='last')]

                    # 如果仍存在超大缺口，尝试全量刷新请求范围
                    gaps_after = _detect_gaps(df_cache_all.index, gap_days=120, max_gaps=1)
                    if gaps_after:
                        full_df = _fetch_with_fallback(req_start_dt, req_end_dt)
                        if full_df is not None and not full_df.empty:
                            full_df = self._normalize_columns(full_df)
                            full_df = self._ensure_datetime_index(full_df)
                            df_cache_all = full_df.copy()

                    if use_cache:
                        df_cache_all.to_csv(file_path)
                    df_out = df_cache_all.loc[req_start_dt:req_end_dt]
                    if use_cache and self._mem_cache_enabled:
                        self._mem_cache[cache_key] = df_out
                        if len(self._mem_cache) > self._mem_cache_max:
                            self._mem_cache.popitem(last=False)
                    return df_out.copy()

                # 无法获取新数据时，回退旧数据
                print(f"[警告] 所有数据源获取失败，使用本地旧数据")
                df_out = df_cache_all.loc[req_start_dt:req_end_dt] if not df_cache_all.empty else pd.DataFrame()
                return df_out.copy()

            # 无本地缓存：全量拉取
            df_new = _fetch_with_fallback(req_start_dt, req_end_dt)
            if df_new is not None and not df_new.empty:
                df_new = self._normalize_columns(df_new)
                df_new = self._ensure_datetime_index(df_new)
                # 补齐中间缺口
                gaps = _detect_gaps(df_new.index)
                if gaps:
                    for g_start, g_end in gaps:
                        if g_end < req_start_dt or g_start > req_end_dt:
                            continue
                        patch = _fetch_with_fallback(g_start, g_end)
                        if patch is not None and not patch.empty:
                            patch = self._normalize_columns(patch)
                            patch = self._ensure_datetime_index(patch)
                            df_new = pd.concat([df_new, patch], axis=0)
                    df_new = self._ensure_datetime_index(df_new)
                    df_new.sort_index(inplace=True)
                    df_new = df_new[~df_new.index.duplicated(keep='last')]
                if use_cache:
                    df_new.to_csv(file_path)
                df_out = df_new.loc[req_start_dt:req_end_dt]
                if use_cache and self._mem_cache_enabled:
                    self._mem_cache[cache_key] = df_out
                    if len(self._mem_cache) > self._mem_cache_max:
                        self._mem_cache.popitem(last=False)
                return df_out.copy()

            return pd.DataFrame()

        try:
            if not df_cache_all.empty:
                df_cache_all = self._ensure_datetime_index(df_cache_all)
                return df_cache_all.loc[req_start_dt:req_end_dt].copy()
        except Exception as e:
            logger.warning("时间索引切片失败 %s: %s", symbol, e)
        return pd.DataFrame()

    def get_index_data(self, index_code, start_date=DEFAULT_START_DATE, end_date=None, use_cache: bool = False):
        """
        获取指数OHLCV数据（优先当前数据源，失败回退Akshare）
        注意：此接口默认不走缓存，以保证情绪类指标的实时性。
        """
        def _to_dt(value, default_dt):
            if value is None or str(value).strip() == "":
                return default_dt
            try:
                dt = pd.to_datetime(value, errors="coerce")
                return default_dt if pd.isna(dt) else dt
            except Exception:
                return default_dt

        req_start_dt = _to_dt(start_date, pd.to_datetime(DEFAULT_START_DATE))
        req_end_dt = _to_dt(end_date, pd.to_datetime(datetime.now().strftime("%Y%m%d")))
        if req_end_dt < req_start_dt:
            req_end_dt = req_start_dt
        start_date = req_start_dt.strftime("%Y%m%d")
        end_date = req_end_dt.strftime("%Y%m%d")

        index_code = str(index_code).strip()
        df = pd.DataFrame()

        # 当前数据源
        if self.data_source and self.data_source.is_available():
            try:
                df = self.data_source.get_index_data(
                    index_code=index_code,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                logger.warning("指数数据获取失败 [%s]: %s", self.data_source_name, e)

        # 智能回退
        if (df is None or df.empty):
            if self.data_source_name == 'diggold':
                fallback_source = EfinanceDataSource()
                source_name = 'efinance'
            elif self.data_source_name == 'efinance':
                fallback_source = DiggoldDataSource()
                source_name = 'diggold'
            else:
                fallback_source = DiggoldDataSource()
                source_name = 'diggold'
            try:
                if fallback_source.is_available():
                    df = fallback_source.get_index_data(
                        index_code=index_code,
                        start_date=start_date,
                        end_date=end_date
                    )
            except Exception as e:
                logger.warning("指数数据回退失败 [%s]: %s", source_name, e)

        if df is None or df.empty:
            return pd.DataFrame()

        df = self._normalize_columns(df)
        df = self._ensure_datetime_index(df)
        return df

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保索引为DatetimeIndex，避免Timestamp切片异常"""
        if df is None or df.empty:
            return df
        data = df.copy()
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, errors="coerce")
            # 移除时区信息，统一为无时区（避免比较时的类型不匹配）
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            data = data[~data.index.isna()]
            data.sort_index(inplace=True)
        except Exception:
            return df
        return data

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一常见列名，保证下游量价特征可稳定获取
        """
        if df is None or df.empty:
            return df
        data = df.copy()
        col_map = {}
        for c in data.columns:
            lc = str(c).lower()
            if lc in ["open", "开盘"]:
                col_map[c] = "Open"
            elif lc in ["high", "最高"]:
                col_map[c] = "High"
            elif lc in ["low", "最低"]:
                col_map[c] = "Low"
            elif lc in ["close", "收盘", "收盘价"]:
                col_map[c] = "Close"
            elif lc in ["volume", "成交量"]:
                col_map[c] = "Volume"
            elif lc in ["amount", "成交额", "成交金额", "成交额(元)"]:
                col_map[c] = "Amount"
            elif lc in ["turnover", "换手率", "换手"]:
                col_map[c] = "Turnover"
        if col_map:
            data = data.rename(columns=col_map)
        return data

    def get_top300_stocks(self):
        """获取全A股列表并按市值排序"""
        # 优先使用当前数据源
        if self.data_source and self.data_source.is_available():
            try:
                stock_list = self.data_source.get_stock_list()
                if not stock_list.empty:
                    # 如果有市值信息，按市值排序
                    if 'market_cap' in stock_list.columns:
                        stock_list = stock_list.sort_values(by='market_cap', ascending=False)
                    return stock_list.head(300)
            except Exception as e:
                print(f"[警告] [{self.data_source_name}] 获取股票列表失败: {e}")
        
        # 回退到akshare
        try:
            df = ak.stock_zh_a_spot_em()
            if '总市值' in df.columns:
                df = df.sort_values(by='总市值', ascending=False)
            df = df.head(300)
            return df[['代码', '名称']].rename(columns={'代码': 'code', '名称': 'name'})
        except Exception as e:
            print(f"[错误] 获取名单失败: {e}")
            return pd.DataFrame()

    def download_batch_data(self, stock_list, start_date=DEFAULT_START_DATE):
        """批量下载"""
        print(f"[批量维护] 正在检查并更新 {len(stock_list)} 只股票...")
        for _, row in tqdm(stock_list.iterrows(), total=len(stock_list)):
            symbol = str(row['code']).zfill(6)
            self.get_stock_data(symbol, start_date=start_date)
            # 稍微快一点，因为大部分可能不需要下载
            time.sleep(0.01)


if __name__ == "__main__":
    loader = DataLoader()
    # 测试更新逻辑
    df = loader.get_stock_data("601899")
    print(f"最新数据日期: {df.index[-1]}")