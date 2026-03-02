"""
数据源抽象层
Data Source Abstraction Layer

统一数据接口，支持多数据源
优先使用：东财掘金SDK > efinance > akshare

Author: VisionQuant Team
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
import time
import random
import os

from src.utils.net_utils import no_proxy_env

# 加载环境变量
from dotenv import load_dotenv
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# 读取掘金 Token
DIGGOLD_TOKEN = os.getenv('DIGGOLD_TOKEN', '')

# 初始化掘金 SDK
DIGGOLD_AVAILABLE = False
if DIGGOLD_TOKEN:
    try:
        from gm.api import set_token, history, get_instruments
        set_token(DIGGOLD_TOKEN)
        DIGGOLD_AVAILABLE = True
        print(f"[数据源] 东财掘金SDK已初始化")
    except ImportError:
        print("[数据源] 掘金SDK未安装，请运行: pip install gm")
    except Exception as e:
        print(f"[数据源] 掘金SDK初始化失败: {e}")
else:
    print("[数据源] 未配置DIGGOLD_TOKEN，将使用备用数据源")


class DataSource(ABC):
    """
    数据源抽象基类

    所有数据源必须实现这些接口
    """

    @abstractmethod
    def get_stock_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取股票OHLCV数据

        Args:
            symbol: 股票代码（6位，如'600519'）
            start_date: 开始日期（格式：'YYYYMMDD'）
            end_date: 结束日期（格式：'YYYYMMDD'）
            adjust: 复权类型（'qfq'前复权, 'hfq'后复权, 'bfq'不复权）

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: DatetimeIndex
        """
        pass

    def get_index_data(
        self,
        index_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取指数数据（可选实现）

        Args:
            index_code: 指数代码（如'000300'沪深300, '000001'上证指数）
            start_date: 开始日期（格式：'YYYYMMDD'）
            end_date: 结束日期（格式：'YYYYMMDD'）

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        return pd.DataFrame()

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查数据源是否可用

        Returns:
            True if available, False otherwise
        """
        pass

    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一数据格式

        将不同数据源的数据格式转换为标准格式
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 确保有必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # 标准化列名（如果存在中文列名）
        column_mapping = {
            '开盘': 'Open', '开盘价': 'Open',
            '收盘': 'Close', '收盘价': 'Close',
            '最高': 'High', '最高价': 'High',
            '最低': 'Low', '最低价': 'Low',
            '成交量': 'Volume', '成交额': 'Amount',
            '日期': 'Date', '时间': 'Date', 'date': 'Date'
        }

        df = df.rename(columns=column_mapping)

        # 确保索引是日期
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass

        # 确保数据类型正确
        for col in required_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 按日期排序
        df.sort_index(inplace=True)

        return df


class DiggoldDataSource(DataSource):
    """
    东财掘金SDK数据源（推荐，最稳定）

    优点：
    - 使用专用协议，无SSL问题
    - 官方SDK，数据最准确
    - 速度最快

    需要配置：DIGGOLD_TOKEN 环境变量
    """

    def __init__(self):
        """初始化掘金数据源"""
        self._available = DIGGOLD_AVAILABLE

    def get_stock_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取股票数据（掘金SDK）- 使用 history_n + count 方式"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            from gm.api import history_n

            symbol = str(symbol).strip().zfill(6)

            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")

            end_date_diggold = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

            if symbol.startswith('6') or symbol.startswith('5'):
                diggold_symbol = f"SHSE.{symbol}"
            else:
                diggold_symbol = f"SZSE.{symbol}"

            if start_date:
                start_dt = datetime.strptime(start_date, "%Y%m%d")
                end_dt = datetime.strptime(end_date, "%Y%m%d")
                days_diff = (end_dt - start_dt).days
                count = min(max(days_diff + 30, 300), 800)
            else:
                count = 500

            df = history_n(
                symbol=diggold_symbol,
                frequency='1d',
                count=count,
                end_time=end_date_diggold,
                adjust=1,
                df=True
            )

            if df is None or df.empty:
                print(f"[diggold] history_n 返回空数据 {symbol}")
                return pd.DataFrame()

            if 'eob' in df.columns:
                df['Date'] = pd.to_datetime(df['eob'])
                df = df.drop(columns=['eob'])
            elif 'bob' in df.columns:
                df['Date'] = pd.to_datetime(df['bob'])
                df = df.drop(columns=['bob'])

            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)

            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_mapping)

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in df.columns]

            if start_date:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                else:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]

            return df[available_cols]

        except Exception as e:
            print(f"掘金SDK获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def get_index_data(
        self,
        index_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """获取指数数据（掘金SDK）"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            from gm.api import history_n

            index_code = str(index_code).strip().zfill(6)

            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")

            end_date_diggold = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

            index_map = {
                '000001': 'SHSE.000001',  # 上证指数
                '000300': 'SHSE.000300',  # 沪深300
                '000016': 'SHSE.000016',  # 上证50
                '000905': 'SHSE.000905',  # 中证500
                '000852': 'SHSE.000852',  # 中证1000
                '399001': 'SZSE.399001',  # 深证成指
                '399006': 'SZSE.399006',  # 创业板指
            }

            if index_code in index_map:
                diggold_symbol = index_map[index_code]
            elif index_code.startswith('0'):
                diggold_symbol = f"SHSE.{index_code}"
            else:
                diggold_symbol = f"SZSE.{index_code}"

            count = 300
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y%m%d")
                end_dt = datetime.strptime(end_date, "%Y%m%d")
                days_diff = (end_dt - start_dt).days
                count = min(max(days_diff + 30, 100), 500)

            df = history_n(
                symbol=diggold_symbol,
                frequency='1d',
                count=count,
                end_time=end_date_diggold,
                adjust=0,
                df=True
            )

            if df is None or df.empty:
                print(f"[diggold] 指数数据返回空 {index_code}")
                return pd.DataFrame()

            if 'eob' in df.columns:
                df['Date'] = pd.to_datetime(df['eob'])
                df = df.drop(columns=['eob'])
            elif 'bob' in df.columns:
                df['Date'] = pd.to_datetime(df['bob'])
                df = df.drop(columns=['bob'])

            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)

            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_mapping)

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in df.columns]

            if start_date:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]

            return df[available_cols]

        except Exception as e:
            print(f"掘金SDK获取指数数据失败 {index_code}: {e}")
            return pd.DataFrame()

    def is_available(self) -> bool:
        """检查掘金SDK是否可用"""
        return self._available


class EfinanceDataSource(DataSource):
    """
    Efinance数据源（备用）

    优点：
    - 免费、无需Token
    - API简洁稳定

    仓库：https://github.com/Micro-sheep/efinance
    """

    def __init__(self):
        """初始化Efinance数据源"""
        try:
            import efinance as ef
            self.ef = ef
            self._available = True
            self._last_request_time = None
            self._min_interval = 1.5
        except ImportError:
            self._available = False
            self.ef = None

    def _enforce_rate_limit(self):
        """强制执行速率限制"""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        time.sleep(random.uniform(0.5, 1.5))
        self._last_request_time = time.time()

    def get_stock_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取股票数据（Efinance）"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            symbol = str(symbol).strip().zfill(6)

            if start_date is None:
                start_date = "20100101"
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")

            self._enforce_rate_limit()

            with no_proxy_env():
                df = self.ef.stock.get_quote_history(
                    stock_codes=symbol,
                    beg=start_date,
                    end=end_date,
                    klt=101,
                    fqt=1
                )

            return self._format_data(df)
        except Exception as e:
            print(f"Efinance获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def is_available(self) -> bool:
        """检查Efinance是否可用"""
        return self._available


class AkshareDataSource(DataSource):
    """
    AkShare数据源（备用）
    """

    def __init__(self):
        """初始化AkShare数据源"""
        try:
            import akshare as ak
            self.ak = ak
            self._available = True
        except ImportError:
            self._available = False
            self.ak = None

    def get_stock_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取股票数据（AkShare）"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            symbol = str(symbol).strip().zfill(6)

            if start_date is None:
                start_date = "20100101"
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")

            with no_proxy_env():
                df = self.ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )

            return self._format_data(df)
        except Exception as e:
            print(f"AkShare获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def is_available(self) -> bool:
        """检查AkShare是否可用"""
        return self._available


# 便捷函数：自动选择最佳数据源
def get_data_source(prefer: str = "auto") -> DataSource:
    """
    获取可用的数据源

    Args:
        prefer: 首选数据源 ('auto', 'diggold', 'efinance', 'akshare')

    Returns:
        可用的数据源实例
    """
    if prefer == "auto":
        # 按优先级自动选择
        sources = [
            DiggoldDataSource(),
            EfinanceDataSource(),
            AkshareDataSource()
        ]
        for source in sources:
            if source.is_available():
                return source
        raise RuntimeError("没有可用的数据源！")

    if prefer == "diggold":
        source = DiggoldDataSource()
        if source.is_available():
            return source

    if prefer == "efinance":
        source = EfinanceDataSource()
        if source.is_available():
            return source

    # 回退到 AkShare
    source = AkshareDataSource()
    if source.is_available():
        return source

    raise RuntimeError("没有可用的数据源！请安装掘金SDK、efinance 或 akshare")


if __name__ == "__main__":
    print("=== 数据源抽象层测试 ===")

    # 测试掘金数据源
    print("\n--- 掘金SDK 数据源 ---")
    diggold_source = DiggoldDataSource()
    if diggold_source.is_available():
        print("掘金SDK数据源可用")
        df = diggold_source.get_stock_data('600519', start_date='20240101', end_date='20240110')
        if not df.empty:
            print(f"成功获取数据，共 {len(df)} 条记录")
            print(df.head())
        else:
            print("获取数据失败")
    else:
        print("掘金SDK数据源不可用")

    # 测试自动选择
    print("\n--- 自动选择数据源 ---")
    try:
        source = get_data_source()
        print(f"已选择数据源: {source.__class__.__name__}")
    except RuntimeError as e:
        print(f"{e}")
