"""VisionQuant Pro - 工业级精简版"""
import streamlit as st
import os, sys, pandas as pd, numpy as np, mplfinance as mpf, plotly.graph_objects as go
from datetime import datetime
import importlib
import logging
import inspect

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit 版本兼容性处理
# 检测 st.image 是否支持 use_container_width 参数（1.12.0+）
try:
    sig = inspect.signature(st.image)
    _USE_CONTAINER_WIDTH_SUPPORTED = 'use_container_width' in sig.parameters
except Exception:
    _USE_CONTAINER_WIDTH_SUPPORTED = False

# 检测 st.button 是否支持 use_container_width 参数
try:
    sig = inspect.signature(st.button)
    _BUTTON_USE_CONTAINER_WIDTH_SUPPORTED = 'use_container_width' in sig.parameters
except Exception:
    _BUTTON_USE_CONTAINER_WIDTH_SUPPORTED = False

# 检测 st.plotly_chart 是否支持 use_container_width 参数
try:
    sig = inspect.signature(st.plotly_chart)
    _PLOTLY_USE_CONTAINER_WIDTH_SUPPORTED = 'use_container_width' in sig.parameters
except Exception:
    _PLOTLY_USE_CONTAINER_WIDTH_SUPPORTED = False

# 检测 st.dataframe 是否支持 use_container_width 参数
try:
    sig = inspect.signature(st.dataframe)
    _DATAFRAME_USE_CONTAINER_WIDTH_SUPPORTED = 'use_container_width' in sig.parameters
except Exception:
    _DATAFRAME_USE_CONTAINER_WIDTH_SUPPORTED = False

# 兼容性包装函数
def _st_image(image, caption=None, width=None, use_container_width=False):
    """兼容性包装 st.image"""
    kwargs = {}
    if caption is not None:
        kwargs['caption'] = caption
    if width is not None:
        kwargs['width'] = width
    if _USE_CONTAINER_WIDTH_SUPPORTED and use_container_width:
        kwargs['use_container_width'] = True
    return st.image(image, **kwargs)

def _st_button(label, key=None, type="secondary", use_container_width=False, on_click=None, args=(), kwargs={}):
    """兼容性包装 st.button"""
    func = st.button
    call_kwargs = {}
    if key is not None:
        call_kwargs['key'] = key
    if on_click is not None:
        call_kwargs['on_click'] = on_click
        call_kwargs['args'] = args
        call_kwargs['kwargs'] = kwargs
    if _BUTTON_USE_CONTAINER_WIDTH_SUPPORTED and use_container_width:
        call_kwargs['use_container_width'] = True
    return func(label, type=type, **call_kwargs)

def _st_plotly_chart(figure_or_data, use_container_width=False, **kwargs):
    """兼容性包装 st.plotly_chart"""
    if _PLOTLY_USE_CONTAINER_WIDTH_SUPPORTED and use_container_width:
        kwargs['use_container_width'] = True
    return st.plotly_chart(figure_or_data, **kwargs)

def _st_dataframe(data, use_container_width=False, **kwargs):
    """兼容性包装 st.dataframe"""
    if _DATAFRAME_USE_CONTAINER_WIDTH_SUPPORTED and use_container_width:
        kwargs['use_container_width'] = True
    return st.dataframe(data, **kwargs)

# 定义项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 网络模式：
# - 默认尊重系统代理（更适合公司网络/本地代理环境）
# - 如需强制直连，可设置环境变量 VQ_FORCE_NO_PROXY=1
if os.getenv("VQ_FORCE_NO_PROXY", "0") == "1":
    for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(_k, None)
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")

try:
    from src.data.data_loader import DataLoader
    from src.data.news_harvester import NewsHarvester
    from src.models.vision_engine import VisionEngine
    from src.strategies.factor_mining import FactorMiner
    from src.strategies.fundamental import FundamentalMiner
    from src.agent.quant_agent import QuantAgent
    from src.utils.visualizer import create_comparison_plot
    from src.utils.pdf_generator import generate_report_pdf
    from src.utils.audio_manager import AudioManager
    from src.strategies.batch_analyzer import BatchAnalyzer
    from src.strategies.portfolio_optimizer import PortfolioOptimizer
    from src.strategies.kline_factor import KLineFactorCalculator
except ImportError as e:
    st.error(f"❌ 系统模块加载失败: {e}")
    st.stop()

def _code_version_key():
    paths = [
        os.path.join(PROJECT_ROOT, "src", "models", "vision_engine.py"),
        os.path.join(PROJECT_ROOT, "src", "strategies", "fundamental.py"),
        os.path.join(PROJECT_ROOT, "src", "data", "data_loader.py"),
        os.path.join(PROJECT_ROOT, "src", "data", "data_source.py"),
        os.path.join(PROJECT_ROOT, "src", "agent", "quant_agent.py"),
    ]
    return "|".join([str(os.path.getmtime(p)) if os.path.exists(p) else "0" for p in paths])

def _find_existing_kline_image(symbol: str, date_str: str, vision_engine=None):
    symbol = str(symbol).zfill(6)
    date_n = str(date_str).replace("-", "")
    if vision_engine is not None:
        try:
            fast_path = vision_engine.find_image_path(symbol, date_n, allow_nearest=True)
            if fast_path:
                return fast_path
        except Exception:
            pass
    img_bases = [
        os.path.join(PROJECT_ROOT, "data", "images_v2"),
        os.path.join(PROJECT_ROOT, "data", "images"),
    ]
    for img_base in img_bases:
        candidates = [
            os.path.join(img_base, f"{symbol}_{date_n}.png"),
            os.path.join(img_base, symbol, f"{symbol}_{date_n}.png"),
            os.path.join(img_base, symbol, f"{date_n}.png"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
    return None

def _render_match_image(symbol: str, date_str: str, loader, out_path: str):
    try:
        df = loader.get_stock_data(symbol)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        dt = pd.to_datetime(str(date_str), errors="coerce")
        if dt is pd.NaT:
            return None
        if dt not in df.index:
            candidates = df.index[df.index <= dt]
            if len(candidates) == 0:
                return None
            dt = candidates.max()
        loc = df.index.get_loc(dt)
        start = max(0, loc - 19)
        window = df.iloc[start:loc + 1].copy()
        if len(window) < 20:
            return None
        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
        mpf.plot(window, type='candle', style=s, savefig=dict(fname=out_path, dpi=50), figsize=(3, 3), axisoff=True)
        return out_path
    except Exception:
        return None

def _compute_market_sentiment(loader, index_code: str = "000300", end_dt=None):
    """
    大盘情绪（后验）：基于沪深300指数的趋势/波动/回撤/量能综合评分
    """
    result = {"ok": False, "index_code": index_code}
    try:
        end_dt = pd.to_datetime(end_dt or datetime.now(), errors="coerce")
        if pd.isna(end_dt):
            end_dt = pd.to_datetime(datetime.now())
        start_dt = end_dt - pd.Timedelta(days=450)
        df = None
        last_err = None
        
        # 方法1: loader.get_index_data
        if hasattr(loader, "get_index_data"):
            try:
                df = loader.get_index_data(
                    index_code=index_code,
                    start_date=start_dt.strftime("%Y%m%d"),
                    end_date=end_dt.strftime("%Y%m%d"),
                )
                if df is not None and not df.empty:
                    logger.info(f"✅ 大盘数据获取成功 [loader.get_index_data]: {len(df)} 条")
            except Exception as e:
                last_err = f"loader.get_index_data: {type(e).__name__}: {e}"
                logger.warning(f"⚠️ {last_err}")
        
        # 方法2: loader.data_source.get_index_data
        if (df is None or df.empty) and hasattr(loader, "data_source") and loader.data_source is not None:
            if hasattr(loader.data_source, "get_index_data"):
                try:
                    df = loader.data_source.get_index_data(
                        index_code=index_code,
                        start_date=start_dt.strftime("%Y%m%d"),
                        end_date=end_dt.strftime("%Y%m%d"),
                    )
                    if df is not None and not df.empty:
                        logger.info(f"✅ 大盘数据获取成功 [data_source.get_index_data]: {len(df)} 条")
                except Exception as e:
                    last_err = f"data_source.get_index_data: {type(e).__name__}: {e}"
                    logger.warning(f"⚠️ {last_err}")
        
        # 方法3: 直接调用 akshare
        if df is None or df.empty:
            try:
                import akshare as ak
                from src.utils.net_utils import no_proxy_env
                with no_proxy_env():
                    df = ak.index_zh_a_hist(
                        symbol=index_code,
                        period="daily",
                        start_date=start_dt.strftime("%Y%m%d"),
                        end_date=end_dt.strftime("%Y%m%d"),
                    )
                if df is not None and not df.empty:
                    logger.info(f"✅ 大盘数据获取成功 [ak.index_zh_a_hist]: {len(df)} 条")
                    # 标准化列名
                    df = loader._normalize_columns(df) if hasattr(loader, "_normalize_columns") else df
                    df = loader._ensure_datetime_index(df) if hasattr(loader, "_ensure_datetime_index") else df
            except Exception as e:
                last_err = f"ak.index_zh_a_hist: {type(e).__name__}: {e}"
                logger.error(f"❌ {last_err}")
        
        if df is None or df.empty:
            result["error"] = f"指数数据不可用: {last_err or '所有方法均失败'}"
            return result
        
        if "Close" not in df.columns:
            result["error"] = f"指数数据缺少Close列，现有列: {list(df.columns)}"
            return result
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty or len(close) < 40:
            result["error"] = "指数数据长度不足"
            return result

        def _safe_ret(series, n):
            if len(series) <= n:
                return None
            return float(series.iloc[-1] / series.iloc[-n - 1] - 1.0)

        ret_1d = _safe_ret(close, 1)
        ret_5d = _safe_ret(close, 5)
        ret_20d = _safe_ret(close, 20)

        returns = close.pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else None

        # 周度（W-FRI）收益与波动
        weekly_close = close.resample("W-FRI").last().dropna()
        weekly_ret_1 = _safe_ret(weekly_close, 1) if len(weekly_close) >= 2 else None
        weekly_ret_4 = _safe_ret(weekly_close, 4) if len(weekly_close) >= 5 else None
        weekly_vol_8 = float(weekly_close.pct_change().tail(8).std() * np.sqrt(52)) if len(weekly_close) >= 10 else None

        # 最大回撤（近60日）
        window_close = close.tail(60)
        roll_max = window_close.cummax()
        dd_60 = float((window_close / roll_max - 1.0).min()) if not window_close.empty else None

        # 量能与振幅（近20日）
        vol_ratio = None
        amp_20 = None
        if "Volume" in df.columns:
            volume = pd.to_numeric(df["Volume"], errors="coerce").dropna()
            if len(volume) >= 20:
                vol_ratio = float(volume.iloc[-1] / (volume.tail(20).mean() + 1e-8))
        if "High" in df.columns and "Low" in df.columns:
            high = pd.to_numeric(df["High"], errors="coerce")
            low = pd.to_numeric(df["Low"], errors="coerce")
            amp = (high - low) / close.reindex(high.index)
            amp_20 = float(amp.tail(20).mean()) if len(amp.dropna()) >= 20 else None

        def _tanh(x, scale=1.0):
            try:
                return float(np.tanh(float(x) * scale))
            except Exception:
                return 0.0

        trend_component = (
            0.5 * _tanh(ret_20d or 0.0, 3.0)
            + 0.3 * _tanh(ret_5d or 0.0, 4.0)
            + 0.2 * _tanh(weekly_ret_4 or 0.0, 2.0)
        )
        vol_component = _tanh(vol_20d or 0.0, 2.0)
        dd_component = _tanh(abs(dd_60 or 0.0), 5.0)
        volume_component = _tanh((vol_ratio or 1.0) - 1.0, 2.0) if vol_ratio is not None else 0.0
        amp_component = _tanh(amp_20 or 0.0, 5.0) if amp_20 is not None else 0.0

        raw_score = (
            0.6 * trend_component
            + 0.2 * volume_component
            - 0.1 * vol_component
            - 0.1 * dd_component
            - 0.05 * amp_component
        )
        score = int(np.clip(50 + raw_score * 50, 0, 100))

        if score >= 70:
            label = "乐观偏多"
        elif score >= 60:
            label = "偏多"
        elif score >= 40:
            label = "中性"
        elif score >= 30:
            label = "偏空"
        else:
            label = "谨慎偏空"

        result.update({
            "ok": True,
            "latest_date": close.index[-1].strftime("%Y-%m-%d"),
            "latest_close": float(close.iloc[-1]),
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "weekly_ret_1": weekly_ret_1,
            "weekly_ret_4": weekly_ret_4,
            "vol_20d": vol_20d,
            "weekly_vol_8": weekly_vol_8,
            "max_drawdown_60d": dd_60,
            "volume_ratio": vol_ratio,
            "amplitude_20d": amp_20,
            "score": score,
            "label": label,
            "formula": "情绪=50+50*(0.6*趋势+0.2*量能-0.1*波动-0.1*回撤-0.05*振幅)",
            "series_dates": close.tail(120).index.strftime("%Y-%m-%d").tolist(),
            "series_values": close.tail(120).round(4).tolist()
        })
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

def _safe_get_fundamentals(fund, symbol, force_live: bool):
    try:
        return fund.get_stock_fundamentals(symbol, force_live=force_live)
    except TypeError:
        # 兼容旧版本（无 force_live 参数）
        return fund.get_stock_fundamentals(symbol)

def _safe_get_industry_peers(fund, symbol, force_live: bool):
    try:
        # 工业化默认：限制重试次数，避免弱网下单次请求阻塞过久
        return fund.get_industry_peers(symbol, max_retries=1, force_live=force_live)
    except TypeError:
        return fund.get_industry_peers(symbol)

def _safe_ai_analyze(agent, *args, **kwargs):
    try:
        return agent.analyze(*args, **kwargs)
    except TypeError:
        kwargs.pop("market_sentiment", None)
        return agent.analyze(*args, **kwargs)

def _augment_matches(matches, query_img_path, query_prices, loader, vision_engine, tmp_dir, filter_trend: bool = True):
    if not matches:
        return matches
    strict_list = []
    relaxed_list = []
    def _segment_signs(prices):
        if prices is None or len(prices) < 6:
            return None
        n = len(prices)
        s3 = n // 3
        s5 = max(1, n // 5)
        def _seg_ret(a, b):
            if b - a < 2:
                return None
            return prices[b - 1] - prices[a]
        segs = {
            "head": (0, s3),
            "mid": (s3, 2 * s3),
            "tail": (2 * s3, n),
            "front": (0, 2 * s3),
            "back": (s3, n),
            "start": (0, s5),
            "end": (n - s5, n)
        }
        signs = {}
        for k, (a, b) in segs.items():
            v = _seg_ret(a, b)
            if v is None:
                continue
            signs[k] = 1 if v >= 0 else -1
        return signs if signs else None

    def _segment_corrs(q, m):
        if q is None or m is None or len(q) < 6 or len(m) < 6:
            return None
        n = min(len(q), len(m))
        s3 = n // 3
        s5 = max(1, n // 5)
        segs = {
            "head": (0, s3),
            "mid": (s3, 2 * s3),
            "tail": (2 * s3, n),
            "front": (0, 2 * s3),
            "back": (s3, n),
            "start": (0, s5),
            "end": (n - s5, n)
        }
        corr_map = {}
        for k, (a, b) in segs.items():
            if b - a < 3:
                continue
            qa = q[a:b]
            ma = m[a:b]
            qn = (qa - qa.mean()) / (qa.std() + 1e-8)
            mn = (ma - ma.mean()) / (ma.std() + 1e-8)
            c = np.corrcoef(qn, mn)[0, 1]
            if not np.isnan(c):
                corr_map[k] = float(c)
        return corr_map if corr_map else None

    query_trend = None
    query_segs = None
    if query_prices is not None and len(query_prices) >= 6:
        query_trend = 1 if query_prices[-1] > query_prices[0] else -1
        query_segs = _segment_signs(query_prices)
    q_pix = vision_engine._load_pixel_vector(query_img_path)
    q_edge = vision_engine._load_edge_vector(query_img_path)
    for i, m in enumerate(matches):
        sym = str(m.get("symbol", "")).zfill(6)
        date_str = m.get("date")
        trend_penalty = 0.0
        seg_penalty = 0.0
        corr_penalty = 0.0
        strict_ok = False
        # 1) 像素/边缘相似度兜底
        if m.get("pixel_sim") is None or m.get("edge_sim") is None:
            path = vision_engine._resolve_image_path(m.get("path"), sym, date_str)
            if not path:
                tmp_path = os.path.join(tmp_dir, f"tmp_match_{sym}_{date_str}.png")
                path = _render_match_image(sym, date_str, loader, tmp_path)
            if path:
                v = vision_engine._load_pixel_vector(path)
                e = vision_engine._load_edge_vector(path)
                pix_cos = vision_engine._cosine_sim(q_pix, v)
                pix_corr = vision_engine._pearson_corr(q_pix, v)
                edge_cos = vision_engine._cosine_sim(q_edge, e) if q_edge is not None else None
                pix_cos = 0.0 if pix_cos is None else pix_cos
                pix_corr = 0.0 if pix_corr is None else pix_corr
                edge_cos = 0.0 if edge_cos is None else edge_cos
                pix_norm = (pix_cos + 1.0) / 2.0
                pix_corr_norm = (pix_corr + 1.0) / 2.0
                edge_norm = (edge_cos + 1.0) / 2.0
                visual_sim = 0.5 * pix_norm + 0.3 * pix_corr_norm + 0.2 * edge_norm
                m["pixel_sim"] = visual_sim
                m["edge_sim"] = edge_norm
            # fallback: 用sim_score填补，避免N/A
            if m.get("pixel_sim") is None:
                m["pixel_sim"] = m.get("sim_score", m.get("score", 0))
            if m.get("edge_sim") is None:
                m["edge_sim"] = m.get("pixel_sim")

        # 2) 相关性与回报相关兜底 + 趋势/段落一致性过滤
        if query_prices is not None:
            try:
                dfp = loader.get_stock_data(sym)
                if dfp is not None and not dfp.empty:
                    dfp.index = pd.to_datetime(dfp.index)
                    dt = pd.to_datetime(str(date_str), errors="coerce")
                    if dt not in dfp.index:
                        candidates = dfp.index[dfp.index <= dt]
                        if len(candidates) == 0:
                            continue
                        dt = candidates.max()
                    loc = dfp.index.get_loc(dt)
                    if loc >= 19:
                            match_prices = dfp.iloc[loc - 19: loc + 1]['Close'].values
                            if filter_trend and query_trend is not None:
                                match_trend = 1 if match_prices[-1] > match_prices[0] else -1
                                m["trend_match"] = 1 if match_trend == query_trend else 0
                                if match_trend != query_trend:
                                    trend_penalty = -0.12
                            seg_agree_ratio = None
                            if filter_trend and query_segs is not None:
                                match_segs = _segment_signs(match_prices)
                                if match_segs is not None:
                                    keys = [k for k in query_segs.keys() if k in match_segs]
                                    if keys:
                                        agree = sum(1 for k in keys if query_segs[k] == match_segs[k])
                                        seg_agree_ratio = agree / len(keys)
                                        m["seg_agree"] = round(seg_agree_ratio, 4)
                                        if seg_agree_ratio < 0.6:
                                            seg_penalty = -0.10
                            qn = (query_prices - query_prices.mean()) / (query_prices.std() + 1e-8)
                            mn = (match_prices - match_prices.mean()) / (match_prices.std() + 1e-8)
                            corr = np.corrcoef(qn, mn)[0, 1]
                            if not np.isnan(corr):
                                m["correlation"] = float(corr)
                                if corr < 0:
                                    corr_penalty = -0.15
                            q_ret = np.diff(query_prices) / (query_prices[:-1] + 1e-8)
                            m_ret = np.diff(match_prices) / (match_prices[:-1] + 1e-8)
                            q_ret = (q_ret - q_ret.mean()) / (q_ret.std() + 1e-8)
                            m_ret = (m_ret - m_ret.mean()) / (m_ret.std() + 1e-8)
                            corr2 = np.corrcoef(q_ret, m_ret)[0, 1]
                            if not np.isnan(corr2):
                                m["ret_corr"] = float(corr2)
                            # 多段相关性（模拟多头注意力）
                            seg_corrs = _segment_corrs(query_prices, match_prices)
                            if seg_corrs:
                                m["seg_corr_head"] = seg_corrs.get("head")
                                m["seg_corr_mid"] = seg_corrs.get("mid")
                                m["seg_corr_tail"] = seg_corrs.get("tail")
                                m["seg_corr_front"] = seg_corrs.get("front")
                                m["seg_corr_back"] = seg_corrs.get("back")
                                m["seg_corr_start"] = seg_corrs.get("start")
                                m["seg_corr_end"] = seg_corrs.get("end")
                                vals = list(seg_corrs.values())
                                if vals:
                                    seg_corr_mean = float(np.mean(vals))
                                    seg_corr_norm = (seg_corr_mean + 1.0) / 2.0
                                    m["seg_corr_mean"] = seg_corr_mean
                                    m["seg_corr_norm"] = seg_corr_norm
                                    if seg_corr_mean < 0:
                                        seg_penalty -= 0.12
            except Exception:
                pass
        # 形态综合评分（强调头/中/尾 + 相关性/回报相关）
        corr = m.get("correlation")
        ret_corr = m.get("ret_corr")
        if corr is None and ret_corr is not None:
            corr = ret_corr
        corr_norm = (float(corr) + 1.0) / 2.0 if corr is not None else 0.5
        seg_score = m.get("seg_score", None)
        seg_agree_ratio = m.get("seg_agree")
        seg_corr_norm = m.get("seg_corr_norm")
        if seg_score is None:
            parts = []
            if seg_agree_ratio is not None:
                parts.append(0.6 * float(seg_agree_ratio))
            if seg_corr_norm is not None:
                parts.append(0.4 * float(seg_corr_norm))
            seg_score = float(np.mean(parts)) if parts else 0.5
            m["seg_score"] = seg_score
        pix = m.get("pixel_sim", m.get("score", m.get("sim_score", 0.5)))
        trend_match = m.get("trend_match", 1)
        if m.get("trend_match") is None:
            m["trend_match"] = trend_match
        seg_corr_norm = seg_corr_norm if seg_corr_norm is not None else 0.5
        shape_score = 0.45 * corr_norm + 0.25 * seg_score + 0.15 * seg_corr_norm + 0.10 * pix + 0.05 * trend_match

        base_score = m.get("score", m.get("sim_score", 0.0))
        m["score"] = float(shape_score) + trend_penalty + seg_penalty + corr_penalty

        # 严格候选：趋势一致 + 段落一致 + 相关性非负
        if trend_match == 1 and (seg_agree_ratio is None or seg_agree_ratio >= 0.7) and (corr is None or corr >= 0) and (m.get("seg_corr_mean") is None or m.get("seg_corr_mean") >= 0):
            strict_ok = True

        if strict_ok:
            strict_list.append(m)
        else:
            relaxed_list.append(m)

    strict_list.sort(key=lambda x: x.get("score", 0), reverse=True)
    relaxed_list.sort(key=lambda x: x.get("score", 0), reverse=True)
    return strict_list + relaxed_list


def _filter_trend_matches(matches, top_k: int = 10, require_corr_nonneg: bool = True):
    """
    严格确保大趋势一致，优先保留趋势一致且相关性非负的候选
    """
    if not matches:
        return matches

    def _is_trend_ok(m):
        if m.get("trend_match") != 1:
            return False
        if require_corr_nonneg:
            corr = m.get("correlation")
            if corr is not None and corr < 0:
                return False
        return True

    strict = [m for m in matches if _is_trend_ok(m)]
    if len(strict) >= top_k:
        return strict[:top_k]

    # 放宽相关性限制（但仍要求趋势一致）
    relaxed = [m for m in matches if m.get("trend_match") == 1 and m not in strict]
    combined = strict + relaxed
    if len(combined) >= top_k:
        return combined[:top_k]

    # 仍不足则回填原始候选，避免Top10空缺
    if len(combined) < top_k:
        seen = {(m.get("symbol"), m.get("date")) for m in combined}
        for m in matches:
            key = (m.get("symbol"), m.get("date"))
            if key in seen:
                continue
            combined.append(m)
            seen.add(key)
            if len(combined) >= top_k:
                break
    return combined[:top_k]

st.set_page_config(page_title="VisionQuant Pro", layout="wide", page_icon="🦄")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef; }
    .agent-box { border-left: 5px solid #ff4b4b; padding: 20px; background-color: #fff1f1; border-radius: 5px; margin-bottom: 20px; }
    .stChatMessage { background-color: #ffffff; border-radius: 12px; padding: 12px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_engines(_code_version: str):
    dl_mod = importlib.import_module("src.data.data_loader")
    ve_mod = importlib.import_module("src.models.vision_engine")
    fm_mod = importlib.import_module("src.strategies.fundamental")
    qa_mod = importlib.import_module("src.agent.quant_agent")
    importlib.reload(dl_mod)
    importlib.reload(ve_mod)
    importlib.reload(fm_mod)
    importlib.reload(qa_mod)
    v = ve_mod.VisionEngine()
    # 延迟加载索引：只在第一次搜索时加载，避免启动时20-30分钟的等待
    # v.reload_index()  # 已移除：索引将在第一次调用search_similar_patterns时自动加载
    return {
        "loader": dl_mod.DataLoader(mem_cache_max=128), "vision": v, "factor": FactorMiner(),
        "fund": fm_mod.FundamentalMiner(), "agent": qa_mod.QuantAgent(), 
        "news": NewsHarvester(), "audio": AudioManager()
    }

eng = load_all_engines(_code_version=_code_version_key())

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_context" not in st.session_state: st.session_state.last_context = ""
if "has_run" not in st.session_state: st.session_state.has_run = False
if "last_voice_text" not in st.session_state: st.session_state.last_voice_text = ""
if "batch_results" not in st.session_state: st.session_state.batch_results = {}
if "portfolio_weights" not in st.session_state: st.session_state.portfolio_weights = {}
if "portfolio_metrics" not in st.session_state: st.session_state.portfolio_metrics = {}
if "current_symbol" not in st.session_state: st.session_state.current_symbol = None
if "ic_summary" not in st.session_state: st.session_state.ic_summary = {}
if "ai_reports" not in st.session_state: st.session_state.ai_reports = {}

# URL 跳转预处理：先写入 session_state，让侧边栏控件同步
url_symbol = st.query_params.get("symbol")
if url_symbol:
    st.session_state["symbol_input"] = url_symbol
    st.session_state["mode_select"] = "🔍 单只股票分析"

from backtest_handlers import run_backtest, run_stratified_backtest_batch
from factor_analysis_handlers import show_factor_analysis as render_factor_analysis
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

with st.sidebar:
    st.title("🦄 VisionQuant Pro")
    st.caption("AI 全栈量化投研系统 v8.8")

    # === 数据源选择 ===
    with st.expander("⚙️ 数据源设置", expanded=False):
        ds_map = {
            "掘金SDK (推荐)": "diggold",
            "AkShare (免费)": "akshare",
            "JQData (聚宽)": "jqdata",
            "RQData (米筐)": "rqdata"
        }
        ds_label = st.selectbox("选择数据源", list(ds_map.keys()), index=0)
        curr_ds = ds_map[ds_label]

        # 数据源状态提示
        if curr_ds == "diggold":
            token = os.getenv("DIGGOLD_TOKEN", "")
            if token:
                st.success("✅ 掘金SDK已配置")
            else:
                st.warning("⚠️ 未配置DIGGOLD_TOKEN")

        # 如果选了付费源，检查/提示输入账号
        if curr_ds in ["jqdata", "rqdata"]:
            st.caption(f"需提供 {curr_ds} 账号 (或设置环境变量)")
            ds_user = st.text_input("用户名", key=f"{curr_ds}_user")
            ds_pass = st.text_input("密码", type="password", key=f"{curr_ds}_pass")
            if st.button("切换/认证", key=f"switch_{curr_ds}"):
                eng["loader"].switch_data_source(curr_ds, username=ds_user, password=ds_pass)
                st.success(f"已尝试切换至 {curr_ds}")
        else:
            # 自动切换到选中的免费数据源
            if eng["loader"].get_current_data_source() != curr_ds:
                eng["loader"].switch_data_source(curr_ds)
                st.success(f"已切换至 {ds_label}")

        st.divider()
    symbol_input = st.text_input("请输入 A 股代码", value="000818", help="输入6位代码", key="symbol_input")
    symbol = symbol_input.strip().zfill(6)
    mode = st.radio("功能模块:", ("🔍 单只股票分析", "📊 批量组合分析"), key="mode_select")

    if mode == "🔍 单只股票分析":
        st.divider()
        st.caption("回测 / 因子有效性分析入口已统一放在“单只股票分析”报告底部 Tab 中（更符合使用路径）。")
    
    elif mode == "📊 批量组合分析":
        batch_input = st.text_area("输入股票代码（每行一个，最多30只）", height=150, key="batch_input")

    st.divider()
    run_btn = _st_button("🚀 开始分析", type="primary", use_container_width=True)

    if st.button("🔄 强制重载", help="清除缓存，重新加载模块"):
        st.cache_resource.clear()
        st.rerun()

url_jump_mode = False
if url_symbol:
    if url_symbol != symbol:
        symbol = url_symbol
        url_jump_mode = True
        mode = "🔍 单只股票分析"
        st.session_state["symbol_input"] = symbol
        st.session_state["mode_select"] = "🔍 单只股票分析"
        if "res" in st.session_state:
            del st.session_state.res
        st.session_state.current_symbol = symbol
        st.session_state.has_run = True
        run_btn = True
    elif "res" not in st.session_state:
        url_jump_mode = True
        mode = "🔍 单只股票分析"
        st.session_state["mode_select"] = "🔍 单只股票分析"
        st.session_state.has_run = True
        run_btn = True
    else:
        st.query_params.clear()
        url_jump_mode = False

if not run_btn and not st.session_state.has_run:
    st.header(f"👋 欢迎使用 VisionQuant Pro")
    st.info(f"当前选中标的: **{symbol}**\n请在左侧侧边栏点击红色按钮启动。")
    st.stop()

if mode == "🔍 单只股票分析":
    if st.session_state.current_symbol != symbol and st.session_state.current_symbol is not None:
        if "res" in st.session_state:
            del st.session_state.res
        st.session_state.has_run = False
        st.session_state.chat_history = []
        st.session_state.last_voice_text = ""
    
    if run_btn:
        st.session_state.has_run = True
        st.session_state.chat_history = []
        st.session_state.last_voice_text = ""
        st.session_state.current_symbol = symbol
        if "res" in st.session_state:
            del st.session_state.res

        progress = st.progress(0)
        status = st.empty()
        status.write("加载行情数据...")
        with st.spinner(f"正在全栈扫描 {symbol}..."):
            try:
                logger.info(f"开始分析股票: {symbol}")
                from datetime import timedelta
                one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
                df = eng["loader"].get_stock_data(symbol, start_date=one_year_ago, use_cache=True)
                if df is None or df.empty:
                    logger.warning(f"本地缓存缺失，尝试实时拉取: {symbol}")
                    df = eng["loader"].get_stock_data(symbol, start_date=one_year_ago, use_cache=False)
                    if df is None or df.empty:
                        st.error("数据获取失败")
                        logger.error(f"数据获取失败: {symbol}")
                        st.stop()
                else:
                    # 非阻塞体验：缓存可用则继续流程，不强制等待实时网络
                    pass
            except Exception as e:
                logger.exception(f"数据获取异常: {symbol}")
                st.error(f"数据获取失败: {str(e)}")
                st.stop()
            progress.progress(20)

            # 数据质量报告
            try:
                quality_report = eng["loader"].quality_checker.check_data_quality(df, symbol)
            except Exception:
                quality_report = {}
            progress.progress(30)

            def _has_valid_num(val):
                try:
                    return val is not None and np.isfinite(float(val)) and float(val) != 0
                except Exception:
                    return False

            def _fund_ok(fd):
                if not isinstance(fd, dict):
                    return False
                ok = fd.get("_ok", {}) or {}
                if ok.get("spot") or ok.get("finance"):
                    return True
                return any([
                    _has_valid_num(fd.get("pe_ttm")),
                    _has_valid_num(fd.get("pb")),
                    _has_valid_num(fd.get("total_mv")),
                ])

            # 基本面：先缓存后实时（优先响应速度）
            fund_data = _safe_get_fundamentals(eng["fund"], symbol, force_live=False)
            if not _fund_ok(fund_data):
                fund_data = _safe_get_fundamentals(eng["fund"], symbol, force_live=True)
            stock_name = fund_data.get('name') or symbol
            status.write("生成查询K线图...")

            # 始终用“最新行情”生成查询K线图（避免历史缓存图导致错配）
            query_window = df.tail(20).copy()
            if query_window is None or query_window.empty:
                st.error("查询窗口不足，无法生成K线图")
                st.stop()
            query_dt = query_window.index[-1]
            query_date_str = query_dt.strftime("%Y%m%d") if query_dt is not None else df.index[-1].strftime("%Y%m%d")
            q_p = os.path.join(PROJECT_ROOT, "data", f"temp_q_{symbol}_{query_date_str}.png")
            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
            mpf.plot(query_window, type='candle', style=s, savefig=dict(fname=q_p, dpi=50), figsize=(3, 3), axisoff=True)
            progress.progress(45)
            
            query_prices = query_window['Close'].values if query_window is not None and len(query_window) >= 20 else None
            # 多尺度检索（日/周/月）+ 动态权重融合
            status.write("相似形态检索中...")
            target_k = 10
            search_k = 80
            try:
                from src.data.multi_scale_generator import MultiScaleChartGenerator
                gen = MultiScaleChartGenerator(figsize=(3, 3), dpi=50)
                q_week = os.path.join(PROJECT_ROOT, "data", "temp_q_week.png")
                q_month = os.path.join(PROJECT_ROOT, "data", "temp_q_month.png")
                df_for_query = df.loc[:query_dt] if query_dt is not None else df
                gen.generate_weekly_chart(df_for_query, weeks=20, output_path=q_week)
                gen.generate_monthly_chart(df_for_query, months=20, output_path=q_month)
                img_paths = {"daily": q_p, "weekly": q_week, "monthly": q_month}
                # 动态融合权重：基于各周期的收益分布质量评分
                try:
                    kline_factor_calc = KLineFactorCalculator(data_loader=eng["loader"])
                    # 仅用于权重估计，使用快速模式减少耗时
                    scale_matches = {
                        "daily": eng["vision"].search_similar_patterns(
                            q_p, top_k=10, query_prices=query_prices, max_date=query_date_str,
                            fast_mode=True, search_k=400, rerank_with_pixels=False
                        ),
                        "weekly": eng["vision"].search_similar_patterns(
                            q_week, top_k=10, max_date=query_date_str,
                            fast_mode=True, search_k=400, rerank_with_pixels=False
                        ),
                        "monthly": eng["vision"].search_similar_patterns(
                            q_month, top_k=10, max_date=query_date_str,
                            fast_mode=True, search_k=400, rerank_with_pixels=False
                        ),
                    }
                    scale_stats = {
                        k: kline_factor_calc.calculate_return_distribution(v, horizon_days=5, query_date=query_date_str)
                        for k, v in scale_matches.items()
                    }
                    scale_weights = kline_factor_calc.estimate_scale_weights(scale_stats)
                except Exception:
                    scale_weights = None

                matches = eng["vision"].search_multi_scale_patterns(
                    img_paths, top_k=search_k, query_prices=query_prices, weights=scale_weights, max_date=query_date_str
                )
            except Exception:
                matches = eng["vision"].search_similar_patterns(
                    q_p, top_k=search_k, query_prices=query_prices, max_date=query_date_str
                )
            progress.progress(65)

            # 补齐相似度字段，减少 N/A
            matches = _augment_matches(matches, q_p, query_prices, eng["loader"], eng["vision"], os.path.join(PROJECT_ROOT, "data"), filter_trend=True)
            matches = _filter_trend_matches(matches, top_k=target_k)
            progress.progress(75)

            # 大盘情绪（沪深300，后验）
            market_sentiment = _compute_market_sentiment(eng["loader"], index_code="000300", end_dt=query_dt)

            def get_future_trajectories(matches, loader):
                trajectories, details = [], []
                for m in matches:
                    try:
                        hdf = loader.get_stock_data(m['symbol'])
                        hdf.index = pd.to_datetime(hdf.index)
                        target_date = pd.to_datetime(m['date'])
                        if target_date in hdf.index:
                            loc = hdf.index.get_loc(target_date)
                            if loc + 5 < len(hdf):
                                subset = hdf.iloc[loc: loc + 6]['Close'].values
                                norm_path = (subset / subset[0] - 1) * 100
                                trajectories.append(norm_path)
                                details.append(f"{m['symbol']} ({m['date']})")
                    except:
                        continue
                return trajectories, details

            trajs, traj_labels = get_future_trajectories(matches, eng["loader"])

            if trajs:
                mean_path = np.mean(np.vstack(trajs), axis=0)
                avg_ret = mean_path[-1]
                traditional_win_rate = np.sum(np.vstack(trajs)[:, -1] > 0) / len(trajs) * 100
            else:
                mean_path, avg_ret, traditional_win_rate = np.zeros(6), 0.0, 50.0

            # Top10多期收益/分布估计
            try:
                from src.utils.top10_analyzer import Top10Analyzer
                analyzer = Top10Analyzer(eng["loader"])
                mh_stats = analyzer.analyze_multi_horizon(matches, horizons=[5, 10, 20])
                dist_stats = analyzer.return_distribution(matches, future_days=20)
            except Exception:
                mh_stats, dist_stats = {}, {}

            try:
                kline_factor_calc = KLineFactorCalculator(data_loader=eng["loader"])
                query_date_for_calc = query_date_str if query_date_str else datetime.now().strftime('%Y%m%d')
                query_df_for_calc = df.loc[:query_dt] if query_dt is not None else df
                hybrid_win_rate_result = kline_factor_calc.calculate_hybrid_win_rate(
                    matches, 
                    query_symbol=symbol,
                    query_date=query_date_for_calc,
                    query_df=query_df_for_calc
                )
                if isinstance(hybrid_win_rate_result, dict):
                    hybrid_win_rate = hybrid_win_rate_result.get('hybrid_win_rate', traditional_win_rate)
                else:
                    hybrid_win_rate = traditional_win_rate
                    hybrid_win_rate_result = None
                logger.info(f"混合胜率计算成功: {symbol}, 胜率={hybrid_win_rate:.1f}%")
            except Exception as e:
                logger.warning(f"混合胜率计算失败，使用传统胜率: {symbol}, 错误={str(e)}")
                hybrid_win_rate = traditional_win_rate
                hybrid_win_rate_result = None
            progress.progress(85)
            
            win_rate = hybrid_win_rate if hybrid_win_rate is not None else traditional_win_rate
            enhanced_factor = None
            enhanced_score = None
            if isinstance(hybrid_win_rate_result, dict):
                enhanced_factor = hybrid_win_rate_result.get("enhanced_factor")
                if isinstance(enhanced_factor, dict):
                    enhanced_score = enhanced_factor.get("final_score")
            # 多因子评分使用增强因子分数（若有），否则回退混合胜率
            win_rate_for_score = enhanced_score if enhanced_score is not None else win_rate

            df_f = eng["factor"]._add_technical_indicators(df)
            news_text = eng["news"].get_latest_news(symbol)
            # 行业对标：先缓存后实时（优先响应速度）
            ind_name, peers_df = _safe_get_industry_peers(eng["fund"], symbol, force_live=False)
            if (
                not ind_name
                or ind_name in ["未知", "上海主板", "深圳主板", "创业板", "科创板"]
                or peers_df is None
                or peers_df.empty
                or len(peers_df) < 2
            ):
                ind_name, peers_df = _safe_get_industry_peers(eng["fund"], symbol, force_live=True)
            if (not ind_name or ind_name == "未知") and fund_data.get("industry"):
                ind_name = fund_data.get("industry")
            progress.progress(95)

            returns = df['Close'].pct_change().dropna()
            total_score, initial_action, s_details = eng["factor"].get_scorecard(
                win_rate_for_score, df_f.iloc[-1], fund_data, returns=returns if not returns.empty else None
            )

            ic_summary = (st.session_state.get("ic_summary") or {}).get(symbol)
            report = None

            c_p = os.path.join(PROJECT_ROOT, "data", "comparison.png")
            create_comparison_plot(q_p, matches, c_p)

            res_dict = {
                "name": stock_name, "c_p": c_p, "trajs": trajs, "mean": mean_path,
                "win": win_rate, "ret": avg_ret, "labels": traj_labels,
                "score": total_score, "act": initial_action, "det": s_details,
                "fund": fund_data, "df_f": df_f, "ind": ind_name, "peers": peers_df,
                "news": news_text, "rep": report,
                "mh_stats": mh_stats, "dist_stats": dist_stats,
                "matches": matches, "q_p": q_p,
                "quality_report": quality_report,
                "market_sentiment": market_sentiment
            }
            if enhanced_factor:
                res_dict["enhanced_factor"] = enhanced_factor
                res_dict["enhanced_score"] = enhanced_score
            
            if hybrid_win_rate_result and hybrid_win_rate is not None:
                res_dict["hybrid_win_rate"] = hybrid_win_rate
                res_dict["traditional_win_rate"] = traditional_win_rate
                res_dict["tb_win_rate"] = hybrid_win_rate_result.get('tb_win_rate', 0)
                res_dict["win_rate_type"] = "混合胜率"
            else:
                res_dict["win_rate_type"] = "传统胜率"
            
            st.session_state.res = res_dict
            progress.progress(100)
            status.empty()
            ic_summary_txt = ""
            if ic_summary:
                ic_summary_txt = f"IC均值: {ic_summary.get('mean_ic')} | IR: {ic_summary.get('ir')} | 显著性: {ic_summary.get('significant')}"

            st.session_state.last_context = f"""
            股票名称: {stock_name} ({symbol})
            当前时间: {datetime.now().strftime('%Y-%m-%d')}
            --- 量化数据 ---
            AI评分: {total_score}/10
            趋势信号: {initial_action}
            形态胜率: {win_rate:.1f}%
            IC摘要: {ic_summary_txt}
            市场情绪(沪深300): {market_sentiment.get('score', 'N/A')} | {market_sentiment.get('label', 'N/A')}
            --- 财务数据 ---
            ROE: {fund_data.get('roe')}%
            PE(TTM): {fund_data.get('pe_ttm')}
            --- 舆情摘要 ---
            {news_text[:500]}
            """

            if url_jump_mode:
                st.session_state.clear_url_after_render = True

    if "res" in st.session_state:
        if st.session_state.get("clear_url_after_render", False):
            st.query_params.clear()
            st.session_state.clear_url_after_render = False
        
        d = st.session_state.res
        display_name = (d.get("name") or "").strip()
        if (not display_name) or (display_name == symbol):
            st.markdown(f"# 📊 深度投研报告: {symbol}")
        else:
            st.markdown(f"# 📊 深度投研报告: {display_name} ({symbol})")

        st.subheader("1. 视觉模式识别")
        with st.expander("ℹ️ 数据来源说明", expanded=False):
            st.markdown("""
            **Top10相似K线匹配**:
            - 使用AttentionCAE模型提取K线形态特征
            - 通过FAISS向量数据库搜索历史相似模式
            - 匹配结果包含：股票代码、日期、相似度分数
            - 计算这些历史模式的未来表现作为预测依据
            """)
        if d.get("quality_report"):
            with st.expander("🧪 数据质量报告", expanded=False):
                qr = d["quality_report"]
                st.write(f"质量评分: {qr.get('score', 'N/A')}")
                st.write(f"样本量: {qr.get('data_points', 'N/A')}")
                st.write(f"时间范围: {qr.get('date_range', {}).get('start')} ~ {qr.get('date_range', {}).get('end')}")
                if qr.get("missing_stats"):
                    st.write(f"缺失率: {qr['missing_stats'].get('missing_ratio', 0)*100:.2f}%")
                    by_col = qr["missing_stats"].get("by_column", {})
                    if by_col:
                        fig_miss = go.Figure()
                        fig_miss.add_trace(go.Bar(x=list(by_col.keys()), y=list(by_col.values())))
                        fig_miss.update_layout(height=250, title="缺失值分布")
                        _st_plotly_chart(fig_miss, use_container_width=True)
                if qr.get("adjust_integrity"):
                    adj = qr["adjust_integrity"]
                    if adj.get("available"):
                        st.write(f"复权完整性: {adj.get('column')} 缺失率 {adj.get('missing_ratio', 0)*100:.2f}%")
                    else:
                        st.write("复权完整性: 未提供复权列")
                if qr.get("warnings"):
                    st.write("警告: " + "; ".join(qr.get("warnings", [])[:5]))
        _st_image(d['c_p'], use_container_width=True)

        # 相似度分解（视觉相似度/相关性）
        if d.get("matches"):
            rows = []
            for m in d["matches"]:
                vector_score = m.get("vector_score")
                corr = m.get("correlation")
                sim_score = m.get("sim_score")
                seg_agree = m.get("seg_agree")
                seg_corr_mean = m.get("seg_corr_mean")
                head_corr = m.get("seg_corr_head")
                mid_corr = m.get("seg_corr_mid")
                tail_corr = m.get("seg_corr_tail")
                if sim_score is None:
                    if vector_score is not None:
                        sim_score = 1.0 / (1.0 + max(float(vector_score), 0.0))
                    else:
                        sim_score = m.get("score", 0)
                corr_norm = None if corr is None else (float(corr) + 1.0) / 2.0
                pix_sim = m.get("pixel_sim")
                edge_sim = m.get("edge_sim")
                ret_corr = m.get("ret_corr")
                rows.append({
                    "股票": f"{m.get('symbol')}",
                    "日期": f"{m.get('date')}",
                    "相似度": round(float(sim_score), 4),
                    "像素相似": round(float(pix_sim), 4) if pix_sim is not None else 0.0,
                    "边缘相似": round(float(edge_sim), 4) if edge_sim is not None else 0.0,
                    "相关性": round(float(corr_norm), 4) if corr_norm is not None else 0.0,
                    "回报相关": round(float((ret_corr+1)/2), 4) if ret_corr is not None else 0.0,
                    "段落一致": round(float(seg_agree), 4) if seg_agree is not None else 0.0,
                    "分段相关均值": round(float(seg_corr_mean), 4) if seg_corr_mean is not None else 0.0,
                    "头段相关": round(float(head_corr), 4) if head_corr is not None else 0.0,
                    "中段相关": round(float(mid_corr), 4) if mid_corr is not None else 0.0,
                    "尾段相关": round(float(tail_corr), 4) if tail_corr is not None else 0.0,
                    "最终分": round(float(m.get("score", 0)), 4)
                })
            with st.expander("🔍 相似度分解（可解释）", expanded=False):
                _st_dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 注意力热力图（如果模型支持）
        try:
            if hasattr(eng["vision"].model, "get_attention_weights"):
                with st.expander("🔥 注意力热力图（解释性）", expanded=False):
                    mode = st.selectbox("显示方式", ["多头(全部)", "单头"], index=0, key="attn_mode")
                    heat_path = os.path.join(PROJECT_ROOT, "data", "temp_attention.png")
                    if mode == "多头(全部)":
                        eng["vision"].generate_attention_heatmap(d.get("q_p"), save_path=heat_path, mode="all")
                        _st_image(heat_path, use_container_width=True)
                    else:
                        num_heads = getattr(eng["vision"].model, "num_attention_heads", 8)
                        head_idx = st.slider("选择注意力头", 0, max(0, num_heads - 1), 0, key="attn_head")
                        eng["vision"].generate_attention_heatmap(d.get("q_p"), save_path=heat_path, head_idx=head_idx, mode="single")
                        _st_image(heat_path, use_container_width=True)
                    if os.path.exists(heat_path):
                        os.remove(heat_path)
        except Exception:
            pass

        # 大盘情绪（后验，不影响匹配）
        st.subheader("1.5 大盘情绪（沪深300，后验）")
        with st.expander("ℹ️ 指标说明", expanded=False):
            st.markdown("""
            - 仅作为**后验补充特征**，不影响K线相似度排序
            - 日频：1D/1W/4W涨跌幅、20日年化波动
            - 周频：4周累计涨跌、8周波动
            - 风险：近60日最大回撤、20日平均振幅
            """)
        sent = d.get("market_sentiment", {}) or {}
        if not sent.get("ok"):
            st.warning(f"⚠️ 大盘情绪暂不可用：{sent.get('error', '未知原因')}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("情绪指数", f"{sent.get('score', 0)}/100", sent.get("label", ""))
            c2.metric("最新收盘", f"{sent.get('latest_close', 0):.2f}")
            c3.metric("1D涨跌", f"{(sent.get('ret_1d') or 0) * 100:.2f}%")
            c4.metric("1W涨跌", f"{(sent.get('ret_5d') or 0) * 100:.2f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("4W涨跌", f"{(sent.get('ret_20d') or 0) * 100:.2f}%")
            c6.metric("20D波动(年化)", f"{(sent.get('vol_20d') or 0) * 100:.2f}%")
            c7.metric("60D最大回撤", f"{(sent.get('max_drawdown_60d') or 0) * 100:.2f}%")
            c8.metric("量能比(20D)", f"{(sent.get('volume_ratio') or 0):.2f}")

            with st.expander("📈 沪深300近120日走势", expanded=False):
                if sent.get("series_dates") and sent.get("series_values"):
                    fig_idx = go.Figure()
                    fig_idx.add_trace(go.Scatter(
                        x=sent["series_dates"],
                        y=sent["series_values"],
                        mode="lines",
                        name="沪深300收盘"
                    ))
                    fig_idx.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                    _st_plotly_chart(fig_idx, use_container_width=True)
            st.caption(f"公式：{sent.get('formula')}")

        if d['trajs']:
            fig = go.Figure()
            for i, p in enumerate(d['trajs']):
                fig.add_trace(go.Scatter(y=p, mode='lines', line=dict(color='rgba(200,200,200,0.5)', width=1),
                                         name=d['labels'][i]))
            fig.add_trace(go.Scatter(y=d['mean'], mode='lines+markers', line=dict(color='#d62728', width=3), name='平均预期'))
            fig.update_layout(title=f"未来5日走势推演 (胜率: {d['win']:.0f}%)", xaxis_title="天数", yaxis_title="收益%", height=400)
            _st_plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("历史胜率", f"{d['win']:.1f}%")
            c2.metric("预期收益", f"{d['ret']:.2f}%")
            if d.get("enhanced_score") is not None:
                c3.metric("增强因子分", f"{d['enhanced_score']:.2f}")
                c4.metric("信号强度", d.get("enhanced_factor", {}).get("signal_level", "N/A"))
            else:
                c3.metric("增强因子分", "N/A")
                c4.metric("信号强度", "N/A")
            
            # 胜率计算公式说明
            if d.get('win_rate_type') == '混合胜率' and 'hybrid_win_rate' in d:
                with c3:
                    with st.expander("📐 胜率计算公式", expanded=False):
                        st.markdown("""
                        **混合胜率 = Triple Barrier胜率 × 70% + 传统胜率 × 30%**
                        
                        - **Triple Barrier胜率**: 基于止盈(+5%)、止损(-3%)、最大持有20天的标签统计
                        - **传统胜率（相似度加权）**: 未来5日收益率>0 的比例，按 Top10 匹配的相似度 `score` 加权
                        - **数据来源**: Top10相似K线模式的历史表现
                        """)
                        if 'tb_win_rate' in d:
                            st.caption(f"TB胜率: {d.get('tb_win_rate', 0):.1f}% | 传统胜率: {d.get('traditional_win_rate', 0):.1f}%")

            # 多期收益曲线（5/10/20）
            mh = d.get("mh_stats", {})
            if mh.get("valid") and mh.get("horizon_stats"):
                hs = mh["horizon_stats"]
                mh_fig = go.Figure()
                for h, stats in hs.items():
                    mh_fig.add_trace(go.Scatter(
                        x=[h], y=[stats.get("avg_return", 0)],
                        mode="markers+text", text=[f"{h}日"],
                        name=f"{h}日"
                    ))
                mh_fig.update_layout(title="多期收益预期（5/10/20日）", xaxis_title="持有期(天)", yaxis_title="均值收益(%)", height=280)
                _st_plotly_chart(mh_fig, use_container_width=True)

            # 收益分布估计
            dist = d.get("dist_stats", {})
            if dist.get("valid"):
                with st.expander("📊 收益分布估计（更严格）", expanded=False):
                    st.write(f"样本数: {dist.get('count')}")
                    st.write(f"均值: {dist.get('mean'):.2f}% | 中位数: {dist.get('median'):.2f}%")
                    st.write(f"分位数: Q05={dist.get('q05'):.2f}%, Q25={dist.get('q25'):.2f}%, Q75={dist.get('q75'):.2f}%")
                    st.write(f"CVaR(5%): {dist.get('cvar'):.2f}%")

            # 复合因子解释（分布 + 情境 + 量价）
            if d.get("enhanced_factor"):
                ef = d["enhanced_factor"]
                with st.expander("🧭 情境感知与量价复合因子（新增）", expanded=False):
                    st.write(f"最佳持有期: {ef.get('best_horizon', 'N/A')} 天")
                    st.write(f"信号强度: {ef.get('signal_level', 'N/A')} | 增强因子分: {ef.get('final_score', 'N/A')}")
                    context = ef.get("context", {})
                    st.caption(f"Regime: {context.get('regime')} | 波动率: {context.get('volatility')} | 流动性评分: {context.get('liquidity_score')}")
                    money = ef.get("money_features", {})
                    if money:
                        st.write("量价/资金特征:")
                        cn_map = {
                            "vol_ratio": "量比(成交量/均量)",
                            "pv_corr": "价量相关性",
                            "obv_slope": "OBV斜率",
                            "cmf": "CMF资金流",
                            "mfi": "MFI资金流指标",
                            "turnover": "换手率(%)",
                            "amount": "成交额",
                            "vwap": "成交均价"
                        }
                        rows = []
                        for k, v in money.items():
                            label = cn_map.get(k, k)
                            if v is None:
                                val = "N/A"
                            else:
                                if k in ["turnover"]:
                                    val = f"{v:.2f}"
                                else:
                                    val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                            rows.append({"指标": label, "数值": val})
                        if rows:
                            _st_dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    dist_map = ef.get("dist_map", {})
                    if dist_map:
                        rows = []
                        for h, stats in dist_map.items():
                            if not stats or not stats.get("valid"):
                                continue
                            rows.append({
                                "持有期": h,
                                "均值": round(stats.get("mean", 0), 2),
                                "胜率": round(stats.get("win_rate", 0), 2),
                                "CVaR": round(stats.get("cvar", 0), 2),
                                "偏度": stats.get("skew"),
                                "峰度": stats.get("kurt"),
                                "赔率": stats.get("odds")
                            })
                        if rows:
                            _st_dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        c_left, c_right = st.columns([1.5, 1])
        with c_left:
            st.subheader("2. 量化多因子看板")
            det = d.get("det", {})

            def _build_weights_payload():
                payload = {
                    "视觉权重": det.get("视觉权重"),
                    "财务权重": det.get("财务权重"),
                    "量化权重": det.get("量化权重"),
                    "regime": det.get("regime"),
                    "权重说明": det.get("权重说明"),
                    "权重更新时间": det.get("权重更新时间"),
                    "source": "dynamic",
                    "error": None
                }
                if payload["视觉权重"] is None or not payload["权重说明"]:
                    try:
                        from src.strategies.regime_manager import RegimeManager
                        rm = RegimeManager(data_loader=eng["loader"])
                        df_tmp = d.get("df_f")
                        returns_tmp = None
                        if df_tmp is not None and "Close" in df_tmp.columns:
                            returns_tmp = pd.to_numeric(df_tmp["Close"], errors="coerce").pct_change().dropna()
                        dyn = rm.calculate_dynamic_weights(
                            returns=returns_tmp if returns_tmp is not None and not returns_tmp.empty else None
                        )
                        w = dyn.get("weights", {})
                        payload.update({
                            "视觉权重": w.get("kline_factor"),
                            "财务权重": w.get("fundamental"),
                            "量化权重": w.get("technical"),
                            "regime": dyn.get("regime"),
                            "权重说明": dyn.get("explain"),
                            "权重更新时间": dyn.get("timestamp"),
                            "source": "dynamic"
                        })
                    except Exception as e:
                        payload["source"] = "fixed"
                        payload["error"] = str(e)

                if payload["视觉权重"] is None:
                    payload.update({
                        "视觉权重": 0.6,
                        "财务权重": 0.2,
                        "量化权重": 0.2,
                        "regime": payload.get("regime") or "unknown",
                        "source": "fixed"
                    })
                return payload

            weights_payload = _build_weights_payload()
            explain = weights_payload.get("权重说明") or {}

            with st.expander("ℹ️ 因子说明", expanded=False):
                st.markdown("""
                **多因子评分系统 (V+F+Q)**:
                - **V (视觉因子)**: K线学习因子胜率
                - **F (基本面因子)**: ROE、PE、PB等
                - **Q (技术因子)**: MA、RSI、MACD等
                - **动态权重**: 根据 Regime + 统计风控因子自动调整
                """)
                st.caption("权重与公式详见下方 🧭 Regime 动态权重（算法与公式）")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI 总评分", f"{d['score']}/10", delta=d['act'])
            fund_ok = (d.get("fund", {}) or {}).get("_ok", {})
            pe_val = d['fund'].get('pe_ttm', 0)
            roe_val = d['fund'].get('roe', 0)
            try:
                pe_val = float(pe_val) if pe_val is not None else 0.0
            except Exception:
                pe_val = 0.0
            try:
                roe_val = float(roe_val) if roe_val is not None else 0.0
            except Exception:
                roe_val = 0.0
            roe_valid = np.isfinite(roe_val) and roe_val != 0
            pe_valid = np.isfinite(pe_val) and pe_val != 0
            roe_suffix = "" if fund_ok.get("finance") else " (估)"
            m2.metric("ROE", f"{roe_val:.2f}%{roe_suffix}" if roe_valid else "N/A")
            m3.metric("PE", f"{pe_val:.2f}" if pe_valid else "N/A")
            m4.metric("趋势", "看涨" if d['df_f'].iloc[-1]['MA_Signal'] > 0 else "看跌")

            with st.expander("📊 杜邦分析 & 因子明细"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**杜邦拆解**")
                    if fund_ok.get("finance"):
                        st.write(f"净利率: {d['fund'].get('net_profit_margin')}%")
                        st.write(f"周转率: {d['fund'].get('asset_turnover')}")
                        st.write(f"权益乘数: {d['fund'].get('leverage')}x")
                with col_b:
                    st.write("**技术因子**")
                    tech_row = None
                    try:
                        df_tmp = d.get("df_f")
                        if df_tmp is not None and not df_tmp.empty:
                            tech_row = df_tmp.iloc[-1]
                    except Exception:
                        tech_row = None
                    if tech_row is not None:
                        rows = []
                        ma_signal = tech_row.get("MA_Signal", None)
                        rsi = tech_row.get("RSI", None)
                        macd_hist = tech_row.get("MACD_Hist", None)
                        rows.append({
                            "指标": "均线趋势",
                            "数值": "看涨" if ma_signal is not None and float(ma_signal) > 0 else "看跌"
                        })
                        if rsi is not None:
                            rows.append({"指标": "RSI", "数值": round(float(rsi), 2)})
                        if macd_hist is not None:
                            rows.append({"指标": "MACD柱", "数值": round(float(macd_hist), 4)})
                        _st_dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("技术因子暂不可用")

            # 解释性评分（V/F/Q贡献）
            st.subheader("🧭 Regime 动态权重（算法与公式）")
            with st.expander("展开查看权重与公式", expanded=True):
                if weights_payload.get("source") == "fixed" and weights_payload.get("error"):
                    st.warning(f"动态权重补算失败，已回退固定权重：{weights_payload.get('error')}")

                st.write(f"当前Regime: {weights_payload.get('regime', 'N/A')}")
                weights_df = pd.DataFrame([
                    {"因子": "视觉(V)", "权重": weights_payload.get("视觉权重")},
                    {"因子": "基本面(F)", "权重": weights_payload.get("财务权重")},
                    {"因子": "技术(Q)", "权重": weights_payload.get("量化权重")},
                ])
                _st_dataframe(weights_df, use_container_width=True, hide_index=True)
                if explain:
                    st.caption(f"权重更新时间: {weights_payload.get('权重更新时间', 'N/A')}")
                    st.write("权重计算逻辑（统计/算法）：")
                    st.write(
                        f"趋势均值(年化): {explain.get('trend_60')} | 波动率(年化): {explain.get('vol_60')} | 最大回撤(120D): {explain.get('max_drawdown_120')}"
                    )
                    st.write(
                        f"偏度: {explain.get('skew_60')} | 峰度: {explain.get('kurt_60')} | 尾部风险: {explain.get('tail_score')}"
                    )
                    st.write(
                        f"趋势得分: {explain.get('trend_score')} | 波动评分: {explain.get('vol_score')} | 回撤评分: {explain.get('dd_score')}"
                    )
                    st.caption(f"公式：{explain.get('formula')}")
                else:
                    st.info("当前暂无可用的权重说明（已回退固定权重）。")
            try:
                v = float(det.get("视觉分(V)", 0))
                f = float(det.get("财务分(F)", 0))
                q = float(det.get("量化分(Q)", 0))
                total = v + f + q if (v + f + q) > 0 else 1.0
                contrib = pd.DataFrame([
                    {"因子": "视觉(V)", "贡献": f"{v/total*100:.1f}%"},
                    {"因子": "基本面(F)", "贡献": f"{f/total*100:.1f}%"},
                    {"因子": "技术(Q)", "贡献": f"{q/total*100:.1f}%"},
                ])
                with st.expander("🧠 评分占比（非权重）", expanded=False):
                    st.caption("说明：这里展示的是分数构成占比，不等同于权重。")
                    _st_dataframe(contrib, use_container_width=True, hide_index=True)
            except Exception:
                pass


        with c_right:
            st.subheader(f"3. 行业对标 ({d['ind']})")
            if d['peers'] is not None and not d['peers'].empty:
                st.dataframe(d['peers'], hide_index=True)
            else:
                st.warning(f"⚠️ 行业对标数据暂不可用（行业: {d['ind']}）")
                if d.get("fund", {}).get("_err"):
                    with st.expander("🔍 诊断信息", expanded=False):
                        st.json(d["fund"].get("_err", []))

        st.divider()
        st.subheader("4. 新闻舆情")
        with st.expander("ℹ️ 数据来源", expanded=False):
            st.markdown("""
            **新闻数据来源**: 
            - 通过akshare接口获取最新新闻
            - 包含公司公告、行业动态、市场资讯
            """)
        st.info(d['news'])

        # ---- 单只股票页：先因子/回测，最后 AI ----
        st.divider()
        tab_bt, tab_fa = st.tabs(["🧪 回测", "📈 因子有效性分析"])

        with tab_bt:
            st.subheader("🧪 策略模拟回测")
            with st.expander("ℹ️ 回测说明", expanded=False):
                st.markdown("""
                **回测策略逻辑**:
                - **仓位计算**: 基于MA60、MA20、MACD和AI胜率阈值
                - **交易成本**: 佣金(0.1%) + 滑点(0.1%) + 市场冲击 + 机会成本
                - **Turnover约束**: 单日最大换手率20%
                - **涨跌停/停牌约束**: 涨停不追、跌停不砍、停牌不交易（A股执行约束）
                - **止损机制**: 达到止损阈值(-8%)时强制平仓
                - **Walk-Forward**: 滚动窗口验证，防止未来函数泄漏
                """)

            cbt1, cbt2, cbt3, cbt4 = st.columns(4)
            with cbt1:
                bt_start_val = st.date_input("开始日期", value=st.session_state.get("bt_start", datetime(2010, 1, 1)), key="bt_start")
            with cbt2:
                bt_end_val = st.date_input("结束日期", value=st.session_state.get("bt_end", datetime.now()), key="bt_end")
            with cbt3:
                bt_cap_val = st.number_input("初始资金", value=st.session_state.get("bt_cap", 100000), key="bt_cap")
            with cbt4:
                bt_ma_val = st.slider("MA周期", 20, 120, st.session_state.get("bt_ma", 60), key="bt_ma")

            cbt5, cbt6, cbt7 = st.columns(3)
            with cbt5:
                bt_stop_val = st.slider("止损%", 1, 20, st.session_state.get("bt_stop", 8), key="bt_stop")
            with cbt6:
                bt_vision_val = st.slider("AI胜率阈值", 40, 80, st.session_state.get("bt_vision", 57), key="bt_vision")
            with cbt7:
                bt_validation_val = st.selectbox("验证模式", ["简单回测", "Walk-Forward验证（严格）"], index=0 if st.session_state.get("bt_validation", "简单回测") == "简单回测" else 1, key="bt_validation")

            if bt_validation_val == "Walk-Forward验证（严格）":
                wf1, wf2 = st.columns(2)
                with wf1:
                    wf_train_months_val = st.slider("训练期(月)", 6, 36, st.session_state.get("wf_train_months", 24), key="wf_train_months")
                with wf2:
                    wf_test_months_val = st.slider("测试期(月)", 1, 12, st.session_state.get("wf_test_months", 6), key="wf_test_months")
            else:
                wf_train_months_val, wf_test_months_val = 24, 6

            enable_stress = st.checkbox(
                "启用Stress Testing",
                value=st.session_state.get("enable_stress", False),
                key="enable_stress",
                help="在极端市场条件下测试策略鲁棒性（2008金融危机、2020疫情崩盘、2015股灾）",
            )
            strict_no_future = st.checkbox(
                "严格无未来函数（更慢）",
                value=st.session_state.get("strict_no_future", True),
                key="strict_no_future",
                help="仅使用当前日期及之前的相似形态，避免未来数据泄漏"
            )
            if strict_no_future:
                cbt8, cbt9 = st.columns(2)
                with cbt8:
                    ai_stride_val = st.slider(
                        "AI评估步长(天)",
                        1, 20,
                        st.session_state.get("ai_stride", 5),
                        key="ai_stride",
                        help="步长越大越快，但精度会下降；设为1表示逐日评估"
                    )
                with cbt9:
                    ai_fast_mode_val = st.checkbox(
                        "快速AI评估（向量近似）",
                        value=st.session_state.get("ai_fast_mode", True),
                        key="ai_fast_mode",
                        help="跳过DTW/相关性计算，显著加速但精度略降"
                    )
            else:
                ai_stride_val, ai_fast_mode_val = 1, False

            if st.button("开始回测", key="backtest_btn"):
                run_backtest(
                    symbol, bt_start_val, bt_end_val, bt_cap_val, bt_ma_val,
                    bt_stop_val, bt_vision_val, bt_validation_val,
                    wf_train_months_val, wf_test_months_val, eng, PROJECT_ROOT,
                    enable_stress_test=enable_stress, strict_no_future=strict_no_future,
                    ai_stride=ai_stride_val, ai_fast_mode=ai_fast_mode_val
                )

        with tab_fa:
            st.subheader("📈 因子有效性分析")
            render_factor_analysis(symbol, d["df_f"], eng, PROJECT_ROOT)

        st.divider()
        st.subheader("5. AI 基金经理终审")
        report = st.session_state.ai_reports.get(symbol)
        ic_summary = (st.session_state.get("ic_summary") or {}).get(symbol)
        if report is None:
            if ic_summary is None:
                st.info("尚未检测到 IC 摘要，建议先运行「因子有效性分析」，再生成 AI 终审。")
            if st.button("生成 AI 终审", key="ai_final_btn"):
                with st.spinner("AI 终审生成中..."):
                    report = _safe_ai_analyze(
                        eng["agent"],
                        symbol, d["score"], d["act"],
                        {"win_rate": d.get("win", 50), "score": 0.9},
                        d["df_f"].iloc[-1].to_dict(), d["fund"], d["news"],
                        ic_summary=ic_summary,
                        market_sentiment=d.get("market_sentiment")
                    )
                    st.session_state.ai_reports[symbol] = report
        if report is None:
            st.warning("AI 终审尚未生成。")
        else:
            action_cn_map = {"BUY": "买入", "SELL": "卖出", "WAIT": "观望"}
            action_cn = action_cn_map.get(report.action, report.action)
            color = "green" if report.action == "BUY" else "red" if report.action == "SELL" else "orange"
        st.markdown(f"""
        <div class="agent-box">
                <h2 style="color:{color}; margin:0">{action_cn}</h2>
                <p>信心: {report.confidence}% | 风险: {report.risk_level}</p>
                <hr><p>{report.reasoning}</p>
        </div>
        """, unsafe_allow_html=True)

        pdf_p = os.path.join(PROJECT_ROOT, "data", f"Report_{symbol}.pdf")
        if report and st.button("📄 导出 PDF"):
            generate_report_pdf(f"{d['name']}({symbol})", report, d['c_p'], pdf_p)
            with open(pdf_p, "rb") as f:
                st.download_button("下载 PDF", f, file_name=f"VQ_{symbol}.pdf")

        st.divider()
        st.subheader("💬 智能对话")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): 
                st.markdown(msg["content"])

        c_mic, c_input = st.columns([1, 8])
        user_voice_text = None
        with c_mic:
            st.write(" ")
            audio = None
            if mic_recorder is not None:
                audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format='wav')
            else:
                st.caption("🎙️语音功能未启用")
        if audio:
            transcribed = eng["audio"].transcribe(audio['bytes'])
            if transcribed and transcribed != st.session_state.last_voice_text:
                user_voice_text = transcribed
                st.session_state.last_voice_text = transcribed
        with c_input:
            text_input = st.chat_input("输入问题...")
        final_input = user_voice_text if user_voice_text else text_input

        if final_input:
            st.session_state.chat_history.append({"role": "user", "content": final_input})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            user_q = st.session_state.chat_history[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    resp = eng["agent"].chat(user_q, st.session_state.last_context)
                    st.markdown(resp)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})

elif mode == "📊 批量组合分析":
    if run_btn:
        symbols = [s.strip().zfill(6) for s in batch_input.split('\n') if s.strip()][:30]
        if len(symbols) == 0:
            st.error("❌ 请输入至少一只股票代码")
            st.stop()
        
        st.session_state.has_run = True
        if "batch_results" in st.session_state:
            del st.session_state.batch_results
        if "multi_tier_result" in st.session_state:
            del st.session_state.multi_tier_result
        if "portfolio_metrics" in st.session_state:
            del st.session_state.portfolio_metrics
        
        batch_analyzer = BatchAnalyzer(eng)
        portfolio_optimizer = PortfolioOptimizer()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, symbol):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"正在分析 {symbol} ({current}/{total})...")
        
        batch_results = batch_analyzer.analyze_batch(symbols, progress_callback=update_progress)
        st.session_state.batch_results = batch_results
        progress_bar.progress(1.0)
        status_text.text("✅ 分析完成")
        progress_bar.empty()
        status_text.empty()

        if not batch_results:
            st.error("批量分析失败或无有效数据")
            st.stop()

        # 统一组合优化（即使没有 BUY 也能输出“增强组合”）
        multi_tier_result = portfolio_optimizer.optimize_multi_tier_portfolio(
            batch_results, eng["loader"], min_weight=0.05, max_weight=0.25, max_positions=10
        )
        st.session_state.multi_tier_result = multi_tier_result

        buy_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'BUY' and v.get('score', 0) >= 7}
        wait_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'WAIT'}
        sell_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'SELL'}

        def _goto_symbol(sym: str):
            if "res" in st.session_state:
                del st.session_state.res
            st.session_state.current_symbol = None
            st.session_state.has_run = False
            st.session_state["symbol_input"] = sym
            st.session_state["mode_select"] = "🔍 单只股票分析"
            st.query_params.update({"symbol": sym, "mode": "detail"})
            st.rerun()

        tier_info = multi_tier_result.get("tier_info", {})
        if tier_info:
            st.info(
                f"组合策略: {tier_info.get('strategy', '-')}"
                f" | 优化器: {tier_info.get('optimizer', 'Black-Litterman')}"
                f" | 说明: {tier_info.get('description', '-')}"
            )

        st.subheader("✅ 组合结果（核心 + 备选）")
        core_weights = multi_tier_result.get('core', {})
        enhanced_weights = multi_tier_result.get('enhanced', {})
        combined_weights = {}
        combined_weights.update(core_weights)
        combined_weights.update(enhanced_weights)

        def _render_weights_table(title, weights):
            st.markdown(f"### {title}")
            if not weights:
                st.info("暂无可用组合")
                return
            rows = []
            for sym, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                data = batch_results.get(sym, {})
                rows.append({
                    "股票代码": sym,
                    "股票名称": data.get("name", sym),
                    "权重": f"{w*100:.1f}%",
                    "评分": f"{data.get('score', 0):.1f}/10",
                    "胜率": f"{data.get('win_rate', 0):.1f}%",
                    "预期收益": f"{data.get('expected_return', 0):.2f}%"
                })
            _st_dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            for sym, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                data = batch_results.get(sym, {})
                c1, c2, c3, c4 = st.columns([3, 1, 1, 4])
                with c1:
                    if _st_button(f"📊 {data.get('name', sym)} ({sym})", key=f"link_{title}_{sym}", use_container_width=True):
                        _goto_symbol(sym)
                with c2:
                    st.write(f"**{data.get('score', 0):.1f}/10**")
                with c3:
                    st.write(f"{w*100:.1f}%")
                with c4:
                    st.write(f"{data.get('action', 'WAIT')} - {data.get('reasoning', '')[:60]}")

        _render_weights_table("核心推荐组合", core_weights)
        _render_weights_table("备选增强组合", enhanced_weights)

        st.subheader("📌 仓位设计与风控设置")
        c1, c2, c3 = st.columns(3)
        c1.metric("最小仓位", "5%")
        c2.metric("最大仓位", "25%")
        c3.metric("最大持仓数", "10")
        st.caption("止盈/止损参考：标签止盈 +5%、标签止损 -3%；回测止损默认 -8%（可在单股回测中调整）")

        if combined_weights:
            st.subheader("📊 组合权重图表")
            # 拥挤交易指标
            hhi = sum([w**2 for w in combined_weights.values()])
            top3 = sum(sorted(combined_weights.values(), reverse=True)[:3])
            st.caption(f"拥挤度(HHI): {hhi:.4f} | 前三集中度: {top3*100:.1f}%")
            
            # 组合指标与风险预算
            try:
                metrics = portfolio_optimizer.calculate_portfolio_metrics(combined_weights, batch_results, eng["loader"])
                if metrics:
                    st.subheader("🧾 组合风险指标")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("期望收益", f"{metrics.get('expected_return', 0):.2f}%")
                    m2.metric("风险(波动)", f"{metrics.get('risk', 0):.2f}%")
                    m3.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    m4.metric("CVaR(5%)", f"{metrics.get('cvar', 0):.2f}%")
                    if metrics.get("risk_budget"):
                        with st.expander("风险预算分解", expanded=False):
                            rb = pd.DataFrame(
                                [{"symbol": k, "risk_contrib": v} for k, v in metrics["risk_budget"].items()]
                            )
                            _st_dataframe(rb, use_container_width=True, hide_index=True)
            except Exception:
                pass

            # 再平衡建议（基于上次权重 + 换手上限）
            prev_weights = st.session_state.get("portfolio_weights", {})
            try:
                rebalance_weights, rebalance_info = portfolio_optimizer.propose_rebalance(
                    prev_weights, combined_weights, max_turnover=0.20
                )
                st.session_state.portfolio_weights = combined_weights
                with st.expander("🔁 再平衡建议（换手≤20%）", expanded=False):
                    st.write(f"预计换手: {rebalance_info.get('turnover', 0)*100:.1f}%")
                    r_df = pd.DataFrame([
                        {"symbol": s, "current": round(prev_weights.get(s, 0)*100, 1), "target": round(rebalance_weights.get(s, 0)*100, 1)}
                        for s in set(prev_weights) | set(rebalance_weights)
                    ])
                    _st_dataframe(r_df, use_container_width=True, hide_index=True)
            except Exception:
                pass
            labels = [f"{batch_results[s].get('name', s)}({s})" for s in combined_weights.keys()]
            values = [combined_weights[s] for s in combined_weights.keys()]
            pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
            pie.update_layout(height=320, title="组合权重分布")
            _st_plotly_chart(pie, use_container_width=True)

            bar = go.Figure()
            bar.add_trace(go.Bar(x=labels, y=[batch_results[s].get('score', 0) for s in combined_weights.keys()],
                                 name="评分", marker_color="#ff4b4b"))
            bar.update_layout(height=300, title="评分对比")
            _st_plotly_chart(bar, use_container_width=True)

            scatter = go.Figure()
            for s in combined_weights.keys():
                scatter.add_trace(go.Scatter(
                    x=[batch_results[s].get('win_rate', 0)],
                    y=[batch_results[s].get('expected_return', 0)],
                    mode='markers+text',
                    text=[s],
                    marker=dict(size=max(8, combined_weights[s]*200), color="#1f77b4"),
                    name=s
                ))
            scatter.update_layout(height=320, title="胜率 vs 预期收益", xaxis_title="胜率(%)", yaxis_title="预期收益(%)")
            _st_plotly_chart(scatter, use_container_width=True)

            st.subheader("🕯️ 组合Top3 K线展示")
            top_syms = [s for s, _ in sorted(combined_weights.items(), key=lambda x: x[1], reverse=True)[:3]]
            if top_syms:
                cols = st.columns(len(top_syms))
                for i, sym in enumerate(top_syms):
                    try:
                        dfk = eng["loader"].get_stock_data(sym)
                        if dfk is None or dfk.empty:
                            continue
                        tmp_img = os.path.join(PROJECT_ROOT, "data", f"temp_batch_k_{sym}.png")
                        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                        sstyle = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
                        mpf.plot(dfk.tail(60), type='candle', style=sstyle,
                                 savefig=dict(fname=tmp_img, dpi=80), figsize=(4, 3), axisoff=True)
                        with cols[i]:
                            _st_image(tmp_img, caption=f"{sym}", use_container_width=True)
                        if os.path.exists(tmp_img):
                            os.remove(tmp_img)
                    except Exception:
                        continue

            # 分层回测（行业/市值/风格 + 显著性）
            with st.expander("🧪 分层回测（行业/市值/风格）", expanded=False):
                if st.button("运行分层回测", key="strat_bt_btn"):
                    strat_df = run_stratified_backtest_batch(list(batch_results.keys()), eng)
                    if strat_df is not None and not strat_df.empty:
                        _st_dataframe(strat_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("分层样本不足或数据不可用")

            # 权重动态变化（简化：基于20日动量的月度再平衡）
            try:
                st.subheader("📈 组合权重动态变化")
                top_syms = list(combined_weights.keys())[:6]
                weight_df = pd.DataFrame()
                for sym in top_syms:
                    dfw = eng["loader"].get_stock_data(sym)
                    if dfw is None or dfw.empty:
                        continue
                    dfw.index = pd.to_datetime(dfw.index)
                    dfw = dfw.tail(180)
                    mom = dfw["Close"].pct_change(20)
                    dfw = dfw.assign(mom=mom)
                    dfw = dfw.resample("M").last().dropna()
                    weight_df[sym] = dfw["mom"]
                if not weight_df.empty:
                    # 归一化为权重
                    weight_df = weight_df.apply(lambda x: x - x.min() + 1e-6)
                    weight_df = weight_df.div(weight_df.sum(axis=1), axis=0)
                    fig_w = go.Figure()
                    for sym in weight_df.columns:
                        fig_w.add_trace(go.Scatter(x=weight_df.index, y=weight_df[sym], mode="lines", name=sym))
                    fig_w.update_layout(height=320, title="月度权重演化（动量驱动）")
                    _st_plotly_chart(fig_w, use_container_width=True)
            except Exception:
                pass

            # 滚动收益热图
            try:
                st.subheader("🧊 滚动收益热图（20日）")
                heat_syms = list(combined_weights.keys())[:8]
                heat_data = []
                heat_index = None
                for sym in heat_syms:
                    dfh = eng["loader"].get_stock_data(sym)
                    if dfh is None or dfh.empty:
                        continue
                    dfh.index = pd.to_datetime(dfh.index)
                    dfh = dfh.tail(200)
                    roll = dfh["Close"].pct_change(20) * 100
                    if heat_index is None:
                        heat_index = roll.index
                    heat_data.append(roll.reindex(heat_index).fillna(0).values)
                if heat_data:
                    heat = go.Figure(data=go.Heatmap(
                        z=np.array(heat_data),
                        x=[d.strftime("%Y-%m-%d") for d in heat_index],
                        y=heat_syms,
                        colorscale="RdYlGn"
                    ))
                    heat.update_layout(height=320)
                    _st_plotly_chart(heat, use_container_width=True)
            except Exception:
                pass
        
        if wait_stocks or sell_stocks:
            st.divider()
            st.subheader("⚠️ 观望/卖出列表")
            all_other = {**wait_stocks, **sell_stocks}
            if all_other:
                for symbol, data in sorted(all_other.items(), key=lambda x: x[1].get('score', 0)):
                    col1, col2, col3 = st.columns([3, 1, 4])
                    with col1:
                        if _st_button(f"📊 {data.get('name', symbol)} ({symbol})",
                                   key=f"link_other_{symbol}", use_container_width=True):
                            _goto_symbol(symbol)
                    with col2:
                        st.write(f"**{data.get('score', 0):.1f}/10**")
                    with col3:
                        st.write(f"{data.get('action', 'WAIT')} - {data.get('reasoning', '')[:50]}")
                    st.divider()
    
    else:
        st.info("👈 请在左侧输入股票代码并点击启动")
