import akshare as ak
import pandas as pd
import numpy as np
import time
import os
import json
import difflib
import threading
from src.utils.net_utils import no_proxy_env

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _call_with_timeout(func, args=(), kwargs=None, timeout: int = 30):
    """
    带超时的函数调用（兼容 Windows）
    """
    if kwargs is None:
        kwargs = {}
    result = [None]
    exception = [None]

    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return None, TimeoutError(f"操作超时 ({timeout}s)")
    if exception[0] is not None:
        return None, exception[0]
    return result[0], None


class TimeoutError(Exception):
    pass


class FundamentalMiner:
    def __init__(self, spot_cache_ttl_sec: int = 300, spot_retry: int = 2):
        # 缓存全市场 spot（ak.stock_zh_a_spot_em 很重，且易波动；缓存能显著降低 N/A）
        self._spot_cache_df = None
        self._spot_cache_ts = 0.0
        self._spot_cache_ttl_sec = spot_cache_ttl_sec
        self._spot_retry = spot_retry
        self._industry_cache = {}
        self._peers_cache = {}
        self._industry_map = {}
        self._industry_cons_cache = {}
        self._industry_map_ts = 0.0
        self._industry_map_ttl_sec = 24 * 3600
        self._industry_board_names = []
        self._industry_board_ts = 0.0
        self._industry_board_ttl_sec = 24 * 3600
        self._industry_name_cache = {}
        self._industry_cache_ts = 0.0
        self._industry_cache_ttl_sec = 7 * 24 * 3600
        self._industry_cache_path = os.path.join(PROJECT_ROOT, "data", "industry_cache.json")
        self._spot_cache_path = os.path.join(PROJECT_ROOT, "data", "spot_cache.csv")
        self._load_industry_cache()
        self._load_spot_cache()

    def _load_industry_cache(self):
        try:
            if os.path.exists(self._industry_cache_path):
                with open(self._industry_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cache_map = data.get("map", {})
                ts = data.get("ts", 0.0)
                if isinstance(cache_map, dict):
                    self._industry_cache.update(cache_map)
                self._industry_cache_ts = float(ts) if ts else 0.0
        except Exception:
            pass

    def _save_industry_cache(self):
        try:
            os.makedirs(os.path.dirname(self._industry_cache_path), exist_ok=True)
            payload = {"ts": time.time(), "map": self._industry_cache}
            with open(self._industry_cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass

    def _load_spot_cache(self):
        try:
            if os.path.exists(self._spot_cache_path):
                df = pd.read_csv(self._spot_cache_path)
                if df is not None and not df.empty:
                    # 标准化代码列
                    code_col = next((c for c in df.columns if '代码' in c or 'code' in str(c).lower()), None)
                    if code_col:
                        df[code_col] = df[code_col].astype(str).str.zfill(6)
                    self._spot_cache_df = df
                    self._spot_cache_ts = time.time()
        except Exception:
            pass

    def _save_spot_cache(self, df: pd.DataFrame):
        try:
            if df is None or df.empty:
                return
            os.makedirs(os.path.dirname(self._spot_cache_path), exist_ok=True)
            df.to_csv(self._spot_cache_path, index=False)
        except Exception:
            pass

    def get_stock_fundamentals(self, symbol, force_live: bool = False):
        """
        获取深度财务指标 (含成长性与安全性分析)
        """
        symbol = str(symbol).strip().zfill(6)
        print(f"[财务分析] 正在透视 {symbol}...")

        # 默认结果结构扩展
        result = {
            "symbol": symbol,
            # 默认不要用 symbol 当 name，否则 UI 会出现 “300286(300286)” 这种重复且掩盖抓取失败
            "name": "",
            "industry": "",
            "pe_ttm": 0.0, "pb": 0.0, "total_mv": 0.0,
            "net_profit": 0.0,
            "roe": 0.0, "net_profit_margin": 0.0, "asset_turnover": 0.6, "leverage": 1.0,
            "debt_asset_ratio": 0.0,
            # === 新增指标 ===
            "gross_margin": 0.0,  # 毛利率
            "current_ratio": 0.0,  # 流动比率 (偿债能力)
            "rev_growth": 0.0,  # 营收增长率
            "profit_growth": 0.0,  # 净利增长率
            "report_date": "最新"
            ,
            # === 状态字段：用于UI层判断“是否成功抓取”，避免把0当真 ===
            "_ok": {"spot": False, "finance": False},
            "_err": []
        }

        def _fill_from_spot(spot_df: pd.DataFrame) -> bool:
            try:
                if spot_df is None or spot_df.empty:
                    return False
                code_col = next((c for c in spot_df.columns if '代码' in c or 'code' in str(c).lower()), None)
                if not code_col:
                    return False
                spot_df = spot_df.copy()
                spot_df[code_col] = spot_df[code_col].astype(str).str.zfill(6)
                target = spot_df[spot_df[code_col] == symbol]
                if target.empty:
                    return False

                def _pick_col(cols, keywords, prefer=None):
                    prefer = prefer or []
                    prefer_groups = []
                    for p in prefer:
                        if isinstance(p, (list, tuple, set)):
                            prefer_groups.append([str(x) for x in p])
                        else:
                            prefer_groups.append([str(p)])
                    for c in cols:
                        for group in prefer_groups:
                            if group and all(k in str(c) for k in group):
                                return c
                    for c in cols:
                        if any(k in str(c) for k in keywords):
                            return c
                    return None

                cols = list(target.columns)
                pe_col = _pick_col(
                    cols,
                    ["市盈率", "PE", "pe", "TTM", "滚动", "动态"],
                    prefer=[["市盈率", "TTM"], ["市盈率", "动"], ["PE", "TTM"]]
                )
                pb_col = _pick_col(cols, ["市净率", "PB", "pb", "市帐率", "P/B"])
                mv_col = _pick_col(cols, ["总市值", "市值", "market", "MV"])
                name_col = next((c for c in target.columns if '名称' in c), None)
                ind_col = next((c for c in target.columns if '行业' in c), None)

                updated = False
                if pe_col:
                    pe_val = self._to_f(target[pe_col].values[0])
                    if self._is_valid_num(pe_val):
                        result["pe_ttm"] = pe_val
                        updated = True
                if pb_col:
                    pb_val = self._to_f(target[pb_col].values[0])
                    if self._is_valid_num(pb_val):
                        result["pb"] = pb_val
                        updated = True
                if mv_col:
                    mv_val = self._to_f(target[mv_col].values[0])
                    if self._is_valid_num(mv_val):
                        result["total_mv"] = round(mv_val / 100000000, 2)
                        updated = True
                if name_col and not result.get("name"):
                    result["name"] = str(target[name_col].values[0]).strip()
                if ind_col and not result.get("industry"):
                    ind_val = str(target[ind_col].values[0]).strip()
                    if ind_val and ind_val not in ["未知", "-"]:
                        result["industry"] = ind_val

                if updated:
                    result["_ok"]["spot"] = True
                return updated
            except Exception as e:
                result["_err"].append(f"spot_df_parse_error: {type(e).__name__}: {e}")
                return False

        try:
            # 1. 实时估值
            try:
                spot_df = self._get_spot_df_live(result) if force_live else self._get_spot_df_cached(result)
                ok = _fill_from_spot(spot_df)
                # 若未拿到有效估值，强制刷新一次
                if not ok and (result.get("pe_ttm", 0) == 0 and result.get("pb", 0) == 0):
                    spot_df = self._get_spot_df_live(result) if force_live else self._get_spot_df_cached(result, force_refresh=True)
                    _fill_from_spot(spot_df)
            except Exception as e:
                result["_err"].append(f"spot_df_error: {type(e).__name__}: {e}")

            # 若 spot 未拿到 name 或 PE/PB，尝试更轻量的个股信息接口兜底（带重试）
            if not result.get("name") or (result.get("pe_ttm", 0) == 0 and result.get("pb", 0) == 0):
                for attempt in range(max(1, self._spot_retry + 1)):
                    try:
                        with no_proxy_env():
                            info_df, err = _call_with_timeout(
                                ak.stock_individual_info_em,
                                kwargs={"symbol": symbol},
                                timeout=15
                            )
                        if err:
                            raise err
                        if info_df is not None and not info_df.empty:
                            # 常见字段：item/value
                            if "item" in info_df.columns and "value" in info_df.columns:
                                # 获取股票名称
                                if not result.get("name"):
                                    name_row = info_df[info_df["item"].astype(str).str.contains("股票简称|名称")]
                                    if not name_row.empty:
                                        result["name"] = str(name_row["value"].values[0]).strip()

                                # 获取行业信息（用于后续行业对标，减少重复请求）
                                try:
                                    ind_row = info_df[info_df["item"].astype(str).str.contains("行业|所属行业")]
                                    if not ind_row.empty:
                                        industry_val = str(ind_row["value"].values[0]).strip()
                                        if industry_val and industry_val not in ["未知", ""]:
                                            result["industry"] = industry_val
                                            self._industry_cache[symbol] = industry_val
                                            if not force_live:
                                                self._save_industry_cache()
                                except Exception:
                                    pass
                                
                                # 尝试获取PE/PB（如果spot_df没有获取到）
                                if result.get("pe_ttm", 0) == 0:
                                    pe_row = info_df[info_df["item"].astype(str).str.contains(
                                        "市盈率|PE|滚动市盈率|TTM|动态市盈率", case=False, regex=True
                                    )]
                                    if not pe_row.empty:
                                        result["pe_ttm"] = self._to_f(pe_row["value"].values[0])
                                
                                if result.get("pb", 0) == 0:
                                    pb_row = info_df[info_df["item"].astype(str).str.contains(
                                        "市净率|PB|P/B", case=False, regex=True
                                    )]
                                    if not pb_row.empty:
                                        result["pb"] = self._to_f(pb_row["value"].values[0])

                                if result.get("total_mv", 0) == 0:
                                    mv_row = info_df[info_df["item"].astype(str).str.contains(
                                        "总市值|市值", case=False, regex=True
                                    )]
                                    if not mv_row.empty:
                                        result["total_mv"] = round(self._to_f(mv_row["value"].values[0]) / 100000000, 2)
                                
                                # 如果获取到关键数据，退出重试循环
                                if result.get("name") or self._is_valid_num(result.get("pe_ttm", 0)):
                                    break
                        if attempt < self._spot_retry:
                            time.sleep(0.5 * (attempt + 1))  # 指数退避
                    except Exception as e:
                        if attempt < self._spot_retry:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        result["_err"].append(f"stock_individual_info_error: {type(e).__name__}: {e}")

            # 1.5 估值指标兜底：从指标接口补齐 PE/PB/市值
            if (result.get("pe_ttm", 0) == 0 or result.get("pb", 0) == 0 or result.get("total_mv", 0) == 0):
                try:
                    extra = self._get_indicator_snapshot(symbol, max_retries=max(1, self._spot_retry))
                    if extra and extra.get("ok"):
                        updated = False
                        if result.get("pe_ttm", 0) == 0 and extra.get("pe_ttm") is not None and self._is_valid_num(extra.get("pe_ttm")):
                            result["pe_ttm"] = extra.get("pe_ttm", 0.0)
                            updated = True
                        if result.get("pb", 0) == 0 and extra.get("pb") is not None and self._is_valid_num(extra.get("pb")):
                            result["pb"] = extra.get("pb", 0.0)
                            updated = True
                        if result.get("total_mv", 0) == 0 and extra.get("total_mv") is not None and self._is_valid_num(extra.get("total_mv")):
                            result["total_mv"] = extra.get("total_mv", 0.0)
                            updated = True
                        if updated:
                            result["_ok"]["spot"] = True
                except Exception as e:
                    result["_err"].append(f"indicator_fallback_error: {type(e).__name__}: {e}")

            # 2. 深度指标：优先使用 THS 财务摘要（经验证可用；EM接口在你环境里全量报错）
            # 工业级优化：添加重试机制
            ths_success = False
            for attempt in range(max(1, self._spot_retry + 1)):
                try:
                    with no_proxy_env():
                        ths_df = ak.stock_financial_abstract_ths(symbol=symbol)
                    if ths_df is not None and not ths_df.empty:
                        # 取最新报告期
                        if "报告期" in ths_df.columns:
                            tmp = ths_df.copy()
                            tmp["报告期_dt"] = pd.to_datetime(tmp["报告期"], errors="coerce")
                            tmp = tmp.sort_values("报告期_dt")
                            latest = tmp.iloc[-1]
                            result["report_date"] = str(latest.get("报告期", result["report_date"]))
                        else:
                            latest = ths_df.iloc[-1]

                        # 关键指标（字段名稳定）
                        result["roe"] = self._to_f(latest.get("净资产收益率"))
                        result["net_profit_margin"] = self._to_f(latest.get("销售净利率"))
                        result["gross_margin"] = self._to_f(latest.get("销售毛利率"))
                        result["current_ratio"] = self._to_f(latest.get("流动比率"))
                        result["debt_asset_ratio"] = self._to_f(latest.get("资产负债率"))
                        # 这些字段有时为 False/空，_to_f 会安全兜底为0.0
                        result["rev_growth"] = self._to_f(latest.get("营业总收入同比增长率"))
                        result["profit_growth"] = self._to_f(latest.get("净利润同比增长率"))

                        if 0 < result["debt_asset_ratio"] < 100:
                            result["leverage"] = round(1 / (1 - result["debt_asset_ratio"] / 100), 2)

                        # 净利润（用于估算PE，若缺失）
                        np_col = next((c for c in ths_df.columns if "归母净利润" in str(c) or "净利润" in str(c)), None)
                        if np_col:
                            np_val = self._to_f(latest.get(np_col))
                            if self._is_valid_num(np_val):
                                result["net_profit"] = round(np_val / 100000000, 2) if np_val > 1e6 else round(np_val, 2)

                        # THS 数据中若包含估值字段，顺便补齐 PE/PB
                        if result.get("pe_ttm", 0) == 0:
                            pe_col = next((c for c in ths_df.columns if "市盈率" in str(c) or "PE" in str(c).upper()), None)
                            if pe_col:
                                pe_val = self._to_f(latest.get(pe_col))
                                if self._is_valid_num(pe_val):
                                    result["pe_ttm"] = pe_val
                                    result["_ok"]["spot"] = True
                        if result.get("pb", 0) == 0:
                            pb_col = next((c for c in ths_df.columns if "市净率" in str(c) or "PB" in str(c).upper()), None)
                            if pb_col:
                                pb_val = self._to_f(latest.get(pb_col))
                                if self._is_valid_num(pb_val):
                                    result["pb"] = pb_val
                                    result["_ok"]["spot"] = True

                        result["_ok"]["finance"] = True
                        ths_success = True
                        break  # 成功获取，退出重试循环
                    else:
                        if attempt < self._spot_retry:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        result["_err"].append("ths_finance_empty")
                except Exception as e:
                    if attempt < self._spot_retry:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    result["_err"].append(f"ths_finance_error (尝试{attempt+1}/{self._spot_retry+1}): {type(e).__name__}: {e}")
            
            # 降级策略：如果THS接口失败，尝试其他财务接口
            if not ths_success:
                # 尝试使用其他财务接口作为降级
                try:
                    # 可以尝试其他接口，但这里先保持原有逻辑
                    pass
                except Exception as e:
                    result["_err"].append(f"finance_fallback_error: {type(e).__name__}: {e}")

            # 3. 若仍拿不到 ROE，则用 PB/PE 推算（标注为推算，不再默默写0）
            if not result["_ok"]["finance"] and result["pe_ttm"] > 0:
                result["roe"] = round((result["pb"] / result["pe_ttm"]) * 100, 2)
                # 只作为兜底推算，不写入 _ok.finance
                result["_err"].append("roe_estimated_by_pb_pe")

            # 3.5 若 PE 仍缺失但已有 PB 与 ROE，反推 PE（兜底）
            if result.get("pe_ttm", 0) == 0 and result.get("pb", 0) > 0 and result.get("roe", 0) > 0:
                try:
                    result["pe_ttm"] = round(result["pb"] / (result["roe"] / 100.0), 2)
                    result["_ok"]["spot"] = True
                    result["_err"].append("pe_estimated_by_pb_roe")
                except Exception:
                    pass

            # 3.6 若 PE 仍缺失但已有市值与净利润，反推 PE（兜底）
            if result.get("pe_ttm", 0) == 0 and result.get("total_mv", 0) > 0 and result.get("net_profit", 0) > 0:
                try:
                    result["pe_ttm"] = round(result["total_mv"] / (result["net_profit"] + 1e-8), 2)
                    result["_ok"]["spot"] = True
                    result["_err"].append("pe_estimated_by_mv_np")
                except Exception:
                    pass

            # 3.7 统一总市值单位（亿元）
            if self._is_valid_num(result.get("total_mv", 0), allow_zero=True):
                try:
                    mv_val = float(result.get("total_mv", 0))
                    if mv_val > 1e6:
                        result["total_mv"] = round(mv_val / 100000000, 2)
                except Exception:
                    pass

            # 4. 最终兜底：若估值字段已取得，补标 spot OK（只有真正获取到有效值才标记）
            if not result["_ok"]["spot"]:
                if (
                    self._is_valid_num(result.get("pe_ttm", 0))
                    or self._is_valid_num(result.get("pb", 0))
                    or self._is_valid_num(result.get("total_mv", 0))
                ):
                    result["_ok"]["spot"] = True
                else:
                    # 如果所有估值字段都是0，明确标记为失败
                    if not result["_err"] or "spot_retry_failed" not in str(result["_err"]):
                        result["_err"].append("估值数据获取失败：PE/PB/市值均为0或不可用")

            # 补齐行业信息（来自缓存/spot）
            if not result.get("industry") and symbol in self._industry_cache:
                result["industry"] = self._industry_cache.get(symbol)
            # 再兜底一次：从行业板块成分反查（避免返回空行业）
            if not result.get("industry"):
                inferred_ind = self._lookup_industry_from_boards(symbol, max_retries=max(1, self._spot_retry), max_scan=300)
                if inferred_ind:
                    result["industry"] = inferred_ind
            if result.get("industry"):
                self._industry_cache[symbol] = result.get("industry")
                if not force_live:
                    self._save_industry_cache()

        except Exception as e:
            result["_err"].append(f"spot_error: {type(e).__name__}: {e}")
            print(f"[警告] 财报异常: {e}")

        return result

    def _get_spot_df_cached(self, result: dict, force_refresh: bool = False):
        """
        获取全市场 spot 数据（带缓存 + 重试 + 超时控制）。
        """
        now = time.time()
        if not force_refresh and self._spot_cache_df is not None and (now - self._spot_cache_ts) < self._spot_cache_ttl_sec:
            return self._spot_cache_df

        last_err = None
        max_retries = max(1, self._spot_retry + 1)
        for i in range(max_retries):
            try:
                with no_proxy_env():
                    df, err = _call_with_timeout(
                        ak.stock_zh_a_spot_em,
                        timeout=30
                    )
                if err:
                    raise err
                if df is None or df.empty:
                    raise RuntimeError("spot_df_empty")
                code_col = next((c for c in df.columns if '代码' in c), None)
                if code_col:
                    df[code_col] = df[code_col].astype(str).str.zfill(6)
                self._spot_cache_df = df
                self._spot_cache_ts = now
                self._save_spot_cache(df)
                return df
            except Exception as e:
                last_err = e
                if i < max_retries - 1:
                    time.sleep(0.5 * (i + 1))

        if self._spot_cache_df is None and os.path.exists(self._spot_cache_path):
            try:
                df = pd.read_csv(self._spot_cache_path)
                if df is not None and not df.empty:
                    self._spot_cache_df = df
                    self._spot_cache_ts = now
                    return df
            except Exception:
                pass

        # 若强制刷新失败但已有内存缓存，直接回退内存缓存，避免PE/PB/行业同时缺失
        if self._spot_cache_df is not None and not self._spot_cache_df.empty:
            if last_err is not None:
                result["_err"].append(f"spot_refresh_failed_use_cache: {type(last_err).__name__}: {last_err}")
            self._spot_cache_ts = now
            return self._spot_cache_df

        if last_err is not None:
            result["_err"].append(f"spot_retry_failed (尝试{max_retries}次): {type(last_err).__name__}: {last_err}")
        return None

    def _get_spot_df_live(self, result: dict):
        """
        强制实时拉取spot，不读写缓存，带超时控制。
        """
        last_err = None
        max_retries = max(1, self._spot_retry + 1)
        for i in range(max_retries):
            try:
                with no_proxy_env():
                    df, err = _call_with_timeout(
                        ak.stock_zh_a_spot_em,
                        timeout=30
                    )
                if err:
                    raise err
                if df is None or df.empty:
                    raise RuntimeError("spot_df_empty")
                code_col = next((c for c in df.columns if '代码' in c), None)
                if code_col:
                    df[code_col] = df[code_col].astype(str).str.zfill(6)
                return df
            except Exception as e:
                last_err = e
                if i < max_retries - 1:
                    time.sleep(0.5 * (i + 1))
        try:
            if os.path.exists(self._spot_cache_path):
                df = pd.read_csv(self._spot_cache_path)
                if df is not None and not df.empty:
                    code_col = next((c for c in df.columns if '代码' in c), None)
                    if code_col:
                        df[code_col] = df[code_col].astype(str).str.zfill(6)
                    if last_err is not None:
                        result["_err"].append(f"spot_live_failed_use_cache: {type(last_err).__name__}: {last_err}")
                    return df
        except Exception:
            pass
        if last_err is not None:
            result["_err"].append(f"spot_live_failed (尝试{max_retries}次): {type(last_err).__name__}: {last_err}")
        return None

    def _get_indicator_snapshot(self, symbol: str, max_retries: int = 2):
        """
        估值指标兜底：尝试从 A 股指标接口提取 PE/PB/市值
        """
        symbol = str(symbol).strip().zfill(6)

        def _symbol_variants(sym: str):
            s = str(sym).strip().zfill(6)
            variants = [s]
            exchange = "SH" if s.startswith(("6", "9")) else "SZ"
            variants.append(f"{s}.{exchange}")
            variants.append(f"{exchange}{s}")
            variants.append(f"{exchange.lower()}{s}")
            return list(dict.fromkeys(variants))

        def _pick_col(df, keywords):
            cols = list(df.columns)
            for c in cols:
                c_low = str(c).lower()
                for kw in keywords:
                    kw_low = kw.lower()
                    if kw_low in c_low or kw in str(c):
                        return c
            return None

        # 扩展接口列表，按优先级尝试
        fetchers = []
        # 优先级1：最常用的接口
        if hasattr(ak, "stock_a_indicator_lg"):
            fetchers.append(("stock_a_indicator_lg", lambda s: ak.stock_a_indicator_lg(symbol=s), True))
        if hasattr(ak, "stock_zh_a_indicator"):
            fetchers.append(("stock_zh_a_indicator", lambda s: ak.stock_zh_a_indicator(symbol=s), True))
        if hasattr(ak, "stock_a_indicator"):
            fetchers.append(("stock_a_indicator", lambda s: ak.stock_a_indicator(symbol=s), True))
        # 优先级2：实时行情接口（作为最后兜底）
        if hasattr(ak, "stock_zh_a_spot_em"):
            fetchers.append(("stock_zh_a_spot_em", lambda s: ak.stock_zh_a_spot_em().query(f"代码 == '{symbol}'"), False))

        symbol_variants = _symbol_variants(symbol)

        for name, fetch, use_variant in fetchers:
            symbols_to_try = symbol_variants if use_variant else [symbol]
            for sym in symbols_to_try:
                for attempt in range(max_retries):
                    try:
                        with no_proxy_env():
                            df = fetch(sym)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        continue
                    if df is None or df.empty:
                        continue
                    # 如果是DataFrame，取第一行或最后一行
                    if isinstance(df, pd.DataFrame):
                        if len(df) > 0:
                            latest = df.iloc[-1] if "trade_date" not in df.columns else df.sort_values("trade_date").iloc[-1]
                        else:
                            continue
                    else:
                        continue

                    pe_col = _pick_col(df, ["pe_ttm", "市盈率", "PE", "TTM", "市盈率TTM"])
                    pb_col = _pick_col(df, ["pb", "市净率", "PB"])
                    mv_col = _pick_col(df, ["total_mv", "总市值", "市值", "流通市值", "market_cap"])

                    pe_ttm = self._to_f(latest.get(pe_col)) if pe_col and pe_col in latest.index else None
                    pb = self._to_f(latest.get(pb_col)) if pb_col and pb_col in latest.index else None
                    total_mv = self._to_f(latest.get(mv_col)) if mv_col and mv_col in latest.index else None
                    if self._is_valid_num(total_mv, allow_zero=True):
                        try:
                            if float(total_mv) > 1e6:
                                total_mv = float(total_mv) / 100000000
                        except Exception:
                            pass

                    # 只有真正获取到有效值才返回
                    if self._is_valid_num(pe_ttm):
                        return {
                            "pe_ttm": pe_ttm,
                            "pb": pb if self._is_valid_num(pb) else None,
                            "total_mv": total_mv if self._is_valid_num(total_mv) else None,
                            "ok": True
                        }
                    if self._is_valid_num(pb):
                        return {
                            "pe_ttm": None,
                            "pb": pb,
                            "total_mv": total_mv if self._is_valid_num(total_mv) else None,
                            "ok": True
                        }
                    if self._is_valid_num(total_mv):
                        return {"pe_ttm": None, "pb": None, "total_mv": total_mv, "ok": True}
        return None

    def _get_industry_board_names(self, max_retries: int = 2):
        now = time.time()
        if self._industry_board_names and (now - self._industry_board_ts) < self._industry_board_ttl_sec:
            return self._industry_board_names
        sources = []
        if hasattr(ak, "stock_board_industry_name_em"):
            sources.append(ak.stock_board_industry_name_em)
        if hasattr(ak, "stock_board_industry_name_ths"):
            sources.append(ak.stock_board_industry_name_ths)
        if not sources:
            return []
        for fetch in sources:
            for attempt in range(max_retries):
                try:
                    with no_proxy_env():
                        df = fetch()
                    if df is None or df.empty:
                        continue
                    name_col = next((c for c in df.columns if "板块" in c or "行业" in c or "名称" in c), None)
                    if not name_col:
                        name_col = df.columns[0]
                    names = [str(x).strip() for x in df[name_col].dropna().tolist() if str(x).strip()]
                    if names:
                        self._industry_board_names = names
                        self._industry_board_ts = now
                        return names
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(0.3 * (attempt + 1))
                        continue
        return []

    def _normalize_industry_name(self, name: str):
        if not name:
            return ""
        n = str(name).strip()
        # 去掉括号内容（如 “计算机设备(申万)”）
        for sep in ["(", "（"]:
            if sep in n:
                n = n.split(sep)[0].strip()
        # 去掉常见后缀
        for suffix in ["行业", "板块", "概念", "指数", "类"]:
            if n.endswith(suffix):
                n = n[: -len(suffix)].strip()
        # 去掉空白
        n = n.replace(" ", "")
        return n

    def _match_industry_board_name(self, industry: str):
        if not industry:
            return None
        industry = self._normalize_industry_name(industry)
        names = self._get_industry_board_names()
        if not names:
            return None
        # 优先完全匹配
        for name in names:
            if industry == self._normalize_industry_name(name):
                return name
        # 其次包含匹配
        for name in names:
            norm = self._normalize_industry_name(name)
            if industry in norm or norm in industry:
                return name
        return None

    def _lookup_industry_from_boards(self, symbol: str, max_retries: int = 2, max_scan: int = 200):
        symbol = str(symbol).strip().zfill(6)
        now = time.time()
        if symbol in self._industry_map and (now - self._industry_map_ts) < self._industry_map_ttl_sec:
            return self._industry_map[symbol]
        if not hasattr(ak, "stock_board_industry_cons_em"):
            return None

        names = self._get_industry_board_names()
        if not names:
            return None

        scanned = 0
        for name in names:
            scanned += 1
            if max_scan and scanned > max_scan:
                break
            codes = self._industry_cons_cache.get(name)
            if codes is None:
                cons_df = None
                for attempt in range(max_retries):
                    try:
                        with no_proxy_env():
                            cons_df = ak.stock_board_industry_cons_em(symbol=name)
                        if cons_df is not None and not cons_df.empty:
                            break
                    except Exception:
                        if attempt < max_retries - 1:
                            time.sleep(0.3 * (attempt + 1))
                            continue
                if cons_df is None or cons_df.empty:
                    continue
                code_col = next((c for c in cons_df.columns if "代码" in c or "证券代码" in c or "stock_code" in c), None)
                if not code_col:
                    continue
                codes = set(cons_df[code_col].astype(str).str.zfill(6).tolist())
                self._industry_cons_cache[name] = codes
            if symbol in codes:
                self._industry_map[symbol] = name
                self._industry_map_ts = now
                return name
        return None

    # ... (get_industry_peers, _find_val, _to_f 保持不变，直接复用原有的即可) ...
    # 为了完整性，这里简写保留辅助函数结构
    def get_industry_peers(self, symbol, max_retries=3, force_live: bool = False):
        """
        工业级优化：获取行业和同行对比数据
        添加重试机制和降级策略
        """
        symbol = str(symbol).strip().zfill(6)
        if (not force_live) and symbol in self._peers_cache:
            cached_ind, cached_peers = self._peers_cache[symbol]
            if cached_peers is not None and len(cached_peers) >= 2 and cached_ind not in ["未知", "上海主板", "深圳主板", "创业板", "科创板", None]:
                return self._peers_cache[symbol]

        industry = None if force_live else self._industry_cache.get(symbol)
        # 1) 个股信息接口（可能不稳定）- 添加重试机制
        if not industry:
            for attempt in range(max_retries):
                try:
                    with no_proxy_env():
                        info_df = ak.stock_individual_info_em(symbol=symbol)
                    if info_df is not None and not info_df.empty and "item" in info_df.columns:
                        row = info_df[info_df["item"].astype(str).str.contains("行业|所属行业")]
                        if not row.empty:
                            industry = str(row["value"].values[0]).strip()
                            break
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # 指数退避
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    print(f"[警告] 获取个股信息失败 ({symbol}): {e}")
                    industry = None

        # 2) 使用缓存的全市场spot兜底
        dummy = {"_err": []}
        spot_df = self._get_spot_df_live(dummy) if force_live else self._get_spot_df_cached(dummy)
        if not industry and spot_df is not None and not spot_df.empty:
            code_col = next((c for c in spot_df.columns if '代码' in c), None)
            ind_col = next((c for c in spot_df.columns if '行业' in c), None)
            if code_col and ind_col:
                row = spot_df[spot_df[code_col].astype(str).str.zfill(6) == symbol]
                if not row.empty:
                    industry = str(row[ind_col].values[0]).strip()

        # 2.5) 行业兜底：从行业板块成分反查
        if not industry or industry == "未知":
            industry = self._lookup_industry_from_boards(symbol, max_retries=max_retries)

        # 3) 最后兜底：若无法确认真实行业，返回“未知”
        # 不再用交易板块前缀冒充行业，避免产生错误同行（如返回银行股）
        if not industry:
            industry = "未知"

        # 保存行业缓存（落盘）
        if industry and industry not in ["未知", "上海主板", "深圳主板", "创业板", "科创板"] and not force_live:
            self._industry_cache[symbol] = industry
            self._save_industry_cache()

        # 4) 构建同行对比
        # 工业级优化：如果spot_df为空，尝试重新获取（带重试）
        try:
            if spot_df is None or spot_df.empty:
                # 尝试重新获取全市场数据（带重试）
                dummy = {"_err": []}
                full_market = self._get_spot_df_live(dummy) if force_live else self._get_spot_df_cached(dummy)
                if full_market is None or full_market.empty:
                    # 最后尝试直接调用（不带缓存）
                    for attempt in range(max_retries):
                        try:
                            with no_proxy_env():
                                full_market = ak.stock_zh_a_spot_em()
                            if full_market is not None and not full_market.empty:
                                break
                            if attempt < max_retries - 1:
                                time.sleep(0.5 * (attempt + 1))
                        except Exception as e:
                            if attempt < max_retries - 1:
                                time.sleep(0.5 * (attempt + 1))
                                continue
                            print(f"[警告] 获取全市场数据失败 (尝试{attempt+1}/{max_retries}): {e}")
                    if full_market is None or full_market.empty:
                        return industry or "未知", pd.DataFrame()
            else:
                full_market = spot_df.copy()

            # 更宽松的列名匹配
            code_col = next((c for c in full_market.columns if '代码' in c or 'code' in str(c).lower()), None)
            name_col = next((c for c in full_market.columns if '名称' in c or 'name' in str(c).lower() or '股票简称' in c), None)
            ind_col = next((c for c in full_market.columns if '行业' in c or 'industry' in str(c).lower()), None)
            mkt_cap_col = next((c for c in full_market.columns if '总市值' in c), None) or next((c for c in full_market.columns if '市值' in c and '流通' not in c), None) or next((c for c in full_market.columns if '市值' in c), None)
            pe_col = next((c for c in full_market.columns if '市盈率' in c and ('动' in c or 'TTM' in c)), None) or next((c for c in full_market.columns if '市盈率' in c), None) or next((c for c in full_market.columns if 'PE' in str(c).upper()), None)
            pb_col = next((c for c in full_market.columns if '市净率' in c), None) or next((c for c in full_market.columns if 'PB' in str(c).upper()), None)

            if code_col:
                full_market[code_col] = full_market[code_col].astype(str).str.zfill(6)

            peers_df = pd.DataFrame()
            cons_df_cached = None
            
            # 优先尝试：通过行业名称获取该行业成分股 (修复：紫金矿业匹配银行问题)
            # 工业级优化：添加重试机制
            if industry and industry not in ["未知", "上海主板", "深圳主板", "创业板", "科创板"]:
                for attempt in range(max_retries):
                    try:
                        # 获取行业成分股代码列表（多名称尝试）
                        board_name = self._match_industry_board_name(industry) or industry
                        norm = self._normalize_industry_name(board_name)
                        board_candidates = []
                        for n in [board_name, norm, f"{norm}行业", f"{norm}板块"]:
                            if n and n not in board_candidates:
                                board_candidates.append(n)
                        # 模糊匹配行业名称（应对“计算机设备(申万)”等差异）
                        try:
                            names = self._get_industry_board_names()
                            if names:
                                for name in names:
                                    if norm and norm in self._normalize_industry_name(name):
                                        if name not in board_candidates:
                                            board_candidates.append(name)
                                close = difflib.get_close_matches(industry, names, n=5, cutoff=0.6)
                                for name in close:
                                    if name not in board_candidates:
                                        board_candidates.append(name)
                        except Exception:
                            pass

                        cons_df = None
                        for bn in board_candidates:
                            if hasattr(ak, "stock_board_industry_cons_em"):
                                with no_proxy_env():
                                    cons_df = ak.stock_board_industry_cons_em(symbol=bn)
                            if (cons_df is None or cons_df.empty) and hasattr(ak, "stock_board_industry_cons_ths"):
                                with no_proxy_env():
                                    cons_df = ak.stock_board_industry_cons_ths(symbol=bn)
                            if (cons_df is None or cons_df.empty) and hasattr(ak, "stock_board_industry_cons_sina"):
                                with no_proxy_env():
                                    cons_df = ak.stock_board_industry_cons_sina(symbol=bn)
                            if cons_df is not None and not cons_df.empty:
                                break

                        # 如果仍失败，尝试用成分反查（扩大扫描范围）
                        if (cons_df is None or cons_df.empty):
                            inferred = self._lookup_industry_from_boards(symbol, max_retries=max_retries, max_scan=800)
                            if inferred:
                                for bn in [inferred, self._normalize_industry_name(inferred), f"{self._normalize_industry_name(inferred)}行业"]:
                                    if hasattr(ak, "stock_board_industry_cons_em"):
                                        with no_proxy_env():
                                            cons_df = ak.stock_board_industry_cons_em(symbol=bn)
                                    if cons_df is not None and not cons_df.empty:
                                        break

                        if cons_df is not None and not cons_df.empty:
                            cons_df_cached = cons_df.copy()
                            cons_code_col = next((c for c in cons_df.columns if '代码' in c or '证券代码' in c or 'stock_code' in c), None)
                            if cons_code_col:
                                cons_codes = cons_df[cons_code_col].astype(str).str.zfill(6).tolist()
                                if cons_codes and code_col:
                                    peers_df = full_market[full_market[code_col].isin(cons_codes)].copy()
                                    if not peers_df.empty:
                                        break  # 成功获取，退出重试循环
                        if attempt < max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))  # 指数退避
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        print(f"[警告] 获取行业成分股失败 ({industry}, 尝试{attempt+1}/{max_retries}): {e}")

            # 兜底1：如果 spot_df 自带行业列，且上面获取成分股失败
            if peers_df.empty and ind_col and industry not in ["未知", "上海主板", "深圳主板", "创业板", "科创板"]:
                norm_ind = self._normalize_industry_name(industry)
                peers_df = full_market[full_market[ind_col].astype(str).apply(self._normalize_industry_name) == norm_ind].copy()

            # 若同行过少，尝试扩大到行业名称匹配
            if not peers_df.empty and len(peers_df) < 3 and ind_col and industry:
                broaden = full_market[full_market[ind_col].astype(str).str.contains(self._normalize_industry_name(industry), na=False)].copy()
                if len(broaden) > len(peers_df):
                    peers_df = broaden
            
            # 移除粗暴的板块前缀兜底，避免将紫金矿业（有色）匹配为市值最高的银行股
            # if peers_df.empty:
            #    peers_df = full_market[full_market[code_col].astype(str).str.startswith(symbol[:2])].copy()

            if peers_df.empty:
                # 如果找不到同行，尝试用全部A股的同名行业（如果spot里有行业列但没匹配上）
                if ind_col and industry:
                     peers_df = full_market[full_market[ind_col].astype(str).str.contains(self._normalize_industry_name(industry), na=False)].copy()
                
                # 成分股兜底：如果 full_market 匹配失败，直接使用行业成分股列表
                if (peers_df.empty or len(peers_df) < 2) and cons_df_cached is not None and not cons_df_cached.empty:
                    # 如果成分股表有代码列，尝试从 full_market 中匹配获取完整数据
                    cons_code_col = next((c for c in cons_df_cached.columns if '代码' in c or '证券代码' in c or 'stock_code' in c), None)
                    if cons_code_col and code_col:
                        cons_codes = cons_df_cached[cons_code_col].astype(str).str.zfill(6).tolist()
                        if cons_codes:
                            matched = full_market[full_market[code_col].isin(cons_codes)].copy()
                            if not matched.empty:
                                peers_df = matched
                    # 如果还是空，直接使用成分股表（但需要重新识别列名）
                    if peers_df.empty:
                        peers_df = cons_df_cached.copy()
                        # 重新识别列名（针对成分股表）
                        code_col = next((c for c in peers_df.columns if '代码' in c or '证券代码' in c or 'stock_code' in c), None)
                        name_col = next((c for c in peers_df.columns if '名称' in c or '股票简称' in c or 'name' in str(c).lower()), None)
                        pe_col = next((c for c in peers_df.columns if '市盈率' in c or 'PE' in str(c).upper()), None)
                        pb_col = next((c for c in peers_df.columns if '市净率' in c or 'PB' in str(c).upper()), None)
                        mkt_cap_col = next((c for c in peers_df.columns if '总市值' in c), None) or next((c for c in peers_df.columns if '市值' in c and '流通' not in c), None)

                if peers_df.empty:
                    print(f"[警告] [{symbol}] 无法获取行业对标数据（行业: {industry}）")
                    return industry, pd.DataFrame()
                
                # 确保排除当前股票，且至少有2个同行
                if code_col and code_col in peers_df.columns and len(peers_df) > 0:
                    peers_df = peers_df[peers_df[code_col] != symbol].copy()

            # 排除当前股票
            if code_col and code_col in peers_df.columns:
                peers_df = peers_df[peers_df[code_col] != symbol].copy()
            
            if peers_df.empty:
                print(f"[警告] [{symbol}] 排除当前股票后，无同行数据")
                return industry, pd.DataFrame()
            
            if mkt_cap_col and mkt_cap_col in peers_df.columns:
                peers_df = peers_df.sort_values(by=mkt_cap_col, ascending=False).head(6).copy()
            else:
                peers_df = peers_df.head(6).copy()

            # 确保所有必需的列都存在
            comparison_data = {}
            if code_col and code_col in peers_df.columns:
                comparison_data["代码"] = peers_df[code_col].astype(str).str.zfill(6).tolist()
            elif code_col:
                comparison_data["代码"] = peers_df.index.astype(str).tolist() if hasattr(peers_df.index, 'tolist') else [str(i) for i in peers_df.index]
            else:
                comparison_data["代码"] = [symbol] * len(peers_df)
            
            if name_col and name_col in peers_df.columns:
                comparison_data["名称"] = peers_df[name_col].astype(str).tolist()
            else:
                comparison_data["名称"] = [""] * len(peers_df)
            
            if pe_col and pe_col in peers_df.columns:
                comparison_data["PE(动)"] = peers_df[pe_col].apply(self._to_f).tolist()
            else:
                comparison_data["PE(动)"] = [0.0] * len(peers_df)
            
            if pb_col and pb_col in peers_df.columns:
                comparison_data["PB"] = peers_df[pb_col].apply(self._to_f).tolist()
            else:
                comparison_data["PB"] = [0.0] * len(peers_df)
            
            if mkt_cap_col and mkt_cap_col in peers_df.columns:
                comparison_data["市值(亿)"] = (peers_df[mkt_cap_col].apply(self._to_f) / 100000000).round(2).tolist()
            else:
                comparison_data["市值(亿)"] = [0.0] * len(peers_df)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['ROE(推算%)'] = np.where(comparison_df['PE(动)'] > 0,
                                                   (comparison_df['PB'] / comparison_df['PE(动)'] * 100).round(2), 0.0)

            if not force_live:
                self._industry_cache[symbol] = industry
                self._peers_cache[symbol] = (industry, comparison_df)
            return industry, comparison_df
        except Exception:
            return industry or "未知", pd.DataFrame()

    def _find_val(self, row, cols, keywords):
        for c in cols:
            if all(k in str(c) for k in keywords): return self._to_f(row[c])
        return 0.0

    def _is_valid_num(self, val, allow_zero: bool = False) -> bool:
        try:
            if val is None:
                return False
            f = float(val)
            if not np.isfinite(f):
                return False
            return True if allow_zero else f != 0
        except Exception:
            return False

    def _to_f(self, val):
        try:
            if val is None:
                return 0.0
            s = str(val).strip()
            if not s:
                return 0.0
            if s in ['-', '--', '—', 'nan', 'NaN', 'N/A', 'NA', 'n/a', 'None', 'null', 'NULL']:
                return 0.0
            s = s.replace('元', '')
            mult = 1.0
            if s.endswith("万亿"):
                mult = 1e12
                s = s[:-2]
            elif s.endswith("亿"):
                mult = 1e8
                s = s[:-1]
            elif s.endswith("万"):
                mult = 1e4
                s = s[:-1]
            s = s.replace('%', '').replace(',', '').replace('倍', '')
            return float(s) * mult
        except:
            return 0.0