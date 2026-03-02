import os
import sys
import time
import logging
from dotenv import load_dotenv

try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 智谱AI配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4.7")

# Google Gemini 配置（备用）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)


class TradeDecision(BaseModel):
    action: str = Field(description="决策: BUY, SELL, or WAIT")
    confidence: int = Field(description="0-100")
    risk_level: str = Field(description="High, Medium, Low")
    reasoning: str = Field(description="分析理由")


class QuantAgent:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=TradeDecision)
        self.chain = None
        self.llm = None
        self._last_error = None
        self._llm_initialized = False  # 延迟初始化标记

        # 提前准备 Prompt
        template = """
你是一位华尔街顶级对冲基金首席风险官 (CRO)。请根据以下多模态数据撰写深度投资备忘录。
**输出必须为中文**，但 action 字段仍使用 BUY/SELL/WAIT。

【数据快照】
股票: {symbol} | 日期: {date}
量化评分: {total_score}/10 | 初步建议: {initial_action}

[技术面]
形态胜率: {win_rate}% | 趋势: {ma_trend} | RSI: {rsi} | MACD: {macd}

[基本面]
PE: {pe_ttm} | PB: {pb} | ROE: {roe}%

[舆情]
{news_summary}

[市场情绪]
{market_sentiment}

[因子有效性]
{ic_summary}

【决策逻辑】
1. **一票否决**：若 ROE<0 或 PE>60，或者新闻有重大利空（立案/调查），强制 **SELL/WAIT**。
2. **趋势共振**：只有当 形态胜率>60% 且 趋势向上 时，才建议 **BUY**。
3. **深度分析**：请详细解释数据之间的冲突（例如：为什么形态好但基本面差要回避？）。
4. **IC反馈**：若IC均值<=0或显著性不足，应降低置信度或给出观望/回避。

{format_instructions}
"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["symbol", "date", "total_score", "initial_action", "win_rate", "ma_trend", "rsi", "macd",
                             "pe_ttm", "pb", "roe", "news_summary", "ic_summary", "market_sentiment"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        # 配置LLM提供者（优先智谱）
        self._llm_providers = []

        if ZHIPU_API_KEY:
            self._llm_providers.append({
                "name": "智谱AI GLM-4.7",
                "type": "zhipu",
                "api_key": ZHIPU_API_KEY,
                "base_url": ZHIPU_BASE_URL,
                "model": ZHIPU_MODEL
            })

        if GOOGLE_API_KEY:
            self._llm_providers.append({
                "name": "Google Gemini",
                "type": "gemini",
                "api_key": GOOGLE_API_KEY,
                "model": "gemini-2.0-flash-exp"
            })

        if not self._llm_providers:
            self._last_error = "缺少 ZHIPU_API_KEY 或 GOOGLE_API_KEY"

    def _init_llm(self):
        """初始化 LLM 连接（延迟调用）"""
        if self._llm_initialized or not self._llm_providers:
            return False

        for provider in self._llm_providers:
            try:
                if provider["type"] == "zhipu":
                    # 使用智谱AI
                    llm = ChatOpenAI(
                        model=provider["model"],
                        api_key=provider["api_key"],
                        base_url=provider["base_url"],
                        temperature=0.4,
                        timeout=60
                    )
                elif provider["type"] == "gemini":
                    # 使用Google Gemini（备用）
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=provider["model"],
                        google_api_key=provider["api_key"],
                        temperature=0.4,
                        transport="rest",
                        timeout=60
                    )
                else:
                    continue

                self.llm = llm
                self.chain = self.prompt | self.llm | self.parser
                self._llm_initialized = True
                logger.info("Agent LLM 已初始化: %s", provider["name"])
                return True
            except Exception as e:
                self._last_error = f"{provider['name']}: {str(e)}"
                logger.warning("LLM初始化失败 [%s]: %s", provider["name"], e)
                continue

        logger.warning("所有LLM连接失败，请检查 API Key 或网络")
        return False

    def _ensure_llm(self):
        """确保 LLM 已初始化（在使用前调用）"""
        if self.chain:
            return True
        return self._init_llm()

    def analyze(self, symbol, total_score, initial_action, visual_data, factor_data, fund_data, news_text="", ic_summary=None, market_sentiment=None):
        if not self._ensure_llm():
            return self._fallback_result(f"API 连接失败: {self._last_error or '请检查网络或额度'}")

        from datetime import datetime
        if isinstance(ic_summary, dict):
            ic_text = (
                f"IC均值: {ic_summary.get('mean_ic', 'N/A')} | "
                f"IR: {ic_summary.get('ir', 'N/A')} | "
                f"IC正相关比例: {ic_summary.get('positive_ratio', 'N/A')} | "
                f"显著性: {ic_summary.get('significant', 'N/A')} | "
                f"样本: {ic_summary.get('samples', 'N/A')}"
            )
        elif isinstance(ic_summary, str) and ic_summary.strip():
            ic_text = ic_summary
        else:
            ic_text = "IC未计算或不可用（未运行因子有效性分析）"

        if isinstance(market_sentiment, dict) and market_sentiment.get("ok"):
            sentiment_text = (
                f"沪深300情绪: {market_sentiment.get('score')}/100 ({market_sentiment.get('label')}) | "
                f"1D: {round((market_sentiment.get('ret_1d') or 0) * 100, 2)}% | "
                f"1W: {round((market_sentiment.get('ret_5d') or 0) * 100, 2)}% | "
                f"4W: {round((market_sentiment.get('ret_20d') or 0) * 100, 2)}% | "
                f"波动: {round((market_sentiment.get('vol_20d') or 0) * 100, 2)}% | "
                f"回撤: {round((market_sentiment.get('max_drawdown_60d') or 0) * 100, 2)}%"
            )
        else:
            sentiment_text = "市场情绪未获取或不可用"

        payload = {
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_score": total_score,
            "initial_action": initial_action,
            "win_rate": visual_data.get('win_rate', 50),
            "ma_trend": factor_data.get('MA_Signal', 0),
            "rsi": round(factor_data.get('RSI', 50), 2),
            "macd": factor_data.get('MACD_Signal', 0),
            "pe_ttm": round(fund_data.get('PE_TTM', 0), 2),
            "pb": round(fund_data.get('PB', 0), 2),
            "roe": round(fund_data.get('ROE', 0), 2),
            "news_summary": news_text[:500] if news_text else "无相关新闻",
            "ic_summary": ic_text,
            "market_sentiment": sentiment_text
        }

        try:
            result = self.chain.invoke(payload)
            return result.model_dump()
        except Exception as e:
            self._last_error = str(e)
            return self._fallback_result(f"AI 分析失败: {str(e)[:100]}")

    def _fallback_result(self, error_msg):
        """当AI不可用时的降级方案"""
        return {
            "action": "WAIT",
            "confidence": 50,
            "risk_level": "Medium",
            "reasoning": f"AI 分析不可用 ({error_msg})，建议等待更多信息或人工分析。"
        }

    def is_available(self) -> bool:
        """检查 Agent 是否可用"""
        return bool(self._llm_providers)
