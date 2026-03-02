# VisionQuant-Pro 项目架构与数据流转

## 项目目录结构

```
VisionQuant-Pro/
├── run.py                    # 启动入口
├── web/
│   ├── app.py                # Streamlit Web 主界面
│   ├── backtest_handlers.py  # 回测处理器
│   └── factor_analysis_handlers.py  # 因子分析处理器
├── src/
│   ├── data/                 # 数据层
│   ├── models/               # 模型层
│   ├── strategies/           # 策略层
│   ├── factor_analysis/      # 因子分析层
│   ├── agent/                # AI Agent层
│   └── utils/                # 工具层
├── config/                   # 配置文件
├── scripts/                  # 训练脚本
└── data/                     # 数据存储
```

## 数据流转架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户界面层                            │
│                           web/app.py (Streamlit)                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              引擎加载层                                      │
│  load_all_engines() → {loader, vision, factor, fund, agent, news, audio}    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   数据获取层     │     │    视觉分析层        │     │    基本面分析层     │
│  DataLoader     │     │   VisionEngine      │     │  FundamentalMiner   │
│                 │     │                     │     │                     │
│ ┌─────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
│ │DataSource   │ │     │ │ AttentionCAE    │ │     │ │ 财务指标获取    │ │
│ │ ├─Diggold   │ │     │ │ (卷积自编码器)  │ │     │ │ PE/PB/ROE      │ │
│ │ ├─Efinance  │ │     │ └────────┬────────┘ │     │ │ 行业对标       │ │
│ │ └─Akshare   │ │     │          │          │     │ └─────────────────┘ │
│ └─────────────┘ │     │          ▼          │     │                     │
│                 │     │ ┌─────────────────┐ │     │ 数据源: 东财/聚宽   │
│ QualityChecker  │     │ │ FAISS 索引      │ │     └─────────────────────┘
│ (数据质量检查)   │     │ │ 相似形态检索    │ │
└────────┬────────┘     │ └────────┬────────┘ │
         │              │          │          │
         │              │          ▼          │
         │              │ ┌─────────────────┐ │
         │              │ │ DTW 距离计算    │ │
         │              │ │ (动态时间规整)  │ │
         │              │ └─────────────────┘ │
         │              └─────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              因子计算层                                      │
│                            FactorMiner                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Q因子(技术)  │  │  V因子(视觉)  │  │  F因子(基本面)│  │  动态权重    │    │
│  │  MA60/RSI    │  │  胜率评分    │  │  PE/PB/ROE   │  │ RegimeManager│    │
│  │  MACD        │  │  相似度      │  │  行业排名    │  │ (市场状态)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI 决策层                                       │
│                            QuantAgent                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LLM (智谱GLM-4 / Google Gemini)                                     │   │
│  │  输入: 技术面 + 基本面 + 舆情 + 因子有效性                           │   │
│  │  输出: TradeDecision {action, confidence, risk_level, reasoning}     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              回测验证层                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │ AdaptiveVisionStrat │  │ TripleBarrierLabeler│  │ StratifiedBacktester│ │
│  │ (自适应双模态策略)   │  │ (三重屏障标签)      │  │ (分层回测)          │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              因子分析层                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ IC Analysis  │  │ DecayAnalysis│  │ CrowdingDet  │  │ RegimeDetect │    │
│  │ (IC分析)     │  │ (衰减分析)   │  │ (拥挤度检测) │  │ (状态检测)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **数据加载器** | `src/data/data_loader.py` | 统一数据获取接口，支持多数据源切换和缓存 |
| **数据源适配** | `src/data/data_source.py` | 抽象数据源接口，实现 Diggold/Efinance/Akshare 适配器 |
| **视觉引擎** | `src/models/vision_engine.py` | K线图像编码、FAISS 相似形态检索、DTW 距离计算 |
| **因子挖掘** | `src/strategies/factor_mining.py` | 计算 Q/V/F 因子，动态权重融合 |
| **基本面分析** | `src/strategies/fundamental.py` | 获取财务指标、行业对标分析 |
| **AI Agent** | `src/agent/quant_agent.py` | LLM 驱动的投资决策生成 |
| **回测引擎** | `src/strategies/backtester.py` | 自适应双模态策略回测 |
| **三重屏障** | `src/data/triple_barrier.py` | 业界标准的金融标签生成方法 |
| **IC 分析** | `src/factor_analysis/ic_analysis.py` | 因子有效性滚动分析 |

## 数据获取流程详解

```
用户请求股票分析 (symbol: 000818)
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  DataLoader.get_stock_data(symbol, start_date, use_cache)  │
└───────────────────────────┬────────────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────┐                 ┌─────────────────────┐
│ 检查本地缓存     │                 │ 缓存未命中/过期      │
│ data/raw/{symbol}.csv │           │ 或请求更新范围       │
└────────┬────────┘                 └──────────┬──────────┘
         │                                     │
         │ 缓存命中                             ▼
         │                          ┌─────────────────────┐
         │                          │ 当前数据源获取       │
         │                          │ DiggoldDataSource   │
         │                          └──────────┬──────────┘
         │                                     │
         │                                     ▼ 失败
         │                          ┌─────────────────────┐
         │                          │ 回退备用数据源       │
         │                          │ EfinanceDataSource  │
         │                          └──────────┬──────────┘
         │                                     │
         └─────────────────────┬───────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ QualityChecker      │
                    │ 数据质量检查         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 返回 DataFrame      │
                    │ [Open,High,Low,     │
                    │  Close,Volume]      │
                    └─────────────────────┘
```

## 当前配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 默认数据源 | `diggold` | 东财掘金 SDK |
| 数据范围 | 最近 365 天 | 动态计算 |
| 复权方式 | `qfq` | 前复权 |
| 缓存目录 | `data/raw/` | 本地 CSV 缓存 |

## 各层详细说明

### 1. 数据层 (src/data/)

**核心文件:**
- `data_loader.py`: 数据加载主入口，支持缓存和多数据源切换
- `data_source.py`: 数据源抽象接口和具体实现
- `quality_checker.py`: 数据质量检查器
- `triple_barrier.py`: 三重屏障标签生成

**数据源优先级:**
1. Diggold (东财掘金SDK) - 推荐，最稳定
2. Efinance - 备用
3. Akshare - 免费备用

### 2. 模型层 (src/models/)

**核心文件:**
- `vision_engine.py`: 视觉引擎主入口
- `attention_cae.py`: 注意力卷积自编码器
- `predict_engine.py`: 预测引擎

**工作流程:**
1. 加载预训练的 AttentionCAE 模型
2. 将 K 线图编码为特征向量
3. 使用 FAISS 进行相似形态检索
4. 计算历史胜率

### 3. 策略层 (src/strategies/)

**核心文件:**
- `factor_mining.py`: 因子挖掘和评分
- `fundamental.py`: 基本面分析
- `backtester.py`: 回测引擎
- `regime_manager.py`: 市场状态管理

**因子体系:**
- Q因子: 技术指标 (MA60, RSI, MACD)
- V因子: 视觉胜率评分
- F因子: 基本面评分 (PE, PB, ROE)

### 4. AI Agent层 (src/agent/)

**核心文件:**
- `quant_agent.py`: LLM 驱动的投资决策

**支持的 LLM:**
- 智谱 GLM-4 (默认)
- Google Gemini (备用)

**输出结构:**
```python
class TradeDecision(BaseModel):
    action: str        # BUY, SELL, WAIT
    confidence: int    # 0-100
    risk_level: str    # High, Medium, Low
    reasoning: str     # 分析理由
```

### 5. 因子分析层 (src/factor_analysis/)

**核心文件:**
- `ic_analysis.py`: IC 滚动分析
- `decay_analysis.py`: 因子衰减分析
- `crowding_detector.py`: 拥挤度检测
- `regime_detector.py`: 市场状态检测

### 6. Web层

**核心文件:**
- `app.py`: Streamlit 主界面
- `backtest_handlers.py`: 回测处理
- `factor_analysis_handlers.py`: 因子分析处理

## 启动流程

```
run.py
    │
    ▼
检查依赖 (streamlit, torch, faiss)
    │
    ▼
检查 .env 配置文件
    │
    ▼
启动 Streamlit 服务 (端口 8501)
    │
    ▼
web/app.py 初始化
    │
    ├─ load_all_engines()
    │   ├─ DataLoader (数据加载器)
    │   ├─ VisionEngine (视觉引擎)
    │   ├─ FactorMiner (因子挖掘)
    │   ├─ FundamentalMiner (基本面)
    │   ├─ QuantAgent (AI Agent)
    │   ├─ NewsHarvester (新闻)
    │   └─ AudioManager (语音)
    │
    └─ 渲染 Web 界面
```

## 环境变量配置 (.env)

```env
# 数据源
DIGGOLD_TOKEN=your_diggold_token

# AI Agent
GOOGLE_API_KEY=your_google_api_key
ZHIPU_API_KEY=your_zhipu_api_key

# 可选: 聚宽/米筐
JQDATA_USERNAME=your_username
JQDATA_PASSWORD=your_password
RQDATA_USERNAME=your_username
RQDATA_PASSWORD=your_password
```

---

*文档生成时间: 2026-03-01*
