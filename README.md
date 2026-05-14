# Clenow Quant

Clenow Smooth Momentum 动量策略回测系统。

基于 Andreas Clenow 著作 *Stocks on the Move* 实现。

## 安装

```bash
pip install -e ".[dev]"
```

或使用 uv:
```bash
uv sync
```

## 快速开始

### 全市场选股 (不限行业)

按Clenow评分排序, 取前20%, 不限行业:

```bash
uv run python run_backtest.py                     # 美股 SP500, 默认参数
uv run python run_backtest.py --market cn         # A股 CSI800
uv run python run_backtest.py --market hk         # 港股 HSI
```

### 行业轮动 (每行业选1支)

每GICS行业选评分最高1支, 分散配置:

```bash
uv run python run_sector_backtest.py              # 美股, 默认参数
uv run python run_sector_backtest.py --market cn  # A股 CSI800
uv run python run_sector_backtest.py --market hk  # 港股 HSI
```

### 每日观察清单 / 实盘订单

**指数成分股选股** (Markdown格式):
```bash
uv run python run_backtest.py --watchlist-only --as-of 2026-05-14 --market cn  # CSI800
uv run python run_backtest.py --watchlist-only --as-of 2026-05-14 --market us  # SP500
```

输出: `output/watchlist.md` (成分股筛选结果)

**全市场选股** (CSV格式):
```bash
uv run python run_backtest.py --watchlist-only --as-of 2026-05-14 --market cn --universe all  # 全A股
uv run python run_backtest.py --watchlist-only --as-of 2026-05-14 --market us --universe all  # 全美股
```

输出: `output/watchlist.csv` (Excel可直接打开)

**参数说明**:
- `--universe index` (默认) — 使用index_constituents表成分股
- `--universe all` — 使用prices表所有tickers (支持CN/US，暂不支持HK)

**选股范围差异**:
| 参数 | CN Universe | US Universe | 输出格式 |
|------|-------------|-------------|----------|
| `--universe index` | CSI800 (~800支) | SP500 (~500支) | Markdown |
| `--universe all` | 全A股 (~5200支) | 全美股 | CSV |

CSV字段: `排名, 代码, 行业, Clenow评分, 回归斜率, R²拟合度, 年化收益率(%), 最新价, 100日均线, 距均线(%)`

实盘订单生成:
```bash
uv run python -m clenow.live --as-of 2026-05-14 --equity 100000 --market us  # CSV订单
```

## 策略

Clenow Smooth Momentum:

1. **评分**: 90日对数线性回归, `exp(slope × 252) - 1) × R²`
2. **排名**: 取评分前20%
3. **仓位**: ATR仓位管理 (`risk_factor=0.005`)
4. **退出**: 突破100日均线强制卖出; 指数<200日均线禁止新开仓

## 市场

| 参数 | 标的池 | 交易日历 | 货币 |
|------|--------|----------|------|
| `us` | SP500 | NYSE | USD |
| `cn` | CSI800 | XSHG | CNY |
| `hk` | HSI | XHKG | HKD |

A股特有: ST过滤、涨跌停检测、停牌/退市处理。

## 架构

**回测引擎 = 实盘引擎。** `compute_target_portfolio` 纯函数, 回测/实盘共享, 无分支。

```
clenow/
├── signals/     # Clenow评分、ATR、趋势过滤
├── portfolio/   # 排名、仓位、持仓跟踪
├── backtest/    # 引擎 + 模拟执行器
├── data/        # DataProvider协议 + Synology MariaDB实现
├── live/        # 每日决策CLI
├── markets/     # 市场参数 (手数、成本、日历)
└── report/      # 权益曲线、指标、交易日志
```

## 测试

```bash
pytest tests/ -q
pytest tests/ -k "cn" -v      # A股相关测试
pytest tests/ -m integration  # 数据库依赖测试
```

## 数据要求

Synology MariaDB 表:
- `prices` (A股/港股后缀 `.SH`/`.SZ`/`.HK`, 美股无后缀)
- `index_constituents` (SP500/CSI800/HSI)
- `index_prices`
- `stocks` (name列用于A股ST检测)

验证:
```bash
python scripts/validate_db.py
```