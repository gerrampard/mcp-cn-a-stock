import asyncio
from io import StringIO
import logging
from typing import Dict, List

from mcp.server.mcpserver import Context, MCPServer
from . import research

logger = logging.getLogger("qtf_mcp")

# Create an MCP server
mcp_app = MCPServer("CnStock")


def normalize_benchmarks(benchmarks: List[str] | str | None = None) -> List[str]:
  """
  规范化基准指数/ETF列表：
  - 未传默认为 ['SH000300']
  - 用户传入若不包含 'SH000300' 则自动追加
  - 支持传入列表或逗号分隔的字符串，统一转大写并去重
  """
  default_benchmark = "SH000300"
  if not benchmarks:
    return [default_benchmark]

  if isinstance(benchmarks, str):
    items = [s.strip().upper() for s in benchmarks.split(",") if s.strip()]
  else:
    items = [s.strip().upper() for s in benchmarks if isinstance(s, str) and s.strip()]

  result: List[str] = []
  for item in items:
    if item not in result:
      result.append(item)

  if default_benchmark not in result:
    result.append(default_benchmark)

  return result if result else [default_benchmark]


async def load_stock_and_benchmarks(
  symbol: str, who: str, benchmarks: List[str] | str | None
) -> tuple[Dict, Dict[str, Dict]]:
  """并发加载目标股票与对比基准数据"""
  target_benchmarks = normalize_benchmarks(benchmarks)
  tasks = [research.load_raw_data(symbol, who)] + [
    research.load_raw_data(bm, who) for bm in target_benchmarks
  ]
  results = await asyncio.gather(*tasks, return_exceptions=True)

  stock_res = results[0]
  if isinstance(stock_res, Exception) or len(stock_res) == 0:
    return {}, {}

  benchmark_datas = {}
  for bm, res in zip(target_benchmarks, results[1:]):
    if not isinstance(res, Exception) and len(res) > 0:
      benchmark_datas[bm] = res
    else:
      logger.warning(f"{who} failed to load benchmark {bm}: {res}")

  return stock_res, benchmark_datas


@mcp_app.tool()
async def brief(
  ctx: Context,
  symbol: str,
  benchmarks: List[str] | None = None,
) -> str:
  """Get brief information for a given stock symbol, including
  - basic data
  - trading data
  - benchmark comparison
  Args:
    symbol (str): Stock symbol, must be in the format of "SH600000" or "SZ000001", you should infer user inputs like stock name to stock symbol
    benchmarks (list[str], optional): Benchmark index or ETF symbols to compare against (e.g. ["SH000300"]). Defaults to ["SH000300"]. It is strongly recommended that you pass in some ETFs that you consider highly relevant, For example, for banks, SH512800 can be chosen as their sector ETF,  as this will help you analyze better. 
  """
  who = ctx.request_context.request.client.host  # type: ignore
  try:
    raw_data, benchmark_datas = await load_stock_and_benchmarks(symbol, who, benchmarks)
    buf = StringIO()
    if len(raw_data) == 0:
      msg = f"No data found for symbol: {symbol}"
      return msg
    research.build_basic_data(buf, symbol, raw_data)
    research.build_trading_data(buf, symbol, raw_data)
    if benchmark_datas:
      research.build_benchmark_data(buf, symbol, raw_data, benchmark_datas)
    return buf.getvalue()
  except:
    msg = f"{who} No data found for symbol: {symbol}"
    logging.info(msg, exc_info=True)
    return msg


@mcp_app.tool()
async def medium(
  ctx: Context,
  symbol: str,
  benchmarks: List[str] | None = None,
) -> str:
  """Get medium information for a given stock symbol, including
  - basic data
  - trading data
  - benchmark comparison
  - financial data
  Args:
    symbol (str): Stock symbol, must be in the format of "SH000001" or "SZ000001", you infer convert user inputs like stock name to stock symbol
    benchmarks (list[str], optional): Benchmark index or ETF symbols to compare against (e.g. ["SH000300"]). Defaults to ["SH000300"]. It is strongly recommended that you pass in some ETFs that you consider highly relevant, For example, for banks, SH512800 can be chosen as their sector ETF,  as this will help you analyze better. 
  """
  who = ctx.request_context.request.client.host  # type: ignore
  try:
    raw_data, benchmark_datas = await load_stock_and_benchmarks(symbol, who, benchmarks)
    buf = StringIO()
    if len(raw_data) == 0:
      msg = f"No data found for symbol: {symbol}"
      return msg
    research.build_basic_data(buf, symbol, raw_data)
    research.build_trading_data(buf, symbol, raw_data)
    if benchmark_datas:
      research.build_benchmark_data(buf, symbol, raw_data, benchmark_datas)
    research.build_financial_data(buf, symbol, raw_data)
    return buf.getvalue()
  except:
    msg = f"{who} No data found for symbol: {symbol}"
    logging.info(msg, exc_info=True)
    return msg


@mcp_app.tool()
async def full(
  ctx: Context,
  symbol: str,
  benchmarks: List[str] | None = None,
) -> str:
  """Get full information for a given stock symbol, including
  - basic data
  - trading data
  - benchmark comparison
  - financial data
  - technical analysis data
  Args:
    symbol (str): Stock symbol, must be in the format of "SH000001" or "SZ000001", you should infer user inputs like stock name to stock symbol
    benchmarks (list[str], optional): Benchmark index or ETF symbols to compare against (e.g. ["SH000300"]). Defaults to ["SH000300"]. It is strongly recommended that you pass in some ETFs that you consider highly relevant, For example, for banks, SH512800 can be chosen as their sector ETF,  as this will help you analyze better. 
  """
  logger.info(f'symbol {symbol}, benchmarks {benchmarks}')
  who = ctx.request_context.request.client.host  # type: ignore
  try:
    raw_data, benchmark_datas = await load_stock_and_benchmarks(symbol, who, benchmarks)
    buf = StringIO()
    if len(raw_data) == 0:
      msg = f"No data found for symbol: {symbol}"
      return msg
    research.build_basic_data(buf, symbol, raw_data)
    research.build_trading_data(buf, symbol, raw_data)
    if benchmark_datas:
      research.build_benchmark_data(buf, symbol, raw_data, benchmark_datas)
    research.build_financial_data(buf, symbol, raw_data)
    research.build_technical_data(buf, symbol, raw_data)
    return buf.getvalue()
  except:
    msg = f"{who} No data found for symbol: {symbol}"
    logging.info(msg, exc_info=True)
    return msg
