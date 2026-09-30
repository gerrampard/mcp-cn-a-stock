from dotenv import load_dotenv
load_dotenv(override=True)

from io import StringIO


from qtf_mcp import research

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def load_data(
  symbol: str,
  start_date: str = "",
  end_date: str = "",
  benchmarks: list[str] | None = None,
) -> str:
  raw_data = await research.load_raw_data(symbol)
  buf = StringIO()
  if len(raw_data) == 0:
    return "No data found for symbol: " + symbol
  research.build_basic_data(buf, symbol, raw_data)
  research.build_trading_data(buf, symbol, raw_data)

  target_bms = list(benchmarks) if benchmarks else ["SH000300"]
  if "SH000300" not in target_bms:
    target_bms.append("SH000300")
  bm_datas = {}
  for bm in target_bms:
    try:
      bm_data = await research.load_raw_data(bm)
      if len(bm_data) > 0:
        bm_datas[bm] = bm_data
    except Exception:
      pass
  if bm_datas:
    research.build_benchmark_data(buf, symbol, raw_data, bm_datas)

  research.build_financial_data(buf, symbol, raw_data)
  research.build_technical_data(buf, symbol, raw_data)
  return buf.getvalue()


if __name__ == "__main__":
  import asyncio

  symbol = "SH600000"
  start_date = "2025-01-01"
  end_date = "2026-01-01"
  result = asyncio.run(load_data(symbol, start_date, end_date, benchmarks=["SH512800"]))
  print(result)
  # asyncio.run(dump_kline(
  #   ["SH600000", "SH600004", "SH600006", "SH600007", "SH600008", "SZ000001", "SZ000002", "SZ000004", "SZ000006", "SZ000007"], 
  #   "/home/jia/repo/msd-rs2/tests/data/stock_kline_1d.csv")
  # )
