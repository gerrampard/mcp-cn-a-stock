import datetime
from io import StringIO
from typing import Dict, TextIO

import numpy as np
from numpy import ndarray
from .indicators import KDJ, MACD, RSI,BBANDS, OBV, ATR

from .datafeed import load_data_msd, is_stock
from .symbols import symbol_with_name, get_symbol_name
import alpha as al


async def load_raw_data(
  symbol: str, who: str = ""
) -> Dict[str, ndarray]:
  return load_data_msd(symbol, n=400, who=who)



def build_stock_data(symbol: str, raw_data: Dict[str, ndarray]) -> str:
  md = StringIO()
  build_basic_data(md, symbol, raw_data)
  build_trading_data(md, symbol, raw_data)
  build_technical_data(md, symbol, raw_data)
  build_financial_data(md, symbol, raw_data)

  return md.getvalue()


def filter_sector(sectors: list[str]) -> list[str]:
  keywords = ["MSCI", "标普", "同花顺", "融资融券", "沪股通"]
  # return sectors not including keywords
  return [s for s in sectors if not any(k in s for k in keywords)]


def est_fin_ratio(last_fin_date: datetime.datetime) -> float:
  if last_fin_date.month == 12:
    return 1
  elif last_fin_date.month == 9:
    return 0.75
  elif last_fin_date.month == 6:
    return 0.5
  elif last_fin_date.month == 3:
    return 0.25
  else:
    return 0


def yearly_fin_index(dates: ndarray) -> int:
  """
  Returns the index of the last December in the dates array.
  If no December is found, returns -1.
  """
  for i in range(len(dates) - 1, -1, -1):
    date = datetime.datetime.fromtimestamp(dates[i].astype(int) / 1_000_000)
    if date.month == 12:
      return i
  return -1

def as_datetime(d) -> datetime.datetime:
  return datetime.datetime.fromtimestamp(d.astype(int) / 1_000_000)

def calc_ttm(data: ndarray, fin_dates: list[datetime.datetime]) -> float:
  total = 0.0
  for i in range(-1, -5, -1):
    if fin_dates[i].month == 3:
      total += data[i]
    else:
      total +=  (data[i] - data[i-1])
  return total
    

def build_basic_data(fp: TextIO, symbol: str, data: Dict[str, ndarray]) -> None:
  print("# 基本数据", file=fp)
  print("", file=fp)
  symbol, name = list(symbol_with_name([symbol]))[0]
  sector = " ".join(filter_sector(data["SECTOR"]))  # type: ignore
  data_date = datetime.datetime.fromtimestamp(data["DATE"][-1].astype(int) / 1_000_000)
  fin_dates = list(map(as_datetime, data["_DS_FINANCE"]["ts"])) if is_stock(symbol) else []
  if is_stock(symbol):
    fin = data["_DS_FINANCE"]
    last_year_index = yearly_fin_index(fin["ts"])
  else:
    last_year_index = -1

  print(f"- 股票代码: {symbol}", file=fp)
  print(f"- 股票名称: {name}", file=fp)
  print(f"- 数据日期: {data_date.strftime('%Y-%m-%d')}", file=fp)
  print(f"- 行业概念: {sector}", file=fp)
  if is_stock(symbol):
    total_shares = data["TCAP"][-1]  # Convert to shares
    circ_shares = data.get("TCAP_A", data["TCAP"])[-1]
    close_price = data["CLOSE2"][-1]
    total_amount = total_shares * close_price
    circ_amount = circ_shares * close_price
    total_mv = total_amount / 1e8
    circ_mv = circ_amount / 1e8
    net_profit = data["NP"][last_year_index]
    pe_static = total_amount / net_profit if net_profit != 0 else float("inf")
    pe_dynamic = total_amount / (data["NP"][-1] / est_fin_ratio(fin_dates[-1])) if net_profit != 0 else float("inf")
    pe_ttm = total_amount / calc_ttm(data["NP"], fin_dates) if net_profit != 0 else float("inf")
    eps_ttm = calc_ttm(data["NP"], fin_dates) / total_shares

    div_array = data.get("DIVIDEND", None)
    if div_array is not None and len(div_array) > 0:
      div_1y = float(np.sum(div_array[-240:])) if len(div_array) >= 240 else float(np.sum(div_array))
      dividend_yield = (div_1y / close_price * 100) if close_price > 0 else 0.0
    else:
      div_1y = 0.0
      dividend_yield = 0.0

    print(f"- 总市值: {total_mv:.2f}亿", file=fp)
    print(f"- 流通市值: {circ_mv:.2f}亿", file=fp)
    print(
      f"- 市盈率(静): {pe_static:.2f}",
      file=fp,
    )
    print(
      f"- 市盈率(动): {pe_dynamic:.2f}",
      file=fp,
    )
    print(
      f"- 市盈率(ttm): {pe_ttm:.2f}",
      file=fp,
    )
    print(
      f"- 每股收益(ttm): {eps_ttm:.2f}",
      file=fp,
    )
    print(
      f"- 市净率: {close_price / data['NAVPS'][-1]:.2f}",
      file=fp,
    )
    print(
      f"- 股息率(ttm): {dividend_yield:.2f}% (近1年分红 {div_1y:.2f}元/股)",
      file=fp,
    )
    print(f"- 净资产收益率: {data['ROE'][-1]:.2f}%", file=fp)
  print("", file=fp)


def today_volume_est_ratio(data: Dict[str, ndarray], now: int = 0) -> float:
  data_dt = datetime.datetime.fromtimestamp(data["DATE"][-1].astype(int) / 1_000_000)
  now_dt = (
    datetime.datetime.now() if now == 0 else datetime.datetime.fromtimestamp(now / 1e9)
  )

  data_date = data_dt.strftime("%Y-%m-%d")
  now_date = now_dt.strftime("%Y-%m-%d")
  if data_date != now_date:
    return 1
  now_time = now_dt.strftime("%H:%M:%S")
  if now_time >= "09:30:00" and now_time < "11:30:00":
    start_dt = now_dt.replace(hour=9, minute=30, second=0)
    minutes = (now_dt - start_dt).seconds / 60
    return 240 / (minutes + 1)
  elif now_time >= "11:30:00" and now_time < "13:00:00":
    return 2
  elif now_time >= "13:00:00" and now_time < "15:00:00":
    start_dt = now_dt.replace(hour=13, minute=0, second=0)
    minutes = (now_dt - start_dt).seconds / 60
    return 240 / (120 + minutes + 1)
  else:
    return 1


FUND_FLOW_FIELDS = [
  ("主力", "main"),
  ("超大单", "super"),
  ("大单", "large"),
  ("中单", "middle"),
  ("小单", "small"),
]


def build_fund_flow(field: tuple[str, str], data: Dict[str, ndarray]) -> str:
  field_amount = field[1] + "_amount"
  field_ratio = field[1] + "_prop"
  value_amount = data.get(field_amount, None)
  value_ratio = data.get(field_ratio, None)
  if value_amount is None or value_ratio is None:
    return ""

  kind = field[0]
  amount = value_amount[-1] / 1e8  # Convert to billions
  ratio = abs(value_ratio[-1])
  in_out = "流入" if amount > 0 else "流出"
  amount = abs(amount)  # Use absolute value for display
  return f"- {kind} {in_out}: {amount:.2f}亿, 占比: {ratio/100:.2%}"


def build_trading_data(fp: TextIO, symbol: str, data: Dict[str, ndarray]) -> None:
  today_vol_est_ratio = today_volume_est_ratio(data)
  close = data["CLOSE"]
  volume = data["VOLUME"]
  amount = data["AMOUNT"] / 1e8
  high = data["HIGH"]
  low = data["LOW"]

  periods = list(filter(lambda n: n <= len(close), [5, 20, 60, 120, 240]))

  print("# 交易数据", file=fp)
  print("", file=fp)

  print("## 价格", file=fp)
  print(f"- 当日: {close[-1]:.3f} 最高: {high[-1]:.3f} 最低: {low[-1]:.3f}", file=fp)
  for p in periods:
    print(
      f"- {p}日均价: {close[-p:].mean():.3f} 最高: {high[-p:].max():.3f} 最低: {low[-p:].min():.3f}",
      file=fp,
    )
  print("", file=fp)

  print("## 振幅", file=fp)
  print(f"- 当日: {(high[-1] / low[-1] - 1):.2%}", file=fp)
  for p in periods:
    print(f"- {p}日振幅: {(high[-p:].max() / low[-p:].min() - 1):.2%}", file=fp)
  print("", file=fp)

  print("## 涨跌幅", file=fp)
  print(f"- 当日: {(close[-1] / close[-2] - 1):.2%}", file=fp)
  for p in periods:
    print(f"- {p}日累计: {(close[-1] / close[-p] - 1) * 100:.2f}%", file=fp)
  print("", file=fp)

  print("## 成交量(万手)", file=fp)
  est_sign = '(盘中)' if  today_vol_est_ratio > 1 else ''
  print(f"- 当日{est_sign}: {volume[-1] / 1e6:.2f}", file=fp)
  if today_vol_est_ratio > 1.0:
    print(f"- 当日(预估,仅参考): {volume[-1] * today_vol_est_ratio / 1e6:.2f}", file=fp)
  for p in periods:
    print(f"- {p}日均量: {volume[-(p+1):-1].mean() / 1e6:.2f}", file=fp)
  print("", file=fp)

  print("## 成交额(亿)", file=fp)
  print(f"- 当日{est_sign}: {amount[-1]:.2f}", file=fp)
  if today_vol_est_ratio > 1.0:
    print(f"- 当日(预估,仅参考): {amount[-1] * today_vol_est_ratio:.2f}", file=fp)
  for p in periods:
    print(f"- {p}日均额(亿): {amount[-(p+1):-1].mean():.2f}", file=fp)
  print("", file=fp)

  print("## 资金流向", file=fp)
  for field in FUND_FLOW_FIELDS:
    value = build_fund_flow(field, data)
    if value:
      print(value, file=fp)

  main_amount = data.get("main_amount", None)
  if main_amount is not None:
    flow_periods = [p for p in [3, 5, 10, 20] if len(main_amount) >= p]
    if flow_periods:
      cum_str = ", ".join([f"{p}日: {main_amount[-p:].sum() / 1e8:+.2f}亿" for p in flow_periods])
      print(f"- 主力多日累计: {cum_str}", file=fp)
  print("", file=fp)

  if is_stock(symbol):
    tcap = data["TCAP_A"][-1]
    print("## 换手率", file=fp)
    print(f"- 当日: {volume[-1] / tcap:.2%}", file=fp)
    for p in periods:
      print(f"- {p}日均换手: {volume[-p:].mean() / tcap:.2%}", file=fp)
      print(f"- {p}日总换手: {volume[-p:].sum() / tcap:.2%}", file=fp)
    print("", file=fp)


COMMON_INDEX_NAMES = {
  "SH000001": "上证指数",
  "SH000300": "沪深300",
  "SZ399001": "深证成指",
  "SZ399006": "创业板指",
  "SH000016": "上证50",
  "SH000905": "中证500",
  "SH000852": "中证1000",
}


def get_display_name(symbol: str) -> str:
  name = get_symbol_name(symbol)
  if not name:
    name = COMMON_INDEX_NAMES.get(symbol, "")
  return f"{symbol} {name}".strip() if name else symbol


def build_benchmark_data(
  fp: TextIO,
  symbol: str,
  stock_data: Dict[str, ndarray],
  benchmark_datas: Dict[str, Dict[str, ndarray]],
) -> None:
  if not benchmark_datas:
    return

  close = stock_data.get("CLOSE")
  if close is None or len(close) < 2:
    return

  periods = list(filter(lambda n: n <= len(close), [5, 20, 60, 120, 240]))
  stock_label = get_display_name(symbol)

  print("# 基准对比", file=fp)
  print("", file=fp)

  headers = ["标的", "当日"] + [f"{p}日累计" for p in periods]
  print("| " + " | ".join(headers) + " |", file=fp)
  print("| --- " * len(headers) + "|", file=fp)

  stock_today = (close[-1] / close[-2] - 1) * 100
  stock_period_changes = {p: (close[-1] / close[-p] - 1) * 100 for p in periods}

  stock_cols = [stock_label, f"{stock_today:+.2f}%"] + [
    f"{stock_period_changes[p]:+.2f}%" for p in periods
  ]
  print("| " + " | ".join(stock_cols) + " |", file=fp)

  for bm_symbol, bm_data in benchmark_datas.items():
    bm_close = bm_data.get("CLOSE")
    if bm_close is None or len(bm_close) < 2:
      continue

    bm_today = (bm_close[-1] / bm_close[-2] - 1) * 100
    bm_period_changes = {
      p: (bm_close[-1] / bm_close[-p] - 1) * 100 if len(bm_close) >= p else float("nan")
      for p in periods
    }

    bm_display = get_display_name(bm_symbol)
    bm_cols = [bm_display, f"{bm_today:+.2f}%"] + [
      f"{bm_period_changes[p]:+.2f}%" if not np.isnan(bm_period_changes[p]) else "-"
      for p in periods
    ]
    print("| " + " | ".join(bm_cols) + " |", file=fp)

    # 相对超额 (Alpha)
    alpha_today = stock_today - bm_today
    alpha_cols = [
      "&nbsp;&nbsp;└ 超额(Alpha)",
      f"{alpha_today:+.2f}%",
    ] + [
      f"{stock_period_changes[p] - bm_period_changes[p]:+.2f}%"
      if not np.isnan(bm_period_changes[p])
      else "-"
      for p in periods
    ]
    print("| " + " | ".join(alpha_cols) + " |", file=fp)

  print("", file=fp)


def build_technical_data(fp: TextIO, symbol: str, data: Dict[str, ndarray]) -> None:
  close = data["CLOSE"]
  high = data["HIGH"]
  low = data["LOW"]
  volume = data["VOLUME"]

  if len(close) < 30:
    return

  kdj_k, kdj_d, kdj_j = KDJ(close, high, low, 9, 3)
  macd_diff, macd_dea = MACD(close, 12, 26, 9)
  macd_bar = (macd_diff - macd_dea) * 2

  rsi_6 = RSI(close, 6)
  rsi_12 = RSI(close, 12)
  rsi_24 = RSI(close, 24)

  bb_upper, bb_middle, bb_lower = BBANDS(close) 
  obv = OBV(close, volume)
  atr = ATR(close, high, low, n=14)

  ma5 = al.MA(close, 5)
  ma10 = al.MA(close, 10)
  ma30 = al.MA(close, 30)
  ma60 = al.MA(close, 60)

  if ma5[-1] > ma10[-1] > ma30[-1] > ma60[-1]:
    trend_desc = "多头排列 (短中期均线强势上行)"
  elif ma5[-1] < ma10[-1] < ma30[-1] < ma60[-1]:
    trend_desc = "空头排列 (短中期均线弱势寻底)"
  elif ma5[-1] > ma10[-1] and ma5[-1] > ma30[-1]:
    trend_desc = "短期企稳偏强，中期蓄势整理"
  elif ma5[-1] < ma10[-1] and ma5[-1] < ma30[-1]:
    trend_desc = "短期承压回调，中期震荡整理"
  else:
    trend_desc = "均线交错纠缠，处于震荡整固格局"

  bar_today = macd_bar[-1]
  bar_prev = macd_bar[-2] if len(macd_bar) >= 2 else bar_today
  if macd_diff[-1] > macd_dea[-1] and macd_diff[-2] <= macd_dea[-2]:
    macd_desc = f"金叉形成，多方动能显现 (Bar: {bar_today:+.2f})"
  elif macd_diff[-1] < macd_dea[-1] and macd_diff[-2] >= macd_dea[-2]:
    macd_desc = f"死叉形成，空方动能释放 (Bar: {bar_today:+.2f})"
  elif bar_today > 0:
    macd_desc = f"红柱区间 (多头主导)，{'红柱继续放大' if bar_today > bar_prev else '红柱有所收敛'} (Bar: {bar_today:+.2f})"
  else:
    macd_desc = f"绿柱区间 (空头主导)，{'绿柱有所收窄 (空方衰竭)' if bar_today > bar_prev else '绿柱持续放大'} (Bar: {bar_today:+.2f})"

  j_val = kdj_j[-1]
  rsi6_val = rsi_6[-1]
  signals = []
  if j_val < 10:
    signals.append(f"KDJ低位超卖 (J={j_val:.1f} < 10)")
  elif j_val > 90:
    signals.append(f"KDJ高位超买 (J={j_val:.1f} > 90)")
  else:
    signals.append(f"KDJ中性 (J={j_val:.1f})")

  if rsi6_val < 20:
    signals.append(f"RSI(6)超卖 ({rsi6_val:.1f} < 20)")
  elif rsi6_val > 80:
    signals.append(f"RSI(6)超买 ({rsi6_val:.1f} > 80)")
  else:
    signals.append(f"RSI(6)常态 ({rsi6_val:.1f})")
  osc_desc = ", ".join(signals)

  c_last = close[-1]
  up_last, mid_last, low_last = bb_upper[-1], bb_middle[-1], bb_lower[-1]
  if c_last >= up_last:
    bb_desc = f"触及/突破上轨 (现价: {c_last:.2f} >= 上轨: {up_last:.2f}，上方阻力加大)"
  elif c_last <= low_last:
    bb_desc = f"触及/跌破下轨 (现价: {c_last:.2f} <= 下轨: {low_last:.2f}，下方支撑显现)"
  elif c_last > mid_last:
    bb_desc = f"中轨上方运行 (中轨: {mid_last:.2f}, 上轨: {up_last:.2f})"
  else:
    bb_desc = f"中轨下方运行 (下轨: {low_last:.2f}, 中轨: {mid_last:.2f})"

  print("# 技术指标", file=fp)
  print("", file=fp)
  print("## 信号摘要", file=fp)
  print(f"- 均线趋势: {trend_desc}", file=fp)
  print(f"- MACD形态: {macd_desc}", file=fp)
  print(f"- 摆动指标: {osc_desc}", file=fp)
  print(f"- 布林带通道: {bb_desc}", file=fp)
  print("", file=fp)

  print("## 指标历史明细(最近30日)", file=fp)
  print("", file=fp)

  date = [
    datetime.datetime.fromtimestamp(d.astype(int) / 1_000_000).strftime("%Y-%m-%d") for d in data["DATE"]
  ]
  columns = [
    ("日期", date),
    ("MA(5)", ma5),
    ("MA(10)", ma10),
    ("MA(30)", ma30),
    ("MA(60)", ma60),
    ("MA(120)", al.MA(close, 120)),
    ("KDJ.K", kdj_k),
    ("KDJ.D", kdj_d),
    ("KDJ.J", kdj_j),
    ("MACD DIF", macd_diff),
    ("MACD DEA", macd_dea),
    ("MACD Bar", macd_bar),
    ("RSI(6)", rsi_6),
    ("RSI(12)", rsi_12),
    ("RSI(24)", rsi_24),
    ("BBands Upper", bb_upper),
    ("BBands Middle", bb_middle),
    ("BBands Lower", bb_lower),
    ("OBV", obv),
    ("ATR", atr),
  ]
  print("| " + " | ".join([c[0] for c in columns]) + " |", file=fp)
  print("| --- " * len(columns) + "|", file=fp)
  for i in range(-1, max(-len(date), -31), -1):
    print(
      "| " + date[i] + "|" + " | ".join([f"{c[1][i]:.2f}" for c in columns[1:]]) + " |",
      file=fp,
    )
  print("", file=fp)


def quarter_label(date: datetime.datetime) -> str:
  if date.month == 12:
    return f'{date.year}年报'
  elif date.month == 9:
    return f'{date.year}年三季报'
  elif date.month == 6:
    return f'{date.year}年中报'
  elif date.month == 3:
    return f'{date.year}年一季报'
  else:
    return ''

def build_financial_data(fp: TextIO, symbol: str, data: Dict[str, ndarray]) -> None:
  if not is_stock(symbol):
    return
  fin = data["_DS_FINANCE"]
  dates = fin["ts"]
  max_years = 5
  print("# 财务数据", file=fp)
  print("", file=fp)
  years = 0
  fields = [
    # name, id, div, is_pct
    ("主营收入(亿元)", "f075", 1_0000_0000, False),
    ("营收同比增长率", "f194", 1, True),
    ("净利润(亿元)", "f097", 1_0000_0000, False),
    ("净利润同比增长率", "f197", 1, True),
    ("扣非营业利润(亿元)", "f088", 1_0000_0000, False),
    ("摊薄每股收益", "f000", 1, False),
    ("扣非每股收益", "f007", 1, False),
    ("每股净资产", "f003", 1, False),
    ("净资产收益率", "f001", 1, True),
    ("经营现金流净额(亿元)", "f108", 1_0000_0000, False),
    ("资产负债率", "f223", 1, True),
  ]

  rows = []
  last_quarter = 0
  for i in range(len(dates) - 1, 0, -1):
    date = datetime.datetime.fromtimestamp(dates[i].astype(int) / 1_000_000)
    is_last = i == (len(dates) - 1)
    if is_last:
      last_quarter = date.month
    if (date.month != 12 and date.month != last_quarter) or (years >= max_years):
      continue
    row = [quarter_label(date)]

    for _, field, div, is_pct in fields:
      if field in fin and len(fin[field]) > i:
        val = fin[field][i] / div
        if np.isnan(val):
          row.append("-")
        elif is_pct:
          row.append(f"{val:.2f}%")
        else:
          row.append(f"{val:.2f}")
      else:
        row.append("-")
    rows.append(row)
    if date.month == 12:
      years += 1

  print("| 指标 | " + " ".join([f"{r[0]} |" for r in rows]), file=fp)
  print("| --- " * (len(rows) + 1) + "|", file=fp)
  for i in range(1, len(rows[0])):
    print(
      f"| {fields[i - 1][0]} | " + " ".join([f"{r[i]} |" for r in rows]),
      file=fp,
    )

  print("", file=fp)
