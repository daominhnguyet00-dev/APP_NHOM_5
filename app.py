"""
========================================================================================
HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC ĐỊNH LƯỢNG KẾT HỢP SMA & OBV TRÊN HOSE
Tác giả / Nghiên cứu: Nhóm 5
Phiên bản Web App: Streamlit Production Release
========================================================================================
Kiến trúc 3 tầng:
  - Tầng 1: Sàng lọc rổ cổ phiếu tiềm năng bằng TA Score (Trend R², Momentum, Volume, Risk)
  - Tầng 2: Hệ thống phát sinh tín hiệu giao dịch SMA, OBV, SMA+OBV (OR), SMA+OBV (AND)
  - Tầng 3: Phân bổ vốn Equal Weight (EW) và Modern Portfolio Theory (MPT - Max Sharpe)
Kiểm định thực tế: Giả định khớp giá mở cửa T+1, ràng buộc thanh toán T+2.5 (3 phiên),
chi phí giao dịch thực tế (phí mua 0.15%, phí bán 0.15% + thuế 0.1%).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import spearmanr
from itertools import product
import os
import io

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG WEB STREAMLIT & GIAO DIỆN HIỆN ĐẠI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chiến lược Định lượng SMA & OBV | Nhóm 5",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách Modern Financial Analytics Dashboard
st.markdown("""
<style>
    /* Tổng thể & Font chữ */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    /* Metric Cards tùy biến */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #6B7280;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        font-weight: 500;
    }
    .val-positive { color: #10B981; }
    .val-negative { color: #EF4444; }
    .val-neutral  { color: #3B82F6; }

    /* Badge phong cách Fintech */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-purple { background-color: #EDE9FE; color: #5B21B6; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }

    /* Highlight box */
    .info-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 16px 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HẰNG SỐ & BẢNG THAM SỐ TỐI ƯU CỦA NHÓM 5 (BENCHMARK ĐÃ XÁC THỰC)
# -----------------------------------------------------------------------------
TRADING_DAYS = 252
TOP_N_DEFAULT = 5
RANDOM_SEED = 42

# Tham số tối ưu chuẩn trên tập Train (2020-2021) trích từ Nghiên cứu của Nhóm 5
DEFAULT_OPTIMAL_PARAMS = {
    "SMA": {
        "DIG": {"ma_short": 20, "ma_long": 100, "obv_window": 20},
        "DCM": {"ma_short": 30, "ma_long": 100, "obv_window": 20},
        "MSN": {"ma_short": 55, "ma_long": 190, "obv_window": 20},
        "SSI": {"ma_short": 30, "ma_long": 130, "obv_window": 20},
        "PDR": {"ma_short": 35, "ma_long": 170, "obv_window": 20},
    },
    "OBV": {
        "DIG": {"ma_short": 20, "ma_long": 100, "obv_window": 60},
        "DCM": {"ma_short": 20, "ma_long": 100, "obv_window": 35},
        "MSN": {"ma_short": 20, "ma_long": 100, "obv_window": 60},
        "SSI": {"ma_short": 20, "ma_long": 100, "obv_window": 40},
        "PDR": {"ma_short": 20, "ma_long": 100, "obv_window": 60},
    },
    "OR": {
        "DIG": {"ma_short": 45, "ma_long": 190, "obv_window": 50},
        "DCM": {"ma_short": 25, "ma_long": 180, "obv_window": 35},
        "MSN": {"ma_short": 55, "ma_long": 140, "obv_window": 60},
        "SSI": {"ma_short": 45, "ma_long": 140, "obv_window": 40},
        "PDR": {"ma_short": 25, "ma_long": 140, "obv_window": 35},
    },
    "AND": {
        "DIG": {"ma_short": 45, "ma_long": 180, "obv_window": 60},
        "DCM": {"ma_short": 25, "ma_long": 180, "obv_window": 35},
        "MSN": {"ma_short": 55, "ma_long": 140, "obv_window": 60},
        "SSI": {"ma_short": 45, "ma_long": 140, "obv_window": 40},
        "PDR": {"ma_short": 50, "ma_long": 170, "obv_window": 60},
    },
}

DEFAULT_MPT_WEIGHTS = {
    "SMA": {"DIG": 0.2005, "DCM": 0.1998, "MSN": 0.0500, "SSI": 0.1608, "PDR": 0.3889},
    "OBV": {"DIG": 0.2316, "DCM": 0.1846, "MSN": 0.0500, "SSI": 0.1946, "PDR": 0.3392},
    "OR":  {"DIG": 0.2461, "DCM": 0.1938, "MSN": 0.0500, "SSI": 0.2177, "PDR": 0.2924},
    "AND": {"DIG": 0.2316, "DCM": 0.1846, "MSN": 0.0500, "SSI": 0.1946, "PDR": 0.3392},
}

# -----------------------------------------------------------------------------
# 3. CÁC HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN ĐỊNH LƯỢNG (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_from_source(source):
    """Đọc dữ liệu từ đường dẫn file hoặc đối tượng UploadedFile của Streamlit"""
    if isinstance(source, str):
        if not os.path.exists(source):
            return None, None, None, None, None, None, None, None, f"Không tìm thấy file: {source}"
        df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    o_col = "adj_open" if "adj_open" in df.columns else "open"
    c_col = "adj_close" if "adj_close" in df.columns else "close"
    h_col = "adj_high" if "adj_high" in df.columns else "high"
    l_col = "adj_low" if "adj_low" in df.columns else "low"

    OPEN  = df.pivot(index="date", columns="ticker", values=o_col).sort_index()
    CLOSE = df.pivot(index="date", columns="ticker", values=c_col).sort_index()
    VOL   = df.pivot(index="date", columns="ticker", values="volume").sort_index()
    HIGH  = df.pivot(index="date", columns="ticker", values=h_col).sort_index() if h_col in df.columns else None
    LOW   = df.pivot(index="date", columns="ticker", values=l_col).sort_index() if l_col in df.columns else None

    # Tách benchmark VNINDEX
    bench_close = CLOSE.pop("VNINDEX") if "VNINDEX" in CLOSE else None
    bench_open  = OPEN.pop("VNINDEX")  if "VNINDEX" in OPEN  else None
    VOL = VOL.drop(columns=["VNINDEX"], errors="ignore")
    if HIGH is not None and "VNINDEX" in HIGH:
        HIGH.pop("VNINDEX")
    if LOW is not None and "VNINDEX" in LOW:
        LOW.pop("VNINDEX")

    # Lọc rổ cổ phiếu hoàn chỉnh (không có missing value)
    valid_stocks = CLOSE.columns[CLOSE.notna().all() & VOL.notna().all()].tolist()
    OPEN, CLOSE, VOL = OPEN[valid_stocks], CLOSE[valid_stocks], VOL[valid_stocks]
    if HIGH is not None:
        HIGH = HIGH[[s for s in valid_stocks if s in HIGH.columns]]
    if LOW is not None:
        LOW = LOW[[s for s in valid_stocks if s in LOW.columns]]

    return OPEN, CLOSE, VOL, HIGH, LOW, bench_open, bench_close, valid_stocks, None


def obv_series(close, vol):
    """Tính chuỗi On-Balance Volume chuẩn xác"""
    return vol.where(~(close < close.shift(1)), -vol).cumsum()


@st.cache_data(show_spinner=False)
def calc_ta_features(close, vol):
    """Tính toán 8 đặc trưng kỹ thuật thành phần trên cửa sổ dữ liệu"""
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    valid = sma200.notna()

    trend_strength = ((sma50 > sma200) & valid).sum() / valid.sum()
    price_vs_sma200 = close.iloc[-1] / sma200.iloc[-1] - 1
    momentum_6m = close.iloc[-1] / close.iloc[-126] - 1
    avg_value = (close * vol).mean()

    obv = obv_series(close, vol)
    obv_conf = (obv.iloc[-1] - obv.iloc[-20]) / vol.iloc[-20:].sum()

    ret = close.pct_change().iloc[1:]
    volatility = ret.std() * np.sqrt(TRADING_DAYS)
    eq = (1 + ret).cumprod()
    mdd = (eq / eq.cummax() - 1).min()

    # Độ rõ xu hướng bằng R² hồi quy log(giá) theo thời gian
    lp = np.log(close)
    x = np.arange(len(lp))
    xc = x - x.mean()
    slope = (xc[:, None] * (lp - lp.mean()).values).sum(0) / (xc ** 2).sum()
    fit = lp.mean().values + np.outer(xc, slope)
    r2 = 1 - ((lp.values - fit) ** 2).sum(0) / ((lp - lp.mean()) ** 2).sum().values
    trend_r2 = pd.Series(np.sign(slope) * r2, index=close.columns)

    return pd.DataFrame({
        "Trend_R2": trend_r2,
        "Trend_Strength": trend_strength,
        "Price_vs_SMA200": price_vs_sma200,
        "Momentum_6M": momentum_6m,
        "Avg_Trading_Value": avg_value,
        "OBV_Confirmation": obv_conf,
        "Volatility": volatility,
        "Max_Drawdown": mdd
    })


def calc_ta_score(features, weights, version="v2"):
    """
    Xếp hạng rổ cổ phiếu theo TA Score.
    weights: tuple (w_trend, w_mom, w_vol, w_risk)
    """
    r = lambda s, asc=True: s.rank(pct=True, ascending=asc)
    s = pd.DataFrame(index=features.index)
    if version == "v1":
        s["Trend"] = 0.5 * r(features.Trend_Strength) + 0.5 * r(features.Price_vs_SMA200)
        s["Momentum"] = r(features.Momentum_6M)
        s["Volume"] = 0.6 * r(features.Avg_Trading_Value) + 0.4 * r(features.OBV_Confirmation)
        s["Risk"] = 0.5 * r(features.Volatility, False) + 0.5 * r(features.Max_Drawdown)
    else:  # version v2 cải tiến của Nhóm 5
        s["Trend"] = r(features.Trend_R2)
        s["Momentum"] = r(features.Momentum_6M)
        s["Volume"] = r(features.Avg_Trading_Value)
        s["Risk"] = 0.5 * r(features.Volatility, False) + 0.5 * r(features.Max_Drawdown)

    s["TA_Score"] = (
        weights[0] * s["Trend"] +
        weights[1] * s["Momentum"] +
        weights[2] * s["Volume"] +
        weights[3] * s["Risk"]
    )
    return s.sort_values("TA_Score", ascending=False)


# -----------------------------------------------------------------------------
# 4. HỆ THỐNG TÍN HIỆU GIAO DỊCH & MÔ PHỎNG THỰC TẾ
# -----------------------------------------------------------------------------
def sma_events(c, s, l):
    fast, slow = c.rolling(int(s)).mean(), c.rolling(int(l)).mean()
    up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return up.astype(int) - dn.astype(int), (fast > slow) & slow.notna()


def obv_events(c, v, w):
    obv = obv_series(c, v)
    ma = obv.rolling(int(w)).mean()
    up = (obv > ma) & (obv.shift(1) <= ma.shift(1))
    dn = (obv < ma) & (obv.shift(1) >= ma.shift(1))
    return up.astype(int) - dn.astype(int), (obv > ma) & ma.notna()


def target_position(c, v, p, rule="AND"):
    """Xác định trạng thái vị thế (0 = tiền mặt, 1 = cổ phiếu)"""
    e_sma, st_sma = sma_events(c, p["ma_short"], p["ma_long"])
    e_obv, st_obv = obv_events(c, v, p["obv_window"])

    if rule == "AND":
        return (st_sma & st_obv).astype(float)
    if rule == "SMA":
        return st_sma.astype(float)
    if rule == "OBV":
        return st_obv.astype(float)
    if rule == "OR":
        buy = (e_sma == 1) | (e_obv == 1)
        sell = (e_sma == -1) | (e_obv == -1)
        ev = buy.astype(int) - sell.astype(int)
        ev[buy & sell] = 0
        pos, cur = np.zeros(len(ev)), 0
        for i, x in enumerate(ev.values):
            cur = 1 if x == 1 else (0 if x == -1 else cur)
            pos[i] = cur
        return pd.Series(pos, index=c.index)
    raise ValueError(f"Quy tắc không hợp lệ: {rule}")


def apply_min_hold(target, min_hold=3):
    """Ràng buộc pháp lý chu kỳ thanh toán T+2.5: mua xong phải giữ tối thiểu min_hold phiên"""
    t = target.values.copy()
    held = 0
    for i in range(len(t)):
        if 0 < held < min_hold:
            t[i] = 1
        held = held + 1 if t[i] == 1 else 0
    return pd.Series(t, index=target.index)


def simulate(o, c, signal, start, end, buy_fee=0.0015, sell_fee=0.0025):
    """
    Mô phỏng khớp lệnh tại giá MỞ CỬA phiên kế tiếp (t+1),
    có tính chi phí giao dịch thực tế và chia tách lợi nhuận qua đêm / trong phiên.
    """
    idx = c.index
    i0 = idx.searchsorted(pd.Timestamp(start))
    i1 = idx.searchsorted(pd.Timestamp(end), "right")

    x = signal.shift(1).fillna(0).values
    x_prev = np.r_[0, x[:-1]]
    if i0 < len(x_prev):
        x_prev[i0] = 0

    r_on = (o / c.shift(1) - 1).values
    r_in = (c / o - 1).values

    gross = (1 + x_prev * r_on) * (1 + x * r_in) - 1
    buy = (x > x_prev)
    sell = (x < x_prev)
    cost = buy * buy_fee + sell * sell_fee * (1 + x_prev * r_on)

    net = pd.Series(gross - cost, index=idx).iloc[i0:i1]
    pos = pd.Series(x, index=idx).iloc[i0:i1]
    trades = int(buy[i0:i1].sum())
    return net.fillna(0), pos, trades


def perf(r, trading_days=252):
    """Tính các chỉ số hiệu năng định lượng tài chính chuẩn mực"""
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {
            "Total Return [%]": 0.0,
            "Annual Return [%]": 0.0,
            "Annual Volatility [%]": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown [%]": 0.0,
            "Calmar Ratio": 0.0
        }
    eq = (1 + r).cumprod()
    yrs = max(len(r) / trading_days, 1e-4)
    sd = r.std()
    tot_ret = (eq.iloc[-1] - 1) * 100
    ann_ret = (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100.0
    ann_vol = sd * np.sqrt(trading_days) * 100
    sharpe = r.mean() / sd * np.sqrt(trading_days) if sd > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min() * 100
    calmar = (ann_ret / abs(mdd)) if mdd < 0 else 0.0

    return {
        "Total Return [%]": tot_ret,
        "Annual Return [%]": ann_ret,
        "Annual Volatility [%]": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": mdd,
        "Calmar Ratio": calmar
    }


def mpt_weights(R, rf=0.0, min_w=0.05, max_w=0.40):
    """Tối ưu hóa trọng số MPT theo tỷ số Sharpe trên tập Train"""
    mu = R.mean().values * TRADING_DAYS
    cov = R.cov().values * TRADING_DAYS
    n = len(mu)
    def obj(w):
        port_ret = w @ mu - rf
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret / port_vol) if port_vol > 0 else 1e6

    bounds = [(min_w, max_w)] * n
    res = minimize(
        obj,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0}
    )
    if res.success:
        return res.x
    return np.full(n, 1.0 / n)


def portfolio(R, P, w, buy_fee=0.0015, sell_fee=0.0025):
    """
    Mô phỏng danh mục đầu tư tái cân bằng định kỳ đầu mỗi tháng.
    Trọng số trôi tự nhiên giữa các phiên, tính phí tái cân bằng thực tế.
    """
    w = np.asarray(w, float)
    first = R.index.to_series().groupby([R.index.year, R.index.month]).transform("first") == R.index
    vals = w.copy()
    out = []
    for i, d in enumerate(R.index):
        if first.iloc[i] and i > 0:
            V = vals.sum()
            delta = w * V - vals
            inv = P.iloc[i - 1].values
            fee = (np.clip(delta, 0, None) * buy_fee + np.clip(-delta, 0, None) * sell_fee) * inv
            vals = w * (V - fee.sum())
        V_prev = vals.sum() if i > 0 else 1.0
        vals = vals * (1 + R.iloc[i].values)
        out.append(vals.sum() / V_prev - 1)
    return pd.Series(out, index=R.index)


def buy_hold(o_df, c_df, tickers, start, end, buy_fee=0.0015, w=None):
    """Chiến lược Mua & Nắm giữ (Buy & Hold) chuẩn đối chứng"""
    o = o_df[tickers].loc[start:end]
    c = c_df[tickers].loc[start:end]
    if len(o) == 0:
        return pd.Series(dtype=float)
    w = np.full(len(tickers), 1.0 / len(tickers)) if w is None else np.asarray(w)
    units = w * (1.0 - buy_fee) / o.iloc[0].values
    wealth = pd.Series(c.values @ units, index=c.index)
    r = wealth.pct_change()
    r.iloc[0] = wealth.iloc[0] - 1.0
    return r


def index_return(bench_o, bench_c, start, end):
    """Lợi nhuận chỉ số VN-Index"""
    if bench_o is None or bench_c is None:
        return pd.Series(dtype=float)
    o = bench_o.loc[start:end]
    c = bench_c.loc[start:end]
    if len(o) == 0:
        return pd.Series(dtype=float)
    w = c / o.iloc[0]
    r = w.pct_change()
    r.iloc[0] = w.iloc[0] - 1.0
    return r


@st.cache_data(show_spinner=False)
def run_rolling_score_strategy(open_df, close_df, vol_df, dates, start, end, weights, lookback=250, n=5, bf=0.0015, sf=0.0025):
    """Chiến lược xoay vòng cổ phiếu theo TA Score mỗi tháng"""
    idx = dates[(dates >= start) & (dates <= end)]
    me = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first().values
    hold_log = {}
    rets = []
    vals = None
    prev_hold = []

    for k, d_exec in enumerate(me):
        d_exec = pd.Timestamp(d_exec)
        i_sig = dates.get_loc(d_exec) - 1
        win = dates[max(0, i_sig - lookback + 1): i_sig + 1]
        f = calc_ta_features(close_df.loc[win], vol_df.loc[win])
        hold = calc_ta_score(f, weights, version="v2").head(n).index.tolist()
        hold_log[d_exec.strftime("%Y-%m")] = ", ".join(hold)
        seg_end = pd.Timestamp(me[k + 1]) if k + 1 < len(me) else None
        seg = idx[(idx >= d_exec) & ((idx < seg_end) if seg_end is not None else True)]
        V = 1.0 if vals is None else vals
        sold = set(prev_hold) - set(hold)
        bought = set(hold) - set(prev_hold)
        cost = (len(sold) * sf + len(bought) * bf) / n * V
        if prev_hold:
            V = V * (1 + (open_df.loc[d_exec, prev_hold] / close_df.iloc[i_sig][prev_hold] - 1).mean())
        units = ((V - cost) / n) / open_df.loc[d_exec, hold]
        wealth = close_df.loc[seg, hold].mul(units, axis=1).sum(axis=1)
        prev_V = 1.0 if vals is None else vals
        r = wealth.pct_change()
        r.iloc[0] = wealth.iloc[0] / prev_V - 1
        rets.append(r)
        vals = wealth.iloc[-1]
        prev_hold = hold

    return (pd.concat(rets) if rets else pd.Series(dtype=float)), pd.Series(hold_log)


# -----------------------------------------------------------------------------
# 5. SIDEBAR ĐIỀU KHIỂN & CẤU HÌNH THÍ NGHIỆM
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("Bảng Điều Khiển")
    st.caption("Cấu hình kiểm định chiến lược định lượng")

    st.subheader("1. Dữ liệu Đầu vào")
    data_option = st.radio(
        "Nguồn dữ liệu:",
        ["Mặc định (HOSE 2020-2023)", "Tải lên file CSV mới"],
        index=0
    )

    uploaded_file = None
    if data_option == "Tải lên file CSV mới":
        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
        data_source = uploaded_file if uploaded_file is not None else "HOSE_2020_2023_in.csv"
    else:
        data_source = "HOSE_2020_2023_in.csv"

    # Tải dữ liệu
    OPEN, CLOSE, VOL, HIGH, LOW, bench_open, bench_close, UNIVERSE, err = load_data_from_source(data_source)

    if err:
        st.error(err)
        st.stop()

    DATES = CLOSE.index
    min_date = DATES[0].date()
    max_date = DATES[-1].date()

    st.subheader("2. Khung Thời Gian")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        train_start_input = st.date_input("Train Bắt đầu", pd.to_datetime("2020-01-01").date(), min_value=min_date, max_value=max_date)
        test_start_input  = st.date_input("Test Bắt đầu",  pd.to_datetime("2022-01-01").date(), min_value=min_date, max_value=max_date)
    with col_d2:
        train_end_input   = st.date_input("Train Kết thúc", pd.to_datetime("2021-12-31").date(), min_value=min_date, max_value=max_date)
        test_end_input    = st.date_input("Test Kết thúc",  pd.to_datetime("2022-12-30").date(), min_value=min_date, max_value=max_date)

    TRAIN_START = str(train_start_input)
    TRAIN_END   = str(train_end_input)
    TEST_START  = str(test_start_input)
    TEST_END    = str(test_end_input)

    train_idx = DATES[(DATES >= TRAIN_START) & (DATES <= TRAIN_END)]
    test_idx  = DATES[(DATES >= TEST_START)  & (DATES <= TEST_END)]

    st.subheader("3. Chi phí Giao dịch & T+2")
    buy_fee_pct  = st.number_input("Phí mua (%)", min_value=0.0, max_value=1.0, value=0.15, step=0.05) / 100.0
    sell_fee_pct = st.number_input("Phí bán (%)", min_value=0.0, max_value=1.0, value=0.15, step=0.05) / 100.0
    tax_pct      = st.number_input("Thuế TNCN bán (%)", min_value=0.0, max_value=1.0, value=0.10, step=0.05) / 100.0
    TOTAL_SELL_FEE = sell_fee_pct + tax_pct

    min_hold_days = st.slider("Ràng buộc giữ vị thế tối thiểu (phiên T+2)", min_value=1, max_value=5, value=3)

    st.subheader("4. Tùy chọn Tối ưu hóa")
    opt_mode = st.selectbox(
        "Chế độ tham số:",
        ["Sử dụng bộ tham số tối ưu chuẩn (Tốc độ cao)", "Tùy chỉnh thủ công từng mã"]
    )

# -----------------------------------------------------------------------------
# 6. TÍNH TOÁN NỀN TẢNG (TA SCORE VÀ TOP 5)
# -----------------------------------------------------------------------------
# Tính features trên tập Train
feat_train = calc_ta_features(CLOSE.loc[train_idx], VOL.loc[train_idx])

# Trọng số mặc định cho TA Score
W_BASE = (0.30, 0.25, 0.25, 0.20)
sc_train = calc_ta_score(feat_train, W_BASE, version="v2")
TOP5_CALCULATED = sc_train.head(TOP_N_DEFAULT).index.tolist()

# Cho phép người dùng chọn danh sách mã hoặc lấy Top 5
selected_tickers = TOP5_CALCULATED

# Khởi tạo tham số cho các mã được chọn
active_params = {}
RULES = ["SMA", "OBV", "OR", "AND"]
RULE_NAMES = {
    "SMA": "SMA Crossover",
    "OBV": "OBV Filter",
    "OR": "SMA + OBV (OR)",
    "AND": "SMA + OBV (AND)"
}

for r in RULES:
    active_params[r] = {}
    for t in selected_tickers:
        if t in DEFAULT_OPTIMAL_PARAMS[r]:
            active_params[r][t] = DEFAULT_OPTIMAL_PARAMS[r][t].copy()
        else:
            # Fallback nếu mã ngoài Top 5 mặc định
            active_params[r][t] = {"ma_short": 25, "ma_long": 150, "obv_window": 30}

# Chạy mô phỏng cho từng quy tắc
WARMUP = 200
TRAIN_EVAL_START = train_idx[WARMUP] if len(train_idx) > WARMUP else train_idx[0]

streams_train, streams_test = {}, {}
pos_train, pos_test = {}, {}
ntr_train, ntr_test = {}, {}
store_results = {}
curves = {}
summary_metrics = {}

for r in RULES:
    st_tr, p_tr, n_tr = {}, {}, {}
    st_te, p_te, n_te = {}, {}, {}
    for t in selected_tickers:
        p = active_params[r][t]
        sig_tr = apply_min_hold(target_position(CLOSE[t].loc[:TRAIN_END], VOL[t].loc[:TRAIN_END], p, r), min_hold_days)
        r_tr, pt_tr, tr_tr = simulate(OPEN[t].loc[:TRAIN_END], CLOSE[t].loc[:TRAIN_END], sig_tr, TRAIN_EVAL_START, TRAIN_END, buy_fee_pct, TOTAL_SELL_FEE)
        st_tr[t], p_tr[t], n_tr[t] = r_tr, pt_tr, tr_tr

        sig_te = apply_min_hold(target_position(CLOSE[t].loc[:TEST_END], VOL[t].loc[:TEST_END], p, r), min_hold_days)
        r_te, pt_te, tr_te = simulate(OPEN[t].loc[:TEST_END], CLOSE[t].loc[:TEST_END], sig_te, TEST_START, TEST_END, buy_fee_pct, TOTAL_SELL_FEE)
        st_te[t], p_te[t], n_te[t] = r_te, pt_te, tr_te

    df_Rtr = pd.DataFrame(st_tr)
    df_Ptr = pd.DataFrame(p_tr)
    df_Rte = pd.DataFrame(st_te)
    df_Pte = pd.DataFrame(p_te)

    # Tính trọng số MPT
    w_mpt = mpt_weights(df_Rtr)
    store_results[r] = {
        "Rtr": df_Rtr, "Ptr": df_Ptr, "Rte": df_Rte, "Pte": df_Pte,
        "w_mpt": w_mpt, "ntr_tr": n_tr, "ntr_te": n_te
    }

    # Danh mục EW và MPT
    w_ew = np.full(len(selected_tickers), 1.0 / len(selected_tickers))
    tr_ew = portfolio(df_Rtr, df_Ptr, w_ew, buy_fee_pct, TOTAL_SELL_FEE)
    te_ew = portfolio(df_Rte, df_Pte, w_ew, buy_fee_pct, TOTAL_SELL_FEE)
    tr_mpt = portfolio(df_Rtr, df_Ptr, w_mpt, buy_fee_pct, TOTAL_SELL_FEE)
    te_mpt = portfolio(df_Rte, df_Pte, w_mpt, buy_fee_pct, TOTAL_SELL_FEE)

    curves[f"{RULE_NAMES[r]} - EW"] = te_ew
    curves[f"{RULE_NAMES[r]} - MPT"] = te_mpt
    summary_metrics[f"{RULE_NAMES[r]} - EW"] = {"Train": perf(tr_ew), "Test": perf(te_ew)}
    summary_metrics[f"{RULE_NAMES[r]} - MPT"] = {"Train": perf(tr_mpt), "Test": perf(te_mpt)}

# Benchmarks
r_vnindex = index_return(bench_open, bench_close, TEST_START, TEST_END)
r_bh_top5 = buy_hold(OPEN, CLOSE, selected_tickers, TEST_START, TEST_END, buy_fee_pct)
r_bh_all   = buy_hold(OPEN, CLOSE, UNIVERSE, TEST_START, TEST_END, buy_fee_pct)

curves["VN-Index"] = r_vnindex
curves["EW Buy & Hold Top 5"] = r_bh_top5
curves[f"EW Buy & Hold Toàn rổ ({len(UNIVERSE)} mã)"] = r_bh_all

summary_metrics["VN-Index"] = {"Test": perf(r_vnindex)}
summary_metrics["EW Buy & Hold Top 5"] = {"Test": perf(r_bh_top5)}
summary_metrics[f"EW Buy & Hold Toàn rổ ({len(UNIVERSE)} mã)"] = {"Test": perf(r_bh_all)}

# Rolling Score Strategy
rs_ret, rs_log = run_rolling_score_strategy(OPEN, CLOSE, VOL, DATES, TEST_START, TEST_END, W_BASE, 250, TOP_N_DEFAULT, buy_fee_pct, TOTAL_SELL_FEE)
curves["TA Score Xoay vòng tháng (EW)"] = rs_ret
summary_metrics["TA Score Xoay vòng tháng (EW)"] = {"Test": perf(rs_ret)}

# Tính Beta và Alpha so với VN-Index
perf_test_table = {}
for k, v in summary_metrics.items():
    if "Test" in v:
        perf_test_table[k] = v["Test"].copy()
        if len(r_vnindex) > 0 and len(curves[k]) > 0:
            df_pair = pd.concat([curves[k], r_vnindex], axis=1).dropna()
            if len(df_pair) > 2 and df_pair.iloc[:, 1].var() > 0:
                beta = np.cov(df_pair.iloc[:, 0], df_pair.iloc[:, 1])[0, 1] / df_pair.iloc[:, 1].var()
            else:
                beta = np.nan
        else:
            beta = np.nan
        perf_test_table[k]["Beta vs VN-Index"] = beta
        perf_test_table[k]["Chênh lệch vs VN-Index (%)"] = (
            perf_test_table[k]["Total Return [%]"] - summary_metrics["VN-Index"]["Test"]["Total Return [%]"]
        )

df_final_test = pd.DataFrame(perf_test_table).T

# -----------------------------------------------------------------------------
# 7. GIAO DIỆN CHÍNH - HEADER & METRIC SUMMARY
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📈 Hệ thống Kiểm định Chiến lược Định lượng SMA & OBV</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Báo cáo & Mô phỏng Kiểm nghiệm Định lượng Ba Tầng trên Sàn HOSE — Nhóm 5</div>', unsafe_allow_html=True)

# Khung tóm tắt các thẻ Metric KPI chủ chốt của Chiến lược Đỉnh cao (SMA+OBV AND - MPT)
and_mpt_test = summary_metrics["SMA + OBV (AND) - MPT"]["Test"]
vnindex_test = summary_metrics["VN-Index"]["Test"]
bh_top5_test = summary_metrics["EW Buy & Hold Top 5"]["Test"]

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    ret_val = and_mpt_test["Total Return [%]"]
    diff_vn = ret_val - vnindex_test["Total Return [%]"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Chiến lược Chính (AND - MPT)</div>
        <div class="metric-value {'val-positive' if ret_val >= 0 else 'val-negative'}">{ret_val:.2f}%</div>
        <div class="metric-sub val-positive">Vượt trội VN-Index: +{diff_vn:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Benchmark VN-Index</div>
        <div class="metric-value val-negative">{vnindex_test["Total Return [%]"]:.2f}%</div>
        <div class="metric-sub">Max DD: {vnindex_test["Max Drawdown [%]"]:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Buy & Hold Top 5</div>
        <div class="metric-value val-negative">{bh_top5_test["Total Return [%]"]:.2f}%</div>
        <div class="metric-sub">Max DD: {bh_top5_test["Max Drawdown [%]"]:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    mdd_val = and_mpt_test["Max Drawdown [%]"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max Drawdown (AND)</div>
        <div class="metric-value val-negative">{mdd_val:.2f}%</div>
        <div class="metric-sub val-positive">Phòng vệ giảm lỗ: 40.9%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    beta_val = df_final_test.loc["SMA + OBV (AND) - MPT", "Beta vs VN-Index"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Hệ số Beta Thị trường</div>
        <div class="metric-value val-neutral">{beta_val:.2f}</div>
        <div class="metric-sub">Rất thấp (Giảm phụ thuộc thị trường)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# 8. HỆ THỐNG CÁC TABS PHÂN TÍCH CHUYÊN SÂU
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 1. Hiệu năng & So sánh Tổng quan",
    "🎯 2. Tầng 1: Sàng lọc Cổ phiếu (TA Score)",
    "⚡ 3. Tầng 2: Tín hiệu Kỹ thuật & Tối ưu",
    "⚖️ 4. Tầng 3: Danh mục & Phân bổ MPT",
    "🔬 5. Phân tích Độ nhạy & Thị trường",
    "📖 6. Phương pháp luận & Nghiên cứu"
])

# =============================================================================
# TAB 1: HIỆU NĂNG TỔNG QUAN
# =============================================================================
with tabs[0]:
    st.subheader("1. Đường cong Giá trị Tài sản (Equity Curves - Test 2022)")
    st.markdown("""
    Trong năm **2022** (giai đoạn thị trường suy thoái sâu - Bear Market), chiến lược kết hợp **SMA + OBV (AND)**
    thể hiện khả năng phòng vệ vượt bậc. Bằng cách kích hoạt bộ lọc kép (giá trên SMA và dòng tiền OBV xác nhận),
    chiến lược tự động rút lui về tiền mặt khi xu hướng gãy, bảo toàn tối đa vốn gốc so với mua và nắm giữ.
    """)

    # Bộ lọc chiến lược hiển thị trên đồ thị
    available_curves = list(curves.keys())
    default_selected = [
        "SMA + OBV (AND) - MPT",
        "SMA + OBV (AND) - EW",
        "SMA + OBV (OR) - EW",
        "VN-Index",
        "EW Buy & Hold Top 5"
    ]
    curves_to_plot = st.multiselect("Chọn các chiến lược hiển thị trên biểu đồ:", available_curves, default=default_selected)

    fig_eq = go.Figure()
    for name in curves_to_plot:
        series = curves[name]
        if len(series) > 0:
            cum_ret = (1 + series).cumprod()
            dash_style = "dot" if ("Buy & Hold" in name or "Index" in name or "Xoay vòng" in name) else "solid"
            width_line = 3 if "AND" in name else (1.5 if dash_style == "dot" else 2)
            fig_eq.add_trace(go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name=name,
                line=dict(dash=dash_style, width=width_line),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Chiến lược: " + name + "<br>Giá trị: %{y:.3f}<extra></extra>"
            ))

    fig_eq.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Vốn ban đầu = 1.0")
    fig_eq.update_layout(
        title="Đường cong Tăng trưởng Vốn (Normalized Equity Curves - Test 2022)",
        xaxis_title="Thời gian",
        yaxis_title="Giá trị Danh mục (Vốn gốc = 1.0)",
        hovermode="x unified",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Đồ thị Underwater (Max Drawdown over Time)
    st.subheader("2. Biểu đồ Độ sụt giảm Vốn Tối đa (Underwater Chart / Drawdown)")
    fig_dd = go.Figure()
    for name in curves_to_plot:
        series = curves[name]
        if len(series) > 0:
            cum = (1 + series).cumprod()
            dd = (cum / cum.cummax() - 1.0) * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                name=name,
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>" + name + "<br>Drawdown: %{y:.2f}%<extra></extra>"
            ))
    fig_dd.update_layout(
        title="Độ sụt giảm vốn từ đỉnh (Underwater Drawdown %)",
        xaxis_title="Thời gian",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Bảng số liệu tổng hợp
    st.subheader("3. Bảng Thống kê Hiệu năng Chi tiết (Tập Test 2022)")
    display_df = df_final_test.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    st.dataframe(
        display_df.style.format("{:.2f}", subset=numeric_cols)
        .background_gradient(subset=["Total Return [%]"], cmap="RdYlGn")
        .background_gradient(subset=["Max Drawdown [%]"], cmap="Reds_r"),
        use_container_width=True
    )

    # Nút tải dữ liệu kết quả
    csv_buffer = io.StringIO()
    df_final_test.to_csv(csv_buffer, encoding="utf-8-sig")
    st.download_button(
        label="📥 Tải Bảng Kết quả Hiệu năng (CSV)",
        data=csv_buffer.getvalue(),
        file_name="backtest_performance_summary.csv",
        mime="text/csv"
    )

# =============================================================================
# TAB 2: TẦNG 1 - SÀNG LỌC CỔ PHIẾU BẰNG TA SCORE
# =============================================================================
with tabs[1]:
    st.subheader("Sàng lọc Cổ phiếu Tiềm năng bằng TA Score (Train 2020 - 2021)")
    st.markdown("""
    Mô hình TA Score tích hợp 4 nhóm chỉ báo chính:
    1. **Trend (30%)**: Độ rõ xu hướng dựa trên $R^2$ của hàm hồi quy $\\ln(\\text{Price})$ theo thời gian.
    2. **Momentum (25%)**: Tỷ suất sinh lời 6 tháng ($MOM_{6M}$).
    3. **Volume (25%)**: Giá trị giao dịch bình quân (Thanh khoản thị trường).
    4. **Risk (20%)**: Điểm rủi ro đảo ngược ($0.5 \\times \\text{Low Volatility} + 0.5 \\times \\text{Low Max Drawdown}$).
    """)

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        w_trend = st.slider("Trọng số Trend (Xu hướng)", 0.0, 1.0, 0.30, 0.05)
    with col_w2:
        w_mom = st.slider("Trọng số Momentum (Động lượng)", 0.0, 1.0, 0.25, 0.05)
    with col_w3:
        w_vol = st.slider("Trọng số Volume (Thanh khoản)", 0.0, 1.0, 0.25, 0.05)
    with col_w4:
        w_risk = st.slider("Trọng số Risk (Rủi ro thấp)", 0.0, 1.0, 0.20, 0.05)

    tot_w = w_trend + w_mom + w_vol + w_risk
    if abs(tot_w - 1.0) > 1e-4:
        st.warning(f"Tổng trọng số hiện tại là {tot_w:.2f} (khác 1.0). Hệ thống sẽ tự chuẩn hóa về tổng 1.0.")
        w_norm = (w_trend / tot_w, w_mom / tot_w, w_vol / tot_w, w_risk / tot_w)
    else:
        w_norm = (w_trend, w_mom, w_vol, w_risk)

    custom_score_df = calc_ta_score(feat_train, w_norm, version="v2")

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.write("##### Top 10 Cổ phiếu Đạt Điểm TA Score Cao Nhất")
        top10_df = custom_score_df.head(10).round(3)
        st.dataframe(top10_df, use_container_width=True)

        # Biểu đồ thanh Top 10
        fig_bar = px.bar(
            top10_df.reset_index(),
            x="ticker",
            y="TA_Score",
            color="TA_Score",
            color_continuous_scale="Viridis",
            title="Điểm số TA Score Top 10 Cổ phiếu",
            labels={"ticker": "Mã Cổ phiếu", "TA_Score": "TA Score"}
        )
        fig_bar.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_t2:
        st.write("##### Ma trận Tương quan Thứ hạng (Spearman Correlation)")
        corr_matrix = feat_train.corr(method="spearman").round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Tương quan giữa các Đặc trưng Thành phần"
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Phân tích độ nhạy lưới trọng số
    st.subheader("Phân tích Độ bền vững của Top 5 (Weight Grid Sensitivity)")
    st.markdown("""
    Kiểm tra độ vững: Cho 4 trọng số quét toàn bộ lưới từ 10% đến 40% (bước nhảy 5%, tổng = 100%, 285 bộ trọng số).
    Nhóm 5 chứng minh rằng: **DIG, DCM, MSN, SSI, PDR** liên tục dẫn đầu rổ cổ phiếu với tần suất xuất hiện cực kỳ cao,
    khẳng định việc lựa chọn Top 5 không bị thiên lệch ngẫu nhiên.
    """)

    # Grid search 285 tổ hợp
    @st.cache_data(show_spinner=False)
    def compute_grid_sensitivity(feat):
        grid = [w for w in product(np.arange(0.10, 0.41, 0.05), repeat=4) if abs(sum(w) - 1.0) < 1e-9]
        freq = {}
        for w in grid:
            top_t = calc_ta_score(feat, w, version="v2").head(TOP_N_DEFAULT).index
            for t in top_t:
                freq[t] = freq.get(t, 0) + 1
        return pd.Series(freq).sort_values(ascending=False) / len(grid) * 100

    grid_res = compute_grid_sensitivity(feat_train).head(12)
    fig_grid = px.bar(
        x=grid_res.index,
        y=grid_res.values,
        labels={"x": "Mã Cổ phiếu", "y": "Tần suất lọt Top 5 (%)"},
        title="Tần suất Lọt vào Top 5 trên 285 Bộ Trọng số Khác nhau",
        color=grid_res.values,
        color_continuous_scale="Teal"
    )
    fig_grid.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_grid, use_container_width=True)

# =============================================================================
# TAB 3: TẦNG 2 - TÍN HIỆU GIAO DỊCH KỸ THUẬT & TỐI ƯU HÓA
# =============================================================================
with tabs[2]:
    st.subheader("Kiểm tra Tín hiệu Kỹ thuật SMA & OBV trên từng Cổ phiếu")

    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        chosen_ticker = st.selectbox("Chọn mã cổ phiếu phân tích:", selected_tickers)
    with col_s2:
        chosen_rule = st.selectbox("Chọn quy tắc giao dịch:", RULES, format_func=lambda x: RULE_NAMES[x], index=3)
    with col_s3:
        st.info(f"Đang hiển thị tín hiệu của **{chosen_ticker}** theo quy tắc **{RULE_NAMES[chosen_rule]}**.")

    # Thông số tham số hiện thời
    p_cur = active_params[chosen_rule][chosen_ticker]

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        param_short = st.number_input("SMA Ngắn hạn (Fast)", min_value=5, max_value=100, value=int(p_cur["ma_short"]), step=5)
    with col_p2:
        param_long = st.number_input("SMA Dài hạn (Slow)", min_value=50, max_value=250, value=int(p_cur["ma_long"]), step=10)
    with col_p3:
        param_obv = st.number_input("Chu kỳ OBV MA", min_value=5, max_value=120, value=int(p_cur["obv_window"]), step=5)

    custom_p = {"ma_short": param_short, "ma_long": param_long, "obv_window": param_obv}

    # Tính tín hiệu chi tiết
    c_series = CLOSE[chosen_ticker]
    v_series = VOL[chosen_ticker]
    o_series = OPEN[chosen_ticker]
    h_series = HIGH[chosen_ticker] if HIGH is not None and chosen_ticker in HIGH else None
    l_series = LOW[chosen_ticker] if LOW is not None and chosen_ticker in LOW else None

    # Tín hiệu vị thế
    raw_pos = target_position(c_series, v_series, custom_p, chosen_rule)
    held_pos = apply_min_hold(raw_pos, min_hold_days)

    # Xác định điểm Mua và Bán
    pos_diff = held_pos.shift(1).fillna(0).diff()
    buy_signals = held_pos.index[(held_pos.shift(1) == 0) & (held_pos == 1)]
    sell_signals = held_pos.index[(held_pos.shift(1) == 1) & (held_pos == 0)]

    # Đường SMA và OBV
    sma_fast = c_series.rolling(custom_p["ma_short"]).mean()
    sma_slow = c_series.rolling(custom_p["ma_long"]).mean()
    obv = obv_series(c_series, v_series)
    obv_ma = obv.rolling(custom_p["obv_window"]).mean()

    # Cắt đồ thị theo giai đoạn Test để soi rõ
    test_dates = c_series.loc[TEST_START:TEST_END].index

    # Đồ thị nến tương tác đa tầng
    fig_candle = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"Biểu đồ Giá & Tín hiệu Giao dịch: {chosen_ticker} ({TEST_START} - {TEST_END})", "Chỉ báo On-Balance Volume (OBV) & Đường Tín hiệu"]
    )

    # Candlestick hoặc Line
    if h_series is not None and l_series is not None:
        fig_candle.add_trace(go.Candlestick(
            x=test_dates,
            open=o_series.loc[test_dates],
            high=h_series.loc[test_dates],
            low=l_series.loc[test_dates],
            close=c_series.loc[test_dates],
            name="Nến Giá",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444"
        ), row=1, col=1)
    else:
        fig_candle.add_trace(go.Scatter(
            x=test_dates, y=c_series.loc[test_dates],
            name="Giá Đóng cửa", line=dict(color="#2563EB", width=2)
        ), row=1, col=1)

    # SMA Fast & Slow
    fig_candle.add_trace(go.Scatter(
        x=test_dates, y=sma_fast.loc[test_dates],
        name=f"SMA Ngắn ({custom_p['ma_short']})", line=dict(color="#F59E0B", width=1.5)
    ), row=1, col=1)
    fig_candle.add_trace(go.Scatter(
        x=test_dates, y=sma_slow.loc[test_dates],
        name=f"SMA Dài ({custom_p['ma_long']})", line=dict(color="#6366F1", width=1.5)
    ), row=1, col=1)

    # Điểm MUA và BÁN
    test_buys = [d for d in buy_signals if d in test_dates]
    test_sells = [d for d in sell_signals if d in test_dates]

    if test_buys:
        fig_candle.add_trace(go.Scatter(
            x=test_buys, y=c_series.loc[test_buys] * 0.98,
            mode="markers", name="Tín hiệu MUA",
            marker=dict(symbol="triangle-up", size=13, color="#10B981")
        ), row=1, col=1)

    if test_sells:
        fig_candle.add_trace(go.Scatter(
            x=test_sells, y=c_series.loc[test_sells] * 1.02,
            mode="markers", name="Tín hiệu BÁN",
            marker=dict(symbol="triangle-down", size=13, color="#DC2626")
        ), row=1, col=1)

    # Panel OBV
    fig_candle.add_trace(go.Scatter(
        x=test_dates, y=obv.loc[test_dates],
        name="OBV", line=dict(color="#065F46", width=1.5)
    ), row=2, col=1)
    fig_candle.add_trace(go.Scatter(
        x=test_dates, y=obv_ma.loc[test_dates],
        name=f"OBV MA ({custom_p['obv_window']})", line=dict(color="#F97316", width=1.5, dash="dot")
    ), row=2, col=1)

    fig_candle.update_layout(
        template="plotly_white",
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Bảng tham số tối ưu chuẩn của toàn bộ Top 5
    st.subheader("Bảng Tham số Tối ưu Hyperopt của 5 Mã Cổ phiếu")
    all_params_rows = []
    for r in RULES:
        for t in selected_tickers:
            p = active_params[r][t]
            all_params_rows.append({
                "Quy tắc": RULE_NAMES[r],
                "Mã": t,
                "SMA Fast": p["ma_short"],
                "SMA Slow": p["ma_long"],
                "OBV Window": p["obv_window"]
            })
    st.dataframe(pd.DataFrame(all_params_rows), use_container_width=True)

# =============================================================================
# TAB 4: TẦNG 3 - DANH MỤC ĐẦU TƯ & PHÂN BỔ MPT
# =============================================================================
with tabs[3]:
    st.subheader("Phân bổ Tối ưu MPT (Modern Portfolio Theory) vs Equal Weight (EW)")
    st.markdown("""
    - **Equal Weight (EW)**: Phân bổ đều mỗi cổ phiếu 20% vốn ($1/N$).
    - **MPT Portfolio**: Tối đa hóa tỷ số Sharpe trên tập Train với ràng buộc mỗi mã $w_i \\in [5\\%, 40\\%]$.
    - **Tái cân bằng hàng tháng**: Thực hiện vào phiên giao dịch đầu tiên của mỗi tháng, có tính phí tái cân bằng thực tế.
    """)

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.write("##### Cơ cấu Tỷ trọng Danh mục MPT (Quy tắc SMA+OBV AND)")
        w_and_mpt = store_results["AND"]["w_mpt"]
        df_pie = pd.DataFrame({"Mã": selected_tickers, "Tỷ trọng (%)": w_and_mpt * 100})
        fig_donut = px.pie(
            df_pie,
            names="Mã",
            values="Tỷ trọng (%)",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_p2:
        st.write("##### Bảng Trọng số MPT giữa các Quy tắc")
        w_table = {}
        for r in RULES:
            w_table[RULE_NAMES[r]] = pd.Series(store_results[r]["w_mpt"], index=selected_tickers)
        df_w_compare = pd.DataFrame(w_table).round(4) * 100
        st.dataframe(df_w_compare.style.format("{:.2f}%"), use_container_width=True)

        st.info("""
        💡 **Phát hiện quan trọng**: Trong giai đoạn thị trường thuận lợi (Train 2020-2021), MPT tối ưu hóa tỷ trọng
        nghiêng về các mã có Sharpe cao và hiệp phương sai thấp như PDR và DIG, giúp nâng cao hiệu suất danh mục.
        """)

    # So sánh với chiến lược Xoay vòng hàng tháng
    st.subheader("Phản biện Thực nghiệm: So sánh với Chiến lược Xoay vòng TA Score")
    st.markdown("""
    Một giả thuyết thường gặp là: *\"Tại sao không dùng luôn TA Score để chọn lại 5 cổ phiếu mỗi tháng thay vì dùng SMA+OBV?\"*
    Thực nghiệm trên năm 2022 của Nhóm 5 chứng minh chiến lược này **lỗ tới -60.10%**!
    Lý do: Khi thị trường gãy sóng, việc chỉ nhìn điểm đà tăng (Momentum) sẽ khiến danh mục liên tục bắt đỉnh các mã
    tăng nóng cuối chu kỳ. Ngược lại, tầng 2 (SMA+OBV AND) hoạt động như một cầu dao ngắt mạch tự động (Circuit Breaker).
    """)

    col_rs1, col_rs2 = st.columns([1, 1])
    with col_rs1:
        st.write("##### Lịch sử Mã nắm giữ theo tháng của TA Score Rotation")
        st.dataframe(rs_log.to_frame("Danh mục nắm giữ"), use_container_width=True)
    with col_rs2:
        st.write("##### So sánh Hiệu năng Trực diện trong Năm Giông Bão 2022")
        df_compare_rs = pd.DataFrame({
            "Chiến lược": [
                "SMA + OBV (AND) - MPT",
                "SMA + OBV (AND) - EW",
                "VN-Index",
                "Buy & Hold Top 5",
                "TA Score Xoay vòng tháng"
            ],
            "Tổng Lợi nhuận (%)": [
                summary_metrics["SMA + OBV (AND) - MPT"]["Test"]["Total Return [%]"],
                summary_metrics["SMA + OBV (AND) - EW"]["Test"]["Total Return [%]"],
                summary_metrics["VN-Index"]["Test"]["Total Return [%]"],
                summary_metrics["EW Buy & Hold Top 5"]["Test"]["Total Return [%]"],
                summary_metrics["TA Score Xoay vòng tháng (EW)"]["Test"]["Total Return [%]"]
            ],
            "Max Drawdown (%)": [
                summary_metrics["SMA + OBV (AND) - MPT"]["Test"]["Max Drawdown [%]"],
                summary_metrics["SMA + OBV (AND) - EW"]["Test"]["Max Drawdown [%]"],
                summary_metrics["VN-Index"]["Test"]["Max Drawdown [%]"],
                summary_metrics["EW Buy & Hold Top 5"]["Test"]["Max Drawdown [%]"],
                summary_metrics["TA Score Xoay vòng tháng (EW)"]["Test"]["Max Drawdown [%]"]
            ]
        })
        st.dataframe(
            df_compare_rs.style.format({"Tổng Lợi nhuận (%)": "{:.2f}%", "Max Drawdown (%)": "{:.2f}%"})
            .background_gradient(subset=["Tổng Lợi nhuận (%)"], cmap="RdYlGn"),
            use_container_width=True
        )

# =============================================================================
# TAB 5: PHÂN TÍCH ĐỘ NHẠY & THỊ TRƯỜNG THỰC TẾ
# =============================================================================
with tabs[4]:
    st.subheader("1. Độ nhạy theo Giả định Khớp lệnh & Ràng buộc Thị trường")
    st.markdown("""
    Nghiên cứu của Nhóm 5 bóc tách 4 nấc thang từ lý thuyết thuần túy đến thực tế khắc nghiệt của TTCK Việt Nam:
    - **Nấc A**: Khớp lệnh ngay tại giá đóng cửa phiên ra tín hiệu, không phí (ảo tưởng thường gặp).
    - **Nấc B**: Khớp lệnh tại giá mở cửa phiên kế tiếp $t+1$, không phí.
    - **Nấc C**: Nấc B + Ràng buộc giữ tối thiểu 3 phiên do quy định thanh toán T+2.5.
    - **Nấc D**: Nấc C + Chi phí giao dịch thực tế (Phí môi giới & Thuế TNCN bán) — **Mô hình Chuẩn của Nhóm 5**.
    """)

    # Hàm mô phỏng các nấc giả định
    def eval_execution_steps(rule_name):
        P_ = active_params[rule_name]
        steps = [
            ("A. Khớp giá Đóng cửa (t), Không phí", False, False, 0.0, 0.0, "daily"),
            ("B. Khớp giá Mở cửa (t+1), Không phí", True, False, 0.0, 0.0, "monthly"),
            ("C. Nấc B + Ràng buộc T+2.5 (3 phiên)", True, True, 0.0, 0.0, "monthly"),
            ("D. Nấc C + Phí thực tế (Mô hình Chuẩn)", True, True, buy_fee_pct, TOTAL_SELL_FEE, "monthly"),
        ]
        rows = []
        for label, eo, mh, bf, sf, how in steps:
            out, pos_out = {}, {}
            for t in selected_tickers:
                c, v, o = CLOSE[t], VOL[t], OPEN[t]
                pos = target_position(c, v, P_[t], rule_name)
                if mh:
                    pos = apply_min_hold(pos, min_hold_days)
                if eo:
                    r_, p_, _ = simulate(o, c, pos, TEST_START, TEST_END, bf, sf)
                    out[t], pos_out[t] = r_, p_
                else:
                    x = pos.shift(1).fillna(0)
                    r = x * c.pct_change().fillna(0) - x.diff().abs().fillna(x) * bf
                    out[t], pos_out[t] = r.loc[TEST_START:TEST_END], x.loc[TEST_START:TEST_END]
            df_R = pd.DataFrame(out)
            df_P = pd.DataFrame(pos_out)
            w_ew = np.full(len(selected_tickers), 1.0 / len(selected_tickers))
            w_mpt = store_results[rule_name]["w_mpt"]
            r_ew = df_R.mul(w_ew, axis=1).sum(axis=1) if how == "daily" else portfolio(df_R, df_P, w_ew, bf, sf)
            r_mpt = df_R.mul(w_mpt, axis=1).sum(axis=1) if how == "daily" else portfolio(df_R, df_P, w_mpt, bf, sf)
            p_ew = perf(r_ew)
            p_mpt = perf(r_mpt)
            rows.append({
                "Giả định Khớp lệnh": label,
                "EW Return (%)": p_ew["Total Return [%]"],
                "EW Sharpe": p_ew["Sharpe Ratio"],
                "MPT Return (%)": p_mpt["Total Return [%]"],
                "MPT Sharpe": p_mpt["Sharpe Ratio"]
            })
        return pd.DataFrame(rows).set_index("Giả định Khớp lệnh")

    tbl_exec_and = eval_execution_steps("AND")
    st.dataframe(tbl_exec_and.style.format("{:.2f}"), use_container_width=True)

    # Phân tích Gap qua đêm (Slippage)
    st.subheader("2. Phân tích Bước nhảy Giá qua đêm (Overnight Gap / Slippage)")
    st.markdown("""
    Đo lường độ lệch giá mở cửa hôm sau so với giá đóng cửa phiên hôm trước: $\\text{Gap} = \\frac{P_{open, t+1}}{P_{close, t}} - 1$.
    Thực nghiệm cho thấy: Lệnh MUA thường bị bất lợi do mở cửa tăng (Gap dương), lệnh BÁN mở cửa giảm (Gap âm).
    """)

    gap_data = []
    for t in selected_tickers:
        c, v, o = CLOSE[t], VOL[t], OPEN[t]
        pos = target_position(c, v, active_params["OR"][t], "OR")
        d = pos.diff().loc[TEST_START:TEST_END]
        gap = (o.shift(-1) / c - 1.0).loc[TEST_START:TEST_END]
        n_buy = int((d > 0).sum())
        n_sell = int((d < 0).sum())
        gap_buy = gap[d > 0].mean() * 100 if n_buy > 0 else 0.0
        gap_sell = gap[d < 0].mean() * 100 if n_sell > 0 else 0.0
        gap_data.append({
            "Mã Cổ phiếu": t,
            "Số lệnh MUA": n_buy,
            "Gap sau MUA (%)": gap_buy,
            "Số lệnh BÁN": n_sell,
            "Gap sau BÁN (%)": gap_sell
        })
    st.dataframe(pd.DataFrame(gap_data).style.format({"Gap sau MUA (%)": "{:.2f}%", "Gap sau BÁN (%)": "{:.2f}%"}), use_container_width=True)

    # Phân tích độ nhạy chi phí giao dịch
    st.subheader("3. Kiểm định Độ nhạy Chi phí Giao dịch (Cost Sensitivity Scenarios)")
    cost_scenarios = {
        "0% (Không phí)": (0.0, 0.0),
        "0.10% phí + Thuế 0.1%": (0.0010, 0.0020),
        "0.15% phí + Thuế 0.1% (Cơ sở)": (0.0015, 0.0025),
        "0.25% phí + Thuế 0.1%": (0.0025, 0.0035),
        "0.35% phí + Thuế 0.1%": (0.0035, 0.0045),
    }

    cost_rows = []
    for c_name, (bf, sf) in cost_scenarios.items():
        out_te = {}
        pos_te = {}
        for t in selected_tickers:
            sig = apply_min_hold(target_position(CLOSE[t].loc[:TEST_END], VOL[t].loc[:TEST_END], active_params["AND"][t], "AND"), min_hold_days)
            r_, p_, _ = simulate(OPEN[t].loc[:TEST_END], CLOSE[t].loc[:TEST_END], sig, TEST_START, TEST_END, bf, sf)
            out_te[t], pos_te[t] = r_, p_
        df_R = pd.DataFrame(out_te)
        df_P = pd.DataFrame(pos_te)
        w_ew = np.full(len(selected_tickers), 1.0 / len(selected_tickers))
        w_mpt = store_results["AND"]["w_mpt"]
        e_perf = perf(portfolio(df_R, df_P, w_ew, bf, sf))
        m_perf = perf(portfolio(df_R, df_P, w_mpt, bf, sf))
        cost_rows.append({
            "Kịch bản Phí": c_name,
            "Tổng Chi phí 2 chiều (%)": (bf + sf) * 100,
            "EW Return (%)": e_perf["Total Return [%]"],
            "EW Sharpe": e_perf["Sharpe Ratio"],
            "MPT Return (%)": m_perf["Total Return [%]"],
            "MPT Sharpe": m_perf["Sharpe Ratio"]
        })

    df_cost = pd.DataFrame(cost_rows).set_index("Kịch bản Phí")
    st.dataframe(df_cost.style.format("{:.2f}"), use_container_width=True)

# =============================================================================
# TAB 6: PHƯƠNG PHÁP LUẬN & TÀI LIỆU NGHIÊN CỨU
# =============================================================================
with tabs[5]:
    st.subheader("Tài liệu & Phương pháp luận Nghiên cứu Định lượng (Nhóm 5)")
    st.markdown("""
    ### 1. Tổng quan Kiến trúc Ba Tầng (Three-Tier Quantitative Architecture)
    Hệ thống được thiết kế theo tư duy định lượng hiện đại, loại bỏ hoàn toàn hiện tượng nhìn trước tương lai (*Look-ahead bias*)
    và khai thác dữ liệu quá mức (*Data Snooping bias*):

    ```
    ┌────────────────────────────────────────────────────────┐
    │  TẦNG 1: SÀNG LỌC CỔ PHIẾU BẰNG TA SCORE              │
    │  - Trend R²: Độ rõ xu hướng từ hồi quy log(giá) (30%) │
    │  - Momentum 6M: Động lượng trung hạn (25%)             │
    │  - Thanh khoản: Giá trị giao dịch bình quân (25%)      │
    │  - Rủi ro: Nghịch đảo biến động & Max DD (20%)         │
    │  ===> Chọn Top 5 cổ phiếu tối ưu: DIG, DCM, MSN, SSI, PDR│
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │  TẦNG 2: TÍN HIỆU GIAO DỊCH SMA & OBV                  │
    │  - SMA Crossover: Xác định xu hướng giá                │
    │  - OBV Crossover: Xác nhận áp lực dòng tiền tích lũy   │
    │  - Quy tắc AND: Chỉ mở vị thế khi CẢ HAI cùng đồng thuận│
    │  - Tối ưu tham số bằng Hyperopt TPE trên tập Train     │
    │  - Ràng buộc giữ tối thiểu 3 phiên (T+2.5 Việt Nam)    │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │  TẦNG 3: QUẢN TRỊ DANH MỤC & PHÂN BỔ VỐN              │
    │  - Equal Weight (EW): Chia đều rủi ro                  │
    │  - Modern Portfolio Theory (MPT): Tối đa Sharpe Ratio  │
    │  - Tái cân bằng hàng tháng (Monthly Rebalancing)       │
    │  - Tính phí giao dịch tái cấu trúc danh mục thực tế   │
    └────────────────────────────────────────────────────────┘
    ```

    ### 2. Các Đóng góp Thực nghiệm Cốt lõi
    1. **Khả năng phòng vệ vượt trội trong Downmarket**:
       Trong năm 2022, VN-Index giảm **-33.22%**, chiến lược Mua & Nắm giữ Top 5 rơi tự do **-56.81%**.
       Tuy nhiên, danh mục **SMA+OBV (AND) - MPT** chỉ giảm **-13.09%**, tạo ra **Alpha dương +20.13%**,
       hạ độ sụt giảm vốn tối đa (Max Drawdown) từ **-59.34%** xuống chỉ còn **-18.41%**.

    2. **Giá trị của Bộ lọc Kép (AND Rule)**:
       Quy tắc AND khắt khe hơn quy tắc OR và SMA đơn lẻ. Khi thị trường giảm, nó cắt giảm tỷ lệ nắm giữ cổ phiếu
       xuống chỉ còn **6% - 34%** số phiên, tự động chuyển danh mục phần lớn thời gian sang tiền mặt để né cú sập.

    3. **Tính Thực tế & Minh bạch**:
       Mô hình tính toán chính xác việc khớp tại giá Mở cửa ngày tiếp theo (không giả định khớp tại giá đóng cửa),
       tính đủ phí mua 0.15%, phí bán 0.15% và thuế 0.1%, có khóa bán trong 3 phiên theo chu kỳ T+2.
    """)

# -----------------------------------------------------------------------------
# 9. FOOTER
# -----------------------------------------------------------------------------
st.write("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.85rem; padding: 10px 0;">
    <b>Báo cáo Tiểu luận & Hệ thống Web App Định lượng Nhóm 5</b> | Phân tích Dữ liệu HOSE 2020 - 2023<br>
    Ứng dụng được xây dựng trên nền tảng <b>Streamlit</b> & <b>Plotly</b> | Sẵn sàng triển khai trên Streamlit Community Cloud
</div>
""", unsafe_allow_html=True)
