import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =========================
# FUNCIONES DE LA ESTRATEGIA
# =========================
def A2(close, window):
    num = (close ** 2).rolling(window).sum()
    den = close.rolling(window).sum().replace(0, np.nan)
    return num / den

def compute_domacd(close, fast, slow, rezago):
    dom = A2(close, fast) - A2(close, slow)
    sig = dom.ewm(span=rezago, adjust=False).mean()
    return dom, sig

def crossover(dom, sig):
    diff = dom - sig
    buy = ((diff.shift(1) <= 0) & (diff > 0)).astype(bool)
    sell = ((diff.shift(1) >= 0) & (diff < 0)).astype(bool)
    return buy, sell

def backtest_pnl(data, stake=100.0):
    in_pos = False
    shares = 0.0
    pnls = []
    dates = []

    last_price = None
    last_date = None

    for row in data.itertuples():
        dt = row[0]
        price = row[1]
        buy = row[2]
        sell = row[3]

        last_price = price
        last_date = dt

        if not in_pos and buy:
            shares = stake / price
            in_pos = True

        elif in_pos and sell:
            pnls.append(shares * price - stake)
            dates.append(dt)
            in_pos = False

    if in_pos and last_price is not None:
        pnls.append(shares * last_price - stake)
        dates.append(last_date)

    return pd.Series(pnls, index=dates, name="PNL")

def compute_drawdown(equity):
    peak = equity.cummax()
    dd = equity - peak
    return dd, dd.min()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="DOMACD Analyzer", layout="centered")
st.title("📈 DOMACD Strategy Analyzer")

st.markdown(
    """
    Análisis de una **estrategia long-only basada en DOMACD**  
    con comparación directa contra **Buy & Hold**.
    """
)

# =========================
# CONTROLES
# =========================
ticker = st.text_input("Ticker", value="AAPL")
interval = st.selectbox("Temporalidad", ["1d", "1h"])

st.subheader("⚙️ Parámetros DOMACD")

fast = st.slider("Fast", 5, 30, 12)
slow = st.slider("Slow", 20, 60, 26)
rezago = st.slider("Signal", 5, 20, 9)

if fast >= slow:
    st.error("Fast debe ser menor que Slow")
    st.stop()

run = st.button("Ejecutar análisis")

# =========================
# EJECUCIÓN
# =========================
if run:

    with st.spinner("Descargando datos..."):
        if interval == "1h":
            df = yf.download(
                ticker,
                interval="1h",
                period="2y",
                auto_adjust=False,
                progress=False
            )
        else:
            df = yf.download(
                ticker,
                start="2020-01-01",
                auto_adjust=False,
                progress=False
            )

    if df.empty or "Close" not in df.columns:
        st.error("No se pudieron descargar datos.")
        st.stop()

    close = df["Close"].dropna()

    dom, sig = compute_domacd(close, fast, slow, rezago)
    buy, sell = crossover(dom, sig)

    data = pd.concat(
        [close, buy, sell],
        axis=1,
        keys=["close", "buy", "sell"]
    ).dropna()

    pnl = backtest_pnl(data)

    if pnl.empty:
        st.warning("No hubo trades con estos parámetros.")
        st.stop()

    equity = pnl.cumsum()
    drawdown, max_dd = compute_drawdown(equity)

    stake = 100.0
    roi_strategy = float(pnl.sum() / stake)

    roi_bh = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    try:
        roi_bh = float(roi_bh)
    except Exception:
        roi_bh = 0.0

    # =========================
    # PESTAÑAS
    # =========================
    tab1, tab2, tab3 = st.tabs(["🧠 Conclusión", "📊 Backtest", "📈 Gráficos"])

    # ---------- TAB 1: BACKTEST ----------
    with tab1:
        st.subheader("Resultados numéricos")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROI Estrategia", f"{roi_strategy*100:.2f}%")
        col2.metric("ROI Buy & Hold", f"{roi_bh*100:.2f}%")
        col3.metric("Max Drawdown", f"{max_dd:.2f}")
        col4.metric("Trades", len(pnl))

    # ---------- TAB 2: GRÁFICOS ----------
    with tab2:
        st.subheader("Equity Curve")

        bh_equity = (close / close.iloc[0] - 1) * stake

        fig1, ax1 = plt.subplots()
        ax1.plot(equity.index, equity.values, label="Estrategia")
        ax1.plot(bh_equity.index, bh_equity.values, linestyle="--", label="Buy & Hold")
        ax1.axhline(0, linestyle="--")
        ax1.legend()
        ax1.grid()
        st.pyplot(fig1)

        st.subheader("Drawdown")

        fig2, ax2 = plt.subplots()
        ax2.plot(drawdown.index, drawdown.values, color="red")
        ax2.axhline(0, linestyle="--")
        ax2.grid()
        st.pyplot(fig2)

    # ---------- TAB 3: CONCLUSIÓN ----------
    with tab3:
        st.subheader("Evaluación final")

        if roi_strategy > roi_bh and max_dd > -stake * 0.3:
            st.success(
                "✅ La estrategia **supera a Buy & Hold** y mantiene un drawdown controlado. "
                "Puede ser adecuada para perfiles que priorizan gestión del riesgo."
            )
        elif roi_strategy > roi_bh:
            st.warning(
                "⚠️ La estrategia supera a Buy & Hold, pero con drawdowns elevados. "
                "Revisar parámetros."
            )
        else:
            st.info(
                "ℹ️ Buy & Hold fue más rentable en este período. "
                "La estrategia puede servir para reducir exposición en mercados volátiles."
            )
