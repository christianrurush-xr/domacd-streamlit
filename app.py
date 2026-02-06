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

def compute_domacd(close, fast=12, slow=26, rezago=9):
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
        dt = row[0]      # índice
        price = row[1]   # close
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

    # cerrar posición abierta al final
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
    Analiza una **estrategia long-only basada en DOMACD**  
    y compárala contra **Buy & Hold**.
    """
)

ticker = st.text_input("Ticker", value="AAPL")
interval = st.selectbox("Temporalidad", ["1d", "1h"])

if st.button("Ejecutar análisis"):

    with st.spinner("Descargando datos..."):
        if interval == "1h":
            df = yf.download(
                ticker,
                interval="1h",
                period="1y",
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
        st.error("No se pudieron descargar datos para este ticker.")
        st.stop()

    close = df["Close"].dropna()

    dom, sig = compute_domacd(close)
    buy, sell = crossover(dom, sig)

    data = pd.concat(
        [close, buy, sell],
        axis=1,
        keys=["close", "buy", "sell"]
    ).dropna()

    pnl_series = backtest_pnl(data)

    if pnl_series.empty:
        st.warning("No hubo trades en este período.")
        st.stop()

    equity = pnl_series.cumsum()
    drawdown, max_dd = compute_drawdown(equity)

    # =========================
    # MÉTRICAS
    # =========================
    stake = 100.0
    roi_strategy = pnl_series.sum() / stake
    roi_bh = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

    st.subheader("📊 Resultados")

    col1, col2, col3 = st.columns(3)
    col1.metric("ROI Estrategia", f"{roi_strategy*100:.2f}%")
    col2.metric("ROI Buy & Hold", f"{roi_bh*100:.2f}%")
    col3.metric("Max Drawdown", f"{max_dd:.2f}")

    # =========================
    # CONCLUSIÓN
    # =========================
    st.subheader("🧠 Conclusión")

    if roi_strategy > roi_bh:
        st.success(
            "La estrategia **supera a Buy & Hold** en este período, "
            "logrando mejor control del riesgo."
        )
    else:
        st.info(
            "En este período, **Buy & Hold fue superior**. "
            "La estrategia puede ser útil si priorizas reducción de drawdowns."
        )

    # =========================
    # GRÁFICOS
    # =========================
    st.subheader("📈 PNL acumulado (Equity Curve)")
    fig1, ax1 = plt.subplots()
    ax1.plot(equity.index, equity.values)
    ax1.axhline(0, linestyle="--")
    ax1.set_ylabel("PNL acumulado")
    ax1.grid()
    st.pyplot(fig1)

    st.subheader("📉 Drawdown")
    fig2, ax2 = plt.subplots()
    ax2.plot(drawdown.index, drawdown.values, color="red")
    ax2.axhline(0, linestyle="--")
    ax2.set_ylabel("Drawdown")
    ax2.grid()
    st.pyplot(fig2)
