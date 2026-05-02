"""
Probabilistic Analysis UI — Enhanced interactive dashboard.
Sections: Returns | Statistics | Correlation | Joint Probability | VaR | Monte Carlo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as scipy_stats
from typing import List

from src.features.probabilistic.analyzer import (
    calculate_full_stats,
    calculate_all_var,
    calculate_joint_probabilities,
    run_monte_carlo,
)

# ── Shared styling ────────────────────────────────────────────────────────────
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
)
_COLORS = ["#0078D4", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#FF5722"]


# ── Master renderer ───────────────────────────────────────────────────────────
def render_probabilistic_ui(tickers: List[str], returns: pd.DataFrame):
    """Entry point — renders all six sections of the Probabilistic Analysis dashboard."""
    st.info(
        "💡 **What is Probabilistic Analysis?** "
        "This section studies stock returns through statistics and probability theory. "
        "It quantifies expected gains, risk exposure, joint movement likelihood, "
        "and simulates thousands of future portfolio paths using Monte Carlo methods."
    )
    _render_returns_section(tickers, returns)
    st.markdown("---")
    _render_statistics_section(returns)
    st.markdown("---")
    _render_correlation_section(returns)
    st.markdown("---")
    _render_joint_probability_section(returns)
    st.markdown("---")
    _render_var_section(tickers, returns)
    st.markdown("---")
    _render_monte_carlo_section(returns)


# ── Section 1: Daily Returns ──────────────────────────────────────────────────
def _render_returns_section(tickers: List[str], returns: pd.DataFrame):
    st.subheader("1. 📈 Daily Returns Analysis")

    # Returns line chart
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=returns.index, y=returns[ticker],
            name=ticker, mode="lines",
            line=dict(color=_COLORS[i % len(_COLORS)], width=1.5),
            opacity=0.85,
        ))
    fig.update_layout(
        title="Daily Returns Over Time",
        xaxis_title="Date", yaxis_title="Daily Return",
        yaxis_tickformat=".2%", hovermode="x unified", **_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # Rolling volatility
    with col1:
        fig_vol = go.Figure()
        for i, ticker in enumerate(tickers):
            rv = returns[ticker].rolling(30).std() * np.sqrt(252)
            fig_vol.add_trace(go.Scatter(
                x=rv.index, y=rv, name=ticker,
                mode="lines", line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            ))
        fig_vol.update_layout(
            title="📉 Rolling 30-Day Annualised Volatility",
            xaxis_title="Date", yaxis_title="Volatility (Annual)",
            yaxis_tickformat=".1%", hovermode="x unified", **_LAYOUT,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # Return distribution histogram
    with col2:
        fig_hist = go.Figure()
        for i, ticker in enumerate(tickers):
            fig_hist.add_trace(go.Histogram(
                x=returns[ticker], name=ticker, opacity=0.7,
                marker_color=_COLORS[i % len(_COLORS)],
                nbinsx=60, histnorm="probability density",
            ))
        fig_hist.update_layout(
            title="📊 Return Distribution (Density)",
            xaxis_title="Daily Return", yaxis_title="Density",
            xaxis_tickformat=".2%", barmode="overlay", **_LAYOUT,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ── Section 2: Descriptive Statistics ────────────────────────────────────────
def _render_statistics_section(returns: pd.DataFrame):
    st.subheader("2. 📐 Descriptive Statistics")
    st.markdown(
        "A comprehensive view of return characteristics including "
        "**expected value, risk, skewness** (asymmetry), and **kurtosis** (tail risk)."
    )

    stats_df = calculate_full_stats(returns)
    fmt = {
        "Expected Return (Daily)": "{:.4%}",
        "Std Dev σ (Daily)": "{:.4%}",
        "Variance": "{:.8f}",
        "Skewness": "{:.3f}",
        "Kurtosis (Excess)": "{:.3f}",
        "Annual Sharpe Ratio": "{:.3f}",
        "Max Daily Gain": "{:.3%}",
        "Max Daily Loss": "{:.3%}",
        "Positive Days %": "{:.1f}%",
    }
    styled = (
        stats_df.style.format(fmt)
        .background_gradient(
            subset=["Expected Return (Daily)", "Annual Sharpe Ratio"], cmap="RdYlGn"
        )
        .background_gradient(subset=["Max Daily Loss"], cmap="RdYlGn_r")
    )
    st.dataframe(styled, use_container_width=True)

    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | Meaning |
|---|---|
| **Expected Return** | Average daily profit/loss |
| **Std Dev (σ)** | How much returns deviate from the mean — higher = riskier |
| **Skewness** | Positive = more big gains, Negative = more big losses |
| **Kurtosis** | Fat tails = more extreme events than a normal distribution |
| **Annual Sharpe Ratio** | Return earned per unit of risk (annualised). Higher is better |
| **Positive Days %** | Percentage of trading days with a positive return |
        """)


# ── Section 3: Correlation & Covariance ──────────────────────────────────────
def _render_correlation_section(returns: pd.DataFrame):
    st.subheader("3. 🔗 Correlation & Covariance")
    st.markdown(
        "**Correlation** (−1 to +1) shows how stocks move together. "
        "Values close to +1 mean stocks move in sync, reducing diversification. "
        "**Covariance** is the raw joint-movement measure."
    )

    corr = returns.corr()
    cov = returns.cov()
    col1, col2 = st.columns(2)

    with col1:
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont={"size": 14},
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
        ))
        fig_corr.update_layout(title="Correlation Matrix", **{k: v for k, v in _LAYOUT.items() if k != "hovermode"})
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        fig_cov = go.Figure(go.Heatmap(
            z=cov.values, x=cov.columns.tolist(), y=cov.index.tolist(),
            colorscale="Blues",
            text=[[f"{v:.6f}" for v in row] for row in cov.values],
            texttemplate="%{text}", textfont={"size": 11},
            hovertemplate="<b>%{y} vs %{x}</b><br>Covariance: %{z:.8f}<extra></extra>",
        ))
        fig_cov.update_layout(title="Covariance Matrix", **{k: v for k, v in _LAYOUT.items() if k != "hovermode"})
        st.plotly_chart(fig_cov, use_container_width=True)

    # Diversification notice
    tickers = returns.columns.tolist()
    if len(tickers) >= 2:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        avg_corr = upper.stack().mean()
        if avg_corr > 0.7:
            st.warning(f"⚠️ **High Correlation ({avg_corr:.2f}):** Stocks move very similarly — limited diversification benefit.")
        elif avg_corr < 0.3:
            st.success(f"✅ **Good Diversification ({avg_corr:.2f}):** Low correlation provides strong diversification.")
        else:
            st.info(f"ℹ️ Average pairwise correlation: **{avg_corr:.2f}** — moderate diversification.")


# ── Section 4: Joint Probability ─────────────────────────────────────────────
def _render_joint_probability_section(returns: pd.DataFrame):
    st.subheader("4. 🎯 Joint Probability Analysis")

    col_ctrl, _ = st.columns([2, 1])
    with col_ctrl:
        threshold_pct = st.slider(
            "Return Threshold (%):",
            min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
            help="P(return > threshold). 0% = probability of any gain.",
            key="joint_prob_threshold",
        )
    threshold = threshold_pct / 100.0

    joint_matrix, individual_probs = calculate_joint_probabilities(returns, threshold)
    tickers = returns.columns.tolist()

    all_mask = (returns > threshold).all(axis=1)
    any_mask = (returns > threshold).any(axis=1)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"P(ALL > {threshold_pct:+.1f}%)",
        f"{all_mask.mean()*100:.1f}%",
        help="All stocks beat the threshold on the same day",
    )
    m2.metric(
        f"P(ANY > {threshold_pct:+.1f}%)",
        f"{any_mask.mean()*100:.1f}%",
        help="At least one stock beats the threshold",
    )
    m3.metric(
        "Days All Beat Threshold",
        f"{int(all_mask.sum())} / {len(returns)}",
    )

    st.markdown(f"*Based on **{len(returns)}** historical trading days.*")

    col1, col2 = st.columns(2)

    # Individual probability bar chart
    with col1:
        fig_bar = go.Figure(go.Bar(
            x=tickers,
            y=[individual_probs[t] * 100 for t in tickers],
            marker_color=[_COLORS[i % len(_COLORS)] for i in range(len(tickers))],
            text=[f"{individual_probs[t]*100:.1f}%" for t in tickers],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title=f"Individual P(Return > {threshold_pct:+.1f}%)",
            xaxis_title="Stock", yaxis_title="Probability (%)",
            yaxis_range=[0, 100], **_LAYOUT,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Pairwise joint probability heatmap
    with col2:
        jm = joint_matrix.astype(float) * 100
        fig_joint = go.Figure(go.Heatmap(
            z=jm.values, x=jm.columns.tolist(), y=jm.index.tolist(),
            colorscale="Viridis", zmin=0, zmax=100,
            text=[[f"{v:.1f}%" for v in row] for row in jm.values],
            texttemplate="%{text}", textfont={"size": 13},
            hovertemplate="<b>%{y} & %{x}</b><br>Joint P: %{z:.2f}%<extra></extra>",
        ))
        fig_joint.update_layout(
            title=f"Pairwise Joint P(Both > {threshold_pct:+.1f}%)",
            **{k: v for k, v in _LAYOUT.items() if k != "hovermode"},
        )
        st.plotly_chart(fig_joint, use_container_width=True)

    # Scatter plot for first pair
    if len(tickers) >= 2:
        t1, t2 = tickers[0], tickers[1]
        fig_scatter = go.Figure(go.Scatter(
            x=returns[t1], y=returns[t2], mode="markers",
            marker=dict(color=_COLORS[0], opacity=0.5, size=5),
            name=f"{t1} vs {t2}",
        ))
        # Add quadrant lines
        fig_scatter.add_vline(x=threshold, line_dash="dash", line_color="gray", opacity=0.6)
        fig_scatter.add_hline(y=threshold, line_dash="dash", line_color="gray", opacity=0.6)
        # Quadrant annotation (top-right = both positive)
        fig_scatter.add_annotation(
            x=returns[t1].quantile(0.85), y=returns[t2].quantile(0.85),
            text=f"Both > {threshold_pct:+.1f}%<br>{all_mask.mean()*100:.1f}%",
            showarrow=False, font=dict(color="#4CAF50", size=12),
        )
        fig_scatter.update_layout(
            title=f"Returns Scatter: {t1} vs {t2}",
            xaxis_title=f"{t1} Daily Return", yaxis_title=f"{t2} Daily Return",
            xaxis_tickformat=".2%", yaxis_tickformat=".2%",
            hovermode="closest", **_LAYOUT,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ── Section 5: Value at Risk ──────────────────────────────────────────────────
def _render_var_section(tickers: List[str], returns: pd.DataFrame):
    st.subheader("5. 📉 Value at Risk (VaR)")
    st.markdown(
        "VaR estimates the **maximum expected loss** at a given confidence level. "
        "Two methods are shown: **Gaussian** (assumes normal distribution) and "
        "**Historical** (uses actual past data — no distribution assumption)."
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        selected_ticker = st.selectbox("Select Stock for Distribution Plot:", tickers, key="var_ticker")
    with col_ctrl2:
        loss_threshold = st.slider(
            "Loss Threshold (%):", min_value=-10.0, max_value=-0.5,
            value=-2.0, step=0.5, key="var_threshold",
            help="Calculate the probability of losing more than this amount in one day.",
        )
    loss_val = loss_threshold / 100.0

    r = returns[selected_ticker].dropna()
    mu, sigma = r.mean(), r.std()
    prob_loss = scipy_stats.norm.cdf(loss_val, loc=mu, scale=sigma)

    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"P(Loss > {loss_threshold:.1f}%)", f"{prob_loss*100:.2f}%", help="Gaussian CDF probability")
    m2.metric("Historical Frequency", f"{(r < loss_val).mean()*100:.2f}%", help="Actual days with loss > threshold")
    m3.metric("Gaussian VaR 95%", f"{(mu + scipy_stats.norm.ppf(0.05) * sigma)*100:.2f}%")
    m4.metric("Historical VaR 95%", f"{np.percentile(r, 5)*100:.2f}%")

    # Normal distribution plot
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    y_pdf = scipy_stats.norm.pdf(x_range, loc=mu, scale=sigma)

    fig_dist = go.Figure()
    # Full curve
    fig_dist.add_trace(go.Scatter(
        x=x_range * 100, y=y_pdf, mode="lines", name="Normal Distribution",
        line=dict(color=_COLORS[0], width=2.5),
    ))
    # Shaded loss region
    mask = x_range <= loss_val
    fig_dist.add_trace(go.Scatter(
        x=np.concatenate([x_range[mask], x_range[mask][::-1]]) * 100,
        y=np.concatenate([y_pdf[mask], np.zeros(mask.sum())]),
        fill="toself", fillcolor="rgba(233,30,99,0.35)",
        line=dict(color="rgba(0,0,0,0)"), name=f"Loss > {loss_threshold:.1f}% region",
    ))
    # Threshold line
    fig_dist.add_vline(
        x=loss_val * 100, line_dash="dash", line_color="#E91E63",
        annotation_text=f"{loss_threshold:.1f}%", annotation_position="top right",
    )
    # Mean line
    fig_dist.add_vline(
        x=mu * 100, line_dash="dot", line_color="#4CAF50",
        annotation_text=f"μ = {mu*100:.3f}%", annotation_position="top left",
    )
    fig_dist.update_layout(
        title=f"{selected_ticker} — Return Distribution with Loss Probability",
        xaxis_title="Daily Return (%)", yaxis_title="Probability Density",
        hovermode="x unified", **_LAYOUT,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.info(
        f"**Interpretation:** Assuming returns follow a normal distribution with "
        f"μ = {mu*100:.4f}% and σ = {sigma*100:.4f}%, the probability of **{selected_ticker}** "
        f"losing more than **{abs(loss_threshold):.1f}%** in a single day is "
        f"**{prob_loss*100:.2f}%** (the red shaded area)."
    )

    # Full VaR comparison table
    st.subheader("VaR Comparison Table — All Stocks")
    var_df = calculate_all_var(returns)
    fmt_var = {col: "{:.3%}" for col in var_df.columns}
    st.dataframe(
        var_df.style.format(fmt_var).background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True,
    )
    with st.expander("📖 Reading the VaR table"):
        st.markdown("""
- **Gaussian VaR 95%**: On any given day, there is a 5% chance the stock loses *more* than this amount (assuming normal returns).
- **Historical VaR 95%**: Same idea, but computed directly from past data — no distribution assumption.
- A more negative number = **higher risk**.
        """)


# ── Section 6: Monte Carlo Simulation ────────────────────────────────────────
def _render_monte_carlo_section(returns: pd.DataFrame):
    st.subheader("6. 🎲 Monte Carlo Simulation")
    st.markdown(
        "Simulate **thousands of possible futures** for an equal-weighted portfolio "
        "by randomly drawing returns from the historical distribution. "
        "This gives a probabilistic view of what might happen over the coming weeks."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.slider("Number of Simulations:", 500, 5000, 1000, step=500, key="mc_sims")
    with col2:
        horizon = st.slider("Forecast Horizon (days):", 10, 90, 30, step=5, key="mc_horizon")

    if st.button("🚀 Run Monte Carlo Simulation", type="primary", key="mc_run"):
        with st.spinner(f"Running {n_sims:,} simulations over {horizon} days..."):
            result = run_monte_carlo(returns, n_simulations=n_sims, horizon=horizon)

        # Key outcome metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probability of Profit", f"{result['prob_profit']*100:.1f}%",
                  help="% of simulations ending with a positive return")
        m2.metric("Expected Return", f"{result['expected_return']*100:.2f}%",
                  help="Median final portfolio return across all simulations")
        m3.metric("VaR 95% (30-day)", f"{result['var_95']*100:.2f}%",
                  help="Worst expected outcome with 95% confidence")
        m4.metric("Best-Case (95th pct)", f"{result['best_case']*100:.2f}%",
                  help="Top 5% of outcomes")

        # Path chart
        days = list(range(1, horizon + 1))
        fig_paths = go.Figure()

        # Sample paths (light, transparent)
        for path in result["sample_paths"]:
            fig_paths.add_trace(go.Scatter(
                x=days, y=path * 100,
                mode="lines", line=dict(color="rgba(0,120,212,0.07)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

        # Percentile bands
        pct = result["percentiles"]
        band_colors = {95: "#4CAF50", 75: "#8BC34A", 50: "#0078D4", 25: "#FF9800", 5: "#E91E63"}
        band_labels = {95: "95th pct", 75: "75th pct", 50: "Median", 25: "25th pct", 5: "5th pct"}
        band_widths = {95: 2.5, 75: 1.5, 50: 3, 25: 1.5, 5: 2.5}
        band_dash = {95: "solid", 75: "dash", 50: "solid", 25: "dash", 5: "solid"}

        for p in [5, 25, 50, 75, 95]:
            fig_paths.add_trace(go.Scatter(
                x=days, y=pct[p] * 100,
                mode="lines", name=band_labels[p],
                line=dict(color=band_colors[p], width=band_widths[p], dash=band_dash[p]),
            ))

        # Zero line
        fig_paths.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        fig_paths.update_layout(
            title=f"Monte Carlo Simulation — {n_sims:,} Paths over {horizon} Days",
            xaxis_title="Trading Day", yaxis_title="Cumulative Return (%)",
            hovermode="x unified", **_LAYOUT,
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        # Final distribution histogram
        fig_final = go.Figure(go.Histogram(
            x=result["final_values"] * 100,
            nbinsx=80, name="Final Returns",
            marker_color=_COLORS[0], opacity=0.8,
            histnorm="probability density",
        ))
        fig_final.add_vline(
            x=result["var_95"] * 100, line_dash="dash", line_color="#E91E63",
            annotation_text="VaR 95%", annotation_position="top right",
        )
        fig_final.add_vline(
            x=result["expected_return"] * 100, line_dash="dash", line_color="#4CAF50",
            annotation_text="Expected", annotation_position="top left",
        )
        fig_final.update_layout(
            title=f"Distribution of Final Portfolio Returns after {horizon} Days",
            xaxis_title="Final Cumulative Return (%)",
            yaxis_title="Density", **_LAYOUT,
        )
        st.plotly_chart(fig_final, use_container_width=True)

        st.success(
            f"✅ **Simulation complete.** Out of {n_sims:,} scenarios, "
            f"**{result['prob_profit']*100:.1f}%** end in profit after {horizon} trading days. "
            f"The median outcome is **{result['expected_return']*100:.2f}%** with a "
            f"worst-case (VaR 95%) of **{result['var_95']*100:.2f}%**."
        )
