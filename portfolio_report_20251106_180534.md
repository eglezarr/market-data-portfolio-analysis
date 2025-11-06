# 📊 PORTFOLIO ANALYSIS REPORT

**Generated:** 2025-11-06 18:05:34
**Analysis Period:** 2018-01-02 to 2025-09-30

---

## 📋 Executive Summary

- **Portfolio Expected Return:** 29.48% annualized
- **Portfolio Volatility:** 30.45% annualized
- **Sharpe Ratio:** 0.886
- **Value at Risk (95%):** $88,434.99
- **Number of Assets:** 12
- **Weight Drift Magnitude:** 12.73% (Buy and Hold strategy)

---

## 📈 Portfolio Overview

**Period Analyzed:** 2018-01-02 to 2025-09-30
**Number of Assets:** 12
**Investment Strategy:** Equal-weighted (Buy and Hold)

### Assets Included:

| Ticker | Initial Weight | Sector |
|--------|----------------|--------|
| AAPL | 8.33% | Technology |
| MSFT | 8.33% | Technology |
| GOOGL | 8.33% | Technology |
| AMZN | 8.33% | Technology |
| TSLA | 8.33% | Technology |
| NVDA | 8.33% | Technology |
| JPM | 8.33% | Financial |
| JNJ | 8.33% | Healthcare |
| PG | 8.33% | Consumer Staples |
| KO | 8.33% | Consumer Staples |
| XOM | 8.33% | Energy |
| MCD | 8.33% | Consumer Discretionary |

---

## 💰 Historical Performance

**Total Return:** 481.08%
**Annualized Return:** 25.59%
**Historical Volatility:** 21.11%
**Historical Sharpe Ratio:** 1.070
**Maximum Drawdown:** -33.18%
**Analysis Period:** 7.72 years (1946 trading days)

---

## 🎲 Monte Carlo Simulation Results

### Simulation Parameters:

- **Number of Simulations:** 10,000
- **Time Horizon:** 252 days (~1.0 year)
- **Initial Investment:** $100,000.00
- **Method:** Geometric Brownian Motion (GBM) with correlations

### Expected Outcomes:

- **Expected Final Value:** $129,480.77
- **Expected Return:** 29.48%
- **Standard Deviation:** 30.45%
- **Volatility (annualized):** 30.45%

### Risk Metrics:

- **Value at Risk (VaR 95%):** $88,434.99
  - *Maximum loss expected with 95% confidence*
- **Conditional VaR (CVaR 95%):** $81,038.77
  - *Expected loss in worst 5% of scenarios*
- **Probability of Loss:** 14.89%

### Scenario Analysis:

- **Best Case (P95):** $185,541.36
- **Expected Case (P50):** $124,983.78
- **Worst Case (P5):** $88,434.99
- **Absolute Best:** $342,632.34
- **Absolute Worst:** $58,647.62

---

## ⚖️ Weight Drift Analysis (Buy and Hold)

In a Buy and Hold strategy without rebalancing, asset weights naturally drift
as different assets have different returns. This analysis shows the average
weight changes across all Monte Carlo simulations.

**Total Drift Magnitude:** 12.73%

| Asset | Initial Weight | Final Weight (Avg) | Change |
|-------|----------------|-------------------|---------|
| AAPL | 8.33% | 8.55% | +0.22% |
| MSFT | 8.33% | 8.53% | +0.20% |
| GOOGL | 8.33% | 8.22% | -0.11% |
| AMZN | 8.33% | 8.08% | -0.25% |
| TSLA | 8.33% | 11.15% | +2.82% |
| NVDA | 8.33% | 11.46% | +3.13% |
| JPM | 8.33% | 7.93% | -0.40% |
| JNJ | 8.33% | 7.05% | -1.28% |
| PG | 8.33% | 7.27% | -1.07% |
| KO | 8.33% | 7.13% | -1.20% |
| XOM | 8.33% | 7.33% | -1.00% |
| MCD | 8.33% | 7.28% | -1.05% |

**Key Observations:**
- **Biggest Weight Gainer:** NVDA (+3.13pp)
- **Biggest Weight Loser:** JNJ (-1.28pp)

---

## 🔍 Risk Analysis

### Individual Asset Risk:

| Asset | Volatility | Beta | Sharpe Ratio | VaR (95%) |
|-------|------------|------|--------------|-----------|
| AAPL | 42.08% | 1.221 | 0.723 | $76,240 |
| MSFT | 38.39% | 1.181 | 0.777 | $79,087 |
| GOOGL | 39.84% | 1.152 | 0.628 | $73,617 |
| AMZN | 44.66% | 1.172 | 0.522 | $67,509 |
| TSLA | 130.31% | 1.622 | 0.604 | $50,929 |
| NVDA | 100.16% | 1.822 | 0.801 | $68,527 |
| JPM | 36.54% | 1.064 | 0.564 | $72,625 |
| JNJ | 21.68% | 0.469 | 0.300 | $77,499 |
| PG | 22.86% | 0.503 | 0.427 | $78,870 |
| KO | 21.53% | 0.540 | 0.358 | $78,448 |
| XOM | 35.27% | 0.813 | 0.319 | $66,281 |
| MCD | 24.86% | 0.636 | 0.407 | $77,077 |

**Portfolio Volatility:** 30.45%
**Portfolio Sharpe Ratio:** 0.886

### Correlation Matrix:

*See correlation heatmap in visualizations for detailed view*

---

## ⚠️ Warnings & Considerations

### Model Assumptions:

- **Geometric Brownian Motion:** Assumes log-normal distribution of returns
- **Constant Parameters:** μ and σ assumed constant over simulation horizon
- **No Rebalancing:** Buy and Hold strategy without portfolio adjustments
- **No Transaction Costs:** Assumes frictionless trading
- **Historical Correlation:** Assumes past correlations persist

### Limitations:

- Monte Carlo simulations are based on historical data
- Past performance does not guarantee future results
- Extreme market events (black swans) may not be captured
- Model does not account for regime changes

### Recommendations:

- Consider regular portfolio rebalancing to maintain target weights
- Monitor for significant changes in asset correlations
- Review risk metrics periodically
- Diversification does not eliminate all risk

---

## 📊 Asset Comparison Table

| Ticker | Type | Exp. Return | Volatility | Sharpe | VaR (95%) | Prob. Loss |
|--------|------|-------------|------------|--------|-----------|------------|
| PORTFOLIO | Portfolio | 29.48% | 30.45% | 0.886 | $88,435 | 14.89% |
| AAPL | Asset | 32.95% | 42.08% | 0.723 | $76,240 | 21.77% |
| MSFT | Asset | 32.34% | 38.39% | 0.777 | $79,087 | 20.25% |
| GOOGL | Asset | 27.51% | 39.84% | 0.628 | $73,617 | 25.92% |
| AMZN | Asset | 25.82% | 44.66% | 0.522 | $67,509 | 30.72% |
| TSLA | Asset | 81.22% | 130.31% | 0.604 | $50,929 | 27.07% |
| NVDA | Asset | 82.71% | 100.16% | 0.801 | $68,527 | 18.09% |
| JPM | Asset | 23.10% | 36.54% | 0.564 | $72,625 | 28.40% |
| JNJ | Asset | 9.00% | 21.68% | 0.300 | $77,499 | 36.60% |
| PG | Asset | 12.26% | 22.86% | 0.427 | $78,870 | 31.70% |
| KO | Asset | 10.20% | 21.53% | 0.358 | $78,448 | 34.08% |
| XOM | Asset | 13.75% | 35.27% | 0.319 | $66,281 | 39.42% |
| MCD | Asset | 12.62% | 24.86% | 0.407 | $77,077 | 33.28% |

---

## 📝 Notes

This report was automatically generated using Monte Carlo simulations.
For visualizations, use the `.plots_report()` method.

**Report generated:** 2025-11-06 18:05:34
