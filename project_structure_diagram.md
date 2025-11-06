# 📐 Project Structure Diagram

**Arquitectura Visual Completa del Proyecto**

---

## 🏗️ Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MARKET DATA PORTFOLIO ANALYSIS                        │
│                         Sistema Completo                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │      run_complete_analysis.py (MAIN)         │
        │  Orquestador Principal - Ejecuta Fase 1-5    │
        └──────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
   [FASE 1-2]               [FASE 3-4]               [FASE 5]
  Datos Crudos            Análisis MC           Visualizaciones
```

---

## 📊 Flujo de Datos Completo

```
┌──────────────┐
│   config.py  │ ◄────────────────┐
│  (settings)  │                  │
└──────────────┘                  │
        │                         │
        │ configura               │
        ▼                         │
┌────────────────────────────┐   │
│  FASE 1: EXTRACCIÓN        │   │
│  main_extraction.py        │   │
└────────────────────────────┘   │
        │                         │
        │ usa                     │
        ▼                         │
┌──────────────────────────┐     │
│   data_extractor.py      │     │
│   (Clase Base ABC)       │     │
└──────────────────────────┘     │
        ▲                         │
        │ herencia                │
        │                         │
   ┌────┼─────┬──────────┐       │
   │         │           │       │
   ▼         ▼           ▼       │
┌─────┐  ┌────────┐  ┌────────┐ │
│Yahoo│  │Finnhub │  │ Alpha  │ │
│.py  │  │.py     │  │Vant.py │ │
└─────┘  └────────┘  └────────┘ │
   │         │           │       │
   └─────────┴───────────┘       │
            │                    │
            ▼                    │
      Raw Data Dict              │
            │                    │
            ▼                    │
┌────────────────────────────┐  │
│  FASE 2: LIMPIEZA          │  │
│  data_cleaner.py           │  │
│  - clean_all_data()        │  │
│  - validate()              │  │
│  - generate_report()       │  │
└────────────────────────────┘  │
            │                    │
            ▼                    │
   Cleaned Data Dict             │
            │                    │
            ▼                    │
┌────────────────────────────┐  │
│  FASE 3: ANÁLISIS          │  │
│  price_series.py           │  │
│  - PriceSeries (activo)    │──┤
│  - Portfolio (consolidado) │  │
└────────────────────────────┘  │
            │                    │
            ▼                    │
     Portfolio Object            │
            │                    │
            ▼                    │
┌────────────────────────────┐  │
│  FASE 4: MONTE CARLO       │  │
│  monte_carlo.py            │  │
│  - MonteCarloSimulator     │  │
│  - ConsolidatedResults     │  │
│  - .report()               │──┤
│  - .plots_report()         │  │
└────────────────────────────┘  │
            │                    │
            ▼                    │
  ConsolidatedResults Object     │
            │                    │
            ▼                    │
┌────────────────────────────┐  │
│  FASE 5: OUTPUTS           │  │
│  - portfolio_report.md     │◄─┘
│  - plots/*.png             │
│  - mc_results.json         │
└────────────────────────────┘
```

---

## 🗂️ Estructura de Archivos por Fase

### 📁 CONFIGURACIÓN

```
config.py ──────────────────► Todos los módulos
  │                            ├─ Tickers
  │                            ├─ Fechas
  │                            ├─ API Keys
  │                            └─ Settings
  │
requirements.txt ───────────► pip install
  └─ pandas, numpy, yfinance, matplotlib...

.env ───────────────────────► config.py
  ├─ FINNHUB_API_KEY
  └─ ALPHAVANTAGE_API_KEY

.gitignore ─────────────────► Git
  └─ Protege .env, data/, logs/
```

---

### 🔌 FASE 1: EXTRACCIÓN

```
data_extractor.py (Abstract Base Class)
    │
    │ define interfaz:
    │ ├─ fetch_stock_data()
    │ ├─ fetch_index_data()
    │ ├─ fetch_multiple_tickers()
    │ └─ _standardize_dataframe()
    │
    ├──► yahoo_extractor.py
    │    └─ Implementa interfaz para Yahoo Finance
    │        ├─ No requiere API key
    │        ├─ Usa librería yfinance
    │        └─ Proporciona Adj Close correcto
    │
    ├──► finnhub_extractor.py
    │    └─ Implementa interfaz para Finnhub API
    │        ├─ Requiere API key
    │        ├─ Rate limit: 60 calls/min
    │        └─ NO proporciona Adj Close
    │
    └──► alphavantage_extractor.py
         └─ Implementa interfaz para AlphaVantage API
             ├─ Requiere API key
             ├─ Rate limit: 5 calls/min (strict!)
             ├─ Delays automáticos (13s)
             └─ Para validación cruzada

main_extraction.py (Orchestrator)
    │
    ├─ extract_data_from_all_sources()
    │   ├─ Instancia cada extractor
    │   ├─ Descarga stocks + indices
    │   └─ Maneja errores y rate limits
    │
    ├─ save_data_to_csv() [opcional]
    │   └─ Guarda raw data en data/raw/
    │
    └─ print_summary()
        └─ Muestra estadísticas de descarga

OUTPUT:
┌────────────────────────────────────┐
│ all_data = {                       │
│   'yahoo': {                       │
│     'stocks': {                    │
│       'AAPL': DataFrame,           │
│       'MSFT': DataFrame, ...       │
│     },                             │
│     'indices': {                   │
│       '^GSPC': DataFrame, ...      │
│     }                              │
│   },                               │
│   'alphavantage': {...}            │
│ }                                  │
└────────────────────────────────────┘
```

---

### 🧹 FASE 2: LIMPIEZA Y VALIDACIÓN

```
data_cleaner.py
    │
    │ Clase: DataCleaner
    │
    ├─ clean_all_data(raw_data, primary_source='yahoo')
    │   │
    │   ├─ _clean_source_data(data, source_name)
    │   │   │
    │   │   └─ _clean_dataframe(df, ticker, source)
    │   │       ├─ Eliminar duplicados
    │   │       ├─ Manejar valores faltantes (interpolation)
    │   │       ├─ _detect_outliers(df, ticker)
    │   │       │   └─ Método IQR (Interquartile Range)
    │   │       ├─ _validate_ohlc_consistency(df, ticker)
    │   │       │   ├─ High >= max(Open, Close, Low)
    │   │       │   └─ Low <= min(Open, Close, High)
    │   │       └─ Ordenar por fecha
    │   │
    │   └─ _cross_validate_sources(cleaned_sources)
    │       └─ _compare_dataframes(df1, df2, ticker, src1, src2)
    │           ├─ Alinear fechas (inner join)
    │           ├─ Calcular diferencias porcentuales
    │           ├─ Calcular correlación
    │           └─ Detectar discrepancias significativas
    │
    ├─ generate_validation_report(save_to_file=True)
    │   └─ Crea VALIDATION_REPORT_*.md
    │       ├─ Estadísticas generales
    │       ├─ Comparación por ticker
    │       ├─ Advertencias (outliers)
    │       └─ Conclusión
    │
    └─ validation_report = {
        'discrepancies': [...],
        'statistics': {...},
        'warnings': [...]
    }

OUTPUT:
┌────────────────────────────────────┐
│ cleaned_data = {                   │
│   'stocks': {                      │
│     'AAPL': DataFrame (clean),     │
│     'MSFT': DataFrame (clean), ... │
│   },                               │
│   'indices': {                     │
│     '^GSPC': DataFrame (clean)     │
│   }                                │
│ }                                  │
│                                    │
│ + VALIDATION_REPORT_*.md (file)    │
└────────────────────────────────────┘
```

---

### 📈 FASE 3: ANÁLISIS ESTADÍSTICO

```
price_series.py
    │
    ├─ Clase: PriceSeries (Activo Individual)
    │   │
    │   ├─ __init__(ticker, data, risk_free_rate)
    │   │   ├─ self.prices (DataFrame)
    │   │   ├─ self.returns (Series)
    │   │   └─ Calcula métricas básicas
    │   │
    │   ├─ Propiedades calculadas:
    │   │   ├─ mean_return_annual
    │   │   ├─ volatility_annual
    │   │   ├─ sharpe_ratio
    │   │   ├─ max_drawdown
    │   │   └─ skewness, kurtosis
    │   │
    │   └─ get_summary()
    │       └─ Diccionario con todas las métricas
    │
    └─ Clase: Portfolio (Portfolio Consolidado)
        │
        ├─ __init__(assets, weights, market_index, risk_free_rate)
        │   │
        │   ├─ self.assets = {ticker: PriceSeries}
        │   ├─ self.weights (equiponderado por defecto)
        │   ├─ self.market_index (PriceSeries para Beta)
        │   └─ Calcula métricas del portfolio
        │
        ├─ Métodos de cálculo:
        │   ├─ calculate_portfolio_returns()
        │   ├─ calculate_portfolio_volatility()
        │   ├─ calculate_portfolio_sharpe()
        │   ├─ calculate_beta()
        │   │   └─ Beta vs market_index (S&P 500)
        │   ├─ calculate_correlation_matrix()
        │   └─ calculate_covariance_matrix()
        │
        ├─ Métodos nuevos (FASE 5):
        │   ├─ get_portfolio_returns()
        │   ├─ get_portfolio_cumulative_returns()
        │   └─ get_portfolio_prices_normalized()
        │
        └─ get_portfolio_summary()
            └─ Diccionario completo con:
                ├─ Portfolio Metrics
                ├─ Individual Assets
                ├─ Weights
                ├─ Correlation Matrix
                └─ Covariance Matrix

main_analysis.py (Orchestrator - Opcional)
    │
    ├─ load_clean_data()
    │   └─ Llama a main_extraction + data_cleaner
    │
    ├─ download_risk_free_rate()
    │   └─ Descarga T-Bills 3M desde Fed
    │
    ├─ create_price_series(cleaned_data, risk_free_rate)
    │   └─ Crea objetos PriceSeries para todos los tickers
    │
    ├─ create_portfolio(price_series_dict, risk_free_rate)
    │   └─ Crea objeto Portfolio
    │
    └─ print_summary(price_series_dict, portfolio)
        └─ Muestra todas las métricas

OUTPUT:
┌────────────────────────────────────┐
│ portfolio = Portfolio(             │
│   assets = {                       │
│     'AAPL': PriceSeries(...),      │
│     'MSFT': PriceSeries(...), ...  │
│   },                               │
│   weights = [0.083, 0.083, ...],   │
│   market_index = PriceSeries(^GSPC)│
│ )                                  │
│                                    │
│ + Todas las métricas calculadas    │
└────────────────────────────────────┘
```

---

### 🎲 FASE 4: SIMULACIONES MONTE CARLO

```
monte_carlo.py
    │
    ├─ @dataclass: SimulationResults
    │   │
    │   ├─ final_values: np.ndarray [n_simulations]
    │   ├─ statistics: dict (mean, std, VaR, CVaR, ...)
    │   ├─ percentiles: dict (p5, p25, p50, p75, p95)
    │   └─ get_summary()
    │
    ├─ @dataclass: ConsolidatedResults
    │   │
    │   ├─ portfolio: Portfolio (NUEVO - Fase 5)
    │   ├─ portfolio_results: SimulationResults
    │   ├─ asset_results: Dict[ticker, SimulationResults]
    │   ├─ parameters: dict
    │   ├─ metadata: dict (weight_drift, timestamps, ...)
    │   │
    │   ├─ get_summary_table()
    │   │   └─ DataFrame comparativo de todo
    │   │
    │   ├─ save_to_json(filename)
    │   │   └─ Guarda resultados consolidados
    │   │
    │   ├─ print_summary()
    │   │   └─ Resumen en consola
    │   │
    │   ├─ report(save_to_file=True, include_warnings=True)
    │   │   │
    │   │   └─ Genera portfolio_report_*.md
    │   │       ├─ Executive Summary
    │   │       ├─ Portfolio Overview
    │   │       ├─ Historical Performance
    │   │       ├─ Monte Carlo Results
    │   │       ├─ Weight Drift Analysis
    │   │       ├─ Risk Analysis
    │   │       ├─ Asset Comparison Table
    │   │       └─ Warnings & Considerations
    │   │
    │   └─ plots_report(show=True, save=True, output_dir="plots")
    │       │
    │       └─ Genera 10 visualizaciones:
    │           ├─ 01_dashboard.png
    │           ├─ 02_historical_prices.png
    │           ├─ 03_monte_carlo_fan_chart.png
    │           ├─ 04_distribution.png
    │           ├─ 05_weight_drift.png
    │           ├─ 06_correlation_heatmap.png
    │           ├─ 07_risk_return_scatter.png
    │           ├─ 08_comparison_table.png
    │           ├─ 09_beta_analysis.png (NUEVO)
    │           └─ 10_max_drawdown.png
    │
    ├─ Clase: MonteCarloSimulator
    │   │
    │   ├─ __init__(portfolio, n_simulations, time_horizon, ...)
    │   │
    │   ├─ simulate_portfolio()
    │   │   │
    │   │   ├─ _get_portfolio_parameters()
    │   │   │   ├─ expected_returns (μ)
    │   │   │   ├─ covariance_matrix (Σ)
    │   │   │   └─ initial_prices
    │   │   │
    │   │   ├─ _simulate_correlated_returns()
    │   │   │   └─ Cholesky Decomposition
    │   │   │       └─ L = cholesky(Σ)
    │   │   │       └─ Returns = μ + L @ Z
    │   │   │
    │   │   ├─ _simulate_asset_prices_vectorized()
    │   │   │   └─ GBM: S_t = S_0 * exp((μ-σ²/2)*t + σ*√t*Z)
    │   │   │
    │   │   ├─ _calculate_portfolio_values()
    │   │   │   └─ Portfolio value = Σ(weight_i * price_i)
    │   │   │
    │   │   ├─ _analyze_weight_drift()
    │   │   │   └─ Tracking de pesos en Buy & Hold
    │   │   │
    │   │   └─ _calculate_statistics(final_values)
    │   │       ├─ Expected Value, Return
    │   │       ├─ Volatility
    │   │       ├─ Sharpe Ratio (NUEVO - Fase 5)
    │   │       ├─ VaR, CVaR
    │   │       ├─ Prob. Loss
    │   │       └─ Percentiles
    │   │
    │   └─ simulate_asset(ticker, asset)
    │       └─ Simulación individual de un activo
    │
    └─ run_monte_carlo(portfolio, n_simulations, ...)
        │
        └─ Función helper que:
            ├─ Instancia MonteCarloSimulator
            ├─ Ejecuta simulate_portfolio()
            ├─ Ejecuta simulate_asset() para cada activo
            └─ Retorna ConsolidatedResults

OUTPUT:
┌────────────────────────────────────┐
│ mc_results = ConsolidatedResults(  │
│   portfolio = Portfolio(...),      │
│   portfolio_results = {...},       │
│   asset_results = {                │
│     'AAPL': SimulationResults(...),│
│     'MSFT': SimulationResults(...)│
│   },                               │
│   metadata = {                     │
│     'weight_drift_analysis': {...} │
│   }                                │
│ )                                  │
│                                    │
│ + mc_results_*.json (file)         │
│ + portfolio_report_*.md (file)     │
│ + plots/*.png (10 files)           │
└────────────────────────────────────┘
```

---

### 📊 FASE 5: VISUALIZACIÓN

```
run_complete_analysis.py (MAIN ORCHESTRATOR)
    │
    ├─ main()
    │   │
    │   ├─ FASE 1: extract_data_from_all_sources()
    │   │   └─ all_data
    │   │
    │   ├─ FASE 2: DataCleaner.clean_all_data()
    │   │   └─ cleaned_data + VALIDATION_REPORT_*.md
    │   │
    │   ├─ FASE 3: create Portfolio
    │   │   ├─ download_risk_free_rate()
    │   │   ├─ create PriceSeries objects
    │   │   └─ Portfolio(assets, market_index, ...)
    │   │
    │   ├─ FASE 4: run_monte_carlo()
    │   │   └─ mc_results (ConsolidatedResults)
    │   │
    │   └─ FASE 5: Outputs
    │       ├─ mc_results.save_to_json()
    │       ├─ mc_results.report() [automático]
    │       └─ return {portfolio, mc_results, ...}
    │
    └─ if __name__ == "__main__":
        └─ results = main()

portfolio_analysis.ipynb (Interactive Analysis)
    │
    ├─ Cell 1: Setup & Imports
    │
    ├─ Cell 2: Load Pre-computed Results
    │   └─ from run_complete_analysis import main
    │       outputs = main()
    │
    ├─ Cell 3: Portfolio Overview
    │   └─ Display portfolio metrics
    │
    ├─ Cell 4: Historical Analysis
    │   └─ Portfolio historical performance
    │
    ├─ Cell 5: Monte Carlo Results
    │   └─ Simulation statistics
    │
    ├─ Cell 6: Weight Drift
    │   └─ Buy & Hold analysis
    │
    ├─ Cell 7: Generate Markdown Report
    │   └─ results.report()
    │
    └─ Cell 8: Generate All Visualizations
        └─ results.plots_report(show=True, save=True)
            └─ 10 gráficos generados

OUTPUTS FINALES:
┌────────────────────────────────────┐
│ Files Generated:                   │
│                                    │
│ 📄 VALIDATION_REPORT_*.md          │
│    └─ Data quality & validation    │
│                                    │
│ 📄 portfolio_report_*.md           │
│    └─ Complete analysis report     │
│                                    │
│ 📄 mc_results_*.json               │
│    └─ Structured data (JSON)       │
│                                    │
│ 📁 plots/                          │
│    ├─ 01_dashboard.png             │
│    ├─ 02_historical_prices.png     │
│    ├─ 03_monte_carlo_fan_chart.png │
│    ├─ 04_distribution.png          │
│    ├─ 05_weight_drift.png          │
│    ├─ 06_correlation_heatmap.png   │
│    ├─ 07_risk_return_scatter.png   │
│    ├─ 08_comparison_table.png      │
│    ├─ 09_beta_analysis.png         │
│    └─ 10_max_drawdown.png          │
└────────────────────────────────────┘
```

---

## 🔄 Herencias y Relaciones

### Herencia de Clases

```
DataExtractor (ABC)
    ├─ YahooExtractor
    ├─ FinnhubExtractor
    └─ AlphaVantageExtractor

(No hay otras herencias - diseño modular)
```

### Composición y Dependencias

```
Portfolio
    ├─ contains: Dict[ticker, PriceSeries]
    ├─ uses: market_index (PriceSeries)
    └─ uses: risk_free_rate (pd.Series)

MonteCarloSimulator
    └─ uses: Portfolio

ConsolidatedResults
    ├─ contains: portfolio (Portfolio)
    ├─ contains: portfolio_results (SimulationResults)
    └─ contains: asset_results (Dict[ticker, SimulationResults])
```

---

## 📊 Flujo de Información

### Data Flow

```
Raw Data (APIs)
    ↓
Extractors → Standardized DataFrames
    ↓
DataCleaner → Clean DataFrames + Validation Report
    ↓
PriceSeries → Individual Asset Analysis
    ↓
Portfolio → Portfolio-level Metrics
    ↓
MonteCarloSimulator → Simulations (10,000 paths)
    ↓
ConsolidatedResults → Aggregated Results
    ↓
├─ .save_to_json() → mc_results_*.json
├─ .report() → portfolio_report_*.md
└─ .plots_report() → plots/*.png (10 files)
```

### Control Flow

```
User
  │
  ├─ python run_complete_analysis.py
  │     │
  │     └─ Ejecuta todas las fases automáticamente
  │
  └─ jupyter notebook portfolio_analysis.ipynb
        │
        └─ Análisis interactivo con visualizaciones en vivo
```

---

## 🎯 Módulos Independientes vs Dependientes

### Módulos Independientes (pueden ejecutarse solos):

✅ **main_extraction.py** - Extrae datos
✅ **data_cleaner.py** - Limpia datos pre-extraídos
✅ **main_analysis.py** - Análisis estadístico completo

### Módulos Dependientes (requieren outputs previos):

⚠️ **monte_carlo.py** - Requiere Portfolio object
⚠️ **portfolio_analysis.ipynb** - Requiere resultados de run_complete_analysis

### Módulo Maestro (ejecuta todo):

🎯 **run_complete_analysis.py** - Pipeline completo end-to-end

---

## 🔧 Puntos de Extensión

### Para añadir nueva fuente de datos:

```python
# 1. Crear nuevo extractor
class NewExtractor(DataExtractor):
    def fetch_stock_data(self, ticker, start, end):
        # Implementar lógica
        pass
    
# 2. Añadir a main_extraction.py
if config.USE_NEW_SOURCE:
    new_data = NewExtractor().fetch_multiple_tickers(...)
    all_data['new_source'] = new_data
```

### Para añadir nueva visualización:

```python
# En monte_carlo.py → plots_report()
# ==================== GRÁFICO 11: NUEVA VIZ ====================
logger.info("[11/11] Nueva Visualización...")
fig11, ax = plt.subplots(...)
# ... código del gráfico ...
```

### Para añadir nueva métrica:

```python
# En price_series.py → PriceSeries o Portfolio
@property
def nueva_metrica(self) -> float:
    """Calcula nueva métrica."""
    return self.returns.nueva_formula()
```

---

## 📐 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────┐
│                     SISTEMA COMPLETO                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CONFIG     │  │  EXTRACTORS  │  │   CLEANER    │     │
│  │              │  │              │  │              │     │
│  │ - Tickers    │→ │ - Yahoo      │→ │ - Validate   │     │
│  │ - Dates      │  │ - Finnhub    │  │ - Clean      │     │
│  │ - API Keys   │  │ - AlphaVant  │  │ - Report     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            ↓                                │
│                  ┌──────────────────┐                       │
│                  │  PRICE SERIES    │                       │
│                  │                  │                       │
│                  │ - PriceSeries    │                       │
│                  │ - Portfolio      │                       │
│                  └──────────────────┘                       │
│                            ↓                                │
│                  ┌──────────────────┐                       │
│                  │  MONTE CARLO     │                       │
│                  │                  │                       │
│                  │ - Simulator      │                       │
│                  │ - Results        │                       │
│                  └──────────────────┘                       │
│                            ↓                                │
│         ┌──────────────────┴──────────────────┐            │
│         ↓                                      ↓            │
│  ┌──────────────┐                    ┌──────────────┐      │
│  │   REPORTS    │                    │     PLOTS    │      │
│  │              │                    │              │      │
│  │ - .md files  │                    │ - .png files │      │
│  │ - .json file │                    │ - Dashboard  │      │
│  └──────────────┘                    └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎬 Secuencia de Ejecución

```
START
  │
  ├─ [1] Load config.py
  │   └─ Tickers, dates, API keys
  │
  ├─ [2] Extract data (Fase 1)
  │   ├─ Yahoo Finance ✓
  │   ├─ Finnhub (optional)
  │   └─ AlphaVantage (optional)
  │
  ├─ [3] Clean & Validate (Fase 2)
  │   ├─ Remove duplicates
  │   ├─ Handle missing values
  │   ├─ Detect outliers
  │   ├─ Cross-validate sources
  │   └─ Generate VALIDATION_REPORT_*.md
  │
  ├─ [4] Download Risk-Free Rate
  │   └─ T-Bills 3M from Fed
  │
  ├─ [5] Create PriceSeries (Fase 3)
  │   ├─ One object per ticker
  │   └─ Calculate individual metrics
  │
  ├─ [6] Create Portfolio (Fase 3)
  │   ├─ Aggregate all assets
  │   ├─ Calculate portfolio metrics
  │   └─ Calculate correlations
  │
  ├─ [7] Run Monte Carlo (Fase 4)
  │   ├─ Portfolio simulation (10,000 paths)
  │   ├─ Individual asset simulations
  │   ├─ Weight drift analysis
  │   └─ Calculate statistics
  │
  ├─ [8] Generate Outputs (Fase 5)
  │   ├─ Save JSON → mc_results_*.json
  │   ├─ Generate Report → portfolio_report_*.md
  │   └─ Generate Plots → plots/*.png (10 files)
  │
  └─ [9] Return results object
      └─ {portfolio, mc_results, price_series}

END
```

---

<div align="center">

**📐 Complete Project Architecture**

*Todas las conexiones, herencias y flujos visualizados*

[⬆ Volver arriba](#-project-structure-diagram)

</div>
