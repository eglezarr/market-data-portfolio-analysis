# 📊 Market Data Extraction & Portfolio Analysis

**Sistema completo de análisis cuantitativo de portfolios con simulaciones Monte Carlo**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Fases del Proyecto](#-fases-del-proyecto)
- [Resultados y Visualizaciones](#-resultados-y-visualizaciones)
- [Documentación Técnica](#-documentación-técnica)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

---

## 🎯 Descripción

Sistema profesional de análisis cuantitativo de portfolios que integra extracción de datos financieros desde múltiples fuentes, limpieza y validación de datos, cálculo de métricas estadísticas avanzadas, simulaciones Monte Carlo y generación automatizada de reportes y visualizaciones.

El proyecto cubre un análisis completo de **12 activos diversificados** (tecnología, finanzas, healthcare, consumo, energía) durante el periodo **2018-2025** (~1,950 observaciones), generando más de **10,000 simulaciones Monte Carlo** para proyecciones de riesgo-retorno.

### Caso de Uso

Este proyecto es ideal para:
- 📚 **Proyectos académicos** de finanzas cuantitativas
- 💼 **Análisis profesional** de portfolios
- 🎓 **Aprendizaje** de técnicas de simulación Monte Carlo
- 📊 **Demostración** de habilidades en Python y finanzas

---

## ✨ Características

### 🔧 Técnicas

- **Extracción Multi-Fuente**: Yahoo Finance, Finnhub, AlphaVantage
- **Validación Cruzada**: Comparación automática entre fuentes
- **Limpieza de Datos**: Detección de outliers, imputación, normalización
- **Análisis Estadístico**: Retornos, volatilidad, Sharpe Ratio, Beta, correlaciones
- **Simulaciones Monte Carlo**: Geometric Brownian Motion (GBM) con correlaciones
- **Visualizaciones Profesionales**: 10 gráficos automatizados de alta calidad
- **Reportes Markdown**: Generación automática de reportes detallados

### 📊 Métricas Calculadas

**Portfolio:**
- Retorno esperado y volatilidad (anualizados)
- Sharpe Ratio
- Value at Risk (VaR) y Conditional VaR (CVaR)
- Maximum Drawdown
- Matriz de correlaciones
- Weight Drift Analysis (Buy and Hold)

**Activos Individuales:**
- Beta vs mercado (S&P 500)
- Alpha de Jensen
- Métricas de riesgo-retorno
- Correlaciones cruzadas

### 🎨 Visualizaciones

1. **Dashboard Ejecutivo** - Métricas clave del portfolio
2. **Evolución de Precios Históricos** - Precios normalizados
3. **Fan Chart Monte Carlo** - Trayectorias simuladas
4. **Distribución de Valores Finales** - Histograma de resultados
5. **Weight Drift Analysis** - Cambio de pesos Buy & Hold
6. **Heatmap de Correlaciones** - Matriz de correlaciones
7. **Riesgo-Retorno Scatter** - Efficient frontier
8. **Tabla Comparativa** - Métricas por activo
9. **Beta Analysis** - Riesgo sistemático vs volatilidad
10. **Maximum Drawdown** - Pérdidas históricas máximas

---

## 📁 Estructura del Proyecto

```
market-data-portfolio-analysis/
│
├── 📋 CONFIGURACIÓN
│   ├── config.py                      # Configuración central (tickers, fechas, API keys)
│   ├── requirements.txt               # Dependencias Python
│   ├── .env                           # API keys (NO incluir en Git)
│   └── .gitignore                     # Archivos a ignorar
│
├── 🔌 FASE 1: EXTRACCIÓN DE DATOS
│   ├── data_extractor.py              # Clase base abstracta
│   ├── yahoo_extractor.py             # Extractor Yahoo Finance
│   ├── finnhub_extractor.py           # Extractor Finnhub API
│   ├── alphavantage_extractor.py      # Extractor AlphaVantage API
│   └── main_extraction.py             # Orquestador de extracción
│
├── 🧹 FASE 2: LIMPIEZA Y VALIDACIÓN
│   └── data_cleaner.py                # Limpieza, homogeneización, validación
│
├── 📈 FASE 3: ANÁLISIS ESTADÍSTICO
│   ├── price_series.py                # Clases PriceSeries y Portfolio
│   └── main_analysis.py               # Orquestador de análisis
│
├── 🎲 FASE 4: SIMULACIONES MONTE CARLO
│   └── monte_carlo.py                 # Motor de simulaciones MC + reportes
│
├── 📊 FASE 5: VISUALIZACIÓN Y REPORTES
│   ├── run_complete_analysis.py       # Script principal (ejecuta todas las fases)
│   └── portfolio_analysis.ipynb       # Jupyter Notebook interactivo
│
└── 📄 OUTPUTS (generados automáticamente)
    ├── VALIDATION_REPORT_*.md         # Reporte de validación de datos
    ├── portfolio_report_*.md          # Reporte completo del análisis
    ├── mc_results_*.json              # Resultados Monte Carlo (formato JSON)
    └── plots/                         # Visualizaciones (10 gráficos .png)
        ├── 01_dashboard.png
        ├── 02_historical_prices.png
        ├── ...
        └── 10_max_drawdown.png
```

---

## 🚀 Instalación

### Requisitos Previos

- **Python 3.8+**
- pip (gestor de paquetes Python)
- (Opcional) Jupyter Notebook para análisis interactivo

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/market-data-portfolio-analysis.git
cd market-data-portfolio-analysis
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `pandas>=2.0.0` - Manipulación de datos
- `numpy>=1.24.0` - Cálculos numéricos
- `yfinance>=0.2.28` - Datos de Yahoo Finance
- `matplotlib>=3.7.0` - Visualizaciones
- `seaborn>=0.12.0` - Gráficos estadísticos
- `jupyter>=1.0.0` - Notebooks interactivos

---

## ⚙️ Configuración

### 1. Crear Archivo `.env`

Crea un archivo `.env` en la raíz del proyecto:

```bash
# API Keys (opcional - solo si usas Finnhub o AlphaVantage)
FINNHUB_API_KEY=tu_clave_finnhub_aqui
ALPHAVANTAGE_API_KEY=tu_clave_alphavantage_aqui
```

**Obtener API Keys gratuitas:**
- Finnhub: https://finnhub.io/
- AlphaVantage: https://www.alphavantage.co/support/#api-key

⚠️ **Nota:** Yahoo Finance NO requiere API key. El proyecto funciona solo con Yahoo si no configuras las otras fuentes.

### 2. Configurar `config.py`

Edita `config.py` para personalizar:

```python
# Periodo de análisis
START_DATE = "2018-01-02"
END_DATE = "2025-10-01"

# Tickers a analizar
STOCK_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
    'JPM', 'JNJ', 'PG', 'KO', 'XOM', 'MCD'
]

# Índices de referencia
INDEX_TICKERS = ['^GSPC', '^DJI', '^IXIC']

# Fuentes a usar
USE_YAHOO = True
USE_FINNHUB = False  # Requiere API key
USE_ALPHAVANTAGE = True  # Requiere API key
```

---

## 💻 Uso

### Opción 1: Ejecución Completa (Recomendado)

Ejecuta todo el análisis de principio a fin:

```bash
python run_complete_analysis.py
```

**Esto ejecuta automáticamente:**
1. ✅ Extracción de datos (Fase 1)
2. ✅ Limpieza y validación (Fase 2)
3. ✅ Análisis estadístico (Fase 3)
4. ✅ Simulaciones Monte Carlo (Fase 4)
5. ✅ Generación de reportes y visualizaciones (Fase 5)

**Archivos generados:**
- `VALIDATION_REPORT_*.md` - Validación de datos
- `portfolio_report_*.md` - Reporte completo
- `mc_results_*.json` - Resultados JSON
- `plots/*.png` - 10 visualizaciones

**Tiempo estimado:** 3-5 minutos (dependiendo de fuentes activas)

---

### Opción 2: Jupyter Notebook (Interactivo)

Para análisis interactivo y visualizaciones en vivo:

```bash
jupyter notebook portfolio_analysis.ipynb
```

**El notebook incluye:**
- Carga de datos pre-procesados
- Análisis exploratorio
- Visualizaciones interactivas
- Generación de reportes personalizados

---

### Opción 3: Ejecución por Fases

Ejecuta cada fase individualmente:

**Fase 1 - Extracción:**
```bash
python main_extraction.py
```

**Fase 2 - Limpieza:**
```python
from data_cleaner import DataCleaner
# Ver código en main_extraction.py
```

**Fase 3 - Análisis:**
```bash
python main_analysis.py
```

**Fase 4 - Monte Carlo:**
```python
from monte_carlo import run_monte_carlo
# Ver código en run_complete_analysis.py
```

---

## 📚 Fases del Proyecto

### 🔌 Fase 1: Extracción de Datos

**Objetivo:** Descargar datos históricos desde múltiples fuentes.

**Fuentes:**
- **Yahoo Finance** (principal - sin API key)
- **Finnhub** (opcional - requiere API key)
- **AlphaVantage** (validación - requiere API key)

**Outputs:**
- DataFrames estandarizados con columnas: Date, Open, High, Low, Close, Adj Close, Volume
- ~1,950 observaciones por activo (2018-2025)

**Características:**
- Manejo automático de rate limits
- Retry logic para peticiones fallidas
- Formato uniforme entre fuentes

---

### 🧹 Fase 2: Limpieza y Validación

**Objetivo:** Limpiar, homogeneizar y validar datos.

**Procesos:**
1. **Eliminación de duplicados**
2. **Manejo de valores faltantes** (interpolación)
3. **Detección de outliers** (método IQR)
4. **Validación OHLC** (High ≥ Low, etc.)
5. **Validación cruzada** entre fuentes (Yahoo vs AlphaVantage)

**Output:**
- `VALIDATION_REPORT_*.md` con estadísticas de consistencia

**Métricas de validación:**
- Diferencia promedio entre fuentes
- Correlación entre fuentes (>0.999 esperado)
- Outliers detectados por activo

---

### 📈 Fase 3: Análisis Estadístico

**Objetivo:** Calcular métricas financieras clave.

**Clases principales:**
- `PriceSeries`: Análisis de un activo individual
- `Portfolio`: Análisis del portfolio completo

**Métricas calculadas:**
- Retornos anualizados
- Volatilidad anualizada
- Sharpe Ratio
- Beta vs mercado (S&P 500)
- Maximum Drawdown
- Matriz de correlaciones
- Matriz de covarianza

**Output:**
- Objetos Python con todas las métricas
- Resumen en formato texto

---

### 🎲 Fase 4: Simulaciones Monte Carlo

**Objetivo:** Proyectar valores futuros del portfolio usando GBM.

**Metodología:**
- **Geometric Brownian Motion (GBM)**
- **Descomposición de Cholesky** para mantener correlaciones
- **10,000 simulaciones** (configurable)
- **Horizonte de 252 días** (~1 año)

**Fórmula GBM:**
```
S_t = S_0 * exp((μ - σ²/2)*t + σ*√t*Z)
```

**Outputs:**
- Expected Value
- Expected Return
- Value at Risk (VaR 95%)
- Conditional VaR (CVaR 95%)
- Probability of Loss
- Percentiles (P5, P25, P50, P75, P95)

**Ventajas:**
- Considera correlaciones entre activos
- Simula portfolio Y cada activo individual
- Análisis de Weight Drift (Buy and Hold)

---

### 📊 Fase 5: Visualización y Reportes

**Objetivo:** Generar reportes y visualizaciones profesionales.

**Reportes generados:**
1. **Portfolio Report** (Markdown)
   - Executive Summary
   - Historical Performance
   - Monte Carlo Results
   - Risk Analysis
   - Asset Comparison

2. **Validation Report** (Markdown)
   - Data quality metrics
   - Cross-validation statistics
   - Outliers detected

**Visualizaciones (10 gráficos):**
- Dashboard ejecutivo
- Precios históricos
- Fan Chart Monte Carlo
- Distribución de valores finales
- Weight Drift
- Correlaciones
- Riesgo-Retorno
- Tabla comparativa
- **Beta Analysis** (riesgo sistemático)
- Maximum Drawdown

**Formatos:**
- PNG (alta resolución, 300 DPI)
- Markdown (reportes)
- JSON (datos estructurados)

---

## 📊 Resultados y Visualizaciones

### Ejemplo de Portfolio Report

```markdown
## Executive Summary

- Portfolio Expected Return: 30.87% annualized
- Portfolio Volatility: 31.12% annualized
- Sharpe Ratio: 0.912
- Value at Risk (95%): $88,247.11
- Number of Assets: 12
```

### Ejemplo de Visualización

**Beta Analysis:**
- Scatter plot de Beta vs Volatilidad
- Color según Sharpe Ratio
- Cuadrantes: Defensive, Aggressive, Stable, Growth
- Identifica activos sobre/infravalorados

### Interpretación de Resultados

**Portfolio Metrics:**
- **Expected Return > 25%**: Excelente performance histórica
- **Sharpe Ratio > 0.9**: Buen retorno ajustado por riesgo
- **VaR 95% = $88K**: En 95% de escenarios, pérdidas < $12K

**Individual Assets:**
- **TSLA, NVDA**: Alta beta (>1.8), alta volatilidad (>100%)
- **JNJ, PG, KO**: Baja beta (<0.6), defensivos
- **AAPL, MSFT**: Balance entre riesgo y retorno

---

## 📖 Documentación Técnica

### Arquitectura del Sistema

**Patrón de Diseño:** Pipeline modular con herencia OOP

**Flujo de datos:**
```
Data Sources → Extractors → Cleaner → PriceSeries → Portfolio → MonteCarlo → Reports
```

**Clases principales:**
- `DataExtractor` (abstracta) → `YahooExtractor`, `FinnhubExtractor`, `AlphaVantageExtractor`
- `DataCleaner` → Limpieza y validación
- `PriceSeries` → Análisis de activo individual
- `Portfolio` → Análisis de portfolio consolidado
- `MonteCarloSimulator` → Simulaciones GBM
- `ConsolidatedResults` → Almacena y presenta resultados

### Decisiones Técnicas

**¿Por qué Yahoo Finance como fuente principal?**
- No requiere API key
- Adjusted Close más preciso
- Mayor confiabilidad histórica

**¿Por qué 10,000 simulaciones?**
- Balance entre precisión y velocidad
- Suficiente para convergencia de distribuciones
- Tiempo de ejecución: ~30 segundos

**¿Por qué GBM?**
- Modelo estándar en finanzas cuantitativas
- Asume log-normalidad de retornos (razonable)
- Fácil de implementar y explicar

### Limitaciones Conocidas

1. **Modelo GBM:**
   - Asume parámetros constantes (μ, σ)
   - No captura regímenes cambiantes
   - No modela eventos extremos (fat tails)

2. **Correlaciones:**
   - Basadas en histórico (pueden cambiar)
   - No captura dependencias no-lineales

3. **Sin costos de transacción:**
   - Modelo asume trading sin fricción
   - No considera slippage o spreads

4. **Buy and Hold:**
   - No simula rebalanceo
   - Weight drift puede ser significativo

