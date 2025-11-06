# 📊 Reporte de Validación de Datos

**Fecha de generación:** 2025-11-06 18:05:01

**Tolerancia configurada:** 5.0%

## 📋 Metodología de Validación

### Comparación: Retornos Diarios

Para validar la calidad de los datos, se compararon los **retornos diarios** 
(cambios porcentuales) de Yahoo Finance con los de AlphaVantage.


**Justificación metodológica:**

- Los **precios absolutos** pueden diferir por ajustes retroactivos diferentes

- Los **retornos diarios** deben ser idénticos si ambas fuentes capturan el mismo mercado

- Más robusto ante diferencias de escala o metodologías de ajuste histórico

- Si retorno de Yahoo = +2.5% y retorno de AlphaVantage = +2.5% → validación perfecta


**Nota:** Diferencias menores a 0.05% en retornos diarios son excelentes.

## 📈 Estadísticas Generales

- **Total de comparaciones:** 12
- **Diferencia promedio en retornos:** 0.0320%
- **Diferencia máxima en retornos:** 96.8946%
- **Correlación promedio de retornos:** 0.893553
- **Correlación mínima de retornos:** 0.692653
- **Tickers con advertencias:** 2

## 📋 Comparación Detallada por Ticker

| Ticker | Fechas Comunes | Diff Retornos (%) | Corr Retornos | Estado |
|--------|----------------|-------------------|---------------|--------|
| MSFT | 1946 | 0.0042 | 0.999815 | ✅ OK |
| AMZN | 1946 | 0.0498 | 0.695856 | ✅ OK |
| TSLA | 1946 | 0.0804 | 0.838975 | ⚠️ WARNING |
| JPM | 1946 | 0.0109 | 0.998869 | ✅ OK |
| NVDA | 1946 | 0.0854 | 0.775464 | ⚠️ WARNING |
| PG | 1946 | 0.0105 | 0.997870 | ✅ OK |
| JNJ | 1946 | 0.0111 | 0.997505 | ✅ OK |
| AAPL | 1946 | 0.0432 | 0.733321 | ✅ OK |
| KO | 1946 | 0.0124 | 0.996812 | ✅ OK |
| XOM | 1946 | 0.0182 | 0.996923 | ✅ OK |
| GOOGL | 1946 | 0.0480 | 0.692653 | ✅ OK |
| MCD | 1946 | 0.0093 | 0.998578 | ✅ OK |

## ⚠️ Advertencias

### AAPL (yahoo)

**Outliers detectados:** 21

### MSFT (yahoo)

**Outliers detectados:** 21

### GOOGL (yahoo)

**Outliers detectados:** 22

### AMZN (yahoo)

**Outliers detectados:** 15

### TSLA (yahoo)

**Outliers detectados:** 17

### NVDA (yahoo)

**Outliers detectados:** 12

### JPM (yahoo)

**Outliers detectados:** 25

### JNJ (yahoo)

**Outliers detectados:** 31

### PG (yahoo)

**Outliers detectados:** 23

### KO (yahoo)

**Outliers detectados:** 28

### XOM (yahoo)

**Outliers detectados:** 14

### MCD (yahoo)

**Outliers detectados:** 22

### ^GSPC (yahoo)

**Outliers detectados:** 26

### ^DJI (yahoo)

**Outliers detectados:** 28

### ^IXIC (yahoo)

**Outliers detectados:** 17

### AAPL (alphavantage)

**Outliers detectados:** 22

### MSFT (alphavantage)

**Outliers detectados:** 21

### GOOGL (alphavantage)

**Outliers detectados:** 23

### AMZN (alphavantage)

**Outliers detectados:** 16

### TSLA (alphavantage)

**Outliers detectados:** 18

### NVDA (alphavantage)

**Outliers detectados:** 14

### JPM (alphavantage)

**Outliers detectados:** 25

### JNJ (alphavantage)

**Outliers detectados:** 28

### PG (alphavantage)

**Outliers detectados:** 23

### KO (alphavantage)

**Outliers detectados:** 28

### XOM (alphavantage)

**Outliers detectados:** 14

### MCD (alphavantage)

**Outliers detectados:** 22

## ✅ Conclusión

**✅ Buena consistencia entre fuentes.**


Las diferencias son aceptables para análisis estadístico.


**Fuente seleccionada para análisis:** Yahoo Finance


**Justificación de la selección:**


Una vez validada la calidad y consistencia de los datos mediante la 
comparación de retornos diarios, se seleccionó **Yahoo Finance** como 
fuente principal por las siguientes razones:


1. **Adjusted Close preciso**: Yahoo Finance proporciona precios ajustados 
que corrigen por:

   - Dividendos pagados

   - Splits y reverse splits de acciones

   - Otros eventos corporativos


2. **Análisis de retornos preciso**: El uso de Adj Close es esencial para 
calcular retornos correctos, ya que el Close sin ajustar muestra caídas 
artificiales en fechas de dividendos o splits.


3. **Cobertura y accesibilidad**: Mayor cobertura histórica sin límites de tasa.


4. **Estándar de industria**: Ampliamente utilizado en análisis cuantitativo.


**Rol de AlphaVantage:**

- ✅ Validación cruzada exitosa: Correlación promedio de retornos de 0.893553

- ✅ Diferencia promedio de retornos de 0.0320% confirma alta consistencia

- ✅ Confirma la captura precisa de movimientos de mercado

- ✅ Proporciona redundancia y confianza en el dataset