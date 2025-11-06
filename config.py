"""
config.py
Archivo de configuración central para el proyecto de extracción de datos.
Aquí se definen los tickers, fechas y API keys.

IMPORTANTE: Las API keys ahora se cargan desde el archivo .env

VERSIÓN AMPLIADA:
- Período: 2018-01-01 a 2025-10-31 (7.8 años, ~1,950 observaciones)
- Activos: 12 empresas diversificadas por sectores
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# ==================== API KEYS ====================
# Las API keys se leen desde el archivo .env
# Crea un archivo .env en la raíz del proyecto con:
# FINNHUB_API_KEY=tu_clave_aqui
# ALPHAVANTAGE_API_KEY=tu_clave_aqui

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")


# ==================== FECHAS ====================
# Formato: 'YYYY-MM-DD'
# 
# IMPORTANTE - Comportamiento de las APIs:
# - Yahoo Finance: usa intervalo [start, end) - NO incluye la fecha final
# - AlphaVantage: usa intervalo [start, end] - SÍ incluye la fecha final
# 
# Con END_DATE = "2025-10-01":
#   - Yahoo descargará hasta 2025-09-30 (último día de trading de septiembre)
#   - Alpha descargará hasta 2025-10-01
#   - La validación cruzada usa inner join → ambos se alinean automáticamente
# 
# Período efectivo: 2018-01-02 a 2025-09-30 (~1,947 días de trading)

START_DATE = "2018-01-02"  # Primer día de trading de 2018
END_DATE = "2025-10-01"    # Ver nota arriba sobre comportamiento de APIs


# ==================== TICKERS ====================

# Lista de acciones a descargar (12 activos diversificados)
STOCK_TICKERS = [
    # TECHNOLOGY (6 activos)
    'AAPL',   # Apple Inc.
    'MSFT',   # Microsoft Corporation
    'GOOGL',  # Alphabet Inc. (Class A)
    'AMZN',   # Amazon.com Inc.
    'TSLA',   # Tesla Inc.
    'NVDA',   # NVIDIA Corporation
    
    # FINANCIAL (1 activo)
    'JPM',    # JPMorgan Chase & Co.
    
    # HEALTHCARE (1 activo)
    'JNJ',    # Johnson & Johnson
    
    # CONSUMER STAPLES (2 activos)
    'PG',     # Procter & Gamble Co.
    'KO',     # The Coca-Cola Company
    
    # ENERGY (1 activo)
    'XOM',    # Exxon Mobil Corporation
    
    # CONSUMER DISCRETIONARY (1 activo)
    'MCD',    # McDonald's Corporation
]

# Lista de índices a descargar
# Nota: Los símbolos pueden variar según la fuente:
# - Yahoo Finance usa ^ como prefijo (^GSPC, ^DJI, ^IXIC)
# - Finnhub puede usar el mismo formato
# - AlphaVantage puede requerir ETFs equivalentes (SPY en lugar de ^GSPC)

INDEX_TICKERS = [
    '^GSPC',  # S&P 500
    '^DJI',   # Dow Jones Industrial Average
    '^IXIC',  # NASDAQ Composite
]


# ==================== CONFIGURACIÓN DE FUENTES ====================

# Fuentes de datos a utilizar (puedes activar/desactivar)
USE_YAHOO = True
USE_FINNHUB = False
USE_ALPHAVANTAGE = True  # ✅ Activado con delays para rate limits

# Nota sobre AlphaVantage:
# - Free tier: 5 llamadas/minuto, 500/día
# - Delays automáticos de 13s implementados (4.6 llamadas/minuto)
# - La descarga completa tomará ~3-4 minutos para 12-15 activos


# ==================== OPCIONES AVANZADAS ====================

# Crear carpeta de salida para guardar los datos si se necesita
OUTPUT_DIR = "data/raw"

# Guardar los datos extraídos en archivos CSV
SAVE_TO_CSV = True

# Mostrar información detallada durante la extracción
VERBOSE = True


# ==================== VALIDACIÓN ====================

# Validación básica (opcional)
if USE_FINNHUB and not FINNHUB_API_KEY:
    print("⚠️  ADVERTENCIA: FINNHUB_API_KEY no encontrada en .env")
    print("    Desactiva Finnhub (USE_FINNHUB=False) o añade tu API key al archivo .env")

if USE_ALPHAVANTAGE and not ALPHAVANTAGE_API_KEY:
    print("⚠️  ADVERTENCIA: ALPHAVANTAGE_API_KEY no encontrada en .env")
    print("    Desactiva AlphaVantage (USE_ALPHAVANTAGE=False) o añade tu API key al archivo .env")


# ==================== INFORMACIÓN DEL DATASET ====================

# Esta información aparecerá en los reportes
DATASET_INFO = {
    'period_start': START_DATE,
    'period_end': END_DATE,
    'period_years': 7.8,
    'n_stocks': len(STOCK_TICKERS),
    'n_indices': len(INDEX_TICKERS),
    'sectors': {
        'Technology': 6,
        'Financial': 1,
        'Healthcare': 1,
        'Consumer Staples': 2,
        'Energy': 1,
        'Retail': 1
    },
    'description': 'Portfolio diversificado de 12 blue-chip stocks con datos de 2018-2025'
}