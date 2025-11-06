"""
main_analysis.py
Script principal para ejecutar el análisis estadístico completo.

Este script orquesta:
1. Carga de datos limpios (desde data_cleaner.py)
2. Descarga de tasa libre de riesgo (T-Bills 3M)
3. Creación de PriceSeries para cada activo
4. Creación de Portfolio
5. Cálculo de métricas y generación de resumen

Uso:
    python main_analysis.py
"""

import pandas as pd
import logging
from typing import Dict
from datetime import datetime

# Importar módulos del proyecto
import config
from data_cleaner import DataCleaner
from main_extraction import extract_data_from_all_sources
from price_series import PriceSeries, Portfolio, download_risk_free_rate

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_clean_data() -> dict:
    """
    Carga y limpia los datos de mercado.
    
    Returns:
        Diccionario con datos limpios: {stocks: {...}, indices: {...}}
    """
    logger.info("\n" + "="*60)
    logger.info("PASO 1: CARGANDO Y LIMPIANDO DATOS")
    logger.info("="*60)
    
    # Extraer datos crudos
    logger.info("\n1.1 Extrayendo datos de fuentes...")
    raw_data = extract_data_from_all_sources()
    
    # Limpiar y validar
    logger.info("\n1.2 Limpiando y validando datos...")
    cleaner = DataCleaner(tolerance=0.05)
    cleaned_data, validation_report = cleaner.clean_all_data(
        raw_data, 
        primary_source='yahoo'
    )
    
    # Generar reporte de validación
    logger.info("\n1.3 Generando reporte de validación...")
    cleaner.generate_validation_report(save_to_file=True)
    
    logger.info("\n✓ Datos cargados y limpios listos para análisis")
    
    return cleaned_data


def create_price_series(
    cleaned_data: dict, 
    risk_free_rate: pd.Series
) -> Dict[str, PriceSeries]:
    """
    Crea objetos PriceSeries para cada activo.
    
    Args:
        cleaned_data: Datos limpios {stocks: {...}, indices: {...}}
        risk_free_rate: Serie temporal de tasa libre de riesgo
        
    Returns:
        Diccionario {ticker: PriceSeries}
    """
    logger.info("\n" + "="*60)
    logger.info("PASO 3: CREANDO PRICE SERIES")
    logger.info("="*60)
    
    price_series_dict = {}
    
    # Crear PriceSeries para acciones
    logger.info("\n3.1 Procesando acciones...")
    for ticker, data in cleaned_data['stocks'].items():
        price_series_dict[ticker] = PriceSeries(
            ticker=ticker,
            data=data,
            risk_free_rate=risk_free_rate
        )
    
    # Crear PriceSeries para índices
    logger.info("\n3.2 Procesando índices...")
    for ticker, data in cleaned_data['indices'].items():
        price_series_dict[ticker] = PriceSeries(
            ticker=ticker,
            data=data,
            risk_free_rate=risk_free_rate
        )
    
    logger.info(f"\n✓ {len(price_series_dict)} PriceSeries creados")
    
    return price_series_dict


def create_portfolio(
    price_series_dict: Dict[str, PriceSeries],
    risk_free_rate: pd.Series
) -> Portfolio:
    """
    Crea un Portfolio con las acciones (excluyendo índices).
    
    Args:
        price_series_dict: Diccionario con todos los PriceSeries
        risk_free_rate: Serie temporal de tasa libre de riesgo
        
    Returns:
        Objeto Portfolio
    """
    logger.info("\n" + "="*60)
    logger.info("PASO 4: CREANDO PORTFOLIO")
    logger.info("="*60)
    
    # Separar acciones de índices
    stocks = {ticker: ps for ticker, ps in price_series_dict.items() 
              if ticker in config.STOCK_TICKERS}
    
    # Obtener índice de mercado (S&P 500)
    market_index = price_series_dict.get('^GSPC', None)
    
    if market_index is None:
        logger.warning("⚠ S&P 500 no encontrado, Beta no se calculará")
    
    # Crear portfolio (pesos equiponderados automáticamente)
    portfolio = Portfolio(
        assets=stocks,
        weights=None,  # None = equiponderado
        market_index=market_index,
        risk_free_rate=risk_free_rate
    )
    
    logger.info("\n✓ Portfolio creado exitosamente")
    
    return portfolio


def print_summary(
    price_series_dict: Dict[str, PriceSeries],
    portfolio: Portfolio
):
    """
    Imprime un resumen completo del análisis.
    
    Args:
        price_series_dict: Diccionario con todos los PriceSeries
        portfolio: Objeto Portfolio
    """
    logger.info("\n" + "="*60)
    logger.info("PASO 5: RESUMEN DE ANÁLISIS")
    logger.info("="*60)
    
    # Resumen del Portfolio
    logger.info("\n" + "="*60)
    logger.info("MÉTRICAS DEL PORTFOLIO")
    logger.info("="*60)
    
    summary = portfolio.get_portfolio_summary()
    
    # Métricas del portfolio
    logger.info("\n📊 Portfolio Consolidado:")
    for key, value in summary['Portfolio Metrics']['Portfolio'].items():
        logger.info(f"  {key}: {value}")
    
    # Pesos
    logger.info("\n⚖️  Pesos de los Activos:")
    for ticker, weight in summary['Weights'].items():
        logger.info(f"  {ticker}: {weight}")
    
    # Métricas individuales
    logger.info("\n" + "="*60)
    logger.info("MÉTRICAS INDIVIDUALES POR ACTIVO")
    logger.info("="*60)
    
    for ticker, metrics in summary['Individual Assets'].items():
        logger.info(f"\n📈 {ticker}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
    
    # Matriz de correlación
    logger.info("\n" + "="*60)
    logger.info("MATRIZ DE CORRELACIÓN")
    logger.info("="*60)
    logger.info(f"\n{summary['Correlation Matrix'].round(4)}")
    
    # Información adicional sobre índices
    logger.info("\n" + "="*60)
    logger.info("ÍNDICES DE REFERENCIA")
    logger.info("="*60)
    
    for ticker in config.INDEX_TICKERS:
        if ticker in price_series_dict:
            ps = price_series_dict[ticker]
            logger.info(f"\n📊 {ticker}:")
            logger.info(f"  Retorno Anualizado: {ps.mean_return_annual:.2f}%")
            logger.info(f"  Volatilidad Anualizada: {ps.volatility_annual:.2f}%")
            logger.info(f"  Sharpe Ratio: {ps.sharpe_ratio:.4f}")
            logger.info(f"  Max Drawdown: {ps.max_drawdown:.2f}%")


def save_summary_to_file(
    price_series_dict: Dict[str, PriceSeries],
    portfolio: Portfolio
):
    """
    Guarda el resumen en un archivo de texto.
    
    Args:
        price_series_dict: Diccionario con todos los PriceSeries
        portfolio: Objeto Portfolio
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ANALYSIS_SUMMARY_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RESUMEN DE ANÁLISIS ESTADÍSTICO\n")
        f.write("="*60 + "\n")
        f.write(f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Periodo analizado: {config.START_DATE} a {config.END_DATE}\n")
        
        # Métricas del Portfolio
        f.write("\n" + "="*60 + "\n")
        f.write("MÉTRICAS DEL PORTFOLIO\n")
        f.write("="*60 + "\n")
        
        summary = portfolio.get_portfolio_summary()
        
        f.write("\nPortfolio Consolidado:\n")
        for key, value in summary['Portfolio Metrics']['Portfolio'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nPesos de los Activos:\n")
        for ticker, weight in summary['Weights'].items():
            f.write(f"  {ticker}: {weight}\n")
        
        # Métricas individuales
        f.write("\n" + "="*60 + "\n")
        f.write("MÉTRICAS INDIVIDUALES POR ACTIVO\n")
        f.write("="*60 + "\n")
        
        for ticker, metrics in summary['Individual Assets'].items():
            f.write(f"\n{ticker}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
        
        # Matriz de correlación
        f.write("\n" + "="*60 + "\n")
        f.write("MATRIZ DE CORRELACIÓN\n")
        f.write("="*60 + "\n")
        f.write(f"\n{summary['Correlation Matrix'].round(4).to_string()}\n")
        
        # Matriz de covarianza
        f.write("\n" + "="*60 + "\n")
        f.write("MATRIZ DE COVARIANZA (anualizada)\n")
        f.write("="*60 + "\n")
        f.write(f"\n{summary['Covariance Matrix'].round(6).to_string()}\n")
    
    logger.info(f"\n✓ Resumen guardado en: {filename}")


def main():
    """
    Función principal que ejecuta el análisis completo.
    """
    logger.info("\n" + "="*80)
    logger.info(" "*20 + "ANÁLISIS ESTADÍSTICO COMPLETO")
    logger.info("="*80)
    logger.info(f"\nPeriodo: {config.START_DATE} a {config.END_DATE}")
    logger.info(f"Acciones: {', '.join(config.STOCK_TICKERS)}")
    logger.info(f"Índices: {', '.join(config.INDEX_TICKERS)}")
    
    try:
        # 1. Cargar datos limpios
        cleaned_data = load_clean_data()
        
        # 2. Descargar tasa libre de riesgo
        logger.info("\n" + "="*60)
        logger.info("PASO 2: DESCARGANDO TASA LIBRE DE RIESGO")
        logger.info("="*60)
        
        risk_free_rate = download_risk_free_rate(
            start_date=config.START_DATE,
            end_date=config.END_DATE
        )
        
        # 3. Crear PriceSeries
        price_series_dict = create_price_series(cleaned_data, risk_free_rate)
        
        # 4. Crear Portfolio
        portfolio = create_portfolio(price_series_dict, risk_free_rate)
        
        # 5. Mostrar resumen
        print_summary(price_series_dict, portfolio)
        
        # 6. Guardar resumen en archivo
        save_summary_to_file(price_series_dict, portfolio)
        
        logger.info("\n" + "="*80)
        logger.info("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        
        logger.info("\nPróximos pasos:")
        logger.info("  1. Revisar el reporte de validación (VALIDATION_REPORT_*.md)")
        logger.info("  2. Revisar el resumen de análisis (ANALYSIS_SUMMARY_*.txt)")
        logger.info("  3. Continuar con Fase 4: Simulaciones Monte Carlo")
        
        return price_series_dict, portfolio
        
    except Exception as e:
        logger.error(f"\n✗ Error durante el análisis: {str(e)}")
        raise


if __name__ == "__main__":
    # Ejecutar análisis completo
    price_series_dict, portfolio = main()
