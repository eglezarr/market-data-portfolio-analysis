"""
main_extraction.py
Script principal para extraer datos de mercado desde múltiples fuentes.

Este script orquesta la extracción de datos de acciones e índices desde
Yahoo Finance, Finnhub y AlphaVantage, manteniendo un formato uniforme.
"""

import pandas as pd
from typing import Dict
import os
import logging

# Importar configuración
import config

# Importar extractores
from yahoo_extractor import YahooExtractor
from finnhub_extractor import FinnhubExtractor
from alphavantage_extractor import AlphaVantageExtractor


# Configurar logging
logging.basicConfig(
    level=logging.INFO if config.VERBOSE else logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_data_from_all_sources():
    """
    Extrae datos desde todas las fuentes configuradas.
    
    Returns:
        Diccionario con la estructura:
        {
            'yahoo': {
                'stocks': {ticker: DataFrame},
                'indices': {ticker: DataFrame}
            },
            'finnhub': {...},
            'alphavantage': {...}
        }
    """
    all_data = {}
    
    # ==================== YAHOO FINANCE ====================
    if config.USE_YAHOO:
        logger.info("=" * 60)
        logger.info("EXTRAYENDO DATOS DESDE YAHOO FINANCE")
        logger.info("=" * 60)
        
        yahoo_extractor = YahooExtractor()
        yahoo_data = {
            'stocks': {},
            'indices': {}
        }
        
        # Extraer acciones
        logger.info(f"\nDescargando {len(config.STOCK_TICKERS)} acciones...")
        yahoo_data['stocks'] = yahoo_extractor.fetch_multiple_tickers(
            tickers=config.STOCK_TICKERS,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            ticker_type='stock'
        )
        
        # Extraer índices
        logger.info(f"\nDescargando {len(config.INDEX_TICKERS)} índices...")
        yahoo_data['indices'] = yahoo_extractor.fetch_multiple_tickers(
            tickers=config.INDEX_TICKERS,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            ticker_type='index'
        )
        
        all_data['yahoo'] = yahoo_data
        logger.info(f"\n✓ Yahoo Finance completado")
    
    # ==================== FINNHUB ====================
    if config.USE_FINNHUB:
        logger.info("\n" + "=" * 60)
        logger.info("EXTRAYENDO DATOS DESDE FINNHUB")
        logger.info("=" * 60)
        
        try:
            finnhub_extractor = FinnhubExtractor(api_key=config.FINNHUB_API_KEY)
            finnhub_data = {
                'stocks': {},
                'indices': {}
            }
            
            # Extraer acciones
            logger.info(f"\nDescargando {len(config.STOCK_TICKERS)} acciones...")
            finnhub_data['stocks'] = finnhub_extractor.fetch_multiple_tickers(
                tickers=config.STOCK_TICKERS,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                ticker_type='stock'
            )
            
            # Extraer índices
            logger.info(f"\nDescargando {len(config.INDEX_TICKERS)} índices...")
            finnhub_data['indices'] = finnhub_extractor.fetch_multiple_tickers(
                tickers=config.INDEX_TICKERS,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                ticker_type='index'
            )
            
            all_data['finnhub'] = finnhub_data
            logger.info(f"\n✓ Finnhub completado")
            
        except Exception as e:
            logger.error(f"✗ Error con Finnhub: {str(e)}")
            logger.info("Continuando con las demás fuentes...")
    
    # ==================== ALPHAVANTAGE ====================
    if config.USE_ALPHAVANTAGE:
        logger.info("\n" + "=" * 60)
        logger.info("EXTRAYENDO DATOS DESDE ALPHAVANTAGE")
        logger.info("=" * 60)
        logger.warning("Nota: AlphaVantage tiene rate limits estrictos (5 llamadas/min)")
        logger.warning("Esta descarga puede tomar varios minutos...")
        
        try:
            av_extractor = AlphaVantageExtractor(api_key=config.ALPHAVANTAGE_API_KEY)
            av_data = {
                'stocks': {},
                'indices': {}
            }
            
            # Extraer acciones
            logger.info(f"\nDescargando {len(config.STOCK_TICKERS)} acciones...")
            av_data['stocks'] = av_extractor.fetch_multiple_tickers(
                tickers=config.STOCK_TICKERS,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                ticker_type='stock'
            )
            
            # Extraer índices
            logger.info(f"\nDescargando {len(config.INDEX_TICKERS)} índices...")
            av_data['indices'] = av_extractor.fetch_multiple_tickers(
                tickers=config.INDEX_TICKERS,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                ticker_type='index'
            )
            
            all_data['alphavantage'] = av_data
            logger.info(f"\n✓ AlphaVantage completado")
            
        except Exception as e:
            logger.error(f"✗ Error con AlphaVantage: {str(e)}")
            logger.info("Continuando con las demás fuentes...")
    
    return all_data


def save_data_to_csv(all_data: dict):
    """
    Guarda todos los datos extraídos en archivos CSV.
    
    Args:
        all_data: Diccionario con todos los datos extraídos
    """
    if not config.SAVE_TO_CSV:
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("GUARDANDO DATOS EN CSV")
    logger.info("=" * 60)
    
    # Crear directorio de salida si no existe
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    for source, data in all_data.items():
        for data_type in ['stocks', 'indices']:
            for ticker, df in data.get(data_type, {}).items():
                # Limpiar el ticker para nombre de archivo (eliminar caracteres especiales)
                clean_ticker = ticker.replace('^', '').replace('.', '_')
                filename = f"{source}_{data_type}_{clean_ticker}.csv"
                filepath = os.path.join(config.OUTPUT_DIR, filename)
                
                df.to_csv(filepath, index=False)
                logger.info(f"✓ Guardado: {filepath}")


def print_summary(all_data: dict):
    """
    Imprime un resumen de los datos extraídos.
    
    Args:
        all_data: Diccionario con todos los datos extraídos
    """
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE LA EXTRACCIÓN")
    logger.info("=" * 60)
    
    for source, data in all_data.items():
        logger.info(f"\n{source.upper()}:")
        logger.info(f"  Acciones: {len(data.get('stocks', {}))} descargadas")
        logger.info(f"  Índices: {len(data.get('indices', {}))} descargados")
        
        # Mostrar primer ticker como ejemplo
        if data.get('stocks'):
            first_ticker = list(data['stocks'].keys())[0]
            first_df = data['stocks'][first_ticker]
            logger.info(f"  Ejemplo ({first_ticker}): {len(first_df)} registros")
            logger.info(f"  Rango: {first_df['Date'].min()} a {first_df['Date'].max()}")


def main():
    """Función principal que ejecuta todo el proceso de extracción."""
    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO EXTRACCIÓN DE DATOS DE MERCADO")
    logger.info("=" * 60)
    logger.info(f"Periodo: {config.START_DATE} a {config.END_DATE}")
    logger.info(f"Acciones: {config.STOCK_TICKERS}")
    logger.info(f"Índices: {config.INDEX_TICKERS}")
    
    # Extraer datos
    all_data = extract_data_from_all_sources()
    
    # Guardar en CSV si está configurado
    save_data_to_csv(all_data)
    
    # Mostrar resumen
    print_summary(all_data)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ PROCESO COMPLETADO")
    logger.info("=" * 60)
    
    return all_data


if __name__ == "__main__":
    # Ejecutar el proceso de extracción
    data = main()
    
    # Los datos están ahora disponibles en la variable 'data'
    # Puedes acceder a ellos así:
    # data['yahoo']['stocks']['AAPL']  -> DataFrame con datos de Apple desde Yahoo
    # data['finnhub']['indices']['^GSPC']  -> DataFrame del S&P 500 desde Finnhub