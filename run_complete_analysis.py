"""
run_complete_analysis.py
Script completo que ejecuta todas las fases del proyecto:

Fase 1: Extracción de datos
Fase 2: Limpieza y validación
Fase 3: Análisis estadístico
Fase 4: Simulaciones Monte Carlo

Este script es el punto de entrada principal para ejecutar todo el análisis.
"""

import logging
from datetime import datetime

# Importar módulos del proyecto
import config
from data_cleaner import DataCleaner
from main_extraction import extract_data_from_all_sources
from price_series import PriceSeries, Portfolio, download_risk_free_rate
from monte_carlo import run_monte_carlo

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta el análisis completo de principio a fin."""
    
    logger.info("\n" + "="*80)
    logger.info(" "*20 + "ANÁLISIS COMPLETO DE PORTFOLIO")
    logger.info("="*80)
    logger.info(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Periodo: {config.START_DATE} a {config.END_DATE}")
    logger.info(f"Activos: {', '.join(config.STOCK_TICKERS)}")
    
    try:
        # ==================== FASE 1: EXTRACCIÓN ====================
        logger.info("\n" + "="*80)
        logger.info("FASE 1: EXTRACCIÓN DE DATOS")
        logger.info("="*80)
        
        raw_data = extract_data_from_all_sources()
        
        # ==================== FASE 2: LIMPIEZA ====================
        logger.info("\n" + "="*80)
        logger.info("FASE 2: LIMPIEZA Y VALIDACIÓN")
        logger.info("="*80)
        
        cleaner = DataCleaner(tolerance=0.05)
        cleaned_data, validation_report = cleaner.clean_all_data(
            raw_data, 
            primary_source='yahoo'
        )
        
        # Generar reporte de validación
        cleaner.generate_validation_report(save_to_file=True)
        
        # ==================== FASE 3: ANÁLISIS ESTADÍSTICO ====================
        logger.info("\n" + "="*80)
        logger.info("FASE 3: ANÁLISIS ESTADÍSTICO")
        logger.info("="*80)
        
        # Descargar tasa libre de riesgo
        logger.info("\nDescargando tasa libre de riesgo (T-Bills 3M)...")
        risk_free_rate = download_risk_free_rate(
            start_date=config.START_DATE,
            end_date=config.END_DATE
        )
        
        # Crear PriceSeries
        logger.info("\nCreando PriceSeries...")
        price_series_dict = {}
        
        for ticker, data in cleaned_data['stocks'].items():
            price_series_dict[ticker] = PriceSeries(
                ticker=ticker,
                data=data,
                risk_free_rate=risk_free_rate
            )
        
        for ticker, data in cleaned_data['indices'].items():
            price_series_dict[ticker] = PriceSeries(
                ticker=ticker,
                data=data,
                risk_free_rate=risk_free_rate
            )
        
        # Crear Portfolio
        logger.info("\nCreando Portfolio...")
        stocks = {ticker: ps for ticker, ps in price_series_dict.items() 
                  if ticker in config.STOCK_TICKERS}
        
        market_index = price_series_dict.get('^GSPC', None)
        
        portfolio = Portfolio(
            assets=stocks,
            weights=None,  # Equiponderado
            market_index=market_index,
            risk_free_rate=risk_free_rate
        )
        
        # ==================== FASE 4: MONTE CARLO ====================
        logger.info("\n" + "="*80)
        logger.info("FASE 4: SIMULACIONES MONTE CARLO")
        logger.info("="*80)
        
        # Ejecutar Monte Carlo
        mc_results = run_monte_carlo(
            portfolio=portfolio,
            n_simulations=10000,
            time_horizon=252,  # 1 año
            initial_investment=100000.0,
            confidence_level=0.95,
            simulate_individuals=True  # Simular también cada activo
        )
        
        # Guardar resultados consolidados en UN SOLO archivo JSON
        logger.info("\n" + "="*60)
        logger.info("GUARDANDO RESULTADOS CONSOLIDADOS")
        logger.info("="*60)
        
        json_file = mc_results.save_to_json()
        
        # ==================== RESUMEN FINAL ====================
        logger.info("\n" + "="*80)
        logger.info("RESUMEN FINAL")
        logger.info("="*80)
        
        # Imprimir resumen consolidado
        mc_results.print_summary()
        
        # ==================== COMPLETADO ====================
        logger.info("\n" + "="*80)
        logger.info("✓ ANÁLISIS COMPLETO FINALIZADO")
        logger.info("="*80)
        
        logger.info("\nArchivos generados:")
        logger.info("  1. VALIDATION_REPORT_*.md - Reporte de validación de datos")
        logger.info(f"  2. {json_file} - Resultados Monte Carlo CONSOLIDADOS")
        logger.info("     (Portfolio + todos los activos en UN SOLO archivo)")
        
        logger.info("\nPróxima fase:")
        logger.info("  Fase 5: Visualizaciones y reportes (Jupyter Notebook)")
        
        return {
            'portfolio': portfolio,
            'price_series': price_series_dict,
            'mc_results': mc_results
        }
        
    except Exception as e:
        logger.error(f"\n✗ Error durante el análisis: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
