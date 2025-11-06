"""
monte_carlo.py
Fase 4: Simulaciones Monte Carlo para Portfolios y Activos

Este módulo implementa simulaciones Monte Carlo usando el método de 
Geometric Brownian Motion (GBM) con correlaciones entre activos.

Metodología:
- GBM: S_t = S_0 * exp((μ - σ²/2)*t + σ*√t*Z)
- Correlaciones: Usando descomposición de Cholesky de la matriz de covarianza
- Múltiples activos simulados simultáneamente manteniendo correlaciones

Clases:
- MonteCarloSimulator: Ejecuta simulaciones
- SimulationResults: Almacena y gestiona resultados
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import logging
import json
from datetime import datetime

# Importar clases del proyecto
from price_series import Portfolio, PriceSeries

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== DATACLASS: CONSOLIDATED RESULTS ====================

@dataclass
class ConsolidatedResults:
    """
    Resultados consolidados de todas las simulaciones Monte Carlo.
    
    Almacena en un solo objeto:
    - Portfolio original con datos históricos
    - Resultados del portfolio completo (simulaciones MC)
    - Resultados de cada activo individual (simulaciones MC)
    - Parámetros de la simulación
    - Metadata (drift, timestamps, etc.)
    
    Este objeto incluye TODO lo necesario para análisis y visualización:
    - Datos históricos (precios, retornos)
    - Resultados de simulaciones
    - Métricas estadísticas
    
    Este objeto puede guardarse/cargarse y permite generar reportes
    y visualizaciones completas.
    
    Attributes:
        portfolio: Portfolio original con activos y datos históricos
        portfolio_results: SimulationResults del portfolio (MC)
        asset_results: Dict {ticker: SimulationResults} de cada activo (MC)
        parameters: Parámetros usados en la simulación
        metadata: Información adicional (timestamp, drift, etc.)
    """
    
    portfolio: Portfolio
    portfolio_results: 'SimulationResults'
    asset_results: Dict[str, 'SimulationResults']
    parameters: dict
    metadata: dict
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """
        Guarda TODOS los resultados en un único archivo JSON consolidado.
        
        Estructura del JSON:
        {
            "metadata": {...},
            "parameters": {...},
            "weight_drift_analysis": {...},  ← IMPORTANTE para documentar Buy and Hold
            "portfolio": {
                "statistics": {...},
                "percentiles": {...}
            },
            "assets": {
                "AAPL": {...},
                "MSFT": {...},
                ...
            }
        }
        
        Args:
            filename: Nombre del archivo (opcional, se genera automático)
            
        Returns:
            Nombre del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monte_carlo_results_{timestamp}.json"
        
        # Construir estructura consolidada
        data = {
            'metadata': self.metadata,
            'parameters': self.parameters,
            'portfolio': {
                'statistics': self.portfolio_results.statistics,
                'percentiles': self.portfolio_results.percentiles,
                'final_values_distribution': {
                    'min': float(self.portfolio_results.final_values.min()),
                    'max': float(self.portfolio_results.final_values.max()),
                    'mean': float(self.portfolio_results.final_values.mean()),
                    'std': float(self.portfolio_results.final_values.std()),
                    'median': float(np.median(self.portfolio_results.final_values)),
                }
            },
            'assets': {}
        }
        
        # Añadir resultados de cada activo
        for ticker, result in self.asset_results.items():
            data['assets'][ticker] = {
                'statistics': result.statistics,
                'percentiles': result.percentiles,
                'final_values_distribution': {
                    'min': float(result.final_values.min()),
                    'max': float(result.final_values.max()),
                    'mean': float(result.final_values.mean()),
                    'std': float(result.final_values.std()),
                    'median': float(np.median(result.final_values)),
                }
            }
        
        # Guardar
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Resultados consolidados guardados en: {filename}")
        logger.info(f"  Portfolio + {len(self.asset_results)} activos en 1 archivo")
        
        # Mostrar información del drift si está disponible
        if 'weight_drift_analysis' in self.metadata:
            drift = self.metadata['weight_drift_analysis']
            logger.info(f"  Magnitud del drift de pesos: {drift['drift_magnitude']*100:.2f}%")
        
        return filename
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Genera una tabla comparativa de todos los resultados.
        
        Returns:
            DataFrame con métricas de portfolio y todos los activos
        """
        rows = []
        
        # Añadir portfolio
        portfolio_stats = self.portfolio_results.statistics
        portfolio_percentiles = self.portfolio_results.percentiles
        
        rows.append({
            'Ticker': 'PORTFOLIO',
            'Type': 'Portfolio',
            'Expected_Return_%': portfolio_stats['expected_return'],
            'Volatility_%': portfolio_stats['std_return'],
            'Sharpe_Ratio': portfolio_stats.get('sharpe_ratio', 0.0),
            'Expected_Value_$': portfolio_stats['expected_value'],
            'VaR_95_$': portfolio_stats['var_95'],
            'CVaR_95_$': portfolio_stats['cvar_95'],
            'Prob_Loss_%': portfolio_stats['prob_loss'],
            'Best_Case_$': portfolio_stats['best_case'],
            'Worst_Case_$': portfolio_stats['worst_case'],
            'P5_$': portfolio_percentiles['p5'],
            'P50_$': portfolio_percentiles['p50'],
            'P95_$': portfolio_percentiles['p95'],
        })
        
        # Añadir cada activo
        for ticker, result in self.asset_results.items():
            stats = result.statistics
            perc = result.percentiles
            
            rows.append({
                'Ticker': ticker,
                'Type': 'Asset',
                'Expected_Return_%': stats['expected_return'],
                'Volatility_%': stats['std_return'],
                'Sharpe_Ratio': stats.get('sharpe_ratio', 0.0),
                'Expected_Value_$': stats['expected_value'],
                'VaR_95_$': stats['var_95'],
                'CVaR_95_$': stats['cvar_95'],
                'Prob_Loss_%': stats['prob_loss'],
                'Best_Case_$': stats['best_case'],
                'Worst_Case_$': stats['worst_case'],
                'P5_$': perc['p5'],
                'P50_$': perc['p50'],
                'P95_$': perc['p95'],
            })
        
        return pd.DataFrame(rows)
    
    def get_weight_drift_analysis(self) -> dict:
        """
        Retorna el análisis de drift de pesos si está disponible.
        
        El drift de pesos documenta cómo los pesos equiponderados iniciales
        cambian durante la simulación debido a la estrategia Buy and Hold.
        
        Returns:
            Dict con:
            - initial_weights: Pesos iniciales
            - average_final_weights: Pesos promedio al final
            - weight_changes: Cambio de cada peso
            - drift_magnitude: Magnitud total del drift
        """
        if 'weight_drift_analysis' in self.metadata:
            return self.metadata['weight_drift_analysis']
        else:
            logger.warning("⚠ Análisis de drift no disponible en estos resultados")
            return None
    
    def get_weight_drift_dataframe(self) -> pd.DataFrame:
        """
        Retorna el análisis de drift en formato DataFrame para análisis.
        
        Returns:
            DataFrame con columnas: Ticker, Initial_Weight, Final_Weight, Change
        """
        drift = self.get_weight_drift_analysis()
        
        if drift is None:
            return pd.DataFrame()
        
        rows = []
        for ticker in drift['initial_weights'].keys():
            rows.append({
                'Ticker': ticker,
                'Initial_Weight_%': drift['initial_weights'][ticker] * 100,
                'Final_Weight_%': drift['average_final_weights'][ticker] * 100,
                'Change_%': drift['weight_changes'][ticker] * 100,
                'Change_pp': drift['weight_changes'][ticker] * 100  # puntos porcentuales
            })
        
        return pd.DataFrame(rows)
    

    def report(
        self,
        save_to_file: bool = True,
        filename: Optional[str] = None,
        include_warnings: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Genera un reporte completo en formato Markdown.
        
        El reporte incluye:
        - Executive Summary
        - Portfolio Overview
        - Historical Performance
        - Monte Carlo Simulation Results
        - Weight Drift Analysis
        - Risk Analysis
        - Warnings & Considerations
        - Asset Comparison Table
        
        Args:
            save_to_file: Si True, guarda el reporte en archivo .md
            filename: Nombre del archivo (opcional, se genera automático)
            include_warnings: Si True, incluye sección de advertencias
            verbose: Si True, imprime confirmación por pantalla
            
        Returns:
            String con el reporte completo en Markdown
        """
        lines = []
        
        # ============================================================
        # HEADER
        # ============================================================
        lines.append("# 📊 PORTFOLIO ANALYSIS REPORT")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Analysis Period:** {self.parameters['start_date']} to {self.parameters['end_date']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # ============================================================
        # EXECUTIVE SUMMARY
        # ============================================================
        lines.append("## 📋 Executive Summary")
        lines.append("")
        
        # Calcular insights clave
        portfolio_stats = self.portfolio_results.statistics
        sharpe = portfolio_stats.get('sharpe_ratio', 0)
        expected_return = portfolio_stats['expected_return']
        volatility = portfolio_stats['std_return']
        var_95 = portfolio_stats['var_95']
        
        # Bullet points con insights
        lines.append(f"- **Portfolio Expected Return:** {expected_return:.2f}% annualized")
        lines.append(f"- **Portfolio Volatility:** {volatility:.2f}% annualized")
        lines.append(f"- **Sharpe Ratio:** {sharpe:.3f}")
        lines.append(f"- **Value at Risk (95%):** ${abs(var_95):,.2f}")
        lines.append(f"- **Number of Assets:** {len(self.portfolio.assets)}")
        
        # Drift insight
        if 'weight_drift_analysis' in self.metadata:
            drift = self.metadata['weight_drift_analysis']
            lines.append(f"- **Weight Drift Magnitude:** {drift['drift_magnitude']*100:.2f}% (Buy and Hold strategy)")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # ============================================================
        # PORTFOLIO OVERVIEW
        # ============================================================
        lines.append("## 📈 Portfolio Overview")
        lines.append("")
        
        lines.append(f"**Period Analyzed:** {self.parameters['start_date']} to {self.parameters['end_date']}")
        lines.append(f"**Number of Assets:** {len(self.portfolio.assets)}")
        lines.append(f"**Investment Strategy:** Equal-weighted (Buy and Hold)")
        lines.append("")
        
        lines.append("### Assets Included:")
        lines.append("")
        lines.append("| Ticker | Initial Weight | Sector |")
        lines.append("|--------|----------------|--------|")
        
        # Mapeo de sectores (simplificado)
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Technology', 'TSLA': 'Technology', 'NVDA': 'Technology',
            'JPM': 'Financial', 'JNJ': 'Healthcare', 'PG': 'Consumer Staples',
            'KO': 'Consumer Staples', 'XOM': 'Energy', 'MCD': 'Consumer Discretionary'
        }
        
        for ticker in self.portfolio.assets.keys():
            weight = self.portfolio.weights[ticker]
            sector = sector_map.get(ticker, 'Other')
            lines.append(f"| {ticker} | {weight*100:.2f}% | {sector} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # ============================================================
        # HISTORICAL PERFORMANCE
        # ============================================================
        lines.append("## 💰 Historical Performance")
        lines.append("")
        
        # Calcular métricas históricas del portfolio
        portfolio_returns = self.portfolio.get_portfolio_returns()
        total_return = ((portfolio_returns + 1).prod() - 1) * 100
        
        # Días de trading
        n_days = len(portfolio_returns)
        n_years = n_days / 252
        
        # Retorno anualizado
        annualized_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100
        
        # Volatilidad histórica
        hist_volatility = portfolio_returns.std() * np.sqrt(252) * 100
        
        # Sharpe histórico
        rf_rate = self.parameters.get('risk_free_rate', 0.03)
        hist_sharpe = (annualized_return - rf_rate*100) / hist_volatility if hist_volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        lines.append(f"**Total Return:** {total_return:.2f}%")
        lines.append(f"**Annualized Return:** {annualized_return:.2f}%")
        lines.append(f"**Historical Volatility:** {hist_volatility:.2f}%")
        lines.append(f"**Historical Sharpe Ratio:** {hist_sharpe:.3f}")
        lines.append(f"**Maximum Drawdown:** {max_drawdown:.2f}%")
        lines.append(f"**Analysis Period:** {n_years:.2f} years ({n_days} trading days)")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # ============================================================
        # MONTE CARLO SIMULATION RESULTS
        # ============================================================
        lines.append("## 🎲 Monte Carlo Simulation Results")
        lines.append("")
        
        lines.append("### Simulation Parameters:")
        lines.append("")
        lines.append(f"- **Number of Simulations:** {self.parameters['n_simulations']:,}")
        lines.append(f"- **Time Horizon:** {self.parameters['time_horizon_days']} days (~{self.parameters['time_horizon_days']/252:.1f} year)")
        lines.append(f"- **Initial Investment:** ${self.parameters['initial_investment']:,.2f}")
        lines.append(f"- **Method:** Geometric Brownian Motion (GBM) with correlations")
        lines.append("")
        
        lines.append("### Expected Outcomes:")
        lines.append("")
        lines.append(f"- **Expected Final Value:** ${portfolio_stats['expected_value']:,.2f}")
        lines.append(f"- **Expected Return:** {portfolio_stats['expected_return']:.2f}%")
        lines.append(f"- **Standard Deviation:** {portfolio_stats['std_return']:.2f}%")
        lines.append(f"- **Volatility (annualized):** {volatility:.2f}%")
        lines.append("")
        
        lines.append("### Risk Metrics:")
        lines.append("")
        lines.append(f"- **Value at Risk (VaR 95%):** ${abs(var_95):,.2f}")
        lines.append(f"  - *Maximum loss expected with 95% confidence*")
        lines.append(f"- **Conditional VaR (CVaR 95%):** ${abs(portfolio_stats['cvar_95']):,.2f}")
        lines.append(f"  - *Expected loss in worst 5% of scenarios*")
        lines.append(f"- **Probability of Loss:** {portfolio_stats['prob_loss']:.2f}%")
        lines.append("")
        
        lines.append("### Scenario Analysis:")
        lines.append("")
        perc = self.portfolio_results.percentiles
        lines.append(f"- **Best Case (P95):** ${perc['p95']:,.2f}")
        lines.append(f"- **Expected Case (P50):** ${perc['p50']:,.2f}")
        lines.append(f"- **Worst Case (P5):** ${perc['p5']:,.2f}")
        lines.append(f"- **Absolute Best:** ${portfolio_stats['best_case']:,.2f}")
        lines.append(f"- **Absolute Worst:** ${portfolio_stats['worst_case']:,.2f}")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # ============================================================
        # WEIGHT DRIFT ANALYSIS
        # ============================================================
        if 'weight_drift_analysis' in self.metadata:
            lines.append("## ⚖️ Weight Drift Analysis (Buy and Hold)")
            lines.append("")
            
            drift = self.metadata['weight_drift_analysis']
            
            lines.append("In a Buy and Hold strategy without rebalancing, asset weights naturally drift")
            lines.append("as different assets have different returns. This analysis shows the average")
            lines.append("weight changes across all Monte Carlo simulations.")
            lines.append("")
            
            lines.append(f"**Total Drift Magnitude:** {drift['drift_magnitude']*100:.2f}%")
            lines.append("")
            
            lines.append("| Asset | Initial Weight | Final Weight (Avg) | Change |")
            lines.append("|-------|----------------|-------------------|---------|")
            
            for ticker in drift['initial_weights'].keys():
                init_w = drift['initial_weights'][ticker] * 100
                final_w = drift['average_final_weights'][ticker] * 100
                change = drift['weight_changes'][ticker] * 100
                change_str = f"+{change:.2f}%" if change > 0 else f"{change:.2f}%"
                lines.append(f"| {ticker} | {init_w:.2f}% | {final_w:.2f}% | {change_str} |")
            
            lines.append("")
            
            # Insights sobre drift
            max_gainer = max(drift['weight_changes'].items(), key=lambda x: x[1])
            max_loser = min(drift['weight_changes'].items(), key=lambda x: x[1])
            
            lines.append("**Key Observations:**")
            lines.append(f"- **Biggest Weight Gainer:** {max_gainer[0]} (+{max_gainer[1]*100:.2f}pp)")
            lines.append(f"- **Biggest Weight Loser:** {max_loser[0]} ({max_loser[1]*100:.2f}pp)")
            lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # ============================================================
        # RISK ANALYSIS
        # ============================================================
        lines.append("## 🔍 Risk Analysis")
        lines.append("")
        
        lines.append("### Individual Asset Risk:")
        lines.append("")
        lines.append("| Asset | Volatility | Beta | Sharpe Ratio | VaR (95%) |")
        lines.append("|-------|------------|------|--------------|-----------|")
        
        # Obtener betas del portfolio histórico (no de simulaciones)
        if self.portfolio and hasattr(self.portfolio, 'calculate_beta'):
            historical_betas = self.portfolio.calculate_beta()
        else:
            historical_betas = {}
        
        for ticker, result in self.asset_results.items():
            vol = result.statistics['std_return']
            # Obtener beta del portfolio histórico, no de las simulaciones
            beta = historical_betas.get(ticker, 0.0)
            sharpe_asset = result.statistics.get('sharpe_ratio', 0)
            var_asset = result.statistics['var_95']
            lines.append(f"| {ticker} | {vol:.2f}% | {beta:.3f} | {sharpe_asset:.3f} | ${abs(var_asset):,.0f} |")
        
        lines.append("")
        lines.append(f"**Portfolio Volatility:** {volatility:.2f}%")
        lines.append(f"**Portfolio Sharpe Ratio:** {sharpe:.3f}")
        lines.append("")
        
        lines.append("### Correlation Matrix:")
        lines.append("")
        lines.append("*See correlation heatmap in visualizations for detailed view*")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # ============================================================
        # WARNINGS & CONSIDERATIONS
        # ============================================================
        if include_warnings:
            lines.append("## ⚠️ Warnings & Considerations")
            lines.append("")
            
            lines.append("### Model Assumptions:")
            lines.append("")
            lines.append("- **Geometric Brownian Motion:** Assumes log-normal distribution of returns")
            lines.append("- **Constant Parameters:** μ and σ assumed constant over simulation horizon")
            lines.append("- **No Rebalancing:** Buy and Hold strategy without portfolio adjustments")
            lines.append("- **No Transaction Costs:** Assumes frictionless trading")
            lines.append("- **Historical Correlation:** Assumes past correlations persist")
            lines.append("")
            
            lines.append("### Limitations:")
            lines.append("")
            lines.append("- Monte Carlo simulations are based on historical data")
            lines.append("- Past performance does not guarantee future results")
            lines.append("- Extreme market events (black swans) may not be captured")
            lines.append("- Model does not account for regime changes")
            lines.append("")
            
            lines.append("### Recommendations:")
            lines.append("")
            lines.append("- Consider regular portfolio rebalancing to maintain target weights")
            lines.append("- Monitor for significant changes in asset correlations")
            lines.append("- Review risk metrics periodically")
            lines.append("- Diversification does not eliminate all risk")
            lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # ============================================================
        # ASSET COMPARISON TABLE
        # ============================================================
        lines.append("## 📊 Asset Comparison Table")
        lines.append("")
        
        # Crear tabla completa
        df_summary = self.get_summary_table()
        
        lines.append("| Ticker | Type | Exp. Return | Volatility | Sharpe | VaR (95%) | Prob. Loss |")
        lines.append("|--------|------|-------------|------------|--------|-----------|------------|")
        
        for _, row in df_summary.iterrows():
            # Obtener Sharpe Ratio (asegurar que existe)
            sharpe = row.get('Sharpe_Ratio', 0.0)
            if pd.isna(sharpe):
                sharpe = 0.0
            
            # ← MOVER ESTA LÍNEA AQUÍ DENTRO (con 4 espacios más de indentación)
            lines.append(
                f"| {row['Ticker']} | {row['Type']} | "
                f"{row['Expected_Return_%']:.2f}% | {row['Volatility_%']:.2f}% | "
                f"{sharpe:.3f} | "
                f"${abs(row['VaR_95_$']):,.0f} | {row['Prob_Loss_%']:.2f}% |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # ============================================================
        # FOOTER
        # ============================================================
        lines.append("## 📝 Notes")
        lines.append("")
        lines.append("This report was automatically generated using Monte Carlo simulations.")
        lines.append(f"For visualizations, use the `.plots_report()` method.")
        lines.append("")
        lines.append(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Unir todas las líneas
        report = "\n".join(lines)
        
        # Guardar en archivo si se solicita
        if save_to_file:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"portfolio_report_{timestamp}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            if verbose:
                logger.info(f"\n✓ Reporte guardado en: {filename}")
        
        return report


    def plots_report(
        self,
        show: bool = True,
        save: bool = False,
        output_dir: str = "plots",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300
    ):
        """
        Genera todas las visualizaciones del análisis de Monte Carlo.
        
        Crea 10 visualizaciones completas:
        TIER 1 (8 gráficos principales):
        1. Dashboard Ejecutivo
        2. Evolución de Precios Históricos  
        3. Trayectorias Monte Carlo (Fan Chart)
        4. Distribución de Valores Finales
        5. Drift de Pesos
        6. Heatmap de Correlaciones
        7. Riesgo-Retorno Scatter
        8. Tabla Comparativa
        
        TIER 2 (2 gráficos adicionales):
        9. Beta Analysis (Riesgo Sistemático)
        10. Maximum Drawdown
        
        Args:
            show: Si True, muestra los gráficos
            save: Si True, guarda los gráficos en archivos
            output_dir: Carpeta donde guardar los gráficos
            figsize: Tamaño de las figuras
            dpi: Resolución de las imágenes
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Crear directorio si se va a guardar
        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("GENERANDO VISUALIZACIONES")
        logger.info("="*60)
        
        # ==================== GRÁFICO 1: DASHBOARD EJECUTIVO ====================
        logger.info("\n[1/10] Dashboard Ejecutivo...")
        fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig1.suptitle('Portfolio Analysis Dashboard', fontsize=16, fontweight='bold')
        
        stats = self.portfolio_results.statistics
        
        # Métrica 1: Retorno Esperado
        axes[0, 0].text(0.5, 0.5, f"{stats['expected_return']:.2f}%", 
                       ha='center', va='center', fontsize=36, fontweight='bold')
        axes[0, 0].text(0.5, 0.2, 'Expected Return\n(Annualized)', ha='center', fontsize=12)
        axes[0, 0].axis('off')
        
        # Métrica 2: Sharpe Ratio
        sharpe = stats.get('sharpe_ratio', 0)
        axes[0, 1].text(0.5, 0.5, f"{sharpe:.3f}", 
                       ha='center', va='center', fontsize=36, fontweight='bold')
        axes[0, 1].text(0.5, 0.2, 'Sharpe Ratio', ha='center', fontsize=14)
        axes[0, 1].axis('off')
        
        # Métrica 3: Volatilidad
        axes[0, 2].text(0.5, 0.5, f"{stats['std_return']:.2f}%", 
                       ha='center', va='center', fontsize=36, fontweight='bold', color='orange')
        axes[0, 2].text(0.5, 0.2, 'Volatility\n(Annualized)', ha='center', fontsize=12)
        axes[0, 2].axis('off')
        
        # Métrica 4: VaR 95%
        axes[1, 0].text(0.5, 0.5, f"${abs(stats['var_95']):,.0f}", 
                       ha='center', va='center', fontsize=30, fontweight='bold', color='red')
        axes[1, 0].text(0.5, 0.2, 'VaR (95%)', ha='center', fontsize=14)
        axes[1, 0].axis('off')
        
        # Métrica 5: Prob. Pérdida
        axes[1, 1].text(0.5, 0.5, f"{stats['prob_loss']:.1f}%", 
                       ha='center', va='center', fontsize=36, fontweight='bold')
        axes[1, 1].text(0.5, 0.2, 'Prob. Loss', ha='center', fontsize=14)
        axes[1, 1].axis('off')
        
        # Métrica 6: Número de simulaciones
        axes[1, 2].text(0.5, 0.5, f"{self.parameters['n_simulations']:,}", 
                       ha='center', va='center', fontsize=36, fontweight='bold', color='green')
        axes[1, 2].text(0.5, 0.2, 'Simulations', ha='center', fontsize=14)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/01_dashboard.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # ==================== GRÁFICO 2: PRECIOS HISTÓRICOS ====================
        logger.info("[2/10] Evolución de Precios Históricos...")
        fig2, ax = plt.subplots(figsize=figsize)
        
        # Obtener precios normalizados
        prices_norm = self.portfolio.get_portfolio_prices_normalized()
        
        # Plot cada activo
        for ticker in self.portfolio.assets.keys():
            ax.plot(prices_norm.index, prices_norm[ticker], label=ticker, alpha=0.6, linewidth=1.5)
        
        # Plot portfolio en negrita
        ax.plot(prices_norm.index, prices_norm['Portfolio'], 
               label='Portfolio', linewidth=3, color='black', linestyle='--')
        
        ax.set_title('Historical Price Evolution (Normalized to 100)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price (Base 100)')
        ax.legend(loc='best', ncol=3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/02_historical_prices.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # ==================== GRÁFICO 3: FAN CHART MONTE CARLO ====================
        logger.info("[3/10] Trayectorias Monte Carlo (Fan Chart)...")
        
        # Nota: Las trayectorias completas no se guardan por memoria
        # Solo graficamos percentiles
        fig3, ax = plt.subplots(figsize=figsize)
        
        days = np.arange(0, self.parameters['time_horizon_days'] + 1)
        init_val = self.parameters['initial_investment']
        
        # Simular algunas trayectorias de ejemplo (100)
        np.random.seed(42)
        sample_paths = np.zeros((100, len(days)))
        sample_paths[:, 0] = init_val
        
        for i in range(100):
            for t in range(1, len(days)):
                ret = np.random.normal(
                    stats['expected_return'] / 100 / 252,
                    stats['std_return'] / 100 / np.sqrt(252)
                )
                sample_paths[i, t] = sample_paths[i, t-1] * (1 + ret)
        
        # Calcular percentiles
        p5 = np.percentile(sample_paths, 5, axis=0)
        p25 = np.percentile(sample_paths, 25, axis=0)
        p50 = np.percentile(sample_paths, 50, axis=0)
        p75 = np.percentile(sample_paths, 75, axis=0)
        p95 = np.percentile(sample_paths, 95, axis=0)
        
        # Plot
        ax.fill_between(days, p5, p95, alpha=0.2, color='blue', label='90% CI (P5-P95)')
        ax.fill_between(days, p25, p75, alpha=0.3, color='blue', label='50% CI (P25-P75)')
        ax.plot(days, p50, 'b-', linewidth=3, label='Median (P50)')
        ax.plot(days, [init_val] * len(days), 'k--', linewidth=1, label='Initial Investment', alpha=0.5)
        
        ax.set_title('Monte Carlo Simulation - Portfolio Value Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/03_monte_carlo_paths.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # ==================== GRÁFICO 4: DISTRIBUCIÓN VALORES FINALES ====================
        logger.info("[4/10] Distribución de Valores Finales...")
        fig4, ax = plt.subplots(figsize=figsize)
        
        final_vals = self.portfolio_results.final_values
        
        ax.hist(final_vals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Líneas verticales
        ax.axvline(stats['expected_value'], color='green', linestyle='--', linewidth=2, label=f"Expected: ${stats['expected_value']:,.0f}")
        ax.axvline(init_val, color='gray', linestyle=':', linewidth=2, label=f"Initial: ${init_val:,.0f}")
        ax.axvline(stats['var_95'], color='red', linestyle='--', linewidth=2, label=f"VaR 95%: ${stats['var_95']:,.0f}")
        
        ax.set_title('Distribution of Final Portfolio Values', fontsize=14, fontweight='bold')
        ax.set_xlabel('Final Value ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/04_distribution.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # ==================== GRÁFICO 5: DRIFT DE PESOS ====================
        logger.info("[5/10] Drift de Pesos...")
        
        if 'weight_drift_analysis' in self.metadata:
            fig5, ax = plt.subplots(figsize=figsize)
            
            drift = self.metadata['weight_drift_analysis']
            tickers = list(drift['initial_weights'].keys())
            
            x = np.arange(len(tickers))
            width = 0.35
            
            initial = [drift['initial_weights'][t] * 100 for t in tickers]
            final = [drift['average_final_weights'][t] * 100 for t in tickers]
            
            ax.bar(x - width/2, initial, width, label='Initial', alpha=0.8)
            ax.bar(x + width/2, final, width, label='Final (Avg)', alpha=0.8)
            
            ax.set_title('Weight Drift Analysis (Buy and Hold)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Weight (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(tickers, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            if save:
                plt.savefig(f"{output_dir}/05_weight_drift.png", dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()
        
        # ==================== GRÁFICO 6: CORRELACIONES ====================
        logger.info("[6/10] Heatmap de Correlaciones...")
        fig6, ax = plt.subplots(figsize=(10, 8))
        
        corr = self.portfolio.correlation_matrix
        
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/06_correlations.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # ==================== GRÁFICO 7: RIESGO-RETORNO ====================
        logger.info("[7/10] Riesgo-Retorno Scatter...")
        fig7, ax = plt.subplots(figsize=figsize)
        
        # Plot activos individuales
        for ticker, result in self.asset_results.items():
            ax.scatter(result.statistics['std_return'], 
                      result.statistics['expected_return'],
                      s=100, alpha=0.6, label=ticker)
        
        # Plot portfolio
        ax.scatter(stats['std_return'], stats['expected_return'],
                  s=300, color='black', marker='D', 
                  label='Portfolio', edgecolors='white', linewidth=2)
        
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.set_xlabel('Volatility (Annual %)')
        ax.set_ylabel('Expected Return (Annual %)')
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/07_risk_return.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # ==================== GRÁFICO 8: TABLA COMPARATIVA ====================
        logger.info("[8/10] Tabla Comparativa...")
        
        df_summary = self.get_summary_table()
        
        # ========== FORMATEAR TODOS LOS NÚMEROS A 2 DECIMALES ==========
        # Crear copia para formatear
        df_formatted = df_summary.copy()
        
        # Columnas que NO deben formatearse (columnas de texto)
        text_columns = ['Asset', 'Ticker', 'Type']
        
        # Formatear cada columna numérica
        for col in df_formatted.columns:
            if col not in text_columns:  # Solo formatear columnas numéricas
                # Convertir a numérico si es string
                try:
                    df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce')
                    # Formatear con 2 decimales y separador de miles
                    df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                except:
                    pass  # Si falla, dejar como está
        
        # ================================================================
        
        fig8, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Crear tabla con datos formateados
        table = ax.table(cellText=df_formatted.values,
                        colLabels=df_formatted.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Estilo
        for i in range(len(df_formatted.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight portfolio row
        for i in range(len(df_formatted.columns)):
            if df_formatted.iloc[0, 0] in ['Portfolio', 'PORTFOLIO']:
                table[(1, i)].set_facecolor('#d4e6f1')
        
        ax.set_title('Portfolio & Assets Comparison', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/08_comparison_table.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # ==================== GRÁFICO 9: BETA ANALYSIS ====================
        logger.info("[9/10] Beta Analysis (Riesgo Sistemático)...")
        
        # Obtener betas del portfolio histórico
        if self.portfolio and hasattr(self.portfolio, 'calculate_beta'):
            betas = self.portfolio.calculate_beta()
        else:
            betas = {}
            logger.warning("⚠️  No se pueden calcular betas - market_index no disponible")
        
        if betas:
            # Preparar datos para el gráfico
            tickers = []
            beta_values = []
            volatilities = []
            sharpe_ratios = []
            
            for ticker, result in self.asset_results.items():
                if ticker in betas:
                    tickers.append(ticker)
                    beta_values.append(betas[ticker])
                    volatilities.append(result.statistics['std_return'])
                    sharpe_ratios.append(result.statistics.get('sharpe_ratio', 0))
            
            # Crear figura
            fig9, ax = plt.subplots(figsize=(14, 10))
            
            # Scatter plot con color según Sharpe Ratio
            scatter = ax.scatter(beta_values, volatilities, 
                                c=sharpe_ratios, cmap='RdYlGn',
                                s=400, alpha=0.7, edgecolors='black', linewidth=2)
            
            # Añadir labels de tickers
            for i, ticker in enumerate(tickers):
                ax.annotate(ticker, (beta_values[i], volatilities[i]),
                           fontsize=11, fontweight='bold',
                           ha='center', va='center')
            
            # Línea vertical en β = 1.0 (riesgo del mercado)
            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label='Market Risk (β=1.0)')
            
            # Regiones de volatilidad
            vol_threshold = 50
            ax.axhspan(0, vol_threshold, alpha=0.05, color='green')
            ax.axhspan(vol_threshold, max(volatilities)*1.1, alpha=0.05, color='red')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Sharpe Ratio', fontsize=12, fontweight='bold')
            
            # Etiquetas y título
            ax.set_xlabel('Beta (Systematic Risk)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Volatility - σ (%)', fontsize=14, fontweight='bold')
            ax.set_title('Risk Analysis: Beta vs Volatility\n' + 
                        'Systematic Risk vs Total Risk',
                        fontsize=16, fontweight='bold', pad=20)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Leyenda
            ax.legend(loc='upper left', fontsize=11)
            
            # Añadir anotaciones de cuadrantes
            max_beta = max(beta_values) * 1.05 if beta_values else 2.0
            max_vol = max(volatilities) * 1.05 if volatilities else 100
            
            # Cuadrante 1: Defensive (Low β, High Vol)
            if max_vol > vol_threshold:
                ax.text(0.3, max_vol*0.92, 'Defensive\n(Low β, High Vol)', 
                       fontsize=9, ha='center', va='top', 
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            # Cuadrante 2: Aggressive (High β, High Vol)
            if max_beta > 1.2 and max_vol > vol_threshold:
                ax.text(max_beta*0.85, max_vol*0.92, 'Aggressive\n(High β, High Vol)', 
                       fontsize=9, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            
            # Cuadrante 3: Stable (Low β, Low Vol)
            ax.text(0.3, 15, 'Stable\n(Low β, Low Vol)', 
                   fontsize=9, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # Cuadrante 4: Growth (High β, Low Vol)
            if max_beta > 1.2:
                ax.text(max_beta*0.85, 15, 'Growth\n(High β, Low Vol)', 
                       fontsize=9, ha='center', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            # Añadir texto explicativo
            explanation = (
                "Beta (β) measures systematic risk:\n"
                "β < 1.0: Less volatile than market (defensive)\n"
                "β = 1.0: Same volatility as market\n"
                "β > 1.0: More volatile than market (aggressive)"
            )
            ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            if save:
                plt.savefig(f"{output_dir}/09_beta_analysis.png", dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()
        else:
            logger.warning("⚠️  Gráfico 9 omitido: No hay datos de beta disponibles")
        
        # ==================== GRÁFICO 10: MAXIMUM DRAWDOWN ====================
        logger.info("[10/10] Maximum Drawdown...")
        fig9, ax = plt.subplots(figsize=figsize)
        
        # Calcular drawdown del portfolio
        portfolio_returns = self.portfolio.get_portfolio_returns()
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=2)
        
        # Marcar máximo drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.scatter([max_dd_date], [max_dd], color='red', s=200, zorder=5, 
                  label=f'Max DD: {max_dd:.2f}%')
        
        ax.set_title('Portfolio Maximum Drawdown Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/10_max_drawdown.png", dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        logger.info("\n✓ Todas las visualizaciones generadas exitosamente")
        
        if save:
            logger.info(f"✓ Gráficos guardados en: {output_dir}/")


    def print_summary(self):
        """Imprime un resumen formateado de todos los resultados."""
        logger.info(f"\n{'='*80}")
        logger.info("RESUMEN CONSOLIDADO - SIMULACIONES MONTE CARLO")
        logger.info(f"{'='*80}")
        
        logger.info(f"\nMetadata:")
        for key, value in self.metadata.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\n{'='*60}")
        logger.info("PORTFOLIO")
        logger.info(f"{'='*60}")
        
        summary = self.portfolio_results.get_summary()
        for key, value in summary.items():
            if key not in ['Tipo', 'Ticker', 'Percentiles']:
                logger.info(f"  {key}: {value}")
        
        logger.info("\n  Percentiles:")
        for key, value in summary['Percentiles'].items():
            logger.info(f"    {key}: {value}")
        
        logger.info(f"\n{'='*60}")
        logger.info("ACTIVOS INDIVIDUALES")
        logger.info(f"{'='*60}")
        
        for ticker, result in self.asset_results.items():
            logger.info(f"\n  {ticker}:")
            logger.info(f"    Retorno Esperado: {result.statistics['expected_return']:.2f}%")
            logger.info(f"    Volatilidad: {result.statistics['std_return']:.2f}%")
            logger.info(f"    Valor Esperado: ${result.statistics['expected_value']:,.2f}")
            logger.info(f"    VaR (95%): ${result.statistics['var_95']:,.2f}")
            logger.info(f"    Prob. Pérdida: {result.statistics['prob_loss']:.2f}%")
    
    def __repr__(self) -> str:
        """Representación en string del objeto."""
        return (f"ConsolidatedResults(portfolio + {len(self.asset_results)} assets, "
                f"n_sims={self.parameters['n_simulations']})")


# ==================== DATACLASS: SIMULATION RESULTS ====================

@dataclass
class SimulationResults:
    """
    Contenedor para resultados de simulaciones Monte Carlo.
    
    Almacena:
    - Trayectorias completas de todas las simulaciones
    - Valores finales
    - Estadísticas agregadas
    - Percentiles y métricas de riesgo
    
    Attributes:
        simulation_type: 'portfolio' o 'asset'
        ticker: Nombre del activo (si es simulación individual)
        initial_value: Inversión inicial
        final_values: Array con valores finales de cada simulación
        paths: Array [n_simulations, time_horizon] con todas las trayectorias
        statistics: Diccionario con estadísticas agregadas
        percentiles: Diccionario con percentiles calculados
        parameters: Parámetros usados en la simulación
    """
    
    simulation_type: str  # 'portfolio' o 'asset'
    ticker: Optional[str]
    initial_value: float
    final_values: np.ndarray
    paths: np.ndarray
    statistics: dict
    percentiles: dict
    parameters: dict
    
    def get_summary(self) -> dict:
        """
        Retorna un resumen completo de los resultados.
        
        Returns:
            Diccionario con todas las métricas formateadas
        """
        return {
            'Tipo': self.simulation_type.upper(),
            'Ticker': self.ticker if self.ticker else 'Portfolio',
            'Inversión Inicial': f"${self.initial_value:,.2f}",
            'Valor Esperado Final': f"${self.statistics['expected_value']:,.2f}",
            'Retorno Esperado': f"{self.statistics['expected_return']:.2f}%",
            'Mejor Escenario': f"${self.statistics['best_case']:,.2f}",
            'Peor Escenario': f"${self.statistics['worst_case']:,.2f}",
            'Probabilidad de Pérdida': f"{self.statistics['prob_loss']:.2f}%",
            'VaR (95%)': f"${self.statistics['var_95']:,.2f}",
            'CVaR (95%)': f"${self.statistics['cvar_95']:,.2f}",
            'Percentiles': {
                'P5': f"${self.percentiles['p5']:,.2f}",
                'P25': f"${self.percentiles['p25']:,.2f}",
                'P50 (Mediana)': f"${self.percentiles['p50']:,.2f}",
                'P75': f"${self.percentiles['p75']:,.2f}",
                'P95': f"${self.percentiles['p95']:,.2f}",
            }
        }
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """
        Guarda las estadísticas en un archivo JSON.
        
        No guarda las trayectorias completas (demasiado pesado),
        solo estadísticas y percentiles.
        
        Args:
            filename: Nombre del archivo (opcional, se genera automático)
            
        Returns:
            Nombre del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ticker_str = self.ticker if self.ticker else 'portfolio'
            filename = f"monte_carlo_{ticker_str}_{timestamp}.json"
        
        # Preparar datos para JSON
        data = {
            'simulation_info': {
                'type': self.simulation_type,
                'ticker': self.ticker,
                'timestamp': datetime.now().isoformat(),
            },
            'parameters': self.parameters,
            'statistics': self.statistics,
            'percentiles': self.percentiles,
            'final_values_sample': {
                'min': float(self.final_values.min()),
                'max': float(self.final_values.max()),
                'mean': float(self.final_values.mean()),
                'std': float(self.final_values.std()),
            }
        }
        
        # Guardar
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Resultados guardados en: {filename}")
        return filename
    
    def __repr__(self) -> str:
        """Representación en string del objeto."""
        ticker_str = self.ticker if self.ticker else 'Portfolio'
        return (f"SimulationResults({ticker_str}, "
                f"n_sims={len(self.final_values)}, "
                f"expected_return={self.statistics['expected_return']:.2f}%)")


# ==================== CLASE: MONTE CARLO SIMULATOR ====================

class MonteCarloSimulator:
    """
    Simulador Monte Carlo para portfolios y activos individuales.
    
    Implementa el método Geometric Brownian Motion (GBM) con correlaciones
    entre activos usando descomposición de Cholesky.
    
    Metodología:
    1. Para cada activo: S_t = S_0 * exp((μ - σ²/2)*Δt + σ*√Δt*Z)
    2. Z se genera correlacionado usando Cholesky de matriz de covarianza
    3. Se mantienen las correlaciones históricas entre activos
    
    Attributes:
        portfolio: Objeto Portfolio con los activos a simular
        n_simulations: Número de simulaciones Monte Carlo
        time_horizon: Horizonte temporal en días
        initial_investment: Inversión inicial en dólares
        confidence_level: Nivel de confianza para VaR/CVaR
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        n_simulations: int = 10000,
        time_horizon: int = 252,
        initial_investment: float = 100000.0,
        confidence_level: float = 0.95
    ):
        """
        Inicializa el simulador Monte Carlo.
        
        Args:
            portfolio: Portfolio a simular
            n_simulations: Número de simulaciones (default: 10,000)
            time_horizon: Días a simular (default: 252 = 1 año)
            initial_investment: Capital inicial (default: $100,000)
            confidence_level: Nivel de confianza (default: 0.95)
        """
        self.portfolio = portfolio
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.initial_investment = initial_investment
        self.confidence_level = confidence_level
        
        # Logging
        logger.info(f"\n{'='*60}")
        logger.info("Inicializando Monte Carlo Simulator")
        logger.info(f"{'='*60}")
        logger.info(f"  Activos: {len(portfolio.assets)}")
        logger.info(f"  Simulaciones: {n_simulations:,}")
        logger.info(f"  Horizonte: {time_horizon} días")
        logger.info(f"  Inversión inicial: ${initial_investment:,.2f}")
        logger.info(f"  Nivel de confianza: {confidence_level*100}%")
    
    def simulate_portfolio(self) -> SimulationResults:
        """
        Simula el portfolio completo usando GBM con correlaciones.
        
        Proceso:
        1. Extrae parámetros del portfolio (retornos, volatilidades, correlaciones)
        2. Genera retornos correlacionados usando Cholesky
        3. Simula evolución de cada activo
        4. Combina según pesos del portfolio
        5. Calcula estadísticas
        
        Returns:
            SimulationResults con todas las métricas
        """
        logger.info(f"\n{'='*60}")
        logger.info("SIMULANDO PORTFOLIO")
        logger.info(f"{'='*60}")
        
        # 1. Extraer parámetros del portfolio
        tickers = list(self.portfolio.assets.keys())
        n_assets = len(tickers)
        
        # Retornos esperados (diarios)
        mean_returns = np.array([
            self.portfolio.assets[ticker].statistics['mean_return_daily']
            for ticker in tickers
        ])
        
        # Volatilidades (diarias)
        volatilities = np.array([
            self.portfolio.assets[ticker].statistics['volatility_daily']
            for ticker in tickers
        ])
        
        # Matriz de correlación
        corr_matrix = self.portfolio.correlation_matrix.values
        
        # Pesos
        weights = np.array([self.portfolio.weights[ticker] for ticker in tickers])
        
        logger.info(f"  Activos: {', '.join(tickers)}")
        logger.info(f"  Generando {self.n_simulations:,} simulaciones...")
        
        # 2. Generar retornos correlacionados
        asset_paths = self._simulate_correlated_assets(
            mean_returns=mean_returns,
            volatilities=volatilities,
            corr_matrix=corr_matrix
        )
        
        # 3. Calcular valor del portfolio en cada simulación
        # asset_paths shape: [n_simulations, n_assets, time_horizon]
        # weights shape: [n_assets]
        
        # Calcular valor de cada activo: inversión_inicial * peso * trayectoria
        asset_values = np.zeros_like(asset_paths)
        for i, ticker in enumerate(tickers):
            asset_values[:, i, :] = self.initial_investment * weights[i] * asset_paths[:, i, :]
        
        # Sumar todos los activos para obtener valor del portfolio
        portfolio_paths = asset_values.sum(axis=1)  # [n_simulations, time_horizon]
        
        # 4. Valores finales
        final_values = portfolio_paths[:, -1]
        
        logger.info(f"  ✓ Simulaciones completadas")
        
        # 5. Calcular estadísticas
        statistics = self._calculate_statistics(final_values)
        percentiles = self._calculate_percentiles(final_values)
        
        # 6. Crear objeto de resultados
        results = SimulationResults(
            simulation_type='portfolio',
            ticker=None,
            initial_value=self.initial_investment,
            final_values=final_values,
            paths=portfolio_paths,
            statistics=statistics,
            percentiles=percentiles,
            parameters=self._get_parameters_dict()
        )
        
        logger.info(f"\n✓ Simulación de portfolio completada")
        self._log_summary(results)
        
        return results
    
    def simulate_asset(self, ticker: str) -> SimulationResults:
        """
        Simula un activo individual usando GBM.
        
        Args:
            ticker: Símbolo del activo a simular
            
        Returns:
            SimulationResults con todas las métricas
        """
        if ticker not in self.portfolio.assets:
            raise ValueError(f"Ticker '{ticker}' no encontrado en el portfolio")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMULANDO ACTIVO: {ticker}")
        logger.info(f"{'='*60}")
        
        asset = self.portfolio.assets[ticker]
        
        # Parámetros del activo
        mean_return = asset.statistics['mean_return_daily']
        volatility = asset.statistics['volatility_daily']
        
        logger.info(f"  Retorno diario: {mean_return*100:.4f}%")
        logger.info(f"  Volatilidad diaria: {volatility*100:.4f}%")
        logger.info(f"  Generando {self.n_simulations:,} simulaciones...")
        
        # Simular usando GBM
        paths = self._simulate_gbm(
            mean_return=mean_return,
            volatility=volatility
        )
        
        # Valores finales
        final_values = self.initial_investment * paths[:, -1]
        
        logger.info(f"  ✓ Simulaciones completadas")
        
        # Calcular estadísticas
        statistics = self._calculate_statistics(final_values)
        percentiles = self._calculate_percentiles(final_values)
        
        
        # Crear objeto de resultados
        results = SimulationResults(
            simulation_type='asset',
            ticker=ticker,
            initial_value=self.initial_investment,
            final_values=final_values,
            paths=self.initial_investment * paths,
            statistics=statistics,
            percentiles=percentiles,
            parameters=self._get_parameters_dict()
        )
        
        logger.info(f"\n✓ Simulación de {ticker} completada")
        self._log_summary(results)
        
        return results
    
    def _simulate_correlated_assets(
        self,
        mean_returns: np.ndarray,
        volatilities: np.ndarray,
        corr_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Simula múltiples activos correlacionados usando Cholesky.
        
        Metodología:
        1. Descomposición de Cholesky de la matriz de correlación: L * L^T = Corr
        2. Generar retornos independientes Z ~ N(0,1)
        3. Correlacionar: Z_corr = L * Z
        4. Aplicar GBM a cada activo con retornos correlacionados
        
        Args:
            mean_returns: Array de retornos esperados (diarios) [n_assets]
            volatilities: Array de volatilidades (diarias) [n_assets]
            corr_matrix: Matriz de correlación [n_assets, n_assets]
            
        Returns:
            Array [n_simulations, n_assets, time_horizon] con trayectorias
        """
        n_assets = len(mean_returns)
        dt = 1.0  # Paso de tiempo diario
        
        # Descomposición de Cholesky
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            logger.warning("  ⚠ Matriz de correlación no es positiva definida, usando aproximación")
            # Añadir pequeña diagonal para regularización
            corr_matrix_reg = corr_matrix + np.eye(n_assets) * 1e-6
            L = np.linalg.cholesky(corr_matrix_reg)
        
        # Inicializar array de trayectorias
        paths = np.ones((self.n_simulations, n_assets, self.time_horizon))
        
        # Generar todas las simulaciones
        for t in range(1, self.time_horizon):
            # Generar retornos independientes N(0,1)
            Z_independent = np.random.randn(self.n_simulations, n_assets)
            
            # Correlacionar usando Cholesky: Z_corr = Z * L^T
            Z_correlated = Z_independent @ L.T
            
            # Aplicar GBM a cada activo
            for i in range(n_assets):
                drift = (mean_returns[i] - 0.5 * volatilities[i]**2) * dt
                diffusion = volatilities[i] * np.sqrt(dt) * Z_correlated[:, i]
                paths[:, i, t] = paths[:, i, t-1] * np.exp(drift + diffusion)
        
        return paths
    
    def _simulate_gbm(
        self,
        mean_return: float,
        volatility: float
    ) -> np.ndarray:
        """
        Simula un activo individual usando Geometric Brownian Motion.
        
        Fórmula: S_t = S_{t-1} * exp((μ - σ²/2)*Δt + σ*√Δt*Z)
        
        Args:
            mean_return: Retorno esperado (diario)
            volatility: Volatilidad (diaria)
            
        Returns:
            Array [n_simulations, time_horizon] con trayectorias normalizadas
        """
        dt = 1.0  # Paso de tiempo diario
        
        # Inicializar trayectorias (empiezan en 1.0)
        paths = np.ones((self.n_simulations, self.time_horizon))
        
        # Generar retornos aleatorios
        Z = np.random.randn(self.n_simulations, self.time_horizon - 1)
        
        # Aplicar GBM
        drift = (mean_return - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt) * Z
        
        # Calcular trayectorias
        for t in range(1, self.time_horizon):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion[:, t-1])
        
        return paths
    
    def _calculate_statistics(self, final_values: np.ndarray) -> dict:
        """
        Calcula estadísticas agregadas de los valores finales.
        
        Args:
            final_values: Array con valores finales de cada simulación
            
        Returns:
            Diccionario con todas las estadísticas
        """
        stats = {}
        
        # Estadísticas básicas
        stats['expected_value'] = float(final_values.mean())
        stats['median_value'] = float(np.median(final_values))
        stats['std_value'] = float(final_values.std())
        stats['min_value'] = float(final_values.min())
        stats['max_value'] = float(final_values.max())
        
        # Retornos
        returns = (final_values - self.initial_investment) / self.initial_investment * 100
        stats['expected_return'] = float(returns.mean())
        stats['median_return'] = float(np.median(returns))
        stats['std_return'] = float(returns.std())
        
        # Mejor y peor caso
        stats['best_case'] = float(final_values.max())
        stats['worst_case'] = float(final_values.min())
        
        # Probabilidad de pérdida
        losses = final_values < self.initial_investment
        stats['prob_loss'] = float(losses.sum() / len(final_values) * 100)
        
        # Value at Risk (VaR) al nivel de confianza especificado
        var_percentile = (1 - self.confidence_level) * 100
        var_value = np.percentile(final_values, var_percentile)
        stats['var_95'] = float(var_value)
        stats['var_loss'] = float(self.initial_investment - var_value)
        
        # Conditional VaR (CVaR) - pérdida esperada dado que se excede el VaR
        var_exceedances = final_values[final_values <= var_value]
        if len(var_exceedances) > 0:
            stats['cvar_95'] = float(var_exceedances.mean())
            stats['cvar_loss'] = float(self.initial_investment - var_exceedances.mean())
        else:
            stats['cvar_95'] = float(var_value)
            stats['cvar_loss'] = float(self.initial_investment - var_value)

        # ========== SHARPE RATIO  ==========
        
        # Obtener tasa libre de riesgo
        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'risk_free_rate'):
            if self.portfolio.risk_free_rate is not None:
                # Calcular tasa libre de riesgo anual promedio
                rf_annual = float(self.portfolio.risk_free_rate.mean() * 252 * 100)  # En %
            else:
                rf_annual = 2.0  # Default 2% si no hay datos
        else:
            rf_annual = 2.0  # Default 2%
        
        # Anualizar retorno y volatilidad para Sharpe Ratio
        annual_return = stats['expected_return']  # Ya está en % anual
        annual_vol = stats['std_return']  # Ya está en % anual
        
        # Calcular Sharpe Ratio
        if annual_vol > 0:
            stats['sharpe_ratio'] = float((annual_return - rf_annual) / annual_vol)
        else:
            stats['sharpe_ratio'] = 0.0

        
        return stats
    
    def _calculate_percentiles(self, final_values: np.ndarray) -> dict:
        """
        Calcula percentiles de los valores finales.
        
        Args:
            final_values: Array con valores finales
            
        Returns:
            Diccionario con percentiles
        """
        percentiles = {}
        
        percentile_levels = [5, 25, 50, 75, 95]
        percentile_names = ['p5', 'p25', 'p50', 'p75', 'p95']
        
        for level, name in zip(percentile_levels, percentile_names):
            percentiles[name] = float(np.percentile(final_values, level))
        
        return percentiles
    
    
    def _get_parameters_dict(self) -> dict:
        """Retorna diccionario con parámetros de la simulación."""
        return {
            'n_simulations': self.n_simulations,
            'time_horizon_days': self.time_horizon,
            'time_horizon_years': self.time_horizon / 252,
            'initial_investment': self.initial_investment,
            'confidence_level': self.confidence_level,
        }
    
    def _log_summary(self, results: SimulationResults):
        """Imprime resumen de resultados."""
        logger.info("\n  Resultados:")
        logger.info(f"    Valor esperado: ${results.statistics['expected_value']:,.2f}")
        logger.info(f"    Retorno esperado: {results.statistics['expected_return']:.2f}%")
        logger.info(f"    Mejor caso: ${results.statistics['best_case']:,.2f}")
        logger.info(f"    Peor caso: ${results.statistics['worst_case']:,.2f}")
        logger.info(f"    Prob. pérdida: {results.statistics['prob_loss']:.2f}%")
        logger.info(f"    VaR (95%): ${results.statistics['var_95']:,.2f}")


# ==================== FUNCIÓN AUXILIAR ====================

def run_monte_carlo(
    portfolio: Portfolio,
    n_simulations: int = 10000,
    time_horizon: int = 252,
    initial_investment: float = 100000.0,
    confidence_level: float = 0.95,
    simulate_individuals: bool = False
) -> ConsolidatedResults:
    """
    Función wrapper para ejecutar simulaciones Monte Carlo fácilmente.
    
    Retorna un objeto ConsolidatedResults que puede guardarse en un único
    archivo JSON, evitando generar múltiples archivos individuales.
    
    Calcula automáticamente el análisis de drift de pesos para documentar
    cómo los pesos equiponderados iniciales cambian durante la simulación
    debido a la estrategia Buy and Hold.
    
    Args:
        portfolio: Portfolio a simular
        n_simulations: Número de simulaciones
        time_horizon: Horizonte temporal en días
        initial_investment: Inversión inicial
        confidence_level: Nivel de confianza
        simulate_individuals: Si True, simula también cada activo individual
        
    Returns:
        ConsolidatedResults con todos los resultados en un solo objeto
    """
    logger.info("\n" + "="*80)
    logger.info(" "*25 + "MONTE CARLO SIMULATION")
    logger.info("="*80)
    
    # Crear simulador
    simulator = MonteCarloSimulator(
        portfolio=portfolio,
        n_simulations=n_simulations,
        time_horizon=time_horizon,
        initial_investment=initial_investment,
        confidence_level=confidence_level
    )
    
    # Simular portfolio
    portfolio_results = simulator.simulate_portfolio()
    
    # Simular activos individuales
    asset_results = {}
    
    if simulate_individuals:
        logger.info("\n" + "="*60)
        logger.info("SIMULANDO ACTIVOS INDIVIDUALES")
        logger.info("="*60)
        
        for ticker in portfolio.assets.keys():
            asset_results[ticker] = simulator.simulate_asset(ticker)
    
    # Calcular análisis de drift de pesos
    logger.info("\n" + "="*60)
    logger.info("CALCULANDO DRIFT DE PESOS")
    logger.info("="*60)
    
    weight_drift = _calculate_weight_drift(
        portfolio=portfolio,
        asset_results=asset_results,
        initial_investment=initial_investment
    )
    
    logger.info("  ✓ Análisis de drift completado")
    logger.info(f"\n  Pesos Iniciales (equiponderados):")
    for ticker, weight in weight_drift['initial_weights'].items():
        logger.info(f"    {ticker}: {weight*100:.2f}%")
    
    logger.info(f"\n  Pesos Finales Promedio (después de {time_horizon} días):")
    for ticker, weight in weight_drift['average_final_weights'].items():
        change = weight_drift['weight_changes'][ticker]
        change_str = f"+{change*100:.2f}%" if change > 0 else f"{change*100:.2f}%"
        logger.info(f"    {ticker}: {weight*100:.2f}% ({change_str})")
    
    # Extraer fechas del portfolio
    first_asset = list(portfolio.assets.values())[0]
    start_date = first_asset.data['Date'].min().strftime('%Y-%m-%d')
    end_date = first_asset.data['Date'].max().strftime('%Y-%m-%d')
    
    # Crear metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'simulation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_assets': len(portfolio.assets),
        'asset_tickers': list(portfolio.assets.keys()),
        'start_date': start_date,
        'end_date': end_date,
        'weight_drift_analysis': weight_drift
    }
    
    # Añadir fechas a parameters también
    params = simulator._get_parameters_dict()
    params['start_date'] = start_date
    params['end_date'] = end_date
    
    # Crear objeto consolidado
    consolidated = ConsolidatedResults(
        portfolio=portfolio,
        portfolio_results=portfolio_results,
        asset_results=asset_results,
        parameters=params,
        metadata=metadata
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ TODAS LAS SIMULACIONES COMPLETADAS")
    logger.info("="*80)
    
    return consolidated


# ==================== FUNCIÓN AUXILIAR: WEIGHT DRIFT ====================

def _calculate_weight_drift(
    portfolio: Portfolio,
    asset_results: Dict[str, 'SimulationResults'],
    initial_investment: float
) -> dict:
    """
    Calcula el drift (desviación) de pesos durante las simulaciones.
    
    En una estrategia Buy and Hold, los pesos iniciales equiponderados
    cambian naturalmente durante la simulación debido a los diferentes
    rendimientos de cada activo. Esta función cuantifica ese cambio.
    
    Metodología:
    1. Para cada simulación, calcula el valor final de cada activo
    2. Calcula el peso final de cada activo en esa simulación
    3. Promedia los pesos finales across todas las simulaciones
    4. Compara con los pesos iniciales
    
    Args:
        portfolio: Portfolio original con pesos iniciales
        asset_results: Dict con SimulationResults de cada activo
        initial_investment: Inversión inicial total
        
    Returns:
        Dict con:
        - initial_weights: Pesos iniciales (equiponderados)
        - average_final_weights: Pesos promedio al final
        - weight_changes: Cambio de cada peso
        - drift_magnitude: Magnitud total del drift
    """
    # Validar que tenemos resultados de activos individuales
    if not asset_results:
        logger.warning("  ⚠ No hay resultados de activos individuales para calcular drift")
        return {
            'initial_weights': portfolio.weights,
            'average_final_weights': portfolio.weights,
            'weight_changes': {ticker: 0.0 for ticker in portfolio.weights.keys()},
            'drift_magnitude': 0.0,
            'note': 'Drift no calculado - no hay simulaciones individuales'
        }
    
    tickers = list(portfolio.weights.keys())
    n_simulations = len(list(asset_results.values())[0].final_values)
    
    # Matriz para almacenar pesos finales: [n_simulations, n_assets]
    final_weights_matrix = np.zeros((n_simulations, len(tickers)))
    
    # Para cada simulación, calcular los pesos finales
    for sim_idx in range(n_simulations):
        # Valor final de cada activo en esta simulación
        asset_final_values = {}
        total_value = 0.0
        
        for i, ticker in enumerate(tickers):
            # El valor final del activo en esta simulación
            asset_value = asset_results[ticker].final_values[sim_idx]
            asset_final_values[ticker] = asset_value
            total_value += asset_value
        
        # Calcular peso de cada activo en esta simulación
        for i, ticker in enumerate(tickers):
            if total_value > 0:
                final_weights_matrix[sim_idx, i] = asset_final_values[ticker] / total_value
            else:
                # Caso extremo: mantener pesos iniciales
                final_weights_matrix[sim_idx, i] = portfolio.weights[ticker]
    
    # Calcular promedio de pesos finales
    average_final_weights = {}
    for i, ticker in enumerate(tickers):
        average_final_weights[ticker] = float(final_weights_matrix[:, i].mean())
    
    # Calcular cambios
    weight_changes = {}
    for ticker in tickers:
        change = average_final_weights[ticker] - portfolio.weights[ticker]
        weight_changes[ticker] = float(change)
    
    # Calcular magnitud total del drift (suma de cambios absolutos)
    drift_magnitude = sum(abs(change) for change in weight_changes.values())
    
    return {
        'initial_weights': {ticker: float(weight) for ticker, weight in portfolio.weights.items()},
        'average_final_weights': average_final_weights,
        'weight_changes': weight_changes,
        'drift_magnitude': float(drift_magnitude),
        'note': f'Promedio de {n_simulations:,} simulaciones. Estrategia: Buy and Hold sin rebalanceo.'
    }


# ==================== FUNCIÓN PRINCIPAL DE EJEMPLO ====================

def main():
    """
    Función de ejemplo para demostrar el uso del simulador Monte Carlo.
    """
    print("="*60)
    print("FASE 4: SIMULACIONES MONTE CARLO")
    print("="*60)
    
    print("\nEste módulo está listo para ser usado.")
    print("\nPara un ejemplo completo, ejecuta:")
    print("  python main_analysis.py")
    print("\nO importa las clases en tu código:")
    print("  from monte_carlo import MonteCarloSimulator, run_monte_carlo")
    print("\nEjemplo de uso:")
    print("  simulator = MonteCarloSimulator(portfolio)")
    print("  results = simulator.simulate_portfolio()")
    print("  results.save_to_json()")


if __name__ == "__main__":
    main()