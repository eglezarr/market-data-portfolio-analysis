"""
price_series.py
Fase 3: Análisis Estadístico de Series de Precios y Carteras

Este módulo contiene las DataClasses principales para el análisis estadístico:
- PriceSeries: Representa una serie de precios individual (acción, índice)
- Portfolio: Representa una cartera de activos con pesos

Funcionalidades:
1. Cálculo automático de retornos y estadísticas básicas
2. Métricas avanzadas: Sharpe Ratio, Beta, VaR, Maximum Drawdown
3. Análisis de correlación y covarianza entre activos
4. Soporte para tasa libre de riesgo (T-Bills 3M)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging
import yfinance as yf
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== FUNCIÓN AUXILIAR: DESCARGAR T-BILLS ====================

def download_risk_free_rate(start_date: str, end_date: str) -> pd.Series:
    """
    Descarga la tasa libre de riesgo: 13 WEEK TREASURY BILL (^IRX).
    
    Argumentación metodológica:
    - Los T-Bills a 3 meses minimizan tanto el riesgo de crédito como 
      el riesgo de duración (volatilidad del precio del bono)
    - Se ajusta a la definición más pura del Ratio de Sharpe
    - La frecuencia diaria coincide con la frecuencia de retornos de activos
    
    Nota sobre Adjusted Close:
    - Si está disponible Adj Close para ^IRX, se usa preferentemente
    - De lo contrario, se usa Close
    - Para T-Bills, la diferencia suele ser mínima
    
    Args:
        start_date: Fecha inicial en formato 'YYYY-MM-DD'
        end_date: Fecha final en formato 'YYYY-MM-DD'
        
    Returns:
        Serie temporal con tasas diarias (en decimal, no porcentaje)
        
    Raises:
        Exception: Si no se pueden descargar los datos
    """
    logger.info("Descargando tasa libre de riesgo (^IRX - T-Bills 3M)...")
    
    try:
        # Descargar datos del T-Bill
        ticker = yf.Ticker("^IRX")
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError("No se pudieron descargar datos de ^IRX")
        
        # Preferir Adj Close si está disponible, sino usar Close
        if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
            rates = data['Adj Close']
            logger.info("  ✓ Usando Adjusted Close para T-Bills")
        else:
            rates = data['Close']
            logger.info("  ✓ Usando Close para T-Bills (Adj Close no disponible)")
        
        # ^IRX está en porcentaje anual, convertir a decimal diario
        # Fórmula: tasa_diaria = (tasa_anual / 100) / 252
        rates_daily = (rates / 100) / 252
        
        # IMPORTANTE: Convertir a timezone-naive para compatibilidad con otros datos
        # Esto evita el error "Cannot join tz-naive with tz-aware DatetimeIndex"
        if rates_daily.index.tz is not None:
            rates_daily.index = rates_daily.index.tz_localize(None)
        
        logger.info(f"  ✓ T-Bills descargados: {len(rates_daily)} observaciones")
        logger.info(f"  Rango: {rates_daily.index.min()} a {rates_daily.index.max()}")
        logger.info(f"  Tasa promedio anual: {rates.mean():.2f}%")
        
        return rates_daily
        
    except Exception as e:
        logger.error(f"  ✗ Error descargando T-Bills: {str(e)}")
        logger.warning("  ⚠ Usando tasa libre de riesgo = 0")
        # Retornar serie de ceros como fallback
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(0.0, index=date_range)


# ==================== DATACLASS: PRICESERIES ====================

@dataclass
class PriceSeries:
    """
    Representa una serie de precios de un activo individual.
    
    Calcula automáticamente:
    - Retornos diarios
    - Estadísticas básicas (media, volatilidad)
    - Métricas avanzadas (Sharpe, Max Drawdown, VaR)
    
    Attributes:
        ticker: Símbolo del activo (ej: 'AAPL')
        data: DataFrame con columnas ['Date', 'Close', 'Adj Close', ...]
        risk_free_rate: Serie temporal con tasa libre de riesgo (opcional)
        returns: Retornos diarios calculados automáticamente
        statistics: Dict con todas las métricas calculadas
    """
    
    ticker: str
    data: pd.DataFrame
    risk_free_rate: Optional[pd.Series] = None
    
    # Campos calculados automáticamente
    returns: pd.Series = field(init=False, repr=False)
    statistics: dict = field(init=False, repr=False)
    
    def __post_init__(self):
        """
        Inicialización automática después de crear la instancia.
        
        Calcula:
        1. Retornos diarios
        2. Todas las estadísticas
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Inicializando PriceSeries: {self.ticker}")
        logger.info(f"{'='*60}")
        
        # Validar datos
        self._validate_data()
        
        # Calcular retornos
        self.returns = self._calculate_returns()
        
        # Calcular estadísticas
        self.statistics = self._calculate_statistics()
        
        logger.info(f"✓ {self.ticker} inicializado correctamente")
        logger.info(f"  Observaciones: {len(self.data)}")
        logger.info(f"  Periodo: {self.data['Date'].min()} a {self.data['Date'].max()}")
        logger.info(f"  Retorno medio diario: {self.mean_return:.4f}%")
        logger.info(f"  Volatilidad diaria: {self.volatility:.4f}%")
    
    def _validate_data(self):
        """Valida que el DataFrame tenga las columnas necesarias."""
        required_columns = ['Date', 'Close']
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en datos de {self.ticker}")
        
        # Preferir Adj Close si está disponible
        if 'Adj Close' not in self.data.columns:
            logger.warning(f"  ⚠ {self.ticker}: 'Adj Close' no disponible, usando 'Close'")
            self.data['Adj Close'] = self.data['Close']
        
        # Asegurar que Date sea datetime
        if self.data['Date'].dtype != 'datetime64[ns]':
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Ordenar por fecha
        self.data = self.data.sort_values('Date').reset_index(drop=True)
    
    def _calculate_returns(self) -> pd.Series:
        """
        Calcula retornos diarios usando Adjusted Close.
        
        Fórmula: r_t = (P_t - P_{t-1}) / P_{t-1}
        
        Returns:
            Serie de retornos diarios (timezone-naive)
        """
        returns = self.data['Adj Close'].pct_change().dropna()
        returns.index = self.data['Date'][1:].values  # Alinear con fechas
        
        # Asegurar que el índice sea DatetimeIndex timezone-naive
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        
        return returns
    
    def _calculate_statistics(self) -> dict:
        """
        Calcula todas las estadísticas del activo.
        
        Returns:
            Diccionario con métricas diarias y anualizadas
        """
        stats = {}
        
        # Métricas básicas (diarias)
        stats['mean_return_daily'] = self.returns.mean()
        stats['volatility_daily'] = self.returns.std()
        stats['min_return'] = self.returns.min()
        stats['max_return'] = self.returns.max()
        stats['skewness'] = self.returns.skew()
        stats['kurtosis'] = self.returns.kurtosis()
        
        # Métricas anualizadas (asumiendo 252 días de trading)
        stats['mean_return_annual'] = stats['mean_return_daily'] * 252
        stats['volatility_annual'] = stats['volatility_daily'] * np.sqrt(252)
        
        # Sharpe Ratio (anualizado)
        stats['sharpe_ratio'] = self._calculate_sharpe_ratio()
        
        # Maximum Drawdown
        stats['max_drawdown'] = self._calculate_max_drawdown()
        
        # Value at Risk (VaR) al 95%
        stats['var_95'] = self._calculate_var(confidence_level=0.95)
        
        # Información adicional
        stats['observations'] = len(self.returns)
        stats['start_date'] = str(self.data['Date'].min())
        stats['end_date'] = str(self.data['Date'].max())
        
        return stats
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calcula el Sharpe Ratio anualizado.
        
        Fórmula: SR = (R_p - R_f) / σ_p
        
        Donde:
        - R_p: Retorno del activo (anualizado)
        - R_f: Tasa libre de riesgo (anualizada)
        - σ_p: Volatilidad del activo (anualizada)
        
        Returns:
            Sharpe Ratio (adimensional)
        """
        # Calcular volatilidad anualizada directamente
        volatility_annual = self.returns.std() * np.sqrt(252)
        mean_return_annual = self.returns.mean() * 252
        
        if self.risk_free_rate is None:
            logger.warning(f"  ⚠ {self.ticker}: Calculando Sharpe sin tasa libre de riesgo")
            rf_annual = 0.0
        else:
            # Alinear fechas entre retornos y tasa libre de riesgo
            aligned_returns, aligned_rf = self._align_returns_with_rf()
            
            # Calcular exceso de retorno
            excess_returns = aligned_returns - aligned_rf
            
            # Anualizar
            rf_annual = aligned_rf.mean() * 252
            mean_excess_return = excess_returns.mean() * 252
            
            # Sharpe Ratio
            if volatility_annual > 0:
                return mean_excess_return / volatility_annual
            else:
                return 0.0
        
        # Si no hay risk_free_rate, usar fórmula simplificada
        if volatility_annual > 0:
            return (mean_return_annual - rf_annual) / volatility_annual
        else:
            return 0.0
    
    def _align_returns_with_rf(self) -> Tuple[pd.Series, pd.Series]:
        """
        Alinea los retornos del activo con la tasa libre de riesgo por fecha.
        
        Returns:
            Tupla (retornos_alineados, rf_alineados)
        """
        # Convertir ambos índices a datetime si no lo son
        returns_with_date = self.returns.copy()
        rf_with_date = self.risk_free_rate.copy()
        
        # Merge por fecha (inner join)
        merged = pd.DataFrame({
            'returns': returns_with_date,
            'rf': rf_with_date
        }).dropna()
        
        return merged['returns'], merged['rf']
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calcula el Maximum Drawdown.
        
        Definición: Máxima pérdida desde un pico hasta un valle.
        
        Fórmula: MDD = min((Trough - Peak) / Peak)
        
        Returns:
            Maximum Drawdown (en decimal, negativo)
        """
        # Calcular valor acumulado de la inversión
        cumulative = (1 + self.returns).cumprod()
        
        # Calcular máximo acumulado hasta cada punto
        running_max = cumulative.cummax()
        
        # Calcular drawdown en cada punto
        drawdown = (cumulative - running_max) / running_max
        
        # Retornar el peor drawdown
        return drawdown.min()
    
    def _calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calcula el Value at Risk (VaR) histórico.
        
        VaR responde: ¿Cuál es la pérdida máxima esperada con X% de confianza?
        
        Args:
            confidence_level: Nivel de confianza (0.95 = 95%)
            
        Returns:
            VaR (en decimal, negativo indica pérdida)
        """
        return self.returns.quantile(1 - confidence_level)
    
    # ==================== PROPIEDADES ====================
    
    @property
    def mean_return(self) -> float:
        """Retorno medio diario (en porcentaje)."""
        return self.statistics['mean_return_daily'] * 100
    
    @property
    def volatility(self) -> float:
        """Volatilidad diaria (en porcentaje)."""
        return self.statistics['volatility_daily'] * 100
    
    @property
    def mean_return_annual(self) -> float:
        """Retorno medio anualizado (en porcentaje)."""
        return self.statistics['mean_return_annual'] * 100
    
    @property
    def volatility_annual(self) -> float:
        """Volatilidad anualizada (en porcentaje)."""
        return self.statistics['volatility_annual'] * 100
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe Ratio anualizado."""
        return self.statistics['sharpe_ratio']
    
    @property
    def max_drawdown(self) -> float:
        """Maximum Drawdown (en porcentaje, negativo)."""
        return self.statistics['max_drawdown'] * 100
    
    @property
    def var_95(self) -> float:
        """Value at Risk al 95% (en porcentaje)."""
        return self.statistics['var_95'] * 100
    
    def get_summary(self) -> dict:
        """
        Retorna un resumen completo de las métricas del activo.
        
        Returns:
            Diccionario con todas las métricas formateadas
        """
        return {
            'Ticker': self.ticker,
            'Observaciones': self.statistics['observations'],
            'Periodo': f"{self.statistics['start_date']} a {self.statistics['end_date']}",
            'Retorno Medio Diario': f"{self.mean_return:.4f}%",
            'Volatilidad Diaria': f"{self.volatility:.4f}%",
            'Retorno Anualizado': f"{self.mean_return_annual:.2f}%",
            'Volatilidad Anualizada': f"{self.volatility_annual:.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.4f}",
            'Maximum Drawdown': f"{self.max_drawdown:.2f}%",
            'VaR (95%)': f"{self.var_95:.4f}%",
            'Skewness': f"{self.statistics['skewness']:.4f}",
            'Kurtosis': f"{self.statistics['kurtosis']:.4f}",
        }
    
    def __repr__(self) -> str:
        """Representación en string del objeto."""
        return (f"PriceSeries(ticker='{self.ticker}', "
                f"observations={len(self.data)}, "
                f"return={self.mean_return_annual:.2f}%, "
                f"volatility={self.volatility_annual:.2f}%)")


# ==================== DATACLASS: PORTFOLIO ====================

@dataclass
class Portfolio:
    """
    Representa una cartera de activos con pesos.
    
    Calcula automáticamente:
    - Matriz de correlación entre activos
    - Matriz de covarianza
    - Retorno y volatilidad de la cartera
    - Beta de cada activo vs índice de mercado
    - Métricas agregadas
    
    Attributes:
        assets: Diccionario {ticker: PriceSeries}
        weights: Diccionario {ticker: peso} (si None, equiponderado)
        market_index: PriceSeries del índice de mercado (ej: S&P 500) para Beta
        risk_free_rate: Serie temporal con tasa libre de riesgo
    """
    
    assets: Dict[str, PriceSeries]
    weights: Optional[Dict[str, float]] = None
    market_index: Optional[PriceSeries] = None
    risk_free_rate: Optional[pd.Series] = None
    
    # Campos calculados automáticamente
    correlation_matrix: pd.DataFrame = field(init=False, repr=False)
    covariance_matrix: pd.DataFrame = field(init=False, repr=False)
    
    def __post_init__(self):
        """
        Inicialización automática después de crear la instancia.
        
        1. Establece pesos equiponderados si no se proporcionan
        2. Valida que los pesos sumen 1
        3. Calcula matrices de correlación y covarianza
        """
        logger.info(f"\n{'='*60}")
        logger.info("Inicializando Portfolio")
        logger.info(f"{'='*60}")
        
        # Establecer pesos equiponderados si no se proporcionan
        if self.weights is None:
            n = len(self.assets)
            self.weights = {ticker: 1.0 / n for ticker in self.assets.keys()}
            logger.info(f"  ✓ Pesos equiponderados: {1/n:.4f} por activo")
        
        # Validar pesos
        self._validate_weights()
        
        # Calcular matrices
        self.correlation_matrix = self._calculate_correlation_matrix()
        self.covariance_matrix = self._calculate_covariance_matrix()
        
        logger.info(f"✓ Portfolio inicializado correctamente")
        logger.info(f"  Activos: {len(self.assets)}")
        logger.info(f"  Retorno esperado: {self.portfolio_return:.2f}%")
        logger.info(f"  Volatilidad: {self.portfolio_volatility:.2f}%")
        logger.info(f"  Sharpe Ratio: {self.portfolio_sharpe_ratio:.4f}")
    
    def _validate_weights(self):
        """Valida que los pesos sean correctos."""
        # Verificar que todos los tickers tengan peso
        for ticker in self.assets.keys():
            if ticker not in self.weights:
                raise ValueError(f"Falta peso para el ticker: {ticker}")
        
        # Verificar que los pesos sumen aproximadamente 1
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Los pesos deben sumar 1. Suma actual: {total_weight:.6f}")
        
        # Verificar que todos los pesos sean positivos
        for ticker, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Peso negativo no permitido para {ticker}: {weight}")
        
        logger.info("  ✓ Pesos validados correctamente")
    
    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calcula la matriz de correlación entre los retornos de los activos.
        
        Returns:
            DataFrame con correlaciones entre todos los pares de activos
        """
        # Crear DataFrame con todos los retornos
        returns_df = pd.DataFrame({
            ticker: asset.returns
            for ticker, asset in self.assets.items()
        })
        
        # Calcular correlación
        corr_matrix = returns_df.corr()
        
        logger.info("  ✓ Matriz de correlación calculada")
        return corr_matrix
    
    def _calculate_covariance_matrix(self) -> pd.DataFrame:
        """
        Calcula la matriz de covarianza entre los retornos de los activos.
        
        Returns:
            DataFrame con covarianzas entre todos los pares de activos
        """
        # Crear DataFrame con todos los retornos
        returns_df = pd.DataFrame({
            ticker: asset.returns
            for ticker, asset in self.assets.items()
        })
        
        # Calcular covarianza (anualizada)
        cov_matrix = returns_df.cov() * 252
        
        logger.info("  ✓ Matriz de covarianza calculada")
        return cov_matrix
    
    def _calculate_portfolio_return(self) -> float:
        """
        Calcula el retorno esperado de la cartera (anualizado).
        
        Fórmula: R_p = Σ(w_i * R_i)
        
        Returns:
            Retorno anualizado de la cartera (en decimal)
        """
        portfolio_return = sum(
            self.weights[ticker] * asset.statistics['mean_return_annual']
            for ticker, asset in self.assets.items()
        )
        return portfolio_return
    
    def _calculate_portfolio_volatility(self) -> float:
        """
        Calcula la volatilidad de la cartera (anualizada).
        
        Fórmula: σ_p = sqrt(w^T * Σ * w)
        
        Donde:
        - w: Vector de pesos
        - Σ: Matriz de covarianza
        
        Returns:
            Volatilidad anualizada de la cartera (en decimal)
        """
        # Crear vector de pesos en el orden correcto
        tickers = list(self.assets.keys())
        weights_vector = np.array([self.weights[ticker] for ticker in tickers])
        
        # Calcular varianza del portfolio: w^T * Σ * w
        portfolio_variance = np.dot(weights_vector, np.dot(self.covariance_matrix.values, weights_vector))
        
        # Volatilidad es la raíz cuadrada de la varianza
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_volatility
    
    def calculate_beta(self) -> Dict[str, float]:
        """
        Calcula el Beta de cada activo respecto al índice de mercado.
        
        Beta mide la sensibilidad del activo respecto al mercado:
        - β = 1: El activo se mueve igual que el mercado
        - β > 1: El activo es más volátil que el mercado
        - β < 1: El activo es menos volátil que el mercado
        
        Fórmula: β_i = Cov(R_i, R_m) / Var(R_m)
        
        Returns:
            Diccionario {ticker: beta}
        """
        if self.market_index is None:
            logger.warning("  ⚠ No se proporcionó índice de mercado para calcular Beta")
            return {ticker: np.nan for ticker in self.assets.keys()}
        
        betas = {}
        market_returns = self.market_index.returns
        market_variance = market_returns.var()
        
        for ticker, asset in self.assets.items():
            # Alinear retornos del activo y del mercado
            aligned = pd.DataFrame({
                'asset': asset.returns,
                'market': market_returns
            }).dropna()
            
            # Calcular covarianza
            covariance = aligned['asset'].cov(aligned['market'])
            
            # Calcular beta
            beta = covariance / market_variance if market_variance > 0 else 0.0
            betas[ticker] = beta
        
        logger.info("  ✓ Betas calculados vs índice de mercado")
        return betas
    
    def calculate_var_portfolio(self, confidence_level: float = 0.95) -> float:
        """
        Calcula el Value at Risk de la cartera completa.
        
        Args:
            confidence_level: Nivel de confianza (0.95 = 95%)
            
        Returns:
            VaR de la cartera (en decimal)
        """
        # Crear DataFrame con retornos ponderados
        returns_df = pd.DataFrame({
            ticker: asset.returns * self.weights[ticker]
            for ticker, asset in self.assets.items()
        })
        
        # Retornos del portfolio
        portfolio_returns = returns_df.sum(axis=1)
        
        # VaR histórico
        var = portfolio_returns.quantile(1 - confidence_level)
        
        return var
    
    # ==================== PROPIEDADES ====================
    
    @property
    def portfolio_return(self) -> float:
        """Retorno anualizado de la cartera (en porcentaje)."""
        return self._calculate_portfolio_return() * 100
    
    @property
    def portfolio_volatility(self) -> float:
        """Volatilidad anualizada de la cartera (en porcentaje)."""
        return self._calculate_portfolio_volatility() * 100
    
    @property
    def portfolio_sharpe_ratio(self) -> float:
        """Sharpe Ratio anualizado de la cartera."""
        if self.risk_free_rate is None:
            rf_annual = 0.0
        else:
            rf_annual = self.risk_free_rate.mean() * 252
        
        if self._calculate_portfolio_volatility() > 0:
            return (self._calculate_portfolio_return() - rf_annual) / self._calculate_portfolio_volatility()
        else:
            return 0.0
    
    @property
    def portfolio_var_95(self) -> float:
        """Value at Risk de la cartera al 95% (en porcentaje)."""
        return self.calculate_var_portfolio(confidence_level=0.95) * 100
    
    def get_portfolio_summary(self) -> dict:
        """
        Retorna un resumen completo de las métricas de la cartera.
        
        Returns:
            Diccionario con métricas de la cartera y de cada activo
        """
        # Métricas del portfolio
        portfolio_metrics = {
            'Portfolio': {
                'Retorno Anualizado': f"{self.portfolio_return:.2f}%",
                'Volatilidad Anualizada': f"{self.portfolio_volatility:.2f}%",
                'Sharpe Ratio': f"{self.portfolio_sharpe_ratio:.4f}",
                'VaR (95%)': f"{self.portfolio_var_95:.4f}%",
            }
        }
        
        # Pesos
        weights_info = {f"Peso {ticker}": f"{weight*100:.2f}%" 
                       for ticker, weight in self.weights.items()}
        
        # Métricas individuales
        individual_metrics = {}
        for ticker, asset in self.assets.items():
            individual_metrics[ticker] = {
                'Retorno Anualizado': f"{asset.mean_return_annual:.2f}%",
                'Volatilidad Anualizada': f"{asset.volatility_annual:.2f}%",
                'Sharpe Ratio': f"{asset.sharpe_ratio:.4f}",
                'Max Drawdown': f"{asset.max_drawdown:.2f}%",
                'VaR (95%)': f"{asset.var_95:.4f}%",
            }
        
        # Betas (si hay índice de mercado)
        if self.market_index is not None:
            betas = self.calculate_beta()
            for ticker, beta in betas.items():
                individual_metrics[ticker]['Beta'] = f"{beta:.4f}"
        
        return {
            'Portfolio Metrics': portfolio_metrics,
            'Weights': weights_info,
            'Individual Assets': individual_metrics,
            'Correlation Matrix': self.correlation_matrix,
            'Covariance Matrix': self.covariance_matrix,
        }
    
    def get_portfolio_returns(self) -> pd.Series:
        """
        Retorna serie temporal de retornos ponderados del portfolio.
        
        Calcula los retornos diarios del portfolio aplicando los pesos
        a los retornos de cada activo individual.
        
        Returns:
            pd.Series con retornos diarios del portfolio
            
        Example:
            >>> portfolio = Portfolio(assets, weights)
            >>> returns = portfolio.get_portfolio_returns()
            >>> returns.plot(title='Portfolio Daily Returns')
        """
        # Crear DataFrame con retornos de todos los activos
        returns_df = pd.DataFrame({
            ticker: ps.returns 
            for ticker, ps in self.assets.items()
        })
        
        # Aplicar pesos
        weighted_returns = returns_df * pd.Series(self.weights)
        
        # Sumar para obtener retorno del portfolio
        portfolio_returns = weighted_returns.sum(axis=1)
        
        return portfolio_returns
    
    def get_portfolio_cumulative_returns(self) -> pd.Series:
        """
        Retorna serie temporal de retornos acumulados del portfolio.
        
        Calcula el valor acumulado del portfolio a lo largo del tiempo,
        partiendo de una base de 100.
        
        Returns:
            pd.Series con valores acumulados (base 100)
            
        Example:
            >>> portfolio = Portfolio(assets, weights)
            >>> cumulative = portfolio.get_portfolio_cumulative_returns()
            >>> cumulative.plot(title='Portfolio Cumulative Performance')
        """
        returns = self.get_portfolio_returns()
        cumulative = (1 + returns).cumprod() * 100
        return cumulative
    
    def get_portfolio_prices_normalized(self) -> pd.DataFrame:
        """
        Retorna DataFrame con precios normalizados de todos los activos.
        
        Normaliza todos los precios a base 100 en la fecha inicial,
        permitiendo comparar el performance relativo de cada activo.
        
        Returns:
            pd.DataFrame con columnas = tickers, valores normalizados a 100
            
        Example:
            >>> portfolio = Portfolio(assets, weights)
            >>> normalized = portfolio.get_portfolio_prices_normalized()
            >>> normalized.plot(title='Normalized Asset Prices')
        """
        # Obtener precios ajustados de cada activo
        prices_dict = {}
        for ticker, ps in self.assets.items():
            adj_close = ps.data.set_index('Date')['Adj Close']
            # Normalizar a 100
            normalized = (adj_close / adj_close.iloc[0]) * 100
            prices_dict[ticker] = normalized
        
        # Crear DataFrame
        prices_df = pd.DataFrame(prices_dict)
        
        # Añadir portfolio
        portfolio_cumulative = self.get_portfolio_cumulative_returns()
        prices_df['Portfolio'] = portfolio_cumulative
        
        return prices_df
    
    def __repr__(self) -> str:
        """Representación en string del objeto."""
        return (f"Portfolio(assets={len(self.assets)}, "
                f"return={self.portfolio_return:.2f}%, "
                f"volatility={self.portfolio_volatility:.2f}%, "
                f"sharpe={self.portfolio_sharpe_ratio:.4f})")


# ==================== FUNCIÓN PRINCIPAL DE EJEMPLO ====================

def main():
    """
    Función de ejemplo para demostrar el uso de PriceSeries y Portfolio.
    
    Este ejemplo muestra cómo:
    1. Cargar datos limpios
    2. Descargar tasa libre de riesgo
    3. Crear PriceSeries individuales
    4. Crear Portfolio
    5. Mostrar métricas
    """
    print("="*60)
    print("FASE 3: ANÁLISIS ESTADÍSTICO")
    print("="*60)
    
    # Este es un ejemplo. En la práctica, se cargarían los datos limpios
    # desde data_cleaner.py
    
    print("\nEste módulo está listo para ser usado.")
    print("\nPara un ejemplo completo, ejecuta:")
    print("  python main_analysis.py")
    print("\nO importa las clases en tu código:")
    print("  from price_series import PriceSeries, Portfolio, download_risk_free_rate")


if __name__ == "__main__":
    main()