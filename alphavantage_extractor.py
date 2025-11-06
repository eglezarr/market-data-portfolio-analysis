"""
alphavantage_extractor.py
Extractor de datos desde AlphaVantage API.
"""

import requests
import pandas as pd
from typing import Optional
import time
from data_extractor import DataExtractor


class AlphaVantageExtractor(DataExtractor):
    """
    Extractor de datos desde AlphaVantage API.
    
    Requiere API key que se puede obtener gratuitamente en:
    https://www.alphavantage.co/support/#api-key
    
    Nota: La versión gratuita tiene limitaciones de rate limit (5 llamadas/minuto)
    """
    
    def __init__(self, api_key: str):
        """
        Inicializa el extractor de AlphaVantage.
        
        Args:
            api_key: Clave de API de AlphaVantage
        """
        super().__init__("AlphaVantage")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.call_count = 0  # Contador de llamadas
        self.delay_seconds = 13  # 13 segundos = máximo 4.6 llamadas/minuto (seguro)
    
    def _wait_for_rate_limit(self):
        """
        Aplica delay para respetar rate limits de AlphaVantage.
        
        AlphaVantage Free Tier: 5 llamadas por minuto máximo.
        Usamos 13 segundos de delay = 4.6 llamadas/minuto (con margen de seguridad).
        """
        if self.call_count > 0:  # No hacer delay en la primera llamada
            self.logger.info(f"  ⏳ Esperando {self.delay_seconds}s para respetar rate limit...")
            time.sleep(self.delay_seconds)
        
        self.call_count += 1
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de una acción desde AlphaVantage.
        
        Args:
            ticker: Símbolo de la acción (ej: 'AAPL')
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            
        Returns:
            DataFrame estandarizado o None si hay error
        """
        if not self.validate_dates(start_date, end_date):
            return None
        
        # IMPORTANTE: Delay ANTES de hacer la petición
        self._wait_for_rate_limit()
        
        try:
            self.logger.info(f"  📡 Llamada #{self.call_count} a AlphaVantage API...")
            
            # Parámetros para la petición
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'outputsize': 'full',  # 'full' para obtener más de 20 años de datos
                'apikey': self.api_key
            }
            
            # Realizar petición
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Verificar si hay errores en la respuesta
            if 'Error Message' in data:
                self.logger.error(f"Error de API para {ticker}: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                self.logger.warning(f"Rate limit alcanzado: {data['Note']}")
                return None
            
            # Extraer los datos de la serie temporal
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                self.logger.warning(f"No se encontraron datos para {ticker}")
                return None
            
            time_series = data[time_series_key]
            
            # Convertir a DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Renombrar columnas al formato estándar
            df_standard = pd.DataFrame({
                'Date': df.index,
                'Open': df['1. open'].astype(float),
                'High': df['2. high'].astype(float),
                'Low': df['3. low'].astype(float),
                'Close': df['4. close'].astype(float),
                'Adj Close': df['5. adjusted close'].astype(float) if '5. adjusted close' in df.columns else df['4. close'].astype(float),
                'Volume': df['5. volume'].astype(float) if '5. volume' in df.columns else df['6. volume'].astype(float)
            })
            
            # Nota: AlphaVantage tiene diferentes formatos según la función usada
            # TIME_SERIES_DAILY no incluye adjusted close
            # TIME_SERIES_DAILY_ADJUSTED sí lo incluye
            # Por ahora usamos Close como Adj Close si no está disponible
            if 'Adj Close' not in df_standard.columns or df_standard['Adj Close'].isna().all():
                df_standard['Adj Close'] = df_standard['Close']
                self.logger.warning(f"AlphaVantage: Usando Close como Adj Close para {ticker}")
            
            # Filtrar por rango de fechas
            df_standard = df_standard[
                (df_standard['Date'] >= start_date) & 
                (df_standard['Date'] <= end_date)
            ]
            
            if df_standard.empty:
                self.logger.warning(f"No hay datos en el rango especificado para {ticker}")
                return None
            
            # Estandarizar formato
            df_standard = self._standardize_dataframe(df_standard)
            
            self.logger.info(f"  ✓ {ticker}: {len(df_standard)} registros descargados")
            
            return df_standard
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error de petición HTTP para {ticker}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error al descargar {ticker} desde AlphaVantage: {str(e)}")
            return None
    
    def fetch_index_data(
        self,
        index_ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de un índice desde AlphaVantage.
        
        Args:
            index_ticker: Símbolo del índice
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            
        Returns:
            DataFrame estandarizado o None si hay error
            
        Note:
            AlphaVantage maneja índices de forma similar a acciones.
            Sin embargo, algunos índices pueden no estar disponibles o
            requerir símbolos específicos (ej: SPY para S&P 500 en lugar de ^GSPC)
        """
        # En AlphaVantage, los índices se manejan igual que las acciones
        # Nota: Algunos índices pueden requerir usar ETFs equivalentes
        return self.fetch_stock_data(index_ticker, start_date, end_date)