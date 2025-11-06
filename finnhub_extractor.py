"""
finnhub_extractor.py
Extractor de datos desde Finnhub API.
"""

import requests
import pandas as pd
from typing import Optional
from datetime import datetime
import time
from data_extractor import DataExtractor


class FinnhubExtractor(DataExtractor):
    """
    Extractor de datos desde Finnhub API.
    
    Requiere API key que se puede obtener gratuitamente en:
    https://finnhub.io/
    """
    
    def __init__(self, api_key: str):
        """
        Inicializa el extractor de Finnhub.
        
        Args:
            api_key: Clave de API de Finnhub
        """
        super().__init__("Finnhub")
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
    
    def _convert_date_to_timestamp(self, date_str: str) -> int:
        """
        Convierte una fecha en formato YYYY-MM-DD a timestamp Unix.
        
        Args:
            date_str: Fecha en formato 'YYYY-MM-DD'
            
        Returns:
            Timestamp Unix
        """
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt.timestamp())
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de una acción desde Finnhub.
        
        Args:
            ticker: Símbolo de la acción (ej: 'AAPL')
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            
        Returns:
            DataFrame estandarizado o None si hay error
        """
        if not self.validate_dates(start_date, end_date):
            return None
        
        try:
            # Convertir fechas a timestamps
            from_timestamp = self._convert_date_to_timestamp(start_date)
            to_timestamp = self._convert_date_to_timestamp(end_date)
            
            # Construir URL para stock candles
            url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': ticker,
                'resolution': 'D',  # D = Daily
                'from': from_timestamp,
                'to': to_timestamp,
                'token': self.api_key
            }
            
            # Realizar petición
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Verificar si hay datos
            if data.get('s') == 'no_data' or not data.get('c'):
                self.logger.warning(f"No se encontraron datos para {ticker}")
                return None
            
            # Crear DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Adj Close': data['c'],  # Finnhub no proporciona Adj Close, usar Close
                'Volume': data['v']
            })
            
            self.logger.warning(f"Finnhub no proporciona Adjusted Close para {ticker}, usando Close como Adj Close")
            
            # Estandarizar formato
            df = self._standardize_dataframe(df)
            
            # Pequeña pausa para respetar rate limits
            time.sleep(0.5)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error de petición HTTP para {ticker}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error al descargar {ticker} desde Finnhub: {str(e)}")
            return None
    
    def fetch_index_data(
        self,
        index_ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de un índice desde Finnhub.
        
        Args:
            index_ticker: Símbolo del índice
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            
        Returns:
            DataFrame estandarizado o None si hay error
            
        Note:
            Finnhub usa símbolos específicos para índices.
            Ejemplos: ^GSPC, ^DJI, ^IXIC
            La API es similar para acciones e índices.
        """
        # En Finnhub, los índices se manejan igual que las acciones
        return self.fetch_stock_data(index_ticker, start_date, end_date)