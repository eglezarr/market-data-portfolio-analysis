"""
yahoo_extractor.py
Extractor de datos desde Yahoo Finance usando la librería yfinance.
"""

import yfinance as yf
import pandas as pd
from typing import Optional
from data_extractor import DataExtractor


class YahooExtractor(DataExtractor):
    """
    Extractor de datos desde Yahoo Finance.
    
    No requiere API key, utiliza la librería yfinance.
    """
    
    def __init__(self):
        """Inicializa el extractor de Yahoo Finance."""
        super().__init__("Yahoo Finance")
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de una acción desde Yahoo Finance.
        
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
            # Descargar datos usando yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                self.logger.warning(f"No se encontraron datos para {ticker}")
                return None
            
            # Resetear el índice para convertir Date en columna
            df = df.reset_index()
            
            # Convertir Date a datetime sin timezone
            # yfinance devuelve datetime64[ns, America/New_York], necesitamos datetime64[ns]
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Preparar DataFrame en formato estandarizado
            # Verificar si existe 'Adj Close' en el DataFrame original
            has_adj_close = 'Adj Close' in df.columns
            
            df_standard = pd.DataFrame({
                'Date': df['Date'],
                'Open': df['Open'].astype(float),
                'High': df['High'].astype(float),
                'Low': df['Low'].astype(float),
                'Close': df['Close'].astype(float),
                'Adj Close': df['Adj Close'].astype(float) if has_adj_close else df['Close'].astype(float),
                'Volume': df['Volume'].astype(float)
            })
            
            # Estandarizar formato (ordena, limpia duplicados, etc.)
            df_standard = self._standardize_dataframe(df_standard)
            
            return df_standard
            
        except Exception as e:
            self.logger.error(f"Error al descargar {ticker} desde Yahoo Finance: {str(e)}")
            return None
    
    def fetch_index_data(
        self,
        index_ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de un índice desde Yahoo Finance.
        
        Args:
            index_ticker: Símbolo del índice (ej: '^GSPC', '^DJI', '^IXIC')
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            
        Returns:
            DataFrame estandarizado o None si hay error
            
        Note:
            Los índices en Yahoo Finance suelen llevar el prefijo '^'
            Ejemplos: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
        """
        # Los índices se obtienen de la misma forma que las acciones en Yahoo Finance
        return self.fetch_stock_data(index_ticker, start_date, end_date)