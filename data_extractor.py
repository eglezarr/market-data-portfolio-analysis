"""
data_extractor.py
Clase base abstracta para extractores de datos bursátiles.
Define la interfaz común que deben implementar todos los extractores.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DataExtractor(ABC):
    """
    Clase base abstracta para extractores de datos de mercado.
    
    Todos los extractores deben heredar de esta clase e implementar
    los métodos abstractos para garantizar una interfaz uniforme.
    """
    
    def __init__(self, source_name: str):
        """
        Inicializa el extractor.
        
        Args:
            source_name: Nombre de la fuente de datos
        """
        self.source_name = source_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de una acción específica.
        
        Args:
            ticker: Símbolo de la acción (ej: 'AAPL')
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            
        Returns:
            DataFrame con columnas: Date, Open, High, Low, Close, Adj Close, Volume
            o None si hay error
        """
        pass
    
    @abstractmethod
    def fetch_index_data(
        self,
        index_ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de un índice específico.
        
        Args:
            index_ticker: Símbolo del índice (ej: '^GSPC' para S&P500)
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            
        Returns:
            DataFrame con columnas: Date, Open, High, Low, Close, Adj Close, Volume
            o None si hay error
        """
        pass
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        ticker_type: str = 'stock'
    ) -> dict:
        """
        Obtiene datos para múltiples tickers.
        
        Args:
            tickers: Lista de símbolos a descargar
            start_date: Fecha inicial en formato 'YYYY-MM-DD'
            end_date: Fecha final en formato 'YYYY-MM-DD'
            ticker_type: 'stock' o 'index'
            
        Returns:
            Diccionario {ticker: DataFrame} con los datos descargados
        """
        results = {}
        
        for ticker in tickers:
            self.logger.info(f"Descargando {ticker} desde {self.source_name}...")
            
            try:
                if ticker_type == 'stock':
                    data = self.fetch_stock_data(ticker, start_date, end_date)
                elif ticker_type == 'index':
                    data = self.fetch_index_data(ticker, start_date, end_date)
                else:
                    self.logger.error(f"Tipo de ticker no válido: {ticker_type}")
                    continue
                
                if data is not None and not data.empty:
                    results[ticker] = data
                    self.logger.info(f"✓ {ticker} descargado correctamente ({len(data)} registros)")
                else:
                    self.logger.warning(f"✗ No se pudieron obtener datos para {ticker}")
                    
            except Exception as e:
                self.logger.error(f"✗ Error descargando {ticker}: {str(e)}")
                continue
        
        return results
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza el formato del DataFrame.
        
        Args:
            df: DataFrame a estandarizar
            
        Returns:
            DataFrame con formato estandarizado
        """
        # Asegurar que las columnas estén en el orden correcto
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        # Verificar que todas las columnas necesarias existen
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Falta la columna requerida: {col}")
        
        # Seleccionar solo las columnas requeridas en el orden correcto
        df = df[required_columns].copy()
        
        # Asegurar que Date sea datetime
        if df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Ordenar por fecha
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['Date'])
        
        return df
    
    def validate_dates(self, start_date: str, end_date: str) -> bool:
        """
        Valida que las fechas tengan el formato correcto.
        
        Args:
            start_date: Fecha inicial
            end_date: Fecha final
            
        Returns:
            True si las fechas son válidas
        """
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start > end:
                self.logger.error("La fecha inicial debe ser anterior a la fecha final")
                return False
            
            return True
            
        except ValueError as e:
            self.logger.error(f"Formato de fecha inválido: {e}")
            return False