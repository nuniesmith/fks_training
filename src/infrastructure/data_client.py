"""
Data Client for fks_training

Provides interface to fetch market data from fks_data service.
Supports multiple methods:
1. Direct adapter usage (if fks_data is available as library)
2. HTTP API calls (if services are separate)
3. Database queries (if shared database)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# Configuration
DATA_SERVICE_URL = os.getenv("FKS_DATA_URL", "http://fks-data:8001")
DATA_SERVICE_TIMEOUT = float(os.getenv("FKS_DATA_TIMEOUT", "30.0"))


class DataClient:
    """
    Client for fetching market data from fks_data service.

    Tries multiple methods in order:
    1. Direct adapter import (if fks_data available)
    2. HTTP API calls
    3. Database queries (if shared DB)
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        use_adapters: bool = True,
        use_api: bool = True,
        use_db: bool = False,
    ):
        """
        Initialize data client.

        Args:
            service_url: URL of fks_data service (default: from env)
            use_adapters: Try to use adapters directly (default: True)
            use_api: Fallback to HTTP API (default: True)
            use_db: Fallback to direct DB queries (default: False)
        """
        self.service_url = service_url or DATA_SERVICE_URL
        self.use_adapters = use_adapters
        self.use_api = use_api
        self.use_db = use_db

        # Try to import adapters
        self._adapter_manager = None
        if self.use_adapters:
            try:
                from fks_data.adapters.multi_provider_manager import MultiProviderManager

                self._adapter_manager = MultiProviderManager()
                logger.info("Successfully initialized adapter manager")
            except ImportError:
                logger.warning(
                    "fks_data adapters not available, will use API or DB fallback"
                )
                self._adapter_manager = None

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT", "BTCUSDT")
            interval: Time interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Start datetime (default: 30 days ago)
            end_time: End datetime (default: now)
            limit: Maximum number of candles (optional)
            provider: Preferred provider (e.g., "binance", "polygon") (optional)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Normalize symbol format
        symbol = self._normalize_symbol(symbol)

        # Set defaults
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        # Try adapters first
        if self._adapter_manager and self.use_adapters:
            try:
                return self._fetch_via_adapters(
                    symbol, interval, start_time, end_time, limit, provider
                )
            except Exception as e:
                logger.warning(f"Adapter fetch failed: {e}, trying API fallback")

        # Try HTTP API
        if self.use_api:
            try:
                return self._fetch_via_api(
                    symbol, interval, start_time, end_time, limit, provider
                )
            except Exception as e:
                logger.warning(f"API fetch failed: {e}, trying DB fallback")

        # Try database (if enabled)
        if self.use_db:
            try:
                return self._fetch_via_db(symbol, interval, start_time, end_time, limit)
            except Exception as e:
                logger.error(f"All data fetch methods failed. Last error: {e}")
                raise

        raise RuntimeError(
            "No data fetch method available. Check fks_data service availability."
        )

    def _fetch_via_adapters(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int],
        provider: Optional[str],
    ) -> pd.DataFrame:
        """Fetch data using adapters directly."""
        if not self._adapter_manager:
            raise RuntimeError("Adapter manager not available")

        # Convert interval to format expected by adapters
        # Most adapters use formats like "1m", "5m", "1h", "1d"
        adapter_interval = interval

        # Determine provider if not specified
        if not provider:
            # Auto-detect based on symbol
            if "/" in symbol or "USDT" in symbol.upper():
                provider = "binance"  # Default for crypto
            else:
                provider = "polygon"  # Default for stocks

        # Fetch data using adapter directly (more reliable than manager)
        from fks_data.adapters import get_adapter
        
        # Determine provider if not specified
        if not provider:
            # Auto-detect based on symbol
            if "/" in symbol or "USDT" in symbol.upper() or "BTC" in symbol.upper():
                provider = "binance"  # Default for crypto
            else:
                provider = "polygon"  # Default for stocks
        
        adapter = get_adapter(provider)
        
        # Fetch data - adapters use different parameter names
        # Binance adapter expects: symbol, interval, start_time (timestamp ms), end_time (timestamp ms), limit
        # Convert datetime to milliseconds for Binance
        start_time_ms = int(start_time.timestamp() * 1000) if start_time else None
        end_time_ms = int(end_time.timestamp() * 1000) if end_time else None
        
        # Try common parameter combinations
        try:
            # Try Binance-style parameters (timestamp in milliseconds)
            result = adapter.fetch(
                symbol=symbol,
                interval=adapter_interval,
                start_time=start_time_ms,
                end_time=end_time_ms,
                limit=limit or 500,
            )
        except TypeError:
            # Try with different parameter names (seconds timestamps)
            try:
                result = adapter.fetch(
                    symbol=symbol,
                    interval=adapter_interval,
                    start=int(start_time.timestamp()) if start_time else None,
                    end=int(end_time.timestamp()) if end_time else None,
                    limit=limit,
                )
            except TypeError:
                # Try minimal parameters (just symbol and interval)
                result = adapter.fetch(
                    symbol=symbol,
                    interval=adapter_interval,
                    limit=limit or 1000,
                )

        # Convert to DataFrame
        if isinstance(result, dict) and "data" in result:
            data = result["data"]
        elif isinstance(result, list):
            data = result
        else:
            raise ValueError(f"Unexpected result format: {type(result)}")

        if not data:
            raise ValueError(f"No data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Normalize column names
        df = self._normalize_dataframe(df)

        return df

    def _fetch_via_api(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int],
        provider: Optional[str],
    ) -> pd.DataFrame:
        """Fetch data via HTTP API."""
        # Try multiple API endpoints
        endpoints = [
            f"{self.service_url}/api/v1/data",
            f"{self.service_url}/data",
        ]

        # Determine provider-specific endpoint
        if provider == "binance":
            endpoints.insert(0, f"{self.service_url}/api/v1/binance/data")
        elif provider == "polygon":
            endpoints.insert(0, f"{self.service_url}/api/v1/polygon/data")

        # Convert datetime to timestamps
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())

        # Try each endpoint
        for endpoint in endpoints:
            try:
                params = {
                    "symbol": symbol,
                    "timeframe": interval,
                    "start": start_ts,
                    "end": end_ts,
                }
                if limit:
                    params["limit"] = limit

                response = httpx.get(
                    endpoint, params=params, timeout=DATA_SERVICE_TIMEOUT
                )
                response.raise_for_status()

                data = response.json()

                # Extract data array
                if isinstance(data, dict):
                    if "data" in data:
                        records = data["data"]
                    elif "records" in data:
                        records = data["records"]
                    else:
                        records = [data]
                elif isinstance(data, list):
                    records = data
                else:
                    raise ValueError(f"Unexpected API response format: {type(data)}")

                if not records:
                    continue  # Try next endpoint

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Normalize column names
                df = self._normalize_dataframe(df)

                return df

            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
                continue

        raise RuntimeError(f"All API endpoints failed for symbol {symbol}")

    def _fetch_via_db(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Fetch data directly from database."""
        # This would require database connection setup
        # For now, raise NotImplementedError
        raise NotImplementedError("Database queries not yet implemented")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format."""
        # Convert "BTC/USDT" to "BTCUSDT" for some providers
        symbol = symbol.replace("/", "").upper()
        return symbol

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard format.

        Expected columns: timestamp (or ts), open, high, low, close, volume
        """
        # Rename timestamp column
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "timestamp"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == "int64":
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            elif df["timestamp"].dtype == "object":
                df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"DataFrame missing required columns: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Select and order columns
        cols = ["timestamp"] + required_cols
        df = df[cols].copy()

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Ensure numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop any rows with NaN in required columns
        df = df.dropna(subset=required_cols)

        return df

