#!/usr/bin/env python3
"""
Real-time Market Data Streaming System for RemoteAlgoTrader
Handles WebSocket connections, real-time price feeds, and market data processing
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
import aiohttp
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import threading
from queue import Queue
import ssl

logger = logging.getLogger(__name__)

class DataSource(Enum):
    ALPACA = "alpaca"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    BINANCE = "binance"

class MarketDataType(Enum):
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    NEWS = "news"
    SENTIMENT = "sentiment"

@dataclass
class MarketData:
    """Represents a market data point"""
    symbol: str
    timestamp: datetime
    data_type: MarketDataType
    price: Optional[float] = None
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    vwap: Optional[float] = None
    source: DataSource = DataSource.ALPACA
    raw_data: Optional[Dict] = None

@dataclass
class BarData:
    """Represents OHLCV bar data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    source: DataSource = DataSource.ALPACA

class MarketDataStream:
    """Real-time market data streaming with multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.websocket_connections: Dict[str, Any] = {}
        self.data_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.price_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.bar_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.is_running = False
        self.reconnect_attempts = defaultdict(int)
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        
        # Alpaca configuration
        self.alpaca_api_key = config.get('alpaca_api_key')
        self.alpaca_secret_key = config.get('alpaca_secret_key')
        self.alpaca_base_url = config.get('alpaca_base_url', 'wss://stream.data.alpaca.markets/v2/iex')
        
        # Polygon configuration
        self.polygon_api_key = config.get('polygon_api_key')
        self.polygon_base_url = config.get('polygon_base_url', 'wss://delayed.polygon.io')
        
        # Data processing
        self.processing_queue = Queue(maxsize=10000)
        self.processing_thread = None
        
    async def start(self, symbols: List[str]):
        """Start streaming market data for specified symbols"""
        try:
            self.is_running = True
            self.symbols = symbols
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_data_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start WebSocket connections
            await asyncio.gather(
                self._start_alpaca_stream(),
                self._start_polygon_stream(),
                self._start_data_processor()
            )
            
        except Exception as e:
            logger.error(f"Failed to start market data stream: {e}")
            raise
    
    async def stop(self):
        """Stop all market data streams"""
        try:
            self.is_running = False
            
            # Close WebSocket connections
            for connection in self.websocket_connections.values():
                await connection.close()
            
            # Stop processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            
            logger.info("Market data streams stopped")
            
        except Exception as e:
            logger.error(f"Error stopping market data streams: {e}")
    
    async def _start_alpaca_stream(self):
        """Start Alpaca WebSocket stream"""
        try:
            if not self.alpaca_api_key or not self.alpaca_secret_key:
                logger.warning("Alpaca credentials not provided, skipping Alpaca stream")
                return
            
            # Authentication message
            auth_message = {
                "action": "auth",
                "key": self.alpaca_api_key,
                "secret": self.alpaca_secret_key
            }
            
            # Subscribe to trades and quotes
            subscribe_message = {
                "action": "subscribe",
                "trades": self.symbols,
                "quotes": self.symbols,
                "bars": self.symbols
            }
            
            uri = f"{self.alpaca_base_url}"
            
            while self.is_running:
                try:
                    async with websockets.connect(uri) as websocket:
                        self.websocket_connections['alpaca'] = websocket
                        
                        # Send authentication
                        await websocket.send(json.dumps(auth_message))
                        auth_response = await websocket.recv()
                        auth_data = json.loads(auth_response)
                        
                        if auth_data.get('T') == 'success':
                            logger.info("Alpaca WebSocket authenticated successfully")
                            
                            # Send subscription
                            await websocket.send(json.dumps(subscribe_message))
                            subscribe_response = await websocket.recv()
                            subscribe_data = json.loads(subscribe_response)
                            
                            if subscribe_data.get('T') == 'subscription':
                                logger.info(f"Subscribed to Alpaca data for {len(self.symbols)} symbols")
                                
                                # Reset reconnect attempts
                                self.reconnect_attempts['alpaca'] = 0
                                
                                # Process incoming messages
                                async for message in websocket:
                                    if not self.is_running:
                                        break
                                    
                                    try:
                                        data = json.loads(message)
                                        await self._process_alpaca_message(data)
                                    except json.JSONDecodeError:
                                        logger.warning("Invalid JSON received from Alpaca")
                                    except Exception as e:
                                        logger.error(f"Error processing Alpaca message: {e}")
                        
                        else:
                            logger.error(f"Alpaca authentication failed: {auth_data}")
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Alpaca WebSocket connection closed")
                except Exception as e:
                    logger.error(f"Alpaca WebSocket error: {e}")
                
                # Reconnect logic
                if self.is_running:
                    await self._handle_reconnect('alpaca')
                    
        except Exception as e:
            logger.error(f"Failed to start Alpaca stream: {e}")
    
    async def _start_polygon_stream(self):
        """Start Polygon WebSocket stream"""
        try:
            if not self.polygon_api_key:
                logger.warning("Polygon API key not provided, skipping Polygon stream")
                return
            
            # Polygon subscription message
            subscribe_message = {
                "action": "subscribe",
                "params": f"T.{','.join(self.symbols)},Q.{','.join(self.symbols)}"
            }
            
            uri = f"{self.polygon_base_url}/stocks"
            
            while self.is_running:
                try:
                    async with websockets.connect(uri) as websocket:
                        self.websocket_connections['polygon'] = websocket
                        
                        # Send subscription
                        await websocket.send(json.dumps(subscribe_message))
                        
                        # Reset reconnect attempts
                        self.reconnect_attempts['polygon'] = 0
                        
                        # Process incoming messages
                        async for message in websocket:
                            if not self.is_running:
                                break
                            
                            try:
                                data = json.loads(message)
                                await self._process_polygon_message(data)
                            except json.JSONDecodeError:
                                logger.warning("Invalid JSON received from Polygon")
                            except Exception as e:
                                logger.error(f"Error processing Polygon message: {e}")
                
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Polygon WebSocket connection closed")
                except Exception as e:
                    logger.error(f"Polygon WebSocket error: {e}")
                
                # Reconnect logic
                if self.is_running:
                    await self._handle_reconnect('polygon')
                    
        except Exception as e:
            logger.error(f"Failed to start Polygon stream: {e}")
    
    async def _process_alpaca_message(self, data: Dict):
        """Process Alpaca WebSocket message"""
        try:
            message_type = data.get('T')
            
            if message_type == 'trade':
                # Process trade data
                trade_data = MarketData(
                    symbol=data.get('S'),
                    timestamp=datetime.fromtimestamp(data.get('t') / 1000000000),
                    data_type=MarketDataType.TRADE,
                    price=data.get('p'),
                    volume=data.get('s'),
                    source=DataSource.ALPACA,
                    raw_data=data
                )
                
                await self._handle_market_data(trade_data)
                
            elif message_type == 'quote':
                # Process quote data
                quote_data = MarketData(
                    symbol=data.get('S'),
                    timestamp=datetime.fromtimestamp(data.get('t') / 1000000000),
                    data_type=MarketDataType.QUOTE,
                    bid=data.get('bp'),
                    ask=data.get('ap'),
                    bid_size=data.get('bs'),
                    ask_size=data.get('as'),
                    source=DataSource.ALPACA,
                    raw_data=data
                )
                
                await self._handle_market_data(quote_data)
                
            elif message_type == 'bar':
                # Process bar data
                bar_data = BarData(
                    symbol=data.get('S'),
                    timestamp=datetime.fromtimestamp(data.get('t') / 1000000000),
                    open=data.get('o'),
                    high=data.get('h'),
                    low=data.get('l'),
                    close=data.get('c'),
                    volume=data.get('v'),
                    vwap=data.get('vw'),
                    source=DataSource.ALPACA
                )
                
                await self._handle_bar_data(bar_data)
                
        except Exception as e:
            logger.error(f"Error processing Alpaca message: {e}")
    
    async def _process_polygon_message(self, data: Dict):
        """Process Polygon WebSocket message"""
        try:
            event_type = data.get('ev')
            
            if event_type == 'T':
                # Trade event
                trade_data = MarketData(
                    symbol=data.get('sym'),
                    timestamp=datetime.fromtimestamp(data.get('t') / 1000),
                    data_type=MarketDataType.TRADE,
                    price=data.get('p'),
                    volume=data.get('s'),
                    source=DataSource.POLYGON,
                    raw_data=data
                )
                
                await self._handle_market_data(trade_data)
                
            elif event_type == 'Q':
                # Quote event
                quote_data = MarketData(
                    symbol=data.get('sym'),
                    timestamp=datetime.fromtimestamp(data.get('t') / 1000),
                    data_type=MarketDataType.QUOTE,
                    bid=data.get('bp'),
                    ask=data.get('ap'),
                    bid_size=data.get('bs'),
                    ask_size=data.get('as'),
                    source=DataSource.POLYGON,
                    raw_data=data
                )
                
                await self._handle_market_data(quote_data)
                
        except Exception as e:
            logger.error(f"Error processing Polygon message: {e}")
    
    async def _handle_market_data(self, data: MarketData):
        """Handle incoming market data"""
        try:
            # Add to cache
            self.price_cache[data.symbol].append(data)
            
            # Add to processing queue
            if not self.processing_queue.full():
                self.processing_queue.put(('market_data', data))
            
            # Notify handlers
            for handler in self.data_handlers[f"{data.symbol}_{data.data_type.value}"]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in market data handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    async def _handle_bar_data(self, data: BarData):
        """Handle incoming bar data"""
        try:
            # Add to cache
            self.bar_cache[data.symbol].append(data)
            
            # Add to processing queue
            if not self.processing_queue.full():
                self.processing_queue.put(('bar_data', data))
            
            # Notify handlers
            for handler in self.data_handlers[f"{data.symbol}_bar"]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in bar data handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling bar data: {e}")
    
    def _process_data_loop(self):
        """Background thread for processing market data"""
        while self.is_running:
            try:
                if not self.processing_queue.empty():
                    data_type, data = self.processing_queue.get(timeout=1)
                    
                    if data_type == 'market_data':
                        self._process_market_data_sync(data)
                    elif data_type == 'bar_data':
                        self._process_bar_data_sync(data)
                        
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                time.sleep(1)
    
    def _process_market_data_sync(self, data: MarketData):
        """Synchronous market data processing"""
        try:
            # Calculate technical indicators
            if data.data_type == MarketDataType.TRADE:
                self._update_technical_indicators(data.symbol, data.price)
            
            # Update price statistics
            self._update_price_statistics(data.symbol, data)
            
        except Exception as e:
            logger.error(f"Error in synchronous market data processing: {e}")
    
    def _process_bar_data_sync(self, data: BarData):
        """Synchronous bar data processing"""
        try:
            # Update OHLCV data
            self._update_ohlcv_data(data.symbol, data)
            
            # Calculate bar-based indicators
            self._calculate_bar_indicators(data.symbol, data)
            
        except Exception as e:
            logger.error(f"Error in synchronous bar data processing: {e}")
    
    def _update_technical_indicators(self, symbol: str, price: float):
        """Update technical indicators for a symbol"""
        try:
            # Get recent prices
            recent_prices = [data.price for data in self.price_cache[symbol] 
                           if data.price is not None][-50:]
            
            if len(recent_prices) < 20:
                return
            
            # Calculate moving averages
            sma_20 = np.mean(recent_prices[-20:])
            sma_50 = np.mean(recent_prices[-50:]) if len(recent_prices) >= 50 else sma_20
            
            # Calculate RSI
            rsi = self._calculate_rsi(recent_prices)
            
            # Store indicators
            self._store_indicators(symbol, {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'last_price': price
            })
            
        except Exception as e:
            logger.error(f"Error updating technical indicators: {e}")
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _update_price_statistics(self, symbol: str, data: MarketData):
        """Update price statistics"""
        try:
            if data.price is None:
                return
            
            # Get recent prices for this symbol
            recent_prices = [d.price for d in self.price_cache[symbol] 
                           if d.price is not None][-100:]
            
            if len(recent_prices) < 10:
                return
            
            # Calculate statistics
            current_price = recent_prices[-1]
            price_change = current_price - recent_prices[-2] if len(recent_prices) > 1 else 0
            price_change_pct = (price_change / recent_prices[-2]) * 100 if len(recent_prices) > 1 else 0
            
            # Store statistics
            self._store_price_stats(symbol, {
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volatility': np.std(recent_prices[-20:]) if len(recent_prices) >= 20 else 0,
                'volume_avg': np.mean([d.volume for d in self.price_cache[symbol][-20:] 
                                     if d.volume is not None]) if len(self.price_cache[symbol]) >= 20 else 0
            })
            
        except Exception as e:
            logger.error(f"Error updating price statistics: {e}")
    
    def _update_ohlcv_data(self, symbol: str, data: BarData):
        """Update OHLCV data"""
        try:
            # Store bar data
            self._store_ohlcv_data(symbol, {
                'timestamp': data.timestamp,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume,
                'vwap': data.vwap
            })
            
        except Exception as e:
            logger.error(f"Error updating OHLCV data: {e}")
    
    def _calculate_bar_indicators(self, symbol: str, data: BarData):
        """Calculate bar-based technical indicators"""
        try:
            # Get recent bars
            recent_bars = list(self.bar_cache[symbol])[-20:]
            
            if len(recent_bars) < 10:
                return
            
            # Calculate indicators
            closes = [bar.close for bar in recent_bars]
            volumes = [bar.volume for bar in recent_bars]
            
            # MACD
            ema_12 = self._calculate_ema(closes, 12)
            ema_26 = self._calculate_ema(closes, 26)
            macd = ema_12 - ema_26
            signal = self._calculate_ema([macd], 9)
            
            # Bollinger Bands
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            std_20 = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes)
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            
            # Store bar indicators
            self._store_bar_indicators(symbol, {
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': macd - signal,
                'bb_upper': bb_upper,
                'bb_middle': sma_20,
                'bb_lower': bb_lower,
                'volume_sma': np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            })
            
        except Exception as e:
            logger.error(f"Error calculating bar indicators: {e}")
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return np.mean(prices)
            
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return np.mean(prices) if prices else 0
    
    def _store_indicators(self, symbol: str, indicators: Dict):
        """Store technical indicators"""
        # This would typically store to a database or cache
        pass
    
    def _store_price_stats(self, symbol: str, stats: Dict):
        """Store price statistics"""
        # This would typically store to a database or cache
        pass
    
    def _store_ohlcv_data(self, symbol: str, data: Dict):
        """Store OHLCV data"""
        # This would typically store to a database or cache
        pass
    
    def _store_bar_indicators(self, symbol: str, indicators: Dict):
        """Store bar-based indicators"""
        # This would typically store to a database or cache
        pass
    
    async def _handle_reconnect(self, source: str):
        """Handle reconnection logic"""
        try:
            self.reconnect_attempts[source] += 1
            
            if self.reconnect_attempts[source] <= self.max_reconnect_attempts:
                delay = self.reconnect_delay * self.reconnect_attempts[source]
                logger.info(f"Reconnecting to {source} in {delay} seconds (attempt {self.reconnect_attempts[source]})")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Max reconnection attempts reached for {source}")
                
        except Exception as e:
            logger.error(f"Error handling reconnection for {source}: {e}")
    
    async def _start_data_processor(self):
        """Start data processing loop"""
        while self.is_running:
            try:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in data processor: {e}")
    
    def add_data_handler(self, symbol: str, data_type: MarketDataType, handler: Callable):
        """Add a data handler for specific symbol and data type"""
        try:
            key = f"{symbol}_{data_type.value}"
            self.data_handlers[key].append(handler)
            logger.info(f"Added data handler for {key}")
        except Exception as e:
            logger.error(f"Error adding data handler: {e}")
    
    def remove_data_handler(self, symbol: str, data_type: MarketDataType, handler: Callable):
        """Remove a data handler"""
        try:
            key = f"{symbol}_{data_type.value}"
            if handler in self.data_handlers[key]:
                self.data_handlers[key].remove(handler)
                logger.info(f"Removed data handler for {key}")
        except Exception as e:
            logger.error(f"Error removing data handler: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            if symbol in self.price_cache and self.price_cache[symbol]:
                latest_data = self.price_cache[symbol][-1]
                return latest_data.price
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get price history for a symbol"""
        try:
            if symbol in self.price_cache:
                return list(self.price_cache[symbol])[-limit:]
            return []
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []
    
    def get_bar_history(self, symbol: str, limit: int = 50) -> List[BarData]:
        """Get bar history for a symbol"""
        try:
            if symbol in self.bar_cache:
                return list(self.bar_cache[symbol])[-limit:]
            return []
        except Exception as e:
            logger.error(f"Error getting bar history for {symbol}: {e}")
            return []
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols"""
        return list(self.price_cache.keys())
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all data sources"""
        return {
            source: source in self.websocket_connections 
            for source in ['alpaca', 'polygon']
        }
