#!/usr/bin/env python3
"""
RemoteAlgoTrader v1.0 - AI-Powered Algorithmic Trading Bot
A sophisticated trading bot using sentiment analysis and technical indicators
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Third-party imports
import torch
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.stream import TradingStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingConfig:
    """Configuration class for trading parameters"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY', 'ENTER YOUR KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', 'ENTER YOUR API')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', 'YOUR_API_KEY')
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.trading_hours = ["10:30:00", "12:00:00", "14:30:00"]
        self.position_size = 5
        self.max_positions = 10
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
        self.risk_per_trade = 0.01  # 1% risk per trade

class SentimentAnalyzer:
    """AI-powered sentiment analysis using DeepSeek model"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True)
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def analyze_sentiment(self, news_text: str, tickers: List[str]) -> Dict[str, str]:
        """Analyze sentiment of news and return ticker-sentiment mapping"""
        try:
            prompt = f"""You are a Professional Stock Advisor (for entertainment purposes). 
            Analyze the sentiment of this news headline and respond with either [POSITIVE] or [NEGATIVE] after </think>.
            Keep your explanation concise—under 200 words—and focus only on the key factors influencing the sentiment.
            
            News: {news_text}
            
            Tickers: {', '.join(tickers)}"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**inputs, max_length=850, do_sample=True, temperature=0.7)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract sentiment from response
            import re
            sentiment_match = re.search(r'\[(POSITIVE|NEGATIVE)\]', output)
            sentiment = sentiment_match.group(1) if sentiment_match else "NEUTRAL"
            
            return {ticker: sentiment for ticker in tickers}
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {ticker: "NEUTRAL" for ticker in tickers}

class NewsCollector:
    """Collect and process financial news"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/reference/news"
    
    def get_news(self, limit: int = 10) -> List[Dict]:
        """Fetch latest financial news"""
        try:
            params = {
                'limit': limit,
                'apiKey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
            
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []

class TechnicalAnalyzer:
    """Technical analysis using various indicators"""
    
    @staticmethod
    def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Ensure we have enough data
            if len(data) < 26:
                return data
            
            # Calculate EMAs
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            
            # Calculate MACD
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['Signal']
            
            # Calculate RSI
            data['RSI'] = ta.rsi(data['Close'], length=14)
            
            # Calculate Bollinger Bands
            bb = ta.bbands(data['Close'], length=20)
            data = pd.concat([data, bb], axis=1)
            
            # Calculate volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return data
    
    @staticmethod
    def generate_signals(data: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals based on technical indicators"""
        if len(data) < 26:
            return {'signal': 'NEUTRAL', 'strength': 0}
        
        try:
            current = data.iloc[-1]
            
            # MACD signal
            macd_bullish = current['MACD'] > current['Signal']
            macd_bearish = current['MACD'] < current['Signal']
            
            # RSI signal
            rsi_bullish = current['RSI'] < 70 and current['RSI'] > 30
            rsi_oversold = current['RSI'] < 30
            rsi_overbought = current['RSI'] > 70
            
            # Bollinger Bands signal
            bb_bullish = current['Close'] > current['BBL_20_2.0']
            bb_bearish = current['Close'] < current['BBU_20_2.0']
            
            # Volume confirmation
            volume_high = current['Volume_Ratio'] > 1.5
            
            # Calculate signal strength
            bullish_signals = sum([macd_bullish, rsi_bullish, bb_bullish])
            bearish_signals = sum([macd_bearish, rsi_overbought, bb_bearish])
            
            if bullish_signals > bearish_signals and rsi_oversold:
                return {'signal': 'BUY', 'strength': bullish_signals}
            elif bearish_signals > bullish_signals and rsi_overbought:
                return {'signal': 'SELL', 'strength': bearish_signals}
            else:
                return {'signal': 'NEUTRAL', 'strength': 0}
                
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}

class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.sentiment_analyzer = SentimentAnalyzer(config.model_name)
        self.news_collector = NewsCollector(config.polygon_api_key)
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize trading client
        try:
            self.trading_client = TradingClient(config.api_key, config.api_secret)
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca. Account status: {account.status}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
        
        # Trading state
        self.trading_active = True
        self.positions = {}
        self.positive_sentiment_stocks = []
        self.negative_sentiment_stocks = []
        
    def collect_sentiment_data(self):
        """Collect news and analyze sentiment"""
        try:
            news_items = self.news_collector.get_news(limit=5)
            
            for news in news_items:
                tickers = news.get('tickers', [])
                description = news.get('description', '')
                
                if tickers and description:
                    sentiment_map = self.sentiment_analyzer.analyze_sentiment(description, tickers)
                    
                    for ticker, sentiment in sentiment_map.items():
                        if sentiment == "POSITIVE" and ticker not in self.positive_sentiment_stocks:
                            self.positive_sentiment_stocks.append(ticker)
                        elif sentiment == "NEGATIVE" and ticker not in self.negative_sentiment_stocks:
                            self.negative_sentiment_stocks.append(ticker)
            
            logger.info(f"Sentiment analysis complete. Positive: {len(self.positive_sentiment_stocks)}, Negative: {len(self.negative_sentiment_stocks)}")
            
        except Exception as e:
            logger.error(f"Failed to collect sentiment data: {e}")
    
    def get_stock_data(self, symbol: str, period: str = '5d', interval: str = '1m') -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if len(data) > 0:
                return self.technical_analyzer.calculate_indicators(data)
            else:
                logger.warning(f"No data received for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol: str, side: str, quantity: int):
        """Execute a trade order"""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            logger.info(f"{side} order submitted for {symbol}: {order.id}")
            
            # Update positions
            if side == "BUY":
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': order.filled_avg_price or 0,
                    'entry_time': datetime.now()
                }
            else:
                if symbol in self.positions:
                    del self.positions[symbol]
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute {side} order for {symbol}: {e}")
            return None
    
    def manage_positions(self):
        """Manage existing positions with stop-loss and take-profit"""
        for symbol, position in list(self.positions.items()):
            try:
                current_data = self.get_stock_data(symbol, period='1d', interval='1m')
                if current_data is None:
                    continue
                
                current_price = current_data['Close'].iloc[-1]
                entry_price = position['entry_price']
                
                # Calculate P&L
                if entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Stop loss
                    if pnl_pct <= -self.config.stop_loss_pct:
                        logger.info(f"Stop loss triggered for {symbol} at {pnl_pct:.2%}")
                        self.execute_trade(symbol, "SELL", position['quantity'])
                    
                    # Take profit
                    elif pnl_pct >= self.config.take_profit_pct:
                        logger.info(f"Take profit triggered for {symbol} at {pnl_pct:.2%}")
                        self.execute_trade(symbol, "SELL", position['quantity'])
                        
            except Exception as e:
                logger.error(f"Failed to manage position for {symbol}: {e}")
    
    def trading_cycle(self):
        """Main trading cycle"""
        logger.info("Starting trading cycle...")
        
        while self.trading_active:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Check if it's time to collect new sentiment data
                if current_time in self.config.trading_hours:
                    logger.info(f"Trading hour reached: {current_time}")
                    self.collect_sentiment_data()
                
                # Manage existing positions
                self.manage_positions()
                
                # Trading logic for positive sentiment stocks
                if self.positive_sentiment_stocks and len(self.positions) < self.config.max_positions:
                    symbol = self.positive_sentiment_stocks[0]
                    data = self.get_stock_data(symbol)
                    
                    if data is not None:
                        signals = self.technical_analyzer.generate_signals(data)
                        
                        if signals['signal'] == 'BUY' and signals['strength'] >= 2:
                            logger.info(f"BUY signal for {symbol} with strength {signals['strength']}")
                            self.execute_trade(symbol, "BUY", self.config.position_size)
                            self.positive_sentiment_stocks.pop(0)
                
                # Trading logic for negative sentiment stocks (short selling)
                if self.negative_sentiment_stocks and len(self.positions) < self.config.max_positions:
                    symbol = self.negative_sentiment_stocks[0]
                    data = self.get_stock_data(symbol)
                    
                    if data is not None:
                        signals = self.technical_analyzer.generate_signals(data)
                        
                        if signals['signal'] == 'SELL' and signals['strength'] >= 2:
                            logger.info(f"SELL signal for {symbol} with strength {signals['strength']}")
                            self.execute_trade(symbol, "SELL", self.config.position_size)
                            self.negative_sentiment_stocks.pop(0)
                
                # Sleep for 1 minute before next cycle
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                self.trading_active = False
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(60)
    
    def run(self):
        """Start the trading bot"""
        try:
            logger.info("Initializing RemoteAlgoTrader v1.0...")
            self.collect_sentiment_data()
            self.trading_cycle()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise

def main():
    """Main entry point"""
    try:
        config = TradingConfig()
        bot = TradingBot(config)
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start trading bot: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
