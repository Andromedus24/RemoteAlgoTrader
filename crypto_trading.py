#!/usr/bin/env python3
"""
Cryptocurrency Trading & Blockchain Integration for RemoteAlgoTrader
Handles crypto trading, blockchain data analysis, DeFi protocols, and cross-chain operations
"""

import os
import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import base64
import requests
import websockets
import aiohttp
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
import ccxt
from web3 import Web3
from eth_account import Account
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

# Set precision for crypto calculations
getcontext().prec = 28

class CryptoExchange(Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    BYBIT = "bybit"
    OKX = "okx"

class Blockchain(Enum):
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"

class DeFiProtocol(Enum):
    UNISWAP = "uniswap"
    SUSHISWAP = "sushiswap"
    AAVE = "aave"
    COMPOUND = "compound"
    CURVE = "curve"
    BALANCER = "balancer"

@dataclass
class CryptoAsset:
    """Represents a cryptocurrency asset"""
    symbol: str
    name: str
    blockchain: Blockchain
    contract_address: Optional[str] = None
    decimals: int = 18
    total_supply: Optional[float] = None
    circulating_supply: Optional[float] = None
    market_cap: Optional[float] = None
    price_usd: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[float] = None

@dataclass
class CryptoOrder:
    """Represents a cryptocurrency order"""
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop"
    quantity: float
    price: Optional[float] = None
    status: str  # "open", "filled", "cancelled"
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    timestamp: datetime = None
    exchange: CryptoExchange = CryptoExchange.BINANCE

@dataclass
class BlockchainTransaction:
    """Represents a blockchain transaction"""
    tx_hash: str
    blockchain: Blockchain
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_used: int
    block_number: int
    timestamp: datetime
    status: str  # "pending", "confirmed", "failed"
    fee: float

@dataclass
class DeFiPosition:
    """Represents a DeFi position"""
    protocol: DeFiProtocol
    blockchain: Blockchain
    asset: str
    quantity: float
    apy: float
    position_type: str  # "lending", "borrowing", "liquidity", "yield_farming"
    entry_time: datetime
    current_value: float
    unrealized_pnl: float

class CryptoExchangeManager:
    """Manages multiple cryptocurrency exchanges"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.api_keys = config.get('api_keys', {})
        self.secret_keys = config.get('secret_keys', {})
        self.passphrases = config.get('passphrases', {})
        
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            for exchange_name in CryptoExchange:
                exchange_id = exchange_name.value
                
                if exchange_id in self.api_keys:
                    # Initialize with API credentials
                    exchange_class = getattr(ccxt, exchange_id)
                    exchange_config = {
                        'apiKey': self.api_keys[exchange_id],
                        'secret': self.secret_keys.get(exchange_id, ''),
                        'password': self.passphrases.get(exchange_id, ''),
                        'sandbox': self.config.get('sandbox_mode', False),
                        'enableRateLimit': True
                    }
                    
                    self.exchanges[exchange_id] = exchange_class(exchange_config)
                    logger.info(f"Initialized {exchange_id} exchange")
                else:
                    # Initialize without credentials (public data only)
                    exchange_class = getattr(ccxt, exchange_id)
                    self.exchanges[exchange_id] = exchange_class({
                        'enableRateLimit': True
                    })
                    logger.info(f"Initialized {exchange_id} exchange (public only)")
                    
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def get_ticker(self, symbol: str, exchange: CryptoExchange = CryptoExchange.BINANCE) -> Dict:
        """Get current ticker for a symbol"""
        try:
            exchange_instance = self.exchanges[exchange.value]
            ticker = await exchange_instance.fetch_ticker(symbol)
            
            return {
                'symbol': ticker['symbol'],
                'last_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'change': ticker['change'],
                'change_percent': ticker['percentage'],
                'timestamp': ticker['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol} on {exchange.value}: {e}")
            return {}
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100,
                       exchange: CryptoExchange = CryptoExchange.BINANCE) -> pd.DataFrame:
        """Get OHLCV data for a symbol"""
        try:
            exchange_instance = self.exchanges[exchange.value]
            ohlcv = await exchange_instance.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV for {symbol} on {exchange.value}: {e}")
            return pd.DataFrame()
    
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                         price: Optional[float] = None, exchange: CryptoExchange = CryptoExchange.BINANCE) -> CryptoOrder:
        """Place an order on the exchange"""
        try:
            exchange_instance = self.exchanges[exchange.value]
            
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': quantity
            }
            
            if price and order_type == 'limit':
                order_params['price'] = price
            
            order = await exchange_instance.create_order(**order_params)
            
            return CryptoOrder(
                id=order['id'],
                symbol=order['symbol'],
                side=order['side'],
                order_type=order['type'],
                quantity=order['amount'],
                price=order.get('price'),
                status=order['status'],
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                exchange=exchange
            )
            
        except Exception as e:
            logger.error(f"Error placing order on {exchange.value}: {e}")
            return None
    
    async def get_balance(self, exchange: CryptoExchange = CryptoExchange.BINANCE) -> Dict[str, float]:
        """Get account balance"""
        try:
            exchange_instance = self.exchanges[exchange.value]
            balance = await exchange_instance.fetch_balance()
            
            return {
                currency: float(info['free']) 
                for currency, info in balance['free'].items() 
                if float(info['free']) > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting balance from {exchange.value}: {e}")
            return {}
    
    async def get_order_status(self, order_id: str, symbol: str,
                              exchange: CryptoExchange = CryptoExchange.BINANCE) -> Dict:
        """Get order status"""
        try:
            exchange_instance = self.exchanges[exchange.value]
            order = await exchange_instance.fetch_order(order_id, symbol)
            
            return {
                'id': order['id'],
                'status': order['status'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'average': order.get('average'),
                'cost': order.get('cost')
            }
            
        except Exception as e:
            logger.error(f"Error getting order status from {exchange.value}: {e}")
            return {}

class BlockchainManager:
    """Manages blockchain connections and operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.web3_instances: Dict[str, Web3] = {}
        self.rpc_urls = config.get('rpc_urls', {})
        self.private_keys = config.get('private_keys', {})
        
        self._initialize_blockchains()
    
    def _initialize_blockchains(self):
        """Initialize blockchain connections"""
        try:
            for blockchain in Blockchain:
                blockchain_name = blockchain.value
                
                if blockchain_name in self.rpc_urls:
                    # Initialize Web3 instance
                    w3 = Web3(Web3.HTTPProvider(self.rpc_urls[blockchain_name]))
                    
                    if w3.is_connected():
                        self.web3_instances[blockchain_name] = w3
                        logger.info(f"Connected to {blockchain_name} blockchain")
                    else:
                        logger.warning(f"Failed to connect to {blockchain_name} blockchain")
                        
        except Exception as e:
            logger.error(f"Error initializing blockchains: {e}")
    
    async def get_balance(self, address: str, blockchain: Blockchain) -> float:
        """Get ETH/BNB balance for an address"""
        try:
            if blockchain.value not in self.web3_instances:
                return 0.0
            
            w3 = self.web3_instances[blockchain.value]
            balance_wei = w3.eth.get_balance(address)
            balance_eth = w3.from_wei(balance_wei, 'ether')
            
            return float(balance_eth)
            
        except Exception as e:
            logger.error(f"Error getting balance for {address} on {blockchain.value}: {e}")
            return 0.0
    
    async def get_token_balance(self, token_address: str, wallet_address: str, 
                               blockchain: Blockchain) -> float:
        """Get ERC-20 token balance"""
        try:
            if blockchain.value not in self.web3_instances:
                return 0.0
            
            w3 = self.web3_instances[blockchain.value]
            
            # ERC-20 token contract ABI (simplified)
            token_abi = [
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                }
            ]
            
            token_contract = w3.eth.contract(address=token_address, abi=token_abi)
            balance_wei = token_contract.functions.balanceOf(wallet_address).call()
            decimals = token_contract.functions.decimals().call()
            
            balance = balance_wei / (10 ** decimals)
            return float(balance)
            
        except Exception as e:
            logger.error(f"Error getting token balance: {e}")
            return 0.0
    
    async def send_transaction(self, from_address: str, to_address: str, value: float,
                              blockchain: Blockchain, gas_price: Optional[int] = None) -> str:
        """Send a transaction"""
        try:
            if blockchain.value not in self.web3_instances:
                raise ValueError(f"Blockchain {blockchain.value} not initialized")
            
            w3 = self.web3_instances[blockchain.value]
            
            # Get private key for the address
            private_key = self.private_keys.get(from_address)
            if not private_key:
                raise ValueError(f"No private key found for address {from_address}")
            
            # Prepare transaction
            nonce = w3.eth.get_transaction_count(from_address)
            
            transaction = {
                'nonce': nonce,
                'to': to_address,
                'value': w3.to_wei(value, 'ether'),
                'gas': 21000,  # Standard gas limit for ETH transfer
                'gasPrice': gas_price or w3.eth.gas_price,
                'chainId': w3.eth.chain_id
            }
            
            # Sign transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return ""
    
    async def get_transaction_receipt(self, tx_hash: str, blockchain: Blockchain) -> Dict:
        """Get transaction receipt"""
        try:
            if blockchain.value not in self.web3_instances:
                return {}
            
            w3 = self.web3_instances[blockchain.value]
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            
            if receipt:
                return {
                    'tx_hash': receipt['transactionHash'].hex(),
                    'block_number': receipt['blockNumber'],
                    'gas_used': receipt['gasUsed'],
                    'status': 'success' if receipt['status'] == 1 else 'failed',
                    'logs': len(receipt['logs'])
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting transaction receipt: {e}")
            return {}
    
    async def get_gas_price(self, blockchain: Blockchain) -> int:
        """Get current gas price"""
        try:
            if blockchain.value not in self.web3_instances:
                return 0
            
            w3 = self.web3_instances[blockchain.value]
            gas_price = w3.eth.gas_price
            
            return gas_price
            
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            return 0

class DeFiManager:
    """Manages DeFi protocol interactions"""
    
    def __init__(self, blockchain_manager: BlockchainManager):
        self.blockchain_manager = blockchain_manager
        self.protocol_contracts = {}
        self._initialize_protocols()
    
    def _initialize_protocols(self):
        """Initialize DeFi protocol contracts"""
        try:
            # This would typically load contract ABIs and addresses
            # For now, using placeholder data
            self.protocol_contracts = {
                DeFiProtocol.UNISWAP: {
                    'router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                    'factory': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'
                },
                DeFiProtocol.AAVE: {
                    'lending_pool': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9'
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing DeFi protocols: {e}")
    
    async def get_uniswap_price(self, token_address: str, weth_address: str = None) -> float:
        """Get token price from Uniswap"""
        try:
            # This would use Uniswap V2 router to get price
            # Simplified implementation
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting Uniswap price: {e}")
            return 0.0
    
    async def swap_tokens(self, token_in: str, token_out: str, amount_in: float,
                         min_amount_out: float, wallet_address: str,
                         blockchain: Blockchain = Blockchain.ETHEREUM) -> str:
        """Swap tokens on Uniswap"""
        try:
            # This would execute a swap on Uniswap
            # Simplified implementation
            return ""
            
        except Exception as e:
            logger.error(f"Error swapping tokens: {e}")
            return ""
    
    async def add_liquidity(self, token_a: str, token_b: str, amount_a: float,
                           amount_b: float, wallet_address: str,
                           blockchain: Blockchain = Blockchain.ETHEREUM) -> str:
        """Add liquidity to Uniswap pool"""
        try:
            # This would add liquidity to a Uniswap pool
            # Simplified implementation
            return ""
            
        except Exception as e:
            logger.error(f"Error adding liquidity: {e}")
            return ""

class CryptoAnalyzer:
    """Cryptocurrency market analysis"""
    
    def __init__(self):
        self.market_data = {}
        self.technical_indicators = {}
    
    def calculate_crypto_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for crypto"""
        try:
            indicators = df.copy()
            
            # Moving averages
            indicators['sma_20'] = indicators['close'].rolling(window=20).mean()
            indicators['sma_50'] = indicators['close'].rolling(window=50).mean()
            indicators['ema_12'] = indicators['close'].ewm(span=12).mean()
            indicators['ema_26'] = indicators['close'].ewm(span=26).mean()
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # RSI
            delta = indicators['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            indicators['bb_middle'] = indicators['close'].rolling(window=20).mean()
            bb_std = indicators['close'].rolling(window=20).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
            indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            indicators['volume_sma'] = indicators['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma']
            
            # Crypto-specific indicators
            indicators['price_change'] = indicators['close'].pct_change()
            indicators['volatility'] = indicators['price_change'].rolling(window=20).std()
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating crypto indicators: {e}")
            return df
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns"""
        try:
            patterns = []
            
            # Double top pattern
            if self._detect_double_top(df):
                patterns.append({
                    'pattern': 'double_top',
                    'confidence': 0.8,
                    'price_level': df['high'].iloc[-1]
                })
            
            # Double bottom pattern
            if self._detect_double_bottom(df):
                patterns.append({
                    'pattern': 'double_bottom',
                    'confidence': 0.8,
                    'price_level': df['low'].iloc[-1]
                })
            
            # Head and shoulders
            if self._detect_head_shoulders(df):
                patterns.append({
                    'pattern': 'head_shoulders',
                    'confidence': 0.7,
                    'price_level': df['low'].iloc[-1]
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _detect_double_top(self, df: pd.DataFrame) -> bool:
        """Detect double top pattern"""
        try:
            # Simplified double top detection
            highs = df['high'].rolling(window=5).max()
            recent_highs = highs.tail(20)
            
            # Look for two similar peaks
            peaks = recent_highs[recent_highs == recent_highs.rolling(window=5).max()]
            
            if len(peaks) >= 2:
                peak_values = peaks.values
                if abs(peak_values[-1] - peak_values[-2]) / peak_values[-1] < 0.02:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting double top: {e}")
            return False
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> bool:
        """Detect double bottom pattern"""
        try:
            # Simplified double bottom detection
            lows = df['low'].rolling(window=5).min()
            recent_lows = lows.tail(20)
            
            # Look for two similar troughs
            troughs = recent_lows[recent_lows == recent_lows.rolling(window=5).min()]
            
            if len(troughs) >= 2:
                trough_values = troughs.values
                if abs(trough_values[-1] - trough_values[-2]) / trough_values[-1] < 0.02:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting double bottom: {e}")
            return False
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> bool:
        """Detect head and shoulders pattern"""
        try:
            # Simplified head and shoulders detection
            # This is a complex pattern that would require more sophisticated analysis
            return False
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return False
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            highs = df['high'].tail(100)
            lows = df['low'].tail(100)
            
            # Find resistance levels (highs)
            resistance_levels = []
            for i in range(len(highs) - 1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    resistance_levels.append(highs.iloc[i])
            
            # Find support levels (lows)
            support_levels = []
            for i in range(len(lows) - 1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    support_levels.append(lows.iloc[i])
            
            return {
                'resistance': sorted(list(set(resistance_levels)), reverse=True)[:5],
                'support': sorted(list(set(support_levels)))[:5]
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {'resistance': [], 'support': []}

class CryptoArbitrageBot:
    """Cryptocurrency arbitrage bot"""
    
    def __init__(self, exchange_manager: CryptoExchangeManager):
        self.exchange_manager = exchange_manager
        self.arbitrage_opportunities = []
    
    async def find_arbitrage_opportunities(self, symbols: List[str], 
                                         min_spread: float = 0.01) -> List[Dict]:
        """Find arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            
            for symbol in symbols:
                prices = {}
                
                # Get prices from different exchanges
                for exchange in CryptoExchange:
                    try:
                        ticker = await self.exchange_manager.get_ticker(symbol, exchange)
                        if ticker and 'last_price' in ticker:
                            prices[exchange.value] = ticker['last_price']
                    except Exception as e:
                        logger.warning(f"Could not get price for {symbol} on {exchange.value}: {e}")
                
                if len(prices) >= 2:
                    # Find best bid and ask
                    min_price = min(prices.values())
                    max_price = max(prices.values())
                    
                    spread = (max_price - min_price) / min_price
                    
                    if spread >= min_spread:
                        opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': min(prices, key=prices.get),
                            'sell_exchange': max(prices, key=prices.get),
                            'buy_price': min_price,
                            'sell_price': max_price,
                            'spread': spread,
                            'potential_profit_pct': spread * 100
                        })
            
            self.arbitrage_opportunities = opportunities
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities: {e}")
            return []
    
    async def execute_arbitrage(self, opportunity: Dict, quantity: float) -> bool:
        """Execute arbitrage trade"""
        try:
            # This would execute the actual arbitrage trades
            # Simplified implementation
            logger.info(f"Executing arbitrage: {opportunity}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return False
