#!/usr/bin/env python3
"""
Advanced Portfolio Management System for RemoteAlgoTrader
Handles position tracking, risk analytics, and performance metrics
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELLED = "cancelled"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    side: str  # "BUY" or "SELL"
    status: PositionStatus
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees: float = 0.0
    strategy: str = "default"
    sentiment_score: Optional[float] = None
    technical_score: Optional[float] = None

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float  # Value at Risk 95%
    current_exposure: float
    sector_exposure: Dict[str, float]
    correlation_matrix: pd.DataFrame

class PortfolioManager:
    """Advanced portfolio management with risk analytics"""
    
    def __init__(self, db_path: str = "portfolio.db"):
        self.db_path = db_path
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% max per position
            'max_sector_exposure': 0.3,  # 30% max per sector
            'max_daily_loss': 0.05,  # 5% max daily loss
            'max_drawdown': 0.15,  # 15% max drawdown
            'var_limit': 0.02,  # 2% VaR limit
        }
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for portfolio tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                side TEXT NOT NULL,
                status TEXT NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                exit_price REAL,
                exit_time TIMESTAMP,
                pnl REAL,
                pnl_pct REAL,
                fees REAL DEFAULT 0.0,
                strategy TEXT DEFAULT 'default',
                sentiment_score REAL,
                technical_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create portfolio_snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_pct REAL NOT NULL,
                exposure REAL NOT NULL,
                risk_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create risk_alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_position(self, position: Position) -> bool:
        """Add a new position to the portfolio"""
        try:
            # Validate position
            if position.quantity <= 0 or position.entry_price <= 0:
                logger.error(f"Invalid position data: {position}")
                return False
            
            # Check risk limits
            if not self._check_risk_limits(position):
                logger.warning(f"Risk limit exceeded for position: {position.symbol}")
                return False
            
            # Add to memory
            self.positions[position.symbol] = position
            
            # Save to database
            self._save_position_to_db(position)
            
            logger.info(f"Position added: {position.symbol} {position.side} {position.quantity} @ {position.entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, exit_time: datetime = None) -> bool:
        """Close a position and calculate P&L"""
        try:
            if symbol not in self.positions:
                logger.error(f"Position not found: {symbol}")
                return False
            
            position = self.positions[symbol]
            exit_time = exit_time or datetime.now()
            
            # Calculate P&L
            if position.side == "BUY":
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # SELL (short)
                pnl = (position.entry_price - exit_price) * position.quantity
            
            pnl_pct = pnl / (position.entry_price * position.quantity)
            
            # Update position
            position.exit_price = exit_price
            position.exit_time = exit_time
            position.pnl = pnl
            position.pnl_pct = pnl_pct
            position.status = PositionStatus.CLOSED
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # Update database
            self._update_position_in_db(position)
            
            logger.info(f"Position closed: {symbol} P&L: ${pnl:.2f} ({pnl_pct:.2%})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False
    
    def update_position_price(self, symbol: str, current_price: float) -> Optional[float]:
        """Update position with current price and return unrealized P&L"""
        try:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            
            # Calculate unrealized P&L
            if position.side == "BUY":
                pnl = (current_price - position.entry_price) * position.quantity
            else:  # SELL (short)
                pnl = (position.entry_price - current_price) * position.quantity
            
            pnl_pct = pnl / (position.entry_price * position.quantity)
            
            # Check stop-loss and take-profit
            if position.stop_loss and pnl_pct <= -position.stop_loss:
                logger.info(f"Stop-loss triggered for {symbol}")
                self.close_position(symbol, current_price)
                return pnl
            
            if position.take_profit and pnl_pct >= position.take_profit:
                logger.info(f"Take-profit triggered for {symbol}")
                self.close_position(symbol, current_price)
                return pnl
            
            return pnl
            
        except Exception as e:
            logger.error(f"Failed to update position price for {symbol}: {e}")
            return None
    
    def get_portfolio_metrics(self, cash_balance: float = 0.0) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Calculate basic metrics
            total_pnl = sum(pos.pnl or 0 for pos in self.closed_positions)
            total_trades = len(self.closed_positions)
            winning_trades = len([pos for pos in self.closed_positions if pos.pnl and pos.pnl > 0])
            losing_trades = len([pos for pos in self.closed_positions if pos.pnl and pos.pnl < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average win/loss
            wins = [pos.pnl for pos in self.closed_positions if pos.pnl and pos.pnl > 0]
            losses = [pos.pnl for pos in self.closed_positions if pos.pnl and pos.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Calculate drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 2%)
            returns = [pos.pnl_pct for pos in self.closed_positions if pos.pnl_pct]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Calculate VaR
            var_95 = self._calculate_var(returns, 0.95)
            
            # Calculate current exposure
            current_exposure = sum(
                pos.quantity * pos.entry_price for pos in self.positions.values()
            )
            
            # Calculate sector exposure
            sector_exposure = self._calculate_sector_exposure()
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix()
            
            # Calculate Calmar ratio
            calmar_ratio = (total_pnl / abs(max_drawdown)) if max_drawdown != 0 else 0
            
            return PortfolioMetrics(
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl / (cash_balance + current_exposure) if (cash_balance + current_exposure) > 0 else 0,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                current_exposure=current_exposure,
                sector_exposure=sector_exposure,
                correlation_matrix=correlation_matrix
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio metrics: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, pd.DataFrame())
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from closed positions"""
        try:
            if not self.closed_positions:
                return 0.0
            
            # Create equity curve
            positions_sorted = sorted(self.closed_positions, key=lambda x: x.exit_time)
            cumulative_pnl = 0
            peak = 0
            max_dd = 0
            
            for pos in positions_sorted:
                cumulative_pnl += pos.pnl or 0
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                
                drawdown = (peak - cumulative_pnl) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Failed to calculate max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
            
            if len(excess_returns) < 2:
                return 0.0
            
            return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            excess_returns = returns_array - risk_free_rate / 252
            
            if len(excess_returns) < 2:
                return 0.0
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            
            return np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to calculate Sortino ratio: {e}")
            return 0.0
    
    def _calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            return np.percentile(returns_array, (1 - confidence_level) * 100)
            
        except Exception as e:
            logger.error(f"Failed to calculate VaR: {e}")
            return 0.0
    
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure (placeholder - would need sector data)"""
        # This is a placeholder - in a real implementation, you'd need sector data
        return {"Technology": 0.4, "Finance": 0.3, "Healthcare": 0.2, "Other": 0.1}
    
    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between positions"""
        try:
            if len(self.closed_positions) < 2:
                return pd.DataFrame()
            
            # Create returns matrix
            symbols = list(set(pos.symbol for pos in self.closed_positions))
            returns_data = {}
            
            for symbol in symbols:
                symbol_positions = [pos for pos in self.closed_positions if pos.symbol == symbol]
                if len(symbol_positions) > 1:
                    returns = [pos.pnl_pct for pos in symbol_positions if pos.pnl_pct is not None]
                    if returns:
                        returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # Pad with zeros to make equal length
            max_length = max(len(returns) for returns in returns_data.values())
            padded_returns = {}
            
            for symbol, returns in returns_data.items():
                padded_returns[symbol] = returns + [0] * (max_length - len(returns))
            
            df = pd.DataFrame(padded_returns)
            return df.corr()
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            return pd.DataFrame()
    
    def _check_risk_limits(self, position: Position) -> bool:
        """Check if position violates risk limits"""
        try:
            # Calculate position value
            position_value = position.quantity * position.entry_price
            
            # Get total portfolio value
            total_value = sum(
                pos.quantity * pos.entry_price for pos in self.positions.values()
            ) + position_value
            
            # Check position size limit
            if total_value > 0 and position_value / total_value > self.risk_limits['max_position_size']:
                logger.warning(f"Position size limit exceeded: {position_value / total_value:.2%}")
                return False
            
            # Check daily loss limit
            today_pnl = sum(
                pos.pnl for pos in self.closed_positions 
                if pos.exit_time and pos.exit_time.date() == datetime.now().date()
            )
            
            if total_value > 0 and abs(today_pnl) / total_value > self.risk_limits['max_daily_loss']:
                logger.warning(f"Daily loss limit exceeded: {abs(today_pnl) / total_value:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check risk limits: {e}")
            return False
    
    def _save_position_to_db(self, position: Position):
        """Save position to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions (
                    symbol, quantity, entry_price, entry_time, side, status,
                    stop_loss, take_profit, strategy, sentiment_score, technical_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.quantity, position.entry_price,
                position.entry_time, position.side, position.status.value,
                position.stop_loss, position.take_profit, position.strategy,
                position.sentiment_score, position.technical_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save position to database: {e}")
    
    def _update_position_in_db(self, position: Position):
        """Update position in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE positions SET
                    status = ?, exit_price = ?, exit_time = ?, pnl = ?, pnl_pct = ?
                WHERE symbol = ? AND status = 'open'
            ''', (
                position.status.value, position.exit_price, position.exit_time,
                position.pnl, position.pnl_pct, position.symbol
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update position in database: {e}")
    
    def get_risk_alerts(self) -> List[Dict]:
        """Get current risk alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT alert_type, message, severity, timestamp, resolved
                FROM risk_alerts
                WHERE resolved = FALSE
                ORDER BY timestamp DESC
            ''')
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'type': row[0],
                    'message': row[1],
                    'severity': row[2],
                    'timestamp': row[3],
                    'resolved': row[4]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get risk alerts: {e}")
            return []
    
    def add_risk_alert(self, alert_type: str, message: str, severity: str = "medium"):
        """Add a new risk alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_alerts (alert_type, message, severity, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (alert_type, message, severity, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"Risk alert added: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to add risk alert: {e}")
    
    def export_portfolio_report(self, filename: str = None) -> str:
        """Export comprehensive portfolio report"""
        try:
            if not filename:
                filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            metrics = self.get_portfolio_metrics()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(metrics),
                'open_positions': [asdict(pos) for pos in self.positions.values()],
                'recent_closed_positions': [
                    asdict(pos) for pos in self.closed_positions[-20:]  # Last 20 positions
                ],
                'risk_alerts': self.get_risk_alerts(),
                'risk_limits': self.risk_limits
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Portfolio report exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export portfolio report: {e}")
            return ""
