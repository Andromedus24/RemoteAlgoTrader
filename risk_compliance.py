#!/usr/bin/env python3
"""
Advanced Risk Management & Compliance System for RemoteAlgoTrader
Provides comprehensive risk monitoring, compliance checking, regulatory reporting, and automated safeguards
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import sqlite3
from pathlib import Path
import hashlib
import hmac
import base64
from collections import defaultdict, deque
import threading
from queue import Queue
import asyncio

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    SUSPENDED = "suspended"

class RiskType(Enum):
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    CONCENTRATION_RISK = "concentration_risk"
    VOLATILITY_RISK = "volatility_risk"

@dataclass
class RiskMetric:
    """Represents a risk metric"""
    risk_type: RiskType
    value: float
    threshold: float
    risk_level: RiskLevel
    timestamp: datetime
    description: str
    source: str

@dataclass
class ComplianceRule:
    """Represents a compliance rule"""
    rule_id: str
    name: str
    description: str
    rule_type: str
    parameters: Dict[str, Any]
    enabled: bool
    priority: int
    created_at: datetime

@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule_id: str
    violation_type: str
    severity: RiskLevel
    message: str
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class PositionLimit:
    """Represents a position limit"""
    symbol: str
    max_position_size: float
    max_position_value: float
    max_daily_trades: int
    max_daily_volume: float
    current_position: float = 0.0
    current_value: float = 0.0
    daily_trades: int = 0
    daily_volume: float = 0.0

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_metrics: Dict[str, RiskMetric] = {}
        self.position_limits: Dict[str, PositionLimit] = {}
        self.risk_alerts: List[Dict] = []
        self.risk_history: deque = deque(maxlen=10000)
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskType.MARKET_RISK: {
                'var_95_limit': 0.02,  # 2% VaR limit
                'max_drawdown_limit': 0.15,  # 15% max drawdown
                'volatility_limit': 0.30,  # 30% volatility limit
                'correlation_limit': 0.8,  # 80% correlation limit
            },
            RiskType.CREDIT_RISK: {
                'exposure_limit': 0.1,  # 10% exposure limit
                'counterparty_limit': 0.05,  # 5% per counterparty
            },
            RiskType.LIQUIDITY_RISK: {
                'illiquid_asset_limit': 0.2,  # 20% illiquid assets
                'cash_reserve_limit': 0.05,  # 5% cash reserve
            },
            RiskType.CONCENTRATION_RISK: {
                'single_position_limit': 0.1,  # 10% single position
                'sector_limit': 0.3,  # 30% per sector
            }
        }
        
        # Initialize database
        self.db_path = config.get('risk_db_path', 'risk_management.db')
        self._init_database()
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._risk_monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _init_database(self):
        """Initialize risk management database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    risk_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    description TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Position limits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    max_position_size REAL NOT NULL,
                    max_position_value REAL NOT NULL,
                    max_daily_trades INTEGER NOT NULL,
                    max_daily_volume REAL NOT NULL,
                    current_position REAL DEFAULT 0.0,
                    current_value REAL DEFAULT 0.0,
                    daily_trades INTEGER DEFAULT 0,
                    daily_volume REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Risk alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Compliance violations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Risk management database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing risk management database: {e}")
    
    def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop"""
        while self.monitoring_active:
            try:
                # Calculate risk metrics
                self._calculate_market_risk()
                self._calculate_credit_risk()
                self._calculate_liquidity_risk()
                self._calculate_concentration_risk()
                
                # Check for risk violations
                self._check_risk_violations()
                
                # Store metrics in database
                self._store_risk_metrics()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('risk_monitoring_interval', 60))
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(60)
    
    def _calculate_market_risk(self):
        """Calculate market risk metrics"""
        try:
            # This would typically use portfolio data
            # For now, using placeholder calculations
            
            # Value at Risk (VaR)
            portfolio_value = 100000  # Placeholder
            returns = np.random.normal(0.001, 0.02, 252)  # Placeholder returns
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Store metrics
            self.risk_metrics['var_95'] = RiskMetric(
                risk_type=RiskType.MARKET_RISK,
                value=abs(var_95),
                threshold=self.risk_thresholds[RiskType.MARKET_RISK]['var_95_limit'],
                risk_level=self._assess_risk_level(abs(var_95), self.risk_thresholds[RiskType.MARKET_RISK]['var_95_limit']),
                timestamp=datetime.now(),
                description="95% Value at Risk",
                source="portfolio_analysis"
            )
            
            self.risk_metrics['volatility'] = RiskMetric(
                risk_type=RiskType.MARKET_RISK,
                value=volatility,
                threshold=self.risk_thresholds[RiskType.MARKET_RISK]['volatility_limit'],
                risk_level=self._assess_risk_level(volatility, self.risk_thresholds[RiskType.MARKET_RISK]['volatility_limit']),
                timestamp=datetime.now(),
                description="Portfolio Volatility",
                source="portfolio_analysis"
            )
            
            self.risk_metrics['max_drawdown'] = RiskMetric(
                risk_type=RiskType.MARKET_RISK,
                value=abs(max_drawdown),
                threshold=self.risk_thresholds[RiskType.MARKET_RISK]['max_drawdown_limit'],
                risk_level=self._assess_risk_level(abs(max_drawdown), self.risk_thresholds[RiskType.MARKET_RISK]['max_drawdown_limit']),
                timestamp=datetime.now(),
                description="Maximum Drawdown",
                source="portfolio_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error calculating market risk: {e}")
    
    def _calculate_credit_risk(self):
        """Calculate credit risk metrics"""
        try:
            # This would typically analyze counterparty exposure
            # For now, using placeholder calculations
            
            total_exposure = 50000  # Placeholder
            portfolio_value = 100000  # Placeholder
            exposure_ratio = total_exposure / portfolio_value
            
            self.risk_metrics['credit_exposure'] = RiskMetric(
                risk_type=RiskType.CREDIT_RISK,
                value=exposure_ratio,
                threshold=self.risk_thresholds[RiskType.CREDIT_RISK]['exposure_limit'],
                risk_level=self._assess_risk_level(exposure_ratio, self.risk_thresholds[RiskType.CREDIT_RISK]['exposure_limit']),
                timestamp=datetime.now(),
                description="Credit Exposure Ratio",
                source="counterparty_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error calculating credit risk: {e}")
    
    def _calculate_liquidity_risk(self):
        """Calculate liquidity risk metrics"""
        try:
            # This would typically analyze asset liquidity
            # For now, using placeholder calculations
            
            illiquid_assets = 15000  # Placeholder
            total_assets = 100000  # Placeholder
            illiquidity_ratio = illiquid_assets / total_assets
            
            cash_reserve = 5000  # Placeholder
            cash_ratio = cash_reserve / total_assets
            
            self.risk_metrics['illiquidity_ratio'] = RiskMetric(
                risk_type=RiskType.LIQUIDITY_RISK,
                value=illiquidity_ratio,
                threshold=self.risk_thresholds[RiskType.LIQUIDITY_RISK]['illiquid_asset_limit'],
                risk_level=self._assess_risk_level(illiquidity_ratio, self.risk_thresholds[RiskType.LIQUIDITY_RISK]['illiquid_asset_limit']),
                timestamp=datetime.now(),
                description="Illiquid Assets Ratio",
                source="asset_analysis"
            )
            
            self.risk_metrics['cash_reserve'] = RiskMetric(
                risk_type=RiskType.LIQUIDITY_RISK,
                value=cash_ratio,
                threshold=self.risk_thresholds[RiskType.LIQUIDITY_RISK]['cash_reserve_limit'],
                risk_level=self._assess_risk_level(cash_ratio, self.risk_thresholds[RiskType.LIQUIDITY_RISK]['cash_reserve_limit'], inverse=True),
                timestamp=datetime.now(),
                description="Cash Reserve Ratio",
                source="cash_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
    
    def _calculate_concentration_risk(self):
        """Calculate concentration risk metrics"""
        try:
            # This would typically analyze position concentration
            # For now, using placeholder calculations
            
            largest_position = 25000  # Placeholder
            total_portfolio = 100000  # Placeholder
            concentration_ratio = largest_position / total_portfolio
            
            self.risk_metrics['position_concentration'] = RiskMetric(
                risk_type=RiskType.CONCENTRATION_RISK,
                value=concentration_ratio,
                threshold=self.risk_thresholds[RiskType.CONCENTRATION_RISK]['single_position_limit'],
                risk_level=self._assess_risk_level(concentration_ratio, self.risk_thresholds[RiskType.CONCENTRATION_RISK]['single_position_limit']),
                timestamp=datetime.now(),
                description="Largest Position Concentration",
                source="position_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
    
    def _assess_risk_level(self, value: float, threshold: float, inverse: bool = False) -> RiskLevel:
        """Assess risk level based on value and threshold"""
        try:
            if inverse:
                # For metrics where lower is better (e.g., cash reserve)
                if value >= threshold:
                    return RiskLevel.LOW
                elif value >= threshold * 0.7:
                    return RiskLevel.MEDIUM
                elif value >= threshold * 0.5:
                    return RiskLevel.HIGH
                else:
                    return RiskLevel.CRITICAL
            else:
                # For metrics where higher is worse
                if value <= threshold * 0.5:
                    return RiskLevel.LOW
                elif value <= threshold * 0.7:
                    return RiskLevel.MEDIUM
                elif value <= threshold:
                    return RiskLevel.HIGH
                else:
                    return RiskLevel.CRITICAL
                    
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return RiskLevel.MEDIUM
    
    def _check_risk_violations(self):
        """Check for risk violations and generate alerts"""
        try:
            for metric_id, metric in self.risk_metrics.items():
                if metric.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    alert = {
                        'alert_type': f"{metric.risk_type.value}_violation",
                        'severity': metric.risk_level.value,
                        'message': f"{metric.description}: {metric.value:.4f} (threshold: {metric.threshold:.4f})",
                        'timestamp': datetime.now(),
                        'metric_id': metric_id,
                        'metric_value': metric.value,
                        'threshold': metric.threshold
                    }
                    
                    self.risk_alerts.append(alert)
                    logger.warning(f"Risk alert: {alert['message']}")
                    
        except Exception as e:
            logger.error(f"Error checking risk violations: {e}")
    
    def _store_risk_metrics(self):
        """Store risk metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_id, metric in self.risk_metrics.items():
                cursor.execute('''
                    INSERT INTO risk_metrics 
                    (risk_type, value, threshold, risk_level, timestamp, description, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.risk_type.value,
                    metric.value,
                    metric.threshold,
                    metric.risk_level.value,
                    metric.timestamp,
                    metric.description,
                    metric.source
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
    
    def add_position_limit(self, symbol: str, max_position_size: float, max_position_value: float,
                          max_daily_trades: int, max_daily_volume: float):
        """Add position limit for a symbol"""
        try:
            limit = PositionLimit(
                symbol=symbol,
                max_position_size=max_position_size,
                max_position_value=max_position_value,
                max_daily_trades=max_daily_trades,
                max_daily_volume=max_daily_volume
            )
            
            self.position_limits[symbol] = limit
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO position_limits 
                (symbol, max_position_size, max_position_value, max_daily_trades, max_daily_volume)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, max_position_size, max_position_value, max_daily_trades, max_daily_volume))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added position limit for {symbol}")
            
        except Exception as e:
            logger.error(f"Error adding position limit: {e}")
    
    def check_position_limit(self, symbol: str, position_size: float, position_value: float) -> bool:
        """Check if position violates limits"""
        try:
            if symbol not in self.position_limits:
                return True  # No limit set
            
            limit = self.position_limits[symbol]
            
            # Check position size limit
            if position_size > limit.max_position_size:
                logger.warning(f"Position size limit exceeded for {symbol}: {position_size} > {limit.max_position_size}")
                return False
            
            # Check position value limit
            if position_value > limit.max_position_value:
                logger.warning(f"Position value limit exceeded for {symbol}: {position_value} > {limit.max_position_value}")
                return False
            
            # Check daily trade limit
            if limit.daily_trades >= limit.max_daily_trades:
                logger.warning(f"Daily trade limit exceeded for {symbol}: {limit.daily_trades} >= {limit.max_daily_trades}")
                return False
            
            # Check daily volume limit
            if limit.daily_volume >= limit.max_daily_volume:
                logger.warning(f"Daily volume limit exceeded for {symbol}: {limit.daily_volume} >= {limit.max_daily_volume}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position limit: {e}")
            return False
    
    def update_position(self, symbol: str, position_size: float, position_value: float, trade_volume: float = 0):
        """Update position information"""
        try:
            if symbol in self.position_limits:
                limit = self.position_limits[symbol]
                limit.current_position = position_size
                limit.current_value = position_value
                limit.daily_trades += 1
                limit.daily_volume += trade_volume
                
                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE position_limits 
                    SET current_position = ?, current_value = ?, daily_trades = ?, daily_volume = ?, last_updated = ?
                    WHERE symbol = ?
                ''', (position_size, position_value, limit.daily_trades, limit.daily_volume, datetime.now(), symbol))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get risk summary"""
        try:
            summary = {
                'total_metrics': len(self.risk_metrics),
                'risk_levels': defaultdict(int),
                'risk_types': defaultdict(int),
                'alerts': len(self.risk_alerts),
                'position_limits': len(self.position_limits)
            }
            
            for metric in self.risk_metrics.values():
                summary['risk_levels'][metric.risk_level.value] += 1
                summary['risk_types'][metric.risk_type.value] += 1
            
            return dict(summary)
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}

class ComplianceManager:
    """Compliance management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violations: List[ComplianceViolation] = []
        self.rule_engine = RuleEngine()
        
        # Initialize database
        self.db_path = config.get('compliance_db_path', 'compliance.db')
        self._init_database()
        
        # Load default rules
        self._load_default_rules()
    
    def _init_database(self):
        """Initialize compliance database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Compliance rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    rule_type TEXT NOT NULL,
                    parameters TEXT,
                    enabled BOOLEAN DEFAULT TRUE,
                    priority INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Compliance violations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Compliance database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing compliance database: {e}")
    
    def _load_default_rules(self):
        """Load default compliance rules"""
        try:
            default_rules = [
                {
                    'rule_id': 'POSITION_LIMIT',
                    'name': 'Position Size Limit',
                    'description': 'Enforce maximum position size limits',
                    'rule_type': 'position_limit',
                    'parameters': {'max_position_pct': 0.1},
                    'priority': 1
                },
                {
                    'rule_id': 'DAILY_LOSS_LIMIT',
                    'name': 'Daily Loss Limit',
                    'description': 'Enforce maximum daily loss limits',
                    'rule_type': 'daily_loss_limit',
                    'parameters': {'max_daily_loss_pct': 0.05},
                    'priority': 1
                },
                {
                    'rule_id': 'CONCENTRATION_LIMIT',
                    'name': 'Concentration Limit',
                    'description': 'Enforce sector concentration limits',
                    'rule_type': 'concentration_limit',
                    'parameters': {'max_sector_pct': 0.3},
                    'priority': 2
                },
                {
                    'rule_id': 'WASH_TRADE_PREVENTION',
                    'name': 'Wash Trade Prevention',
                    'description': 'Prevent wash trading activities',
                    'rule_type': 'wash_trade_prevention',
                    'parameters': {'min_hold_time': 3600},
                    'priority': 1
                },
                {
                    'rule_id': 'MARKET_MANIPULATION',
                    'name': 'Market Manipulation Detection',
                    'description': 'Detect potential market manipulation',
                    'rule_type': 'market_manipulation',
                    'parameters': {'max_order_size_pct': 0.01},
                    'priority': 1
                }
            ]
            
            for rule_data in default_rules:
                rule = ComplianceRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    rule_type=rule_data['rule_type'],
                    parameters=rule_data['parameters'],
                    enabled=True,
                    priority=rule_data['priority'],
                    created_at=datetime.now()
                )
                
                self.compliance_rules[rule.rule_id] = rule
            
            logger.info(f"Loaded {len(default_rules)} default compliance rules")
            
        except Exception as e:
            logger.error(f"Error loading default rules: {e}")
    
    def add_compliance_rule(self, rule: ComplianceRule):
        """Add a new compliance rule"""
        try:
            self.compliance_rules[rule.rule_id] = rule
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO compliance_rules 
                (rule_id, name, description, rule_type, parameters, enabled, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.name,
                rule.description,
                rule.rule_type,
                json.dumps(rule.parameters),
                rule.enabled,
                rule.priority
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added compliance rule: {rule.name}")
            
        except Exception as e:
            logger.error(f"Error adding compliance rule: {e}")
    
    def check_compliance(self, trade_data: Dict) -> List[ComplianceViolation]:
        """Check compliance for a trade"""
        try:
            violations = []
            
            for rule in self.compliance_rules.values():
                if not rule.enabled:
                    continue
                
                # Check rule based on type
                if rule.rule_type == 'position_limit':
                    violation = self._check_position_limit_rule(rule, trade_data)
                elif rule.rule_type == 'daily_loss_limit':
                    violation = self._check_daily_loss_limit_rule(rule, trade_data)
                elif rule.rule_type == 'concentration_limit':
                    violation = self._check_concentration_limit_rule(rule, trade_data)
                elif rule.rule_type == 'wash_trade_prevention':
                    violation = self._check_wash_trade_rule(rule, trade_data)
                elif rule.rule_type == 'market_manipulation':
                    violation = self._check_market_manipulation_rule(rule, trade_data)
                else:
                    continue
                
                if violation:
                    violations.append(violation)
                    self.violations.append(violation)
                    
                    # Store violation in database
                    self._store_violation(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return []
    
    def _check_position_limit_rule(self, rule: ComplianceRule, trade_data: Dict) -> Optional[ComplianceViolation]:
        """Check position limit rule"""
        try:
            max_position_pct = rule.parameters.get('max_position_pct', 0.1)
            position_value = trade_data.get('position_value', 0)
            portfolio_value = trade_data.get('portfolio_value', 1)
            
            position_pct = position_value / portfolio_value
            
            if position_pct > max_position_pct:
                return ComplianceViolation(
                    rule_id=rule.rule_id,
                    violation_type='position_limit_exceeded',
                    severity=RiskLevel.HIGH,
                    message=f"Position limit exceeded: {position_pct:.2%} > {max_position_pct:.2%}",
                    timestamp=datetime.now(),
                    details={
                        'position_pct': position_pct,
                        'max_position_pct': max_position_pct,
                        'position_value': position_value,
                        'portfolio_value': portfolio_value
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking position limit rule: {e}")
            return None
    
    def _check_daily_loss_limit_rule(self, rule: ComplianceRule, trade_data: Dict) -> Optional[ComplianceViolation]:
        """Check daily loss limit rule"""
        try:
            max_daily_loss_pct = rule.parameters.get('max_daily_loss_pct', 0.05)
            daily_pnl = trade_data.get('daily_pnl', 0)
            portfolio_value = trade_data.get('portfolio_value', 1)
            
            daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
            
            if daily_loss_pct > max_daily_loss_pct:
                return ComplianceViolation(
                    rule_id=rule.rule_id,
                    violation_type='daily_loss_limit_exceeded',
                    severity=RiskLevel.CRITICAL,
                    message=f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {max_daily_loss_pct:.2%}",
                    timestamp=datetime.now(),
                    details={
                        'daily_loss_pct': daily_loss_pct,
                        'max_daily_loss_pct': max_daily_loss_pct,
                        'daily_pnl': daily_pnl,
                        'portfolio_value': portfolio_value
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit rule: {e}")
            return None
    
    def _check_concentration_limit_rule(self, rule: ComplianceRule, trade_data: Dict) -> Optional[ComplianceViolation]:
        """Check concentration limit rule"""
        try:
            max_sector_pct = rule.parameters.get('max_sector_pct', 0.3)
            sector_exposure = trade_data.get('sector_exposure', {})
            
            for sector, exposure_pct in sector_exposure.items():
                if exposure_pct > max_sector_pct:
                    return ComplianceViolation(
                        rule_id=rule.rule_id,
                        violation_type='concentration_limit_exceeded',
                        severity=RiskLevel.MEDIUM,
                        message=f"Sector concentration limit exceeded: {sector} {exposure_pct:.2%} > {max_sector_pct:.2%}",
                        timestamp=datetime.now(),
                        details={
                            'sector': sector,
                            'exposure_pct': exposure_pct,
                            'max_sector_pct': max_sector_pct,
                            'sector_exposure': sector_exposure
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking concentration limit rule: {e}")
            return None
    
    def _check_wash_trade_rule(self, rule: ComplianceRule, trade_data: Dict) -> Optional[ComplianceViolation]:
        """Check wash trade prevention rule"""
        try:
            min_hold_time = rule.parameters.get('min_hold_time', 3600)  # 1 hour in seconds
            symbol = trade_data.get('symbol', '')
            current_time = datetime.now()
            
            # This would check against recent trades for the same symbol
            # Simplified implementation
            return None
            
        except Exception as e:
            logger.error(f"Error checking wash trade rule: {e}")
            return None
    
    def _check_market_manipulation_rule(self, rule: ComplianceRule, trade_data: Dict) -> Optional[ComplianceViolation]:
        """Check market manipulation rule"""
        try:
            max_order_size_pct = rule.parameters.get('max_order_size_pct', 0.01)
            order_size = trade_data.get('order_size', 0)
            market_volume = trade_data.get('market_volume', 1)
            
            order_size_pct = order_size / market_volume
            
            if order_size_pct > max_order_size_pct:
                return ComplianceViolation(
                    rule_id=rule.rule_id,
                    violation_type='market_manipulation_suspected',
                    severity=RiskLevel.HIGH,
                    message=f"Large order size detected: {order_size_pct:.4%} > {max_order_size_pct:.4%}",
                    timestamp=datetime.now(),
                    details={
                        'order_size_pct': order_size_pct,
                        'max_order_size_pct': max_order_size_pct,
                        'order_size': order_size,
                        'market_volume': market_volume
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking market manipulation rule: {e}")
            return None
    
    def _store_violation(self, violation: ComplianceViolation):
        """Store violation in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO compliance_violations 
                (rule_id, violation_type, severity, message, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                violation.rule_id,
                violation.violation_type,
                violation.severity.value,
                violation.message,
                violation.timestamp,
                json.dumps(violation.details)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing violation: {e}")
    
    def get_compliance_status(self) -> ComplianceStatus:
        """Get overall compliance status"""
        try:
            if not self.violations:
                return ComplianceStatus.COMPLIANT
            
            # Check for critical violations
            critical_violations = [v for v in self.violations if v.severity == RiskLevel.CRITICAL and not v.resolved]
            if critical_violations:
                return ComplianceStatus.SUSPENDED
            
            # Check for high severity violations
            high_violations = [v for v in self.violations if v.severity == RiskLevel.HIGH and not v.resolved]
            if high_violations:
                return ComplianceStatus.VIOLATION
            
            # Check for medium severity violations
            medium_violations = [v for v in self.violations if v.severity == RiskLevel.MEDIUM and not v.resolved]
            if medium_violations:
                return ComplianceStatus.WARNING
            
            return ComplianceStatus.COMPLIANT
            
        except Exception as e:
            logger.error(f"Error getting compliance status: {e}")
            return ComplianceStatus.WARNING
    
    def resolve_violation(self, violation_id: str, resolution_notes: str = ""):
        """Resolve a compliance violation"""
        try:
            for violation in self.violations:
                if violation.rule_id == violation_id and not violation.resolved:
                    violation.resolved = True
                    violation.resolution_time = datetime.now()
                    
                    # Update database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        UPDATE compliance_violations 
                        SET resolved = TRUE, resolution_time = ?
                        WHERE rule_id = ? AND timestamp = ?
                    ''', (datetime.now(), violation.rule_id, violation.timestamp))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"Resolved violation: {violation_id}")
                    break
                    
        except Exception as e:
            logger.error(f"Error resolving violation: {e}")

class RuleEngine:
    """Rule engine for complex compliance rules"""
    
    def __init__(self):
        self.rules = {}
        self.facts = {}
    
    def add_rule(self, rule_name: str, conditions: Dict, actions: List[str]):
        """Add a rule to the engine"""
        try:
            self.rules[rule_name] = {
                'conditions': conditions,
                'actions': actions
            }
            
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
    
    def evaluate_rules(self, facts: Dict) -> List[str]:
        """Evaluate all rules against facts"""
        try:
            triggered_actions = []
            
            for rule_name, rule in self.rules.items():
                if self._evaluate_conditions(rule['conditions'], facts):
                    triggered_actions.extend(rule['actions'])
            
            return triggered_actions
            
        except Exception as e:
            logger.error(f"Error evaluating rules: {e}")
            return []
    
    def _evaluate_conditions(self, conditions: Dict, facts: Dict) -> bool:
        """Evaluate rule conditions"""
        try:
            for condition_key, expected_value in conditions.items():
                if condition_key not in facts:
                    return False
                
                if facts[condition_key] != expected_value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating conditions: {e}")
            return False
