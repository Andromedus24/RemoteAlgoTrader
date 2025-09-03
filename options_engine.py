#!/usr/bin/env python3
"""
Advanced Options & Derivatives Trading Engine for RemoteAlgoTrader
Handles options trading, Greeks calculations, volatility analysis, and derivatives strategies
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
from scipy.stats import norm
from scipy.optimize import minimize
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionStrategy(Enum):
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    BULL_SPREAD = "bull_spread"
    BEAR_SPREAD = "bear_spread"
    BUTTERFLY_SPREAD = "butterfly_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

@dataclass
class OptionContract:
    """Represents an option contract"""
    symbol: str
    underlying: str
    option_type: OptionType
    strike_price: float
    expiration_date: datetime
    current_price: float
    bid_price: float
    ask_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    last_updated: datetime

@dataclass
class OptionPosition:
    """Represents an option position"""
    contract: OptionContract
    quantity: int
    entry_price: float
    entry_time: datetime
    strategy: OptionStrategy
    current_price: float
    unrealized_pnl: float
    delta_exposure: float
    gamma_exposure: float
    theta_exposure: float
    vega_exposure: float

@dataclass
class Greeks:
    """Option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class OptionsPricingEngine:
    """Black-Scholes options pricing engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> Tuple[float, Greeks]:
        """Calculate Black-Scholes option price and Greeks"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == OptionType.CALL:
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                
                # Greeks for call
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
                vega = S * np.sqrt(T) * norm.pdf(d1) / 100
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
                
            else:  # PUT
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
                # Greeks for put
                delta = norm.cdf(d1) - 1
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                vega = S * np.sqrt(T) * norm.pdf(d1) / 100
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            greeks = Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
            
            return price, greeks
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return 0.0, Greeks(0, 0, 0, 0, 0)
    
    def implied_volatility(self, S: float, K: float, T: float, r: float, option_price: float, option_type: OptionType) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            def objective(sigma):
                price, _ = self.black_scholes(S, K, T, r, sigma, option_type)
                return price - option_price
            
            # Initial guess
            sigma = 0.3
            
            # Newton-Raphson iteration
            for _ in range(100):
                price, greeks = self.black_scholes(S, K, T, r, sigma, option_type)
                vega = greeks.vega * 100  # Convert back to per vol point
                
                if abs(vega) < 1e-10:
                    break
                
                diff = (price - option_price) / vega
                sigma -= diff
                
                if abs(diff) < 1e-6:
                    break
                
                sigma = max(0.001, sigma)  # Ensure positive volatility
            
            return sigma
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return 0.3
    
    def binomial_pricing(self, S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType, steps: int = 100) -> float:
        """Binomial option pricing model"""
        try:
            dt = T / steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            
            # Build price tree
            price_tree = np.zeros((steps + 1, steps + 1))
            for i in range(steps + 1):
                for j in range(i + 1):
                    price_tree[i, j] = S * (u ** (i - j)) * (d ** j)
            
            # Build option value tree
            option_tree = np.zeros((steps + 1, steps + 1))
            
            # Terminal values
            for j in range(steps + 1):
                if option_type == OptionType.CALL:
                    option_tree[steps, j] = max(price_tree[steps, j] - K, 0)
                else:
                    option_tree[steps, j] = max(K - price_tree[steps, j], 0)
            
            # Backward induction
            for i in range(steps - 1, -1, -1):
                for j in range(i + 1):
                    option_tree[i, j] = np.exp(-r * dt) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1])
            
            return option_tree[0, 0]
            
        except Exception as e:
            logger.error(f"Error in binomial pricing: {e}")
            return 0.0

class VolatilityAnalyzer:
    """Volatility analysis and forecasting"""
    
    def __init__(self):
        self.historical_vol = {}
        self.volatility_surface = {}
    
    def calculate_historical_volatility(self, prices: List[float], window: int = 30) -> float:
        """Calculate historical volatility"""
        try:
            if len(prices) < window + 1:
                return 0.0
            
            returns = np.diff(np.log(prices))
            rolling_vol = np.std(returns[-window:]) * np.sqrt(252)  # Annualized
            
            return rolling_vol
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return 0.0
    
    def calculate_implied_volatility_surface(self, options_data: List[OptionContract]) -> Dict:
        """Calculate implied volatility surface"""
        try:
            surface = defaultdict(dict)
            
            for option in options_data:
                T = (option.expiration_date - datetime.now()).days / 365
                surface[option.strike_price][T] = option.implied_volatility
            
            return dict(surface)
            
        except Exception as e:
            logger.error(f"Error calculating IV surface: {e}")
            return {}
    
    def forecast_volatility(self, historical_vol: List[float], method: str = 'garch') -> float:
        """Forecast future volatility"""
        try:
            if method == 'simple':
                # Simple moving average
                return np.mean(historical_vol[-30:])
            
            elif method == 'ewma':
                # Exponentially weighted moving average
                alpha = 0.94
                weights = np.array([alpha * (1 - alpha) ** i for i in range(len(historical_vol))])
                weights = weights / np.sum(weights)
                return np.sum(weights * historical_vol)
            
            elif method == 'garch':
                # Simplified GARCH(1,1) model
                omega = 0.000001
                alpha = 0.1
                beta = 0.8
                
                vol_forecast = omega / (1 - alpha - beta)
                return vol_forecast
            
            else:
                return np.mean(historical_vol[-30:])
                
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return 0.3

class OptionsStrategyEngine:
    """Options strategy engine for complex strategies"""
    
    def __init__(self, pricing_engine: OptionsPricingEngine):
        self.pricing_engine = pricing_engine
        self.positions: List[OptionPosition] = []
    
    def create_long_call(self, underlying: str, strike: float, expiration: datetime, quantity: int = 1) -> OptionPosition:
        """Create long call position"""
        try:
            # This would typically fetch real option data
            # For now, using placeholder data
            option_contract = OptionContract(
                symbol=f"{underlying}{expiration.strftime('%y%m%d')}C{int(strike*1000)}",
                underlying=underlying,
                option_type=OptionType.CALL,
                strike_price=strike,
                expiration_date=expiration,
                current_price=5.50,
                bid_price=5.45,
                ask_price=5.55,
                volume=100,
                open_interest=500,
                implied_volatility=0.25,
                delta=0.65,
                gamma=0.02,
                theta=-0.05,
                vega=0.15,
                rho=0.03,
                last_updated=datetime.now()
            )
            
            position = OptionPosition(
                contract=option_contract,
                quantity=quantity,
                entry_price=5.50,
                entry_time=datetime.now(),
                strategy=OptionStrategy.LONG_CALL,
                current_price=5.50,
                unrealized_pnl=0.0,
                delta_exposure=quantity * option_contract.delta,
                gamma_exposure=quantity * option_contract.gamma,
                theta_exposure=quantity * option_contract.theta,
                vega_exposure=quantity * option_contract.vega
            )
            
            self.positions.append(position)
            return position
            
        except Exception as e:
            logger.error(f"Error creating long call position: {e}")
            return None
    
    def create_long_put(self, underlying: str, strike: float, expiration: datetime, quantity: int = 1) -> OptionPosition:
        """Create long put position"""
        try:
            option_contract = OptionContract(
                symbol=f"{underlying}{expiration.strftime('%y%m%d')}P{int(strike*1000)}",
                underlying=underlying,
                option_type=OptionType.PUT,
                strike_price=strike,
                expiration_date=expiration,
                current_price=3.25,
                bid_price=3.20,
                ask_price=3.30,
                volume=75,
                open_interest=300,
                implied_volatility=0.28,
                delta=-0.35,
                gamma=0.03,
                theta=-0.04,
                vega=0.12,
                rho=-0.02,
                last_updated=datetime.now()
            )
            
            position = OptionPosition(
                contract=option_contract,
                quantity=quantity,
                entry_price=3.25,
                entry_time=datetime.now(),
                strategy=OptionStrategy.LONG_PUT,
                current_price=3.25,
                unrealized_pnl=0.0,
                delta_exposure=quantity * option_contract.delta,
                gamma_exposure=quantity * option_contract.gamma,
                theta_exposure=quantity * option_contract.theta,
                vega_exposure=quantity * option_contract.vega
            )
            
            self.positions.append(position)
            return position
            
        except Exception as e:
            logger.error(f"Error creating long put position: {e}")
            return None
    
    def create_bull_spread(self, underlying: str, lower_strike: float, upper_strike: float, 
                          expiration: datetime, quantity: int = 1) -> List[OptionPosition]:
        """Create bull spread (long call + short call)"""
        try:
            # Long call at lower strike
            long_call = self.create_long_call(underlying, lower_strike, expiration, quantity)
            
            # Short call at upper strike
            short_call_contract = OptionContract(
                symbol=f"{underlying}{expiration.strftime('%y%m%d')}C{int(upper_strike*1000)}",
                underlying=underlying,
                option_type=OptionType.CALL,
                strike_price=upper_strike,
                expiration_date=expiration,
                current_price=2.75,
                bid_price=2.70,
                ask_price=2.80,
                volume=80,
                open_interest=400,
                implied_volatility=0.22,
                delta=0.45,
                gamma=0.025,
                theta=-0.04,
                vega=0.13,
                rho=0.025,
                last_updated=datetime.now()
            )
            
            short_call = OptionPosition(
                contract=short_call_contract,
                quantity=-quantity,  # Negative for short position
                entry_price=2.75,
                entry_time=datetime.now(),
                strategy=OptionStrategy.SHORT_CALL,
                current_price=2.75,
                unrealized_pnl=0.0,
                delta_exposure=-quantity * short_call_contract.delta,
                gamma_exposure=-quantity * short_call_contract.gamma,
                theta_exposure=-quantity * short_call_contract.theta,
                vega_exposure=-quantity * short_call_contract.vega
            )
            
            self.positions.append(short_call)
            return [long_call, short_call]
            
        except Exception as e:
            logger.error(f"Error creating bull spread: {e}")
            return []
    
    def create_iron_condor(self, underlying: str, put_strike_low: float, put_strike_high: float,
                          call_strike_low: float, call_strike_high: float, expiration: datetime,
                          quantity: int = 1) -> List[OptionPosition]:
        """Create iron condor strategy"""
        try:
            positions = []
            
            # Short put spread
            short_put_low = self._create_short_put(underlying, put_strike_low, expiration, quantity)
            long_put_high = self._create_long_put(underlying, put_strike_high, expiration, quantity)
            
            # Short call spread
            short_call_low = self._create_short_call(underlying, call_strike_low, expiration, quantity)
            long_call_high = self._create_long_call(underlying, call_strike_high, expiration, quantity)
            
            positions.extend([short_put_low, long_put_high, short_call_low, long_call_high])
            self.positions.extend(positions)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error creating iron condor: {e}")
            return []
    
    def _create_short_put(self, underlying: str, strike: float, expiration: datetime, quantity: int) -> OptionPosition:
        """Create short put position"""
        try:
            option_contract = OptionContract(
                symbol=f"{underlying}{expiration.strftime('%y%m%d')}P{int(strike*1000)}",
                underlying=underlying,
                option_type=OptionType.PUT,
                strike_price=strike,
                expiration_date=expiration,
                current_price=2.50,
                bid_price=2.45,
                ask_price=2.55,
                volume=60,
                open_interest=250,
                implied_volatility=0.26,
                delta=-0.30,
                gamma=0.025,
                theta=-0.03,
                vega=0.10,
                rho=-0.015,
                last_updated=datetime.now()
            )
            
            position = OptionPosition(
                contract=option_contract,
                quantity=-quantity,  # Negative for short
                entry_price=2.50,
                entry_time=datetime.now(),
                strategy=OptionStrategy.SHORT_PUT,
                current_price=2.50,
                unrealized_pnl=0.0,
                delta_exposure=-quantity * option_contract.delta,
                gamma_exposure=-quantity * option_contract.gamma,
                theta_exposure=-quantity * option_contract.theta,
                vega_exposure=-quantity * option_contract.vega
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Error creating short put position: {e}")
            return None
    
    def _create_short_call(self, underlying: str, strike: float, expiration: datetime, quantity: int) -> OptionPosition:
        """Create short call position"""
        try:
            option_contract = OptionContract(
                symbol=f"{underlying}{expiration.strftime('%y%m%d')}C{int(strike*1000)}",
                underlying=underlying,
                option_type=OptionType.CALL,
                strike_price=strike,
                expiration_date=expiration,
                current_price=1.75,
                bid_price=1.70,
                ask_price=1.80,
                volume=45,
                open_interest=200,
                implied_volatility=0.20,
                delta=0.25,
                gamma=0.02,
                theta=-0.025,
                vega=0.08,
                rho=0.015,
                last_updated=datetime.now()
            )
            
            position = OptionPosition(
                contract=option_contract,
                quantity=-quantity,  # Negative for short
                entry_price=1.75,
                entry_time=datetime.now(),
                strategy=OptionStrategy.SHORT_CALL,
                current_price=1.75,
                unrealized_pnl=0.0,
                delta_exposure=-quantity * option_contract.delta,
                gamma_exposure=-quantity * option_contract.gamma,
                theta_exposure=-quantity * option_contract.theta,
                vega_exposure=-quantity * option_contract.vega
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Error creating short call position: {e}")
            return None
    
    def calculate_portfolio_greeks(self) -> Greeks:
        """Calculate total portfolio Greeks"""
        try:
            total_delta = sum(pos.delta_exposure for pos in self.positions)
            total_gamma = sum(pos.gamma_exposure for pos in self.positions)
            total_theta = sum(pos.theta_exposure for pos in self.positions)
            total_vega = sum(pos.vega_exposure for pos in self.positions)
            
            # Rho calculation would require more complex logic
            total_rho = 0.0
            
            return Greeks(
                delta=total_delta,
                gamma=total_gamma,
                theta=total_theta,
                vega=total_vega,
                rho=total_rho
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            return Greeks(0, 0, 0, 0, 0)
    
    def calculate_portfolio_pnl(self, current_underlying_price: float) -> float:
        """Calculate total portfolio P&L"""
        try:
            total_pnl = 0.0
            
            for position in self.positions:
                # Simplified P&L calculation
                price_change = position.current_price - position.entry_price
                position_pnl = position.quantity * price_change * 100  # Options are typically 100 shares
                total_pnl += position_pnl
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error calculating portfolio P&L: {e}")
            return 0.0
    
    def risk_analysis(self) -> Dict:
        """Perform risk analysis on options portfolio"""
        try:
            greeks = self.calculate_portfolio_greeks()
            total_pnl = self.calculate_portfolio_pnl(0)  # Placeholder price
            
            # Calculate risk metrics
            max_loss = self._calculate_max_loss()
            max_profit = self._calculate_max_profit()
            breakeven_points = self._calculate_breakeven_points()
            
            return {
                'total_delta': greeks.delta,
                'total_gamma': greeks.gamma,
                'total_theta': greeks.theta,
                'total_vega': greeks.vega,
                'total_pnl': total_pnl,
                'max_loss': max_loss,
                'max_profit': max_profit,
                'breakeven_points': breakeven_points,
                'position_count': len(self.positions),
                'risk_level': self._assess_risk_level(greeks)
            }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {}
    
    def _calculate_max_loss(self) -> float:
        """Calculate maximum possible loss"""
        try:
            max_loss = 0.0
            
            for position in self.positions:
                if position.strategy in [OptionStrategy.LONG_CALL, OptionStrategy.LONG_PUT]:
                    # Long options: max loss is premium paid
                    max_loss += abs(position.quantity) * position.entry_price * 100
                elif position.strategy in [OptionStrategy.SHORT_CALL, OptionStrategy.SHORT_PUT]:
                    # Short options: max loss is unlimited (for calls) or strike price (for puts)
                    if position.contract.option_type == OptionType.CALL:
                        max_loss += float('inf')  # Unlimited loss
                    else:
                        max_loss += abs(position.quantity) * position.contract.strike_price * 100
            
            return max_loss
            
        except Exception as e:
            logger.error(f"Error calculating max loss: {e}")
            return 0.0
    
    def _calculate_max_profit(self) -> float:
        """Calculate maximum possible profit"""
        try:
            max_profit = 0.0
            
            for position in self.positions:
                if position.strategy in [OptionStrategy.LONG_CALL, OptionStrategy.LONG_PUT]:
                    # Long options: unlimited profit potential
                    max_profit += float('inf')
                elif position.strategy in [OptionStrategy.SHORT_CALL, OptionStrategy.SHORT_PUT]:
                    # Short options: max profit is premium received
                    max_profit += abs(position.quantity) * position.entry_price * 100
            
            return max_profit
            
        except Exception as e:
            logger.error(f"Error calculating max profit: {e}")
            return 0.0
    
    def _calculate_breakeven_points(self) -> List[float]:
        """Calculate breakeven points for strategies"""
        try:
            breakeven_points = []
            
            # This is a simplified calculation
            # In practice, you'd need to analyze each strategy separately
            
            for position in self.positions:
                if position.strategy == OptionStrategy.LONG_CALL:
                    breakeven = position.contract.strike_price + position.entry_price
                    breakeven_points.append(breakeven)
                elif position.strategy == OptionStrategy.LONG_PUT:
                    breakeven = position.contract.strike_price - position.entry_price
                    breakeven_points.append(breakeven)
            
            return list(set(breakeven_points))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error calculating breakeven points: {e}")
            return []
    
    def _assess_risk_level(self, greeks: Greeks) -> str:
        """Assess overall risk level based on Greeks"""
        try:
            risk_score = 0
            
            # Delta risk
            if abs(greeks.delta) > 100:
                risk_score += 3
            elif abs(greeks.delta) > 50:
                risk_score += 2
            elif abs(greeks.delta) > 20:
                risk_score += 1
            
            # Gamma risk
            if abs(greeks.gamma) > 10:
                risk_score += 3
            elif abs(greeks.gamma) > 5:
                risk_score += 2
            elif abs(greeks.gamma) > 2:
                risk_score += 1
            
            # Theta risk (time decay)
            if greeks.theta < -50:
                risk_score += 2
            elif greeks.theta < -20:
                risk_score += 1
            
            # Vega risk (volatility)
            if abs(greeks.vega) > 100:
                risk_score += 2
            elif abs(greeks.vega) > 50:
                risk_score += 1
            
            if risk_score >= 8:
                return "HIGH"
            elif risk_score >= 5:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return "UNKNOWN"
    
    def get_positions_summary(self) -> Dict:
        """Get summary of all positions"""
        try:
            summary = {
                'total_positions': len(self.positions),
                'strategies': defaultdict(int),
                'underlyings': defaultdict(int),
                'total_premium_paid': 0.0,
                'total_premium_received': 0.0
            }
            
            for position in self.positions:
                summary['strategies'][position.strategy.value] += 1
                summary['underlyings'][position.contract.underlying] += 1
                
                if position.quantity > 0:  # Long position
                    summary['total_premium_paid'] += position.quantity * position.entry_price * 100
                else:  # Short position
                    summary['total_premium_received'] += abs(position.quantity) * position.entry_price * 100
            
            summary['strategies'] = dict(summary['strategies'])
            summary['underlyings'] = dict(summary['underlyings'])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting positions summary: {e}")
            return {}
