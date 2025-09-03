#!/usr/bin/env python3
"""
Advanced Backtesting & Strategy Optimization Engine for RemoteAlgoTrader
Provides comprehensive backtesting, walk-forward analysis, Monte Carlo simulation, and strategy optimization
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    SIMPLE = "simple"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"

class OptimizationMethod(Enum):
    GRID_SEARCH = "grid_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    PARTICLE_SWARM = "particle_swarm"

@dataclass
class BacktestResult:
    """Results from backtesting"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_history: List[Dict]
    parameters: Dict[str, Any]

@dataclass
class OptimizationResult:
    """Results from strategy optimization"""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict]
    parameter_importance: Dict[str, float]
    convergence_curve: List[float]
    method: OptimizationMethod

class BacktestEngine:
    """Advanced backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage = 0.0005  # 0.05% slippage
        self.results = []
        
    def run_backtest(self, data: pd.DataFrame, strategy_func: Callable, 
                    parameters: Dict[str, Any], mode: BacktestMode = BacktestMode.SIMPLE) -> BacktestResult:
        """Run backtest with specified mode"""
        try:
            if mode == BacktestMode.SIMPLE:
                return self._simple_backtest(data, strategy_func, parameters)
            elif mode == BacktestMode.WALK_FORWARD:
                return self._walk_forward_backtest(data, strategy_func, parameters)
            elif mode == BacktestMode.MONTE_CARLO:
                return self._monte_carlo_backtest(data, strategy_func, parameters)
            elif mode == BacktestMode.STRESS_TEST:
                return self._stress_test_backtest(data, strategy_func, parameters)
            else:
                raise ValueError(f"Unknown backtest mode: {mode}")
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def _simple_backtest(self, data: pd.DataFrame, strategy_func: Callable, 
                        parameters: Dict[str, Any]) -> BacktestResult:
        """Simple backtest implementation"""
        try:
            # Initialize variables
            capital = self.initial_capital
            position = 0
            trades = []
            equity_curve = []
            current_price = 0
            
            # Run strategy on data
            for i in range(len(data)):
                current_data = data.iloc[:i+1]
                if len(current_data) < 50:  # Need minimum data for indicators
                    continue
                
                current_price = current_data['Close'].iloc[-1]
                
                # Get strategy signal
                signal = strategy_func(current_data, parameters)
                
                # Execute trades
                if signal == 1 and position <= 0:  # Buy signal
                    if position < 0:  # Close short position
                        trade_pnl = (entry_price - current_price) * abs(position)
                        capital += trade_pnl
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_data.index[-1],
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': trade_pnl,
                            'type': 'short_close'
                        })
                    
                    # Open long position
                    position = capital * 0.95 / current_price  # Use 95% of capital
                    entry_price = current_price
                    entry_time = current_data.index[-1]
                    
                elif signal == -1 and position >= 0:  # Sell signal
                    if position > 0:  # Close long position
                        trade_pnl = (current_price - entry_price) * position
                        capital += trade_pnl
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_data.index[-1],
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': trade_pnl,
                            'type': 'long_close'
                        })
                    
                    # Open short position
                    position = -capital * 0.95 / current_price
                    entry_price = current_price
                    entry_time = current_data.index[-1]
                
                # Calculate current equity
                if position > 0:
                    current_equity = capital + (current_price - entry_price) * position
                elif position < 0:
                    current_equity = capital + (entry_price - current_price) * abs(position)
                else:
                    current_equity = capital
                
                equity_curve.append(current_equity)
            
            # Close final position
            if position != 0:
                if position > 0:
                    final_pnl = (current_price - entry_price) * position
                else:
                    final_pnl = (entry_price - current_price) * abs(position)
                capital += final_pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': data.index[-1],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'pnl': final_pnl,
                    'type': 'final_close'
                })
            
            # Calculate performance metrics
            equity_series = pd.Series(equity_curve, index=data.index[50:])
            return self._calculate_performance_metrics(equity_series, trades, parameters)
            
        except Exception as e:
            logger.error(f"Error in simple backtest: {e}")
            return None
    
    def _walk_forward_backtest(self, data: pd.DataFrame, strategy_func: Callable,
                              parameters: Dict[str, Any]) -> BacktestResult:
        """Walk-forward analysis backtest"""
        try:
            # Split data into training and testing periods
            total_days = len(data)
            training_days = int(total_days * 0.7)
            testing_days = int(total_days * 0.2)
            
            all_trades = []
            all_equity = []
            
            for i in range(training_days, total_days - testing_days, testing_days):
                # Training period
                train_data = data.iloc[i-training_days:i]
                
                # Optimize parameters on training data
                optimized_params = self._optimize_parameters(train_data, strategy_func, parameters)
                
                # Test period
                test_data = data.iloc[i:i+testing_days]
                
                # Run backtest on test data with optimized parameters
                test_result = self._simple_backtest(test_data, strategy_func, optimized_params)
                
                if test_result:
                    all_trades.extend(test_result.trade_history)
                    all_equity.extend(test_result.equity_curve.values)
            
            # Combine results
            combined_equity = pd.Series(all_equity)
            return self._calculate_performance_metrics(combined_equity, all_trades, parameters)
            
        except Exception as e:
            logger.error(f"Error in walk-forward backtest: {e}")
            return None
    
    def _monte_carlo_backtest(self, data: pd.DataFrame, strategy_func: Callable,
                             parameters: Dict[str, Any], num_simulations: int = 1000) -> BacktestResult:
        """Monte Carlo simulation backtest"""
        try:
            # Get base backtest result
            base_result = self._simple_backtest(data, strategy_func, parameters)
            if not base_result:
                return None
            
            # Generate Monte Carlo simulations
            returns = base_result.equity_curve.pct_change().dropna()
            
            simulation_results = []
            for _ in range(num_simulations):
                # Bootstrap sample returns
                bootstrapped_returns = np.random.choice(returns.values, size=len(returns), replace=True)
                bootstrapped_equity = (1 + pd.Series(bootstrapped_returns)).cumprod() * self.initial_capital
                
                # Calculate metrics for this simulation
                sim_result = self._calculate_performance_metrics(bootstrapped_equity, [], parameters)
                simulation_results.append(sim_result)
            
            # Aggregate results
            total_returns = [r.total_return for r in simulation_results]
            sharpe_ratios = [r.sharpe_ratio for r in simulation_results]
            max_drawdowns = [r.max_drawdown for r in simulation_results]
            
            # Calculate confidence intervals
            ci_95_total_return = np.percentile(total_returns, [2.5, 97.5])
            ci_95_sharpe = np.percentile(sharpe_ratios, [2.5, 97.5])
            ci_95_drawdown = np.percentile(max_drawdowns, [2.5, 97.5])
            
            # Return base result with Monte Carlo statistics
            base_result.monte_carlo_stats = {
                'mean_total_return': np.mean(total_returns),
                'std_total_return': np.std(total_returns),
                'ci_95_total_return': ci_95_total_return,
                'mean_sharpe': np.mean(sharpe_ratios),
                'ci_95_sharpe': ci_95_sharpe,
                'mean_drawdown': np.mean(max_drawdowns),
                'ci_95_drawdown': ci_95_drawdown,
                'num_simulations': num_simulations
            }
            
            return base_result
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo backtest: {e}")
            return None
    
    def _stress_test_backtest(self, data: pd.DataFrame, strategy_func: Callable,
                             parameters: Dict[str, Any]) -> BacktestResult:
        """Stress testing backtest"""
        try:
            # Run base backtest
            base_result = self._simple_backtest(data, strategy_func, parameters)
            if not base_result:
                return None
            
            # Stress test scenarios
            stress_scenarios = {
                'high_volatility': self._apply_volatility_shock(data),
                'market_crash': self._apply_market_crash(data),
                'high_commission': self._apply_commission_shock(),
                'high_slippage': self._apply_slippage_shock()
            }
            
            stress_results = {}
            for scenario_name, scenario_data in stress_scenarios.items():
                if isinstance(scenario_data, pd.DataFrame):
                    stress_results[scenario_name] = self._simple_backtest(scenario_data, strategy_func, parameters)
                else:
                    # For commission/slippage shocks, modify parameters
                    modified_params = parameters.copy()
                    if scenario_name == 'high_commission':
                        modified_params['commission_rate'] = 0.005  # 0.5%
                    elif scenario_name == 'high_slippage':
                        modified_params['slippage'] = 0.002  # 0.2%
                    
                    stress_results[scenario_name] = self._simple_backtest(data, strategy_func, modified_params)
            
            # Add stress test results to base result
            base_result.stress_test_results = stress_results
            return base_result
            
        except Exception as e:
            logger.error(f"Error in stress test backtest: {e}")
            return None
    
    def _apply_volatility_shock(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility shock to data"""
        try:
            shocked_data = data.copy()
            returns = data['Close'].pct_change()
            
            # Increase volatility by 50%
            shocked_returns = returns * 1.5
            shocked_prices = (1 + shocked_returns).cumprod() * data['Close'].iloc[0]
            
            shocked_data['Close'] = shocked_prices
            shocked_data['High'] = shocked_data['Close'] * 1.02
            shocked_data['Low'] = shocked_data['Close'] * 0.98
            shocked_data['Open'] = shocked_data['Close'].shift(1)
            
            return shocked_data
            
        except Exception as e:
            logger.error(f"Error applying volatility shock: {e}")
            return data
    
    def _apply_market_crash(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply market crash scenario"""
        try:
            crashed_data = data.copy()
            
            # Simulate 20% market crash in the middle of the period
            crash_start = len(data) // 2
            crash_end = crash_start + 30
            
            crash_factor = 0.8  # 20% drop
            crashed_data.loc[crash_start:crash_end, 'Close'] *= crash_factor
            crashed_data.loc[crash_start:crash_end, 'High'] *= crash_factor
            crashed_data.loc[crash_start:crash_end, 'Low'] *= crash_factor
            crashed_data.loc[crash_start:crash_end, 'Open'] *= crash_factor
            
            return crashed_data
            
        except Exception as e:
            logger.error(f"Error applying market crash: {e}")
            return data
    
    def _apply_commission_shock(self) -> Dict:
        """Apply commission shock"""
        return {'commission_rate': 0.005}
    
    def _apply_slippage_shock(self) -> Dict:
        """Apply slippage shock"""
        return {'slippage': 0.002}
    
    def _calculate_performance_metrics(self, equity_curve: pd.Series, trades: List[Dict],
                                     parameters: Dict[str, Any]) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic metrics
            total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
            annualized_return = self._calculate_annualized_return(equity_curve)
            
            # Risk metrics
            returns = equity_curve.pct_change().dropna()
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade metrics
            if trades:
                winning_trades = [t for t in trades if t['pnl'] > 0]
                losing_trades = [t for t in trades if t['pnl'] < 0]
                
                win_rate = len(winning_trades) / len(trades)
                profit_factor = (sum(t['pnl'] for t in winning_trades) / 
                               abs(sum(t['pnl'] for t in losing_trades))) if losing_trades else float('inf')
                
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
                largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
                
                # Calculate average trade duration
                durations = []
                for trade in trades:
                    duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / (24 * 3600)
                    durations.append(duration)
                avg_trade_duration = np.mean(durations) if durations else 0
            else:
                win_rate = 0
                profit_factor = 0
                avg_win = 0
                avg_loss = 0
                largest_win = 0
                largest_loss = 0
                avg_trade_duration = 0
            
            # Drawdown curve
            drawdown_curve = self._calculate_drawdown_curve(equity_curve)
            
            return BacktestResult(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(trades),
                winning_trades=len(winning_trades) if trades else 0,
                losing_trades=len(losing_trades) if trades else 0,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_trade_duration=avg_trade_duration,
                equity_curve=equity_curve,
                drawdown_curve=drawdown_curve,
                trade_history=trades,
                parameters=parameters
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return None
    
    def _calculate_annualized_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return"""
        try:
            total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            
            if total_days > 0:
                annualized_return = (1 + total_return) ** (365 / total_days) - 1
                return annualized_return
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating annualized return: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252
            return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_deviation = np.std(downside_returns)
            return np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            return drawdown.min()
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_drawdown_curve(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown curve"""
        try:
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            return drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown curve: {e}")
            return pd.Series()

class StrategyOptimizer:
    """Strategy optimization engine"""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
        self.optimization_history = []
        
    def optimize_strategy(self, data: pd.DataFrame, strategy_func: Callable,
                        parameter_ranges: Dict[str, Tuple], 
                        method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM,
                        objective: str = 'sharpe_ratio') -> OptimizationResult:
        """Optimize strategy parameters"""
        try:
            if method == OptimizationMethod.GENETIC_ALGORITHM:
                return self._genetic_optimization(data, strategy_func, parameter_ranges, objective)
            elif method == OptimizationMethod.GRID_SEARCH:
                return self._grid_search_optimization(data, strategy_func, parameter_ranges, objective)
            elif method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                return self._bayesian_optimization(data, strategy_func, parameter_ranges, objective)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            return None
    
    def _genetic_optimization(self, data: pd.DataFrame, strategy_func: Callable,
                            parameter_ranges: Dict[str, Tuple], objective: str) -> OptimizationResult:
        """Genetic algorithm optimization"""
        try:
            def objective_function(params):
                # Convert parameter array to dictionary
                param_dict = {}
                param_names = list(parameter_ranges.keys())
                for i, name in enumerate(param_names):
                    param_dict[name] = params[i]
                
                # Run backtest
                result = self.backtest_engine._simple_backtest(data, strategy_func, param_dict)
                if not result:
                    return -999  # Penalty for failed backtest
                
                # Return objective value (negative for minimization)
                if objective == 'sharpe_ratio':
                    return -result.sharpe_ratio
                elif objective == 'total_return':
                    return -result.total_return
                elif objective == 'calmar_ratio':
                    return -result.calmar_ratio
                else:
                    return -result.sharpe_ratio
            
            # Prepare bounds for differential evolution
            bounds = list(parameter_ranges.values())
            
            # Run optimization
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42
            )
            
            # Convert best parameters back to dictionary
            best_params = {}
            param_names = list(parameter_ranges.keys())
            for i, name in enumerate(param_names):
                best_params[name] = result.x[i]
            
            return OptimizationResult(
                best_parameters=best_params,
                best_score=-result.fun,  # Convert back to positive
                optimization_history=self.optimization_history,
                parameter_importance=self._calculate_parameter_importance(best_params),
                convergence_curve=[],  # Would need to track during optimization
                method=OptimizationMethod.GENETIC_ALGORITHM
            )
            
        except Exception as e:
            logger.error(f"Error in genetic optimization: {e}")
            return None
    
    def _grid_search_optimization(self, data: pd.DataFrame, strategy_func: Callable,
                                parameter_ranges: Dict[str, Tuple], objective: str) -> OptimizationResult:
        """Grid search optimization"""
        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            best_score = float('-inf')
            best_params = None
            optimization_history = []
            
            for params in param_combinations:
                # Run backtest
                result = self.backtest_engine._simple_backtest(data, strategy_func, params)
                if not result:
                    continue
                
                # Calculate objective score
                if objective == 'sharpe_ratio':
                    score = result.sharpe_ratio
                elif objective == 'total_return':
                    score = result.total_return
                elif objective == 'calmar_ratio':
                    score = result.calmar_ratio
                else:
                    score = result.sharpe_ratio
                
                optimization_history.append({
                    'parameters': params,
                    'score': score,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_return': result.total_return,
                    'max_drawdown': result.max_drawdown
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            return OptimizationResult(
                best_parameters=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                parameter_importance=self._calculate_parameter_importance(best_params),
                convergence_curve=[h['score'] for h in optimization_history],
                method=OptimizationMethod.GRID_SEARCH
            )
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {e}")
            return None
    
    def _bayesian_optimization(self, data: pd.DataFrame, strategy_func: Callable,
                             parameter_ranges: Dict[str, Tuple], objective: str) -> OptimizationResult:
        """Bayesian optimization"""
        try:
            # This would implement Bayesian optimization using scikit-optimize
            # For now, using a simplified version
            return self._genetic_optimization(data, strategy_func, parameter_ranges, objective)
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return None
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, Tuple]) -> List[Dict]:
        """Generate parameter combinations for grid search"""
        try:
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            # Generate all combinations
            combinations = []
            for values in np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_values)):
                param_dict = {}
                for i, name in enumerate(param_names):
                    param_dict[name] = values[i]
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {e}")
            return []
    
    def _calculate_parameter_importance(self, best_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate parameter importance"""
        try:
            # This would typically use sensitivity analysis
            # For now, returning equal importance
            importance = {}
            for param in best_params:
                importance[param] = 1.0 / len(best_params)
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return {}

class BacktestAnalyzer:
    """Advanced backtest analysis and reporting"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_backtest_results(self, results: List[BacktestResult]) -> Dict:
        """Analyze multiple backtest results"""
        try:
            analysis = {
                'summary_stats': self._calculate_summary_stats(results),
                'risk_metrics': self._calculate_risk_metrics(results),
                'trade_analysis': self._analyze_trades(results),
                'equity_analysis': self._analyze_equity_curves(results),
                'parameter_sensitivity': self._analyze_parameter_sensitivity(results)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing backtest results: {e}")
            return {}
    
    def _calculate_summary_stats(self, results: List[BacktestResult]) -> Dict:
        """Calculate summary statistics"""
        try:
            total_returns = [r.total_return for r in results]
            sharpe_ratios = [r.sharpe_ratio for r in results]
            max_drawdowns = [r.max_drawdown for r in results]
            win_rates = [r.win_rate for r in results]
            
            return {
                'mean_total_return': np.mean(total_returns),
                'std_total_return': np.std(total_returns),
                'mean_sharpe': np.mean(sharpe_ratios),
                'std_sharpe': np.std(sharpe_ratios),
                'mean_drawdown': np.mean(max_drawdowns),
                'std_drawdown': np.std(max_drawdowns),
                'mean_win_rate': np.mean(win_rates),
                'std_win_rate': np.std(win_rates)
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}
    
    def _calculate_risk_metrics(self, results: List[BacktestResult]) -> Dict:
        """Calculate risk metrics"""
        try:
            # Value at Risk
            total_returns = [r.total_return for r in results]
            var_95 = np.percentile(total_returns, 5)
            var_99 = np.percentile(total_returns, 1)
            
            # Expected Shortfall
            es_95 = np.mean([r for r in total_returns if r <= var_95])
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': es_95,
                'skewness': stats.skew(total_returns),
                'kurtosis': stats.kurtosis(total_returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _analyze_trades(self, results: List[BacktestResult]) -> Dict:
        """Analyze trade patterns"""
        try:
            all_trades = []
            for result in results:
                all_trades.extend(result.trade_history)
            
            if not all_trades:
                return {}
            
            trade_durations = []
            trade_returns = []
            
            for trade in all_trades:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / (24 * 3600)
                trade_durations.append(duration)
                trade_returns.append(trade['pnl'])
            
            return {
                'total_trades': len(all_trades),
                'avg_trade_duration': np.mean(trade_durations),
                'std_trade_duration': np.std(trade_durations),
                'avg_trade_return': np.mean(trade_returns),
                'std_trade_return': np.std(trade_returns),
                'best_trade': max(trade_returns),
                'worst_trade': min(trade_returns)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def _analyze_equity_curves(self, results: List[BacktestResult]) -> Dict:
        """Analyze equity curves"""
        try:
            # Combine all equity curves
            combined_equity = pd.concat([r.equity_curve for r in results], axis=1)
            
            return {
                'mean_equity': combined_equity.mean(axis=1),
                'std_equity': combined_equity.std(axis=1),
                'min_equity': combined_equity.min(axis=1),
                'max_equity': combined_equity.max(axis=1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing equity curves: {e}")
            return {}
    
    def _analyze_parameter_sensitivity(self, results: List[BacktestResult]) -> Dict:
        """Analyze parameter sensitivity"""
        try:
            # This would analyze how changes in parameters affect performance
            # Simplified implementation
            return {}
            
        except Exception as e:
            logger.error(f"Error analyzing parameter sensitivity: {e}")
            return {}
    
    def generate_report(self, results: List[BacktestResult], filename: str = None) -> str:
        """Generate comprehensive backtest report"""
        try:
            analysis = self.analyze_backtest_results(results)
            
            if not filename:
                filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Generate HTML report
            html_content = self._generate_html_report(analysis, results)
            
            with open(filename, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Backtest report generated: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def _generate_html_report(self, analysis: Dict, results: List[BacktestResult]) -> str:
        """Generate HTML report"""
        try:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtest Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                    .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>Backtest Report</h1>
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <div class="metric">Mean Return: {:.2%}</div>
                    <div class="metric">Mean Sharpe: {:.2f}</div>
                    <div class="metric">Mean Drawdown: {:.2%}</div>
                </div>
                <div class="section">
                    <h2>Risk Metrics</h2>
                    <div class="metric">VaR 95%: {:.2%}</div>
                    <div class="metric">VaR 99%: {:.2%}</div>
                </div>
            </body>
            </html>
            """.format(
                analysis.get('summary_stats', {}).get('mean_total_return', 0),
                analysis.get('summary_stats', {}).get('mean_sharpe', 0),
                analysis.get('summary_stats', {}).get('mean_drawdown', 0),
                analysis.get('risk_metrics', {}).get('var_95', 0),
                analysis.get('risk_metrics', {}).get('var_99', 0)
            )
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return "<html><body>Error generating report</body></html>"
