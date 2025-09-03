#!/usr/bin/env python3
"""
Web Dashboard & Real-time Monitoring System for RemoteAlgoTrader
Provides web interface for monitoring trading performance, portfolio status, and system health
"""

import os
import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue
import sqlite3
from pathlib import Path

# Web framework imports
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Data visualization
import plotly.graph_objs as go
import plotly.utils
import plotly.express as px
from plotly.subplots import make_subplots

# Database
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DashboardStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    uptime: float
    active_connections: int
    timestamp: datetime

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_pnl: float
    daily_pnl: float
    win_rate: float
    total_trades: int
    active_positions: int
    portfolio_value: float
    cash_balance: float
    timestamp: datetime

class WebDashboard:
    """Web dashboard for real-time monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.get('secret_key', 'your-secret-key-here')
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize login manager
        self.login_manager = LoginManager()
        self.login_manager.init_app(self.app)
        self.login_manager.login_view = 'login'
        
        # Data storage
        self.system_metrics: List[SystemMetrics] = []
        self.trading_metrics: List[TradingMetrics] = []
        self.alerts: List[Dict] = []
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        
        # Real-time data queue
        self.data_queue = Queue(maxsize=1000)
        self.update_thread = None
        self.is_running = False
        
        # Database connection
        self.db_path = config.get('db_path', 'dashboard.db')
        self._init_database()
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
    def _init_database(self):
        """Initialize dashboard database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_usage REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    disk_usage REAL NOT NULL,
                    network_latency REAL NOT NULL,
                    uptime REAL NOT NULL,
                    active_connections INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trading metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_pnl REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    active_positions INTEGER NOT NULL,
                    portfolio_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Dashboard database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing dashboard database: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.login_manager.user_loader
        def load_user(user_id):
            return self._get_user_by_id(user_id)
        
        @self.app.route('/')
        @login_required
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login page"""
            if request.method == 'POST':
                username = request.form['username']
                password = request.form['password']
                
                user = self._authenticate_user(username, password)
                if user:
                    login_user(user)
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password')
            
            return render_template('login.html')
        
        @self.app.route('/logout')
        @login_required
        def logout():
            """Logout"""
            logout_user()
            return redirect(url_for('login'))
        
        @self.app.route('/api/metrics')
        @login_required
        def get_metrics():
            """Get current metrics"""
            return jsonify({
                'system': self._get_latest_system_metrics(),
                'trading': self._get_latest_trading_metrics(),
                'alerts': self._get_recent_alerts(),
                'positions': self._get_active_positions(),
                'trades': self._get_recent_trades()
            })
        
        @self.app.route('/api/system-metrics')
        @login_required
        def get_system_metrics():
            """Get system metrics history"""
            hours = request.args.get('hours', 24, type=int)
            metrics = self._get_system_metrics_history(hours)
            return jsonify(metrics)
        
        @self.app.route('/api/trading-metrics')
        @login_required
        def get_trading_metrics():
            """Get trading metrics history"""
            days = request.args.get('days', 30, type=int)
            metrics = self._get_trading_metrics_history(days)
            return jsonify(metrics)
        
        @self.app.route('/api/portfolio')
        @login_required
        def get_portfolio():
            """Get portfolio data"""
            portfolio = self._get_portfolio_data()
            return jsonify(portfolio)
        
        @self.app.route('/api/positions')
        @login_required
        def get_positions():
            """Get positions data"""
            positions = self._get_positions_data()
            return jsonify(positions)
        
        @self.app.route('/api/trades')
        @login_required
        def get_trades():
            """Get trades data"""
            trades = self._get_trades_data()
            return jsonify(trades)
        
        @self.app.route('/api/alerts')
        @login_required
        def get_alerts():
            """Get alerts data"""
            alerts = self._get_alerts_data()
            return jsonify(alerts)
        
        @self.app.route('/api/charts/portfolio-performance')
        @login_required
        def get_portfolio_performance_chart():
            """Get portfolio performance chart"""
            chart_data = self._create_portfolio_performance_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/charts/position-distribution')
        @login_required
        def get_position_distribution_chart():
            """Get position distribution chart"""
            chart_data = self._create_position_distribution_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/charts/trade-analysis')
        @login_required
        def get_trade_analysis_chart():
            """Get trade analysis chart"""
            chart_data = self._create_trade_analysis_chart()
            return jsonify(chart_data)
        
        @self.app.route('/api/system/status')
        @login_required
        def get_system_status():
            """Get system status"""
            status = self._get_system_status()
            return jsonify(status)
        
        @self.app.route('/api/system/restart', methods=['POST'])
        @login_required
        def restart_system():
            """Restart trading system"""
            success = self._restart_trading_system()
            return jsonify({'success': success})
        
        @self.app.route('/api/system/stop', methods=['POST'])
        @login_required
        def stop_system():
            """Stop trading system"""
            success = self._stop_trading_system()
            return jsonify({'success': success})
        
        @self.app.route('/api/settings', methods=['GET', 'POST'])
        @login_required
        def settings():
            """Settings page"""
            if request.method == 'POST':
                # Update settings
                settings_data = request.json
                success = self._update_settings(settings_data)
                return jsonify({'success': success})
            
            # Get current settings
            settings = self._get_settings()
            return jsonify(settings)
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info(f"Client connected: {request.sid}")
            emit('status', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_metrics')
        def handle_metrics_subscription():
            """Handle metrics subscription"""
            # Send initial metrics
            metrics = {
                'system': self._get_latest_system_metrics(),
                'trading': self._get_latest_trading_metrics()
            }
            emit('metrics_update', metrics)
    
    def start(self, host: str = '0.0.0.0', port: int = 5000):
        """Start the web dashboard"""
        try:
            self.is_running = True
            
            # Start update thread
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            
            # Start Flask app
            logger.info(f"Starting web dashboard on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=False)
            
        except Exception as e:
            logger.error(f"Error starting web dashboard: {e}")
            raise
    
    def stop(self):
        """Stop the web dashboard"""
        try:
            self.is_running = False
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            
            logger.info("Web dashboard stopped")
            
        except Exception as e:
            logger.error(f"Error stopping web dashboard: {e}")
    
    def _update_loop(self):
        """Background loop for updating metrics and sending real-time updates"""
        while self.is_running:
            try:
                # Update system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Update trading metrics
                trading_metrics = self._collect_trading_metrics()
                self.trading_metrics.append(trading_metrics)
                
                # Store in database
                self._store_metrics(system_metrics, trading_metrics)
                
                # Send real-time updates
                self._send_real_time_updates(system_metrics, trading_metrics)
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            import psutil
            
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network latency (placeholder)
            network_latency = 50.0  # ms
            
            # Uptime
            uptime = time.time() - psutil.boot_time()
            
            # Active connections (placeholder)
            active_connections = len(self.socketio.server.manager.rooms)
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                uptime=uptime,
                active_connections=active_connections,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, datetime.now())
    
    def _collect_trading_metrics(self) -> TradingMetrics:
        """Collect current trading metrics"""
        try:
            # This would typically get data from the trading system
            # For now, using placeholder data
            
            total_pnl = 1250.50
            daily_pnl = 45.75
            win_rate = 0.68
            total_trades = 156
            active_positions = 8
            portfolio_value = 25000.00
            cash_balance = 5000.00
            
            return TradingMetrics(
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                win_rate=win_rate,
                total_trades=total_trades,
                active_positions=active_positions,
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return TradingMetrics(0, 0, 0, 0, 0, 0, 0, datetime.now())
    
    def _store_metrics(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """Store metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store system metrics
            cursor.execute('''
                INSERT INTO system_metrics 
                (cpu_usage, memory_usage, disk_usage, network_latency, uptime, active_connections)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                system_metrics.cpu_usage,
                system_metrics.memory_usage,
                system_metrics.disk_usage,
                system_metrics.network_latency,
                system_metrics.uptime,
                system_metrics.active_connections
            ))
            
            # Store trading metrics
            cursor.execute('''
                INSERT INTO trading_metrics 
                (total_pnl, daily_pnl, win_rate, total_trades, active_positions, portfolio_value, cash_balance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trading_metrics.total_pnl,
                trading_metrics.daily_pnl,
                trading_metrics.win_rate,
                trading_metrics.total_trades,
                trading_metrics.active_positions,
                trading_metrics.portfolio_value,
                trading_metrics.cash_balance
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def _send_real_time_updates(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """Send real-time updates to connected clients"""
        try:
            update_data = {
                'system': asdict(system_metrics),
                'trading': asdict(trading_metrics),
                'timestamp': datetime.now().isoformat()
            }
            
            self.socketio.emit('real_time_update', update_data)
            
        except Exception as e:
            logger.error(f"Error sending real-time updates: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        try:
            # Keep only last 1000 entries in memory
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
            
            if len(self.trading_metrics) > 1000:
                self.trading_metrics = self.trading_metrics[-1000:]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def _get_latest_system_metrics(self) -> Dict:
        """Get latest system metrics"""
        if self.system_metrics:
            return asdict(self.system_metrics[-1])
        return {}
    
    def _get_latest_trading_metrics(self) -> Dict:
        """Get latest trading metrics"""
        if self.trading_metrics:
            return asdict(self.trading_metrics[-1])
        return {}
    
    def _get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT alert_type, message, severity, timestamp
                FROM alerts
                WHERE resolved = FALSE
                ORDER BY timestamp DESC
                LIMIT 10
            ''')
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'type': row[0],
                    'message': row[1],
                    'severity': row[2],
                    'timestamp': row[3]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    def _get_active_positions(self) -> List[Dict]:
        """Get active positions"""
        # Placeholder - would get from trading system
        return [
            {'symbol': 'AAPL', 'quantity': 100, 'entry_price': 150.25, 'current_price': 152.50, 'pnl': 225.00},
            {'symbol': 'GOOGL', 'quantity': 50, 'entry_price': 2750.00, 'current_price': 2780.00, 'pnl': 1500.00},
            {'symbol': 'TSLA', 'quantity': 75, 'entry_price': 800.00, 'current_price': 820.00, 'pnl': 1500.00}
        ]
    
    def _get_recent_trades(self) -> List[Dict]:
        """Get recent trades"""
        # Placeholder - would get from trading system
        return [
            {'symbol': 'MSFT', 'side': 'BUY', 'quantity': 50, 'price': 300.00, 'timestamp': '2024-01-15 10:30:00'},
            {'symbol': 'AMZN', 'side': 'SELL', 'quantity': 25, 'price': 3500.00, 'timestamp': '2024-01-15 09:45:00'},
            {'symbol': 'NVDA', 'side': 'BUY', 'quantity': 30, 'price': 500.00, 'timestamp': '2024-01-15 09:15:00'}
        ]
    
    def _get_system_metrics_history(self, hours: int) -> List[Dict]:
        """Get system metrics history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT cpu_usage, memory_usage, disk_usage, network_latency, uptime, active_connections, timestamp
                FROM system_metrics
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp
            '''.format(hours))
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    'cpu_usage': row[0],
                    'memory_usage': row[1],
                    'disk_usage': row[2],
                    'network_latency': row[3],
                    'uptime': row[4],
                    'active_connections': row[5],
                    'timestamp': row[6]
                })
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics history: {e}")
            return []
    
    def _get_trading_metrics_history(self, days: int) -> List[Dict]:
        """Get trading metrics history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT total_pnl, daily_pnl, win_rate, total_trades, active_positions, portfolio_value, cash_balance, timestamp
                FROM trading_metrics
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days))
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    'total_pnl': row[0],
                    'daily_pnl': row[1],
                    'win_rate': row[2],
                    'total_trades': row[3],
                    'active_positions': row[4],
                    'portfolio_value': row[5],
                    'cash_balance': row[6],
                    'timestamp': row[7]
                })
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting trading metrics history: {e}")
            return []
    
    def _create_portfolio_performance_chart(self) -> Dict:
        """Create portfolio performance chart"""
        try:
            # Get historical data
            metrics = self._get_trading_metrics_history(30)
            
            if not metrics:
                return {'error': 'No data available'}
            
            # Create chart
            dates = [m['timestamp'] for m in metrics]
            portfolio_values = [m['portfolio_value'] for m in metrics]
            pnl_values = [m['total_pnl'] for m in metrics]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Value', 'Cumulative P&L'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=portfolio_values, name='Portfolio Value', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=pnl_values, name='Cumulative P&L', line=dict(color='green')),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Portfolio Performance',
                height=600,
                showlegend=True
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error creating portfolio performance chart: {e}")
            return {'error': str(e)}
    
    def _create_position_distribution_chart(self) -> Dict:
        """Create position distribution chart"""
        try:
            positions = self._get_active_positions()
            
            if not positions:
                return {'error': 'No positions available'}
            
            symbols = [p['symbol'] for p in positions]
            values = [p['quantity'] * p['current_price'] for p in positions]
            
            fig = go.Figure(data=[go.Pie(labels=symbols, values=values)])
            fig.update_layout(title='Position Distribution')
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error creating position distribution chart: {e}")
            return {'error': str(e)}
    
    def _create_trade_analysis_chart(self) -> Dict:
        """Create trade analysis chart"""
        try:
            trades = self._get_recent_trades()
            
            if not trades:
                return {'error': 'No trades available'}
            
            # Group by symbol
            symbol_data = {}
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in symbol_data:
                    symbol_data[symbol] = {'buys': 0, 'sells': 0}
                
                if trade['side'] == 'BUY':
                    symbol_data[symbol]['buys'] += 1
                else:
                    symbol_data[symbol]['sells'] += 1
            
            symbols = list(symbol_data.keys())
            buys = [symbol_data[s]['buys'] for s in symbols]
            sells = [symbol_data[s]['sells'] for s in symbols]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Buys', x=symbols, y=buys, marker_color='green'))
            fig.add_trace(go.Bar(name='Sells', x=symbols, y=sells, marker_color='red'))
            
            fig.update_layout(
                title='Trade Analysis by Symbol',
                barmode='group'
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error creating trade analysis chart: {e}")
            return {'error': str(e)}
    
    def _get_system_status(self) -> Dict:
        """Get system status"""
        try:
            latest_system = self._get_latest_system_metrics()
            
            if not latest_system:
                return {'status': DashboardStatus.OFFLINE.value}
            
            # Determine status based on metrics
            cpu_usage = latest_system.get('cpu_usage', 0)
            memory_usage = latest_system.get('memory_usage', 0)
            
            if cpu_usage > 90 or memory_usage > 90:
                status = DashboardStatus.ERROR.value
            elif cpu_usage > 70 or memory_usage > 70:
                status = DashboardStatus.WARNING.value
            else:
                status = DashboardStatus.ONLINE.value
            
            return {
                'status': status,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'uptime': latest_system.get('uptime', 0),
                'active_connections': latest_system.get('active_connections', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': DashboardStatus.ERROR.value}
    
    def _restart_trading_system(self) -> bool:
        """Restart trading system"""
        try:
            # This would typically restart the trading system
            logger.info("Trading system restart requested")
            return True
        except Exception as e:
            logger.error(f"Error restarting trading system: {e}")
            return False
    
    def _stop_trading_system(self) -> bool:
        """Stop trading system"""
        try:
            # This would typically stop the trading system
            logger.info("Trading system stop requested")
            return True
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
            return False
    
    def _get_settings(self) -> Dict:
        """Get current settings"""
        return {
            'trading_enabled': True,
            'max_positions': 10,
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'update_interval': 5
        }
    
    def _update_settings(self, settings: Dict) -> bool:
        """Update settings"""
        try:
            # This would typically update system settings
            logger.info(f"Settings updated: {settings}")
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False
    
    def _get_user_by_id(self, user_id: int):
        """Get user by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, username, email, role FROM users WHERE id = ?', (user_id,))
            user_data = cursor.fetchone()
            
            conn.close()
            
            if user_data:
                return User(user_data[0], user_data[1], user_data[2], user_data[3])
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def _authenticate_user(self, username: str, password: str):
        """Authenticate user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, username, password_hash, email, role FROM users WHERE username = ?', (username,))
            user_data = cursor.fetchone()
            
            conn.close()
            
            if user_data and check_password_hash(user_data[2], password):
                return User(user_data[0], user_data[1], user_data[3], user_data[4])
            return None
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None

class User(UserMixin):
    """User model for authentication"""
    
    def __init__(self, id: int, username: str, email: str, role: str):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
