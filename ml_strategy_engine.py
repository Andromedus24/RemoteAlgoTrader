#!/usr/bin/env python3
"""
Advanced Machine Learning Strategy Engine for RemoteAlgoTrader
Implements multiple ML models for trading strategy generation and optimization
"""

import os
import json
import logging
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"
    CNN_LSTM = "cnn_lstm"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class MLPrediction:
    """Represents a machine learning prediction"""
    symbol: str
    timestamp: datetime
    prediction: SignalType
    confidence: float
    model_type: ModelType
    features: Dict[str, float]
    probability: Optional[float] = None
    model_version: str = "1.0"

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    training_time: float
    prediction_time: float
    last_updated: datetime
    version: str

class FeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical features"""
        try:
            features = df.copy()
            
            # Price-based features
            features['price_change'] = features['Close'].pct_change()
            features['price_change_2d'] = features['Close'].pct_change(2)
            features['price_change_5d'] = features['Close'].pct_change(5)
            features['price_change_10d'] = features['Close'].pct_change(10)
            
            # Moving averages
            features['sma_5'] = features['Close'].rolling(window=5).mean()
            features['sma_10'] = features['Close'].rolling(window=10).mean()
            features['sma_20'] = features['Close'].rolling(window=20).mean()
            features['sma_50'] = features['Close'].rolling(window=50).mean()
            
            features['ema_5'] = features['Close'].ewm(span=5).mean()
            features['ema_10'] = features['Close'].ewm(span=10).mean()
            features['ema_20'] = features['Close'].ewm(span=20).mean()
            features['ema_50'] = features['Close'].ewm(span=50).mean()
            
            # Price relative to moving averages
            features['price_vs_sma_5'] = features['Close'] / features['sma_5'] - 1
            features['price_vs_sma_10'] = features['Close'] / features['sma_10'] - 1
            features['price_vs_sma_20'] = features['Close'] / features['sma_20'] - 1
            features['price_vs_sma_50'] = features['Close'] / features['sma_50'] - 1
            
            # RSI
            features['rsi_14'] = self._calculate_rsi(features['Close'], 14)
            features['rsi_21'] = self._calculate_rsi(features['Close'], 21)
            
            # MACD
            ema_12 = features['Close'].ewm(span=12).mean()
            ema_26 = features['Close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            bb_20 = features['Close'].rolling(window=20)
            features['bb_upper'] = bb_20.mean() + (bb_20.std() * 2)
            features['bb_lower'] = bb_20.mean() - (bb_20.std() * 2)
            features['bb_position'] = (features['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Volume features
            features['volume_sma_5'] = features['Volume'].rolling(window=5).mean()
            features['volume_sma_20'] = features['Volume'].rolling(window=20).mean()
            features['volume_ratio'] = features['Volume'] / features['volume_sma_20']
            
            # Volatility features
            features['volatility_5d'] = features['Close'].rolling(window=5).std()
            features['volatility_20d'] = features['Close'].rolling(window=20).std()
            features['volatility_ratio'] = features['volatility_5d'] / features['volatility_20d']
            
            # Momentum features
            features['momentum_5d'] = features['Close'] / features['Close'].shift(5) - 1
            features['momentum_10d'] = features['Close'] / features['Close'].shift(10) - 1
            features['momentum_20d'] = features['Close'] / features['Close'].shift(20) - 1
            
            # Support and resistance
            features['high_20d'] = features['High'].rolling(window=20).max()
            features['low_20d'] = features['Low'].rolling(window=20).min()
            features['support_resistance_ratio'] = (features['Close'] - features['low_20d']) / (features['high_20d'] - features['low_20d'])
            
            # Time-based features
            features['day_of_week'] = features.index.dayofweek
            features['month'] = features.index.month
            features['quarter'] = features.index.quarter
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = features['Close'].shift(lag)
                features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'close_mean_{window}'] = features['Close'].rolling(window=window).mean()
                features[f'close_std_{window}'] = features['Close'].rolling(window=window).std()
                features[f'volume_mean_{window}'] = features['Volume'].rolling(window=window).mean()
                features[f'volume_std_{window}'] = features['Volume'].rolling(window=window).std()
            
            # Remove NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index)
    
    def create_target_variable(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.Series:
        """Create target variable for ML models"""
        try:
            # Calculate future returns
            future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
            
            # Create binary classification target
            target = pd.Series(index=df.index, dtype=int)
            target[future_returns > threshold] = 1  # Buy signal
            target[future_returns < -threshold] = -1  # Sell signal
            target[(future_returns >= -threshold) & (future_returns <= threshold)] = 0  # Hold
            
            return target
            
        except Exception as e:
            logger.error(f"Error creating target variable: {e}")
            return pd.Series(index=df.index, dtype=int)
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for ML models"""
        try:
            # Select feature columns (exclude target and basic OHLCV)
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if target_col:
                exclude_cols.append(target_col)
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            self.feature_names = feature_cols
            
            X = df[feature_cols].values
            y = df[target_col].values if target_col else None
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]), np.array([])

class MLStrategyEngine:
    """Advanced machine learning strategy engine"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.feature_engineer = FeatureEngineer()
        self.model_performances: Dict[str, ModelPerformance] = {}
        
        # Model configurations
        self.model_configs = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            ModelType.XGBOOST: {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            ModelType.LIGHTGBM: {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            ModelType.LOGISTIC_REGRESSION: {
                'random_state': 42,
                'max_iter': 1000
            },
            ModelType.SVM: {
                'kernel': 'rbf',
                'random_state': 42
            }
        }
    
    def train_model(self, model_type: ModelType, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> Any:
        """Train a machine learning model"""
        try:
            start_time = time.time()
            
            if model_type == ModelType.RANDOM_FOREST:
                model = RandomForestClassifier(**self.model_configs[model_type])
            elif model_type == ModelType.GRADIENT_BOOSTING:
                model = GradientBoostingClassifier(**self.model_configs[model_type])
            elif model_type == ModelType.XGBOOST:
                model = xgb.XGBClassifier(**self.model_configs[model_type])
            elif model_type == ModelType.LIGHTGBM:
                model = lgb.LGBMClassifier(**self.model_configs[model_type])
            elif model_type == ModelType.LOGISTIC_REGRESSION:
                model = LogisticRegression(**self.model_configs[model_type])
            elif model_type == ModelType.SVM:
                model = SVC(**self.model_configs[model_type], probability=True)
            elif model_type == ModelType.LSTM:
                model = self._create_lstm_model(X_train.shape[1])
            elif model_type == ModelType.CNN_LSTM:
                model = self._create_cnn_lstm_model(X_train.shape[1])
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            if model_type in [ModelType.LSTM, ModelType.CNN_LSTM]:
                # For neural networks, reshape data and use validation
                X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1])) if X_val is not None else None
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(
                        filepath=str(self.model_dir / f"{model_type.value}_best.h5"),
                        monitor='val_loss',
                        save_best_only=True
                    )
                ]
                
                model.fit(
                    X_train_reshaped, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val_reshaped, y_val) if X_val is not None else None,
                    callbacks=callbacks,
                    verbose=0
                )
            else:
                # For traditional ML models
                if X_val is not None and y_val is not None:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Store model
            self.models[model_type.value] = model
            
            # Save model
            self._save_model(model, model_type)
            
            logger.info(f"Trained {model_type.value} model in {training_time:.2f} seconds")
            return model
            
        except Exception as e:
            logger.error(f"Error training {model_type.value} model: {e}")
            return None
    
    def _create_lstm_model(self, n_features: int) -> Sequential:
        """Create LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: buy, sell, hold
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_lstm_model(self, n_features: int) -> Sequential:
        """Create CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(1, n_features)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, model_type: ModelType, X: np.ndarray) -> MLPrediction:
        """Make prediction using trained model"""
        try:
            if model_type.value not in self.models:
                raise ValueError(f"Model {model_type.value} not trained")
            
            model = self.models[model_type.value]
            start_time = time.time()
            
            # Prepare input
            if model_type in [ModelType.LSTM, ModelType.CNN_LSTM]:
                X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
                prediction = model.predict(X_reshaped, verbose=0)
            else:
                prediction = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                else:
                    probabilities = None
            
            prediction_time = time.time() - start_time
            
            # Convert prediction to signal
            if model_type in [ModelType.LSTM, ModelType.CNN_LSTM]:
                signal_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                probability = prediction[0][signal_idx]
            else:
                signal_idx = prediction[0] if isinstance(prediction, np.ndarray) else prediction
                confidence = probabilities[0][signal_idx] if probabilities is not None else 0.5
                probability = probabilities[0][signal_idx] if probabilities is not None else None
            
            # Map to signal type
            signal_map = {1: SignalType.BUY, -1: SignalType.SELL, 0: SignalType.HOLD}
            signal = signal_map.get(signal_idx, SignalType.HOLD)
            
            # Create features dictionary
            features = dict(zip(self.feature_engineer.feature_names, X[0]))
            
            return MLPrediction(
                symbol="",  # Will be set by caller
                timestamp=datetime.now(),
                prediction=signal,
                confidence=confidence,
                model_type=model_type,
                features=features,
                probability=probability,
                model_version="1.0"
            )
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_type.value}: {e}")
            return None
    
    def evaluate_model(self, model_type: ModelType, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Evaluate model performance"""
        try:
            if model_type.value not in self.models:
                raise ValueError(f"Model {model_type.value} not trained")
            
            model = self.models[model_type.value]
            
            # Make predictions
            if model_type in [ModelType.LSTM, ModelType.CNN_LSTM]:
                X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                y_pred_proba = model.predict(X_test_reshaped, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            performance = ModelPerformance(
                model_type=model_type,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                confusion_matrix=cm,
                training_time=0,  # Will be updated
                prediction_time=0,  # Will be updated
                last_updated=datetime.now(),
                version="1.0"
            )
            
            self.model_performances[model_type.value] = performance
            
            logger.info(f"Model {model_type.value} evaluation - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_type.value}: {e}")
            return None
    
    def ensemble_predict(self, X: np.ndarray, weights: Dict[str, float] = None) -> MLPrediction:
        """Make ensemble prediction using multiple models"""
        try:
            if not self.models:
                raise ValueError("No models trained for ensemble prediction")
            
            predictions = []
            confidences = []
            model_types = []
            
            for model_type_str, model in self.models.items():
                model_type = ModelType(model_type_str)
                prediction = self.predict(model_type, X)
                
                if prediction:
                    predictions.append(prediction.prediction.value)
                    confidences.append(prediction.confidence)
                    model_types.append(model_type)
            
            if not predictions:
                raise ValueError("No valid predictions from ensemble")
            
            # Weighted voting
            if weights:
                weighted_votes = defaultdict(float)
                for pred, conf, model_type in zip(predictions, confidences, model_types):
                    weight = weights.get(model_type.value, 1.0)
                    weighted_votes[pred] += conf * weight
                
                final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
                final_confidence = max(weighted_votes.values())
            else:
                # Simple voting
                vote_counts = defaultdict(int)
                for pred, conf in zip(predictions, confidences):
                    vote_counts[pred] += 1
                
                final_prediction = max(vote_counts.items(), key=lambda x: x[1])[0]
                final_confidence = np.mean(confidences)
            
            # Map back to signal type
            signal_map = {'buy': SignalType.BUY, 'sell': SignalType.SELL, 'hold': SignalType.HOLD}
            final_signal = signal_map.get(final_prediction, SignalType.HOLD)
            
            return MLPrediction(
                symbol="",  # Will be set by caller
                timestamp=datetime.now(),
                prediction=final_signal,
                confidence=final_confidence,
                model_type=ModelType.RANDOM_FOREST,  # Placeholder
                features={},  # Will be populated
                probability=final_confidence,
                model_version="ensemble"
            )
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return None
    
    def _save_model(self, model: Any, model_type: ModelType):
        """Save trained model"""
        try:
            model_path = self.model_dir / f"{model_type.value}.pkl"
            
            if model_type in [ModelType.LSTM, ModelType.CNN_LSTM]:
                # Save Keras model
                model.save(str(self.model_dir / f"{model_type.value}.h5"))
            else:
                # Save traditional ML model
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save feature engineer
            feature_engineer_path = self.model_dir / f"{model_type.value}_feature_engineer.pkl"
            with open(feature_engineer_path, 'wb') as f:
                pickle.dump(self.feature_engineer, f)
            
            logger.info(f"Saved {model_type.value} model to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving {model_type.value} model: {e}")
    
    def load_model(self, model_type: ModelType) -> bool:
        """Load trained model"""
        try:
            model_path = self.model_dir / f"{model_type.value}.pkl"
            feature_engineer_path = self.model_dir / f"{model_type.value}_feature_engineer.pkl"
            
            if model_type in [ModelType.LSTM, ModelType.CNN_LSTM]:
                # Load Keras model
                model_file = self.model_dir / f"{model_type.value}.h5"
                if model_file.exists():
                    model = load_model(str(model_file))
                    self.models[model_type.value] = model
            else:
                # Load traditional ML model
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.models[model_type.value] = model
            
            # Load feature engineer
            if feature_engineer_path.exists():
                with open(feature_engineer_path, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
            
            logger.info(f"Loaded {model_type.value} model")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {model_type.value} model: {e}")
            return False
    
    def get_model_performance(self, model_type: ModelType) -> Optional[ModelPerformance]:
        """Get model performance metrics"""
        return self.model_performances.get(model_type.value)
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        return list(self.models.keys())
    
    def retrain_model(self, model_type: ModelType, new_data: pd.DataFrame) -> bool:
        """Retrain model with new data"""
        try:
            # Create features
            features_df = self.feature_engineer.create_technical_features(new_data)
            target = self.feature_engineer.create_target_variable(features_df)
            
            # Prepare data
            X, y = self.feature_engineer.prepare_features(features_df, 'target')
            
            if len(X) == 0:
                logger.error("No valid data for retraining")
                return False
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Retrain model
            model = self.train_model(model_type, X_train, y_train, X_test, y_test)
            
            if model is not None:
                # Evaluate new model
                performance = self.evaluate_model(model_type, X_test, y_test)
                logger.info(f"Retrained {model_type.value} model successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error retraining {model_type.value} model: {e}")
            return False
