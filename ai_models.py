#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG AI Surrogate Models
Machine learning models for torsion prediction and surrogate simulation

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import os

# Optional imports for different ML frameworks
try:
    import sklearn
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor
    )
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


# ------------------------------
# DATA CLASSES
# ------------------------------
@dataclass
class AIModelConfig:
    """Configuration for AI models"""
    model_type: str = "random_forest"  # random_forest, neural_network, gradient_boosting
    framework: str = "sklearn"  # sklearn, pytorch, tensorflow
    features: List[str] = field(default_factory=lambda: ['x', 'y', 'z', 'phase', 'w'])
    target: str = "tau"
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    random_seed: int = 42
    normalize: bool = True
    
    # Model-specific parameters
    n_estimators: int = 100
    max_depth: int = 20
    learning_rate: float = 0.01
    batch_size: int = 1024
    epochs: int = 50
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    
    # Training
    use_gpu: bool = True
    early_stopping: bool = True
    patience: int = 10
    save_model: bool = True
    model_file: Optional[str] = None


# ------------------------------
# BASE MODEL CLASS
# ------------------------------
class BaseSurrogateModel:
    """Base class for surrogate models"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = config.features
        self.target_name = config.target
        self.training_history = {}
        
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.config.validation_fraction + self.config.test_fraction,
            random_state=self.config.random_seed
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.config.test_fraction / (self.config.validation_fraction + self.config.test_fraction),
            random_state=self.config.random_seed
        )
        
        # Normalize if requested
        if self.config.normalize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the model - to be overridden"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions - to be overridden"""
        raise NotImplementedError
    
    def save(self, filename: str):
        """Save model to disk"""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }, filename)
        logger.info(f"Model saved to {filename}")
    
    def load(self, filename: str):
        """Load model from disk"""
        import joblib
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.config = data['config']
        self.feature_names = data['feature_names']
        self.target_name = data['target_name']
        logger.info(f"Model loaded from {filename}")


# ------------------------------
# SKLEARN MODELS
# ------------------------------
class SklearnSurrogateModel(BaseSurrogateModel):
    """Surrogate model using scikit-learn"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        self._create_model()
    
    def _create_model(self):
        """Create the sklearn model based on config"""
        if self.config.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_seed,
                n_jobs=-1
            )
        elif self.config.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_seed
            )
        elif self.config.model_type == "neural_network":
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(self.config.hidden_layers),
                learning_rate_init=self.config.learning_rate,
                max_iter=self.config.epochs,
                random_state=self.config.random_seed
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train sklearn model"""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        
        # Train
        logger.info(f"Training {self.config.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        
        results = {
            'train_mse': float(mean_squared_error(y_train, y_train_pred)),
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'val_mse': float(mean_squared_error(y_val, y_val_pred)),
            'val_r2': float(r2_score(y_val, y_val_pred)),
            'test_mse': float(mean_squared_error(y_test, y_test_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred))
        }
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            results['feature_importance'] = dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
        
        logger.info(f"Training results: {results}")
        self.training_history = results
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)


# ------------------------------
# PYTORCH MODELS
# ------------------------------
class PyTorchSurrogateModel(BaseSurrogateModel):
    """Surrogate model using PyTorch"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self._create_model()
    
    def _create_model(self):
        """Create PyTorch neural network"""
        class TorsionNet(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout):
                super().__init__()
                layers_list = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_layers:
                    layers_list.append(nn.Linear(prev_dim, hidden_dim))
                    layers_list.append(nn.ReLU())
                    layers_list.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim
                
                layers_list.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, x):
                return self.network(x)
        
        self.model = TorsionNet(
            input_dim=len(self.feature_names),
            hidden_layers=self.config.hidden_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train PyTorch model"""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).view(-1, 1).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")
            
            # Early stopping
            if self.config.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(X_train_t).cpu().numpy()
            y_val_pred = self.model(X_val_t).cpu().numpy()
            y_test_pred = self.model(X_test_t).cpu().numpy()
        
        results = {
            'train_mse': float(mean_squared_error(y_train, y_train_pred)),
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'val_mse': float(mean_squared_error(y_val, y_val_pred)),
            'val_r2': float(r2_score(y_val, y_val_pred)),
            'test_mse': float(mean_squared_error(y_test, y_test_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred)),
            'training_history': history
        }
        
        logger.info(f"Training results: {results}")
        self.training_history = results
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.flatten()


# ------------------------------
# TENSORFLOW/KERAS MODELS
# ------------------------------
class TensorFlowSurrogateModel(BaseSurrogateModel):
    """Surrogate model using TensorFlow/Keras"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self._create_model()
    
    def _create_model(self):
        """Create Keras model"""
        model = keras.Sequential()
        model.add(layers.Input(shape=(len(self.feature_names),)))
        
        for units in self.config.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.config.dropout))
        
        model.add(layers.Dense(1))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Keras model"""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        
        # Callbacks
        callbacks = []
        if self.config.early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.patience,
                    restore_best_weights=True
                )
            )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_train_pred = self.model.predict(X_train, verbose=0).flatten()
        y_val_pred = self.model.predict(X_val, verbose=0).flatten()
        y_test_pred = self.model.predict(X_test, verbose=0).flatten()
        
        results = {
            'train_mse': float(mean_squared_error(y_train, y_train_pred)),
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'val_mse': float(mean_squared_error(y_val, y_val_pred)),
            'val_r2': float(r2_score(y_val, y_val_pred)),
            'test_mse': float(mean_squared_error(y_test, y_test_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred)),
            'training_history': history.history
        }
        
        logger.info(f"Training results: {results}")
        self.training_history = results
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled, verbose=0).flatten()


# ------------------------------
# MODEL FACTORY
# ------------------------------
class SurrogateModelFactory:
    """Factory for creating surrogate models"""
    
    @staticmethod
    def create_model(config: AIModelConfig) -> BaseSurrogateModel:
        """Create appropriate model based on configuration"""
        if config.framework == "sklearn":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            return SklearnSurrogateModel(config)
        elif config.framework == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            return PyTorchSurrogateModel(config)
        elif config.framework == "tensorflow":
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available")
            return TensorFlowSurrogateModel(config)
        else:
            raise ValueError(f"Unknown framework: {config.framework}")


# ------------------------------
# ENSEMBLE MODEL
# ------------------------------
class EnsembleSurrogateModel(BaseSurrogateModel):
    """Ensemble of multiple surrogate models"""
    
    def __init__(self, configs: List[AIModelConfig]):
        self.configs = configs
        self.models = []
        self.weights = []
        self.scaler = None
        
        for config in configs:
            model = SurrogateModelFactory.create_model(config)
            self.models.append(model)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train all models in ensemble"""
        results = {}
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            model_results = model.train(X, y)
            results[f'model_{i}'] = model_results
            
            # Use validation performance as weight
            self.weights.append(1.0 / model_results['val_mse'])
        
        # Normalize weights
        self.weights = np.array(self.weights)
        self.weights /= self.weights.sum()
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction (weighted average)"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)


# ------------------------------
# FEATURE ENGINEERING
# ------------------------------
class FeatureEngineer:
    """Create features from raw simulation data"""
    
    @staticmethod
    def create_features(
        points: np.ndarray,
        phase: np.ndarray,
        tau: Optional[np.ndarray] = None,
        hub_potential: Optional[np.ndarray] = None,
        w_coord: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create feature matrix from raw data
        """
        features = []
        
        # Basic coordinates
        features.append(points[:, :3])  # x, y, z
        
        # Phase
        features.append(phase.reshape(-1, 1))
        
        # Phase-derived features
        phase_sin = np.sin(phase).reshape(-1, 1)
        phase_cos = np.cos(phase).reshape(-1, 1)
        features.append(phase_sin)
        features.append(phase_cos)
        
        # Radius
        r = np.linalg.norm(points[:, :3], axis=1).reshape(-1, 1)
        features.append(r)
        
        # Angular coordinates
        theta = np.arccos(points[:, 2] / (r.flatten() + 1e-10)).reshape(-1, 1)
        phi = np.arctan2(points[:, 1], points[:, 0]).reshape(-1, 1)
        features.append(theta)
        features.append(phi)
        
        # 4th dimension if available
        if w_coord is not None:
            features.append(w_coord.reshape(-1, 1))
        
        # Hub potential if available
        if hub_potential is not None:
            features.append(hub_potential.reshape(-1, 1))
        
        return np.hstack(features)


# ------------------------------
# UNIT TESTS
# ------------------------------
def test_ai_models():
    """Test AI surrogate models"""
    # Create synthetic data
    n_samples = 10000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(n_samples)
    
    # Test sklearn model
    config = AIModelConfig(
        model_type="random_forest",
        framework="sklearn",
        n_estimators=10,  # Small for testing
        max_depth=5
    )
    
    model = SurrogateModelFactory.create_model(config)
    results = model.train(X, y)
    
    assert 'test_r2' in results
    assert results['test_r2'] > 0
    
    # Test predictions
    X_test = np.random.randn(100, n_features)
    y_pred = model.predict(X_test)
    
    assert len(y_pred) == 100
    
    print("AI models test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ai_models()