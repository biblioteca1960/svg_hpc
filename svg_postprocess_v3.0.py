#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG HPC Post-Processing - Advanced Analysis Pipeline
Combines VTU chunks, computes physical quantities, and trains AI models

Author: L. Morató de Dalmases
Version: 3.0
Date: February 2026
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
from scipy import stats, signal
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpi4py import MPI

# Optional imports
try:
    import pyvista as pv
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    pv = None

try:
    from sklearn.ensemble import RandomForestRegressor
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


# ------------------------------
# MPI INITIALIZATION
# ------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ------------------------------
# PHYSICAL CONSTANTS
# ------------------------------
class SVGConstants:
    DELTA_DEG = 6.8
    DELTA = np.deg2rad(DELTA_DEG)
    TAU = DELTA / np.sqrt(3)
    ETA_EFF = 1.34e-19
    BETA_CMB = 0.351  # degrees
    TAU_E = 7.7e-3    # Tesla
    N_TF = 4.0         # Tully-Fisher exponent


# ------------------------------
# DATA LOADER
# ------------------------------
class SVGDataLoader:
    """Load and combine SVG simulation data"""
    
    def __init__(self, data_dir="output", step=None):
        self.data_dir = Path(data_dir)
        self.step = step
        self.points = None
        self.phase = None
        self.tau = None
        self.eta_eff = None
        self.hub_potential = None
        self.w_coord = None
        
    def load_step(self, step):
        """Load all data for a specific time step"""
        self.step = step
        
        # Find all VTU files for this step
        pattern = f"svg_rank*_step{step:06d}_chunk*.vtu"
        files = sorted(glob.glob(str(self.data_dir / pattern)))
        
        if not files:
            raise FileNotFoundError(f"No VTU files found for step {step}")
        
        print(f"Found {len(files)} VTU files for step {step}")
        
        # Load data from all files
        all_points = []
        all_phase = []
        all_tau = []
        all_eta = []
        all_hub = []
        all_w = []
        
        for f in files:
            if VTK_AVAILABLE:
                grid = pv.read(f)
                all_points.append(grid.points)
                all_phase.append(grid.point_data["phase"])
                all_tau.append(grid.point_data["tau"])
                all_eta.append(grid.point_data["eta_eff"])
                all_hub.append(grid.point_data["hub_potential"])
                all_w.append(grid.point_data["w_coord"])
            else:
                # Fallback: load from CSV if available
                csv_file = f.replace('.vtu', '.csv')
                if os.path.exists(csv_file):
                    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
                    all_points.append(data[:, :3])
                    all_phase.append(data[:, 3])
                    all_tau.append(data[:, 4])
                    # Others not available in CSV
        
        self.points = np.vstack(all_points)
        self.phase = np.hstack(all_phase)
        self.tau = np.hstack(all_tau)
        self.eta_eff = np.hstack(all_eta) if all_eta else None
        self.hub_potential = np.hstack(all_hub) if all_hub else None
        self.w_coord = np.hstack(all_w) if all_w else None
        
        print(f"Loaded {len(self.points)} points")
        return self
    
    def load_hubs(self, hubs_file="hubs_history.h5"):
        """Load hub detection history"""
        if os.path.exists(hubs_file):
            with h5py.File(hubs_file, 'r') as f:
                hubs = f['hubs'][:]
            return hubs
        return None


# ------------------------------
# PHYSICAL ANALYZER
# ------------------------------
class SVGAnalyzer:
    """Analyze SVG simulation data"""
    
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def compute_residual_torsion(self):
        """Compute residual torsion field"""
        phase_mean = np.mean(self.data.phase)
        phase_std = np.std(self.data.phase)
        
        # Residual torsion: τ_res = τ * (φ - φ̄)²
        residual_tau = self.data.tau * (self.data.phase - phase_mean)**2
        
        self.results['residual_tau'] = residual_tau
        self.results['phase_mean'] = phase_mean
        self.results['phase_std'] = phase_std
        self.results['tau_mean'] = np.mean(residual_tau)
        self.results['tau_std'] = np.std(residual_tau)
        
        return residual_tau
    
    def compute_power_spectrum(self, n_bins=50):
        """Compute power spectrum of phase fluctuations"""
        # Simple 1D power spectrum approximation
        # In production: use 3D FFT
        
        # Sort points by radius
        r = np.linalg.norm(self.data.points, axis=1)
        sort_idx = np.argsort(r)
        
        phase_sorted = self.data.phase[sort_idx]
        
        # Compute FFT
        fft = np.fft.fft(phase_sorted - np.mean(phase_sorted))
        power = np.abs(fft[:len(fft)//2])**2
        k = np.fft.fftfreq(len(phase_sorted))[:len(fft)//2]
        
        # Bin in k-space
        k_bins = np.logspace(np.log10(k[1]), np.log10(k[-1]), n_bins)
        power_binned = []
        
        for i in range(len(k_bins)-1):
            mask = (k >= k_bins[i]) & (k < k_bins[i+1])
            if np.any(mask):
                power_binned.append(np.mean(power[mask]))
            else:
                power_binned.append(0)
        
        k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
        
        self.results['power_spectrum'] = {
            'k': k_centers.tolist(),
            'power': power_binned
        }
        
        # Fit power law
        valid = np.array(power_binned) > 0
        if np.sum(valid) > 5:
            log_k = np.log(k_centers[valid])
            log_p = np.log(np.array(power_binned)[valid])
            
            slope, intercept = np.polyfit(log_k, log_p, 1)
            self.results['power_law_index'] = -slope  # P(k) ∝ k^{-n}
            self.results['power_law_fit'] = {
                'slope': slope,
                'intercept': intercept
            }
        
        return self.results['power_spectrum']
    
    def detect_hubs_final(self, threshold=0.8):
        """Final hub detection"""
        if self.data.hub_potential is None:
            return None
        
        hub_mask = self.data.hub_potential > threshold
        hub_indices = np.where(hub_mask)[0]
        
        hubs = []
        for idx in hub_indices:
            hubs.append({
                'x': float(self.data.points[idx, 0]),
                'y': float(self.data.points[idx, 1]),
                'z': float(self.data.points[idx, 2]),
                'phase': float(self.data.phase[idx]),
                'potential': float(self.data.hub_potential[idx]),
                'tau': float(self.data.tau[idx])
            })
        
        self.results['hubs'] = hubs
        self.results['n_hubs'] = len(hubs)
        self.results['hub_density'] = len(hubs) / len(self.data.points)
        
        return hubs
    
    def compute_cmb_birefringence(self):
        """Compute synthetic CMB birefringence"""
        # Simplified model: β ∝ ∫ τ · ∇φ dl
        # Here we use a statistical approximation
        
        # Phase gradient magnitude
        if len(self.data.points) > 10000:
            # Use subset for gradient computation
            idx = np.random.choice(len(self.data.points), 10000, replace=False)
            points_sub = self.data.points[idx]
            phase_sub = self.data.phase[idx]
            
            # Build KD-tree
            tree = cKDTree(points_sub)
            
            gradients = []
            for i, p in enumerate(points_sub):
                # Find neighbors
                distances, neighbors = tree.query(p, k=min(10, len(points_sub)))
                if len(neighbors) > 1:
                    # Compute gradient via least squares
                    A = points_sub[neighbors[1:]] - p
                    b = phase_sub[neighbors[1:]] - phase_sub[i]
                    try:
                        grad, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        gradients.append(grad)
                    except:
                        pass
            
            if gradients:
                grad_mag = np.linalg.norm(gradients, axis=1)
                mean_grad = np.mean(grad_mag)
                
                # β ∝ τ · |∇φ|
                beta = np.rad2deg(SVGConstants.TAU * mean_grad * 1e3)  # Scale factor
                self.results['cmb_birefringence'] = float(beta)
                self.results['cmb_birefringence_target'] = SVGConstants.BETA_CMB
                self.results['cmb_error'] = abs(beta - SVGConstants.BETA_CMB)
        
        return self.results.get('cmb_birefringence')
    
    def compute_tully_fisher(self):
        """Compute synthetic Tully-Fisher relation"""
        # Group points into "galaxies" via clustering
        if len(self.data.points) > 100000:
            from sklearn.cluster import KMeans
            
            # Use subset for clustering
            idx = np.random.choice(len(self.data.points), 100000, replace=False)
            points_sub = self.data.points[idx]
            phase_sub = self.data.phase[idx]
            tau_sub = self.data.tau[idx]
            
            # Cluster into "galaxies"
            n_clusters = min(100, len(points_sub) // 1000)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(points_sub)
            
            masses = []
            velocities = []
            
            for i in range(n_clusters):
                cluster_mask = labels == i
                if np.sum(cluster_mask) < 10:
                    continue
                
                # Mass proxy: number of nodes * average torsion
                mass = np.sum(cluster_mask) * np.mean(tau_sub[cluster_mask])
                
                # Velocity proxy: phase gradient
                cluster_points = points_sub[cluster_mask]
                cluster_phase = phase_sub[cluster_mask]
                
                if len(cluster_points) > 1:
                    # Approximate velocity dispersion
                    v_disp = np.std(cluster_phase) * np.mean(tau_sub[cluster_mask])
                    
                    masses.append(mass)
                    velocities.append(v_disp)
            
            if masses and velocities:
                masses = np.array(masses)
                velocities = np.array(velocities)
                
                # Fit Tully-Fisher: v ∝ M^{1/n}
                log_m = np.log(masses)
                log_v = np.log(velocities)
                
                slope, intercept = np.polyfit(log_m, log_v, 1)
                n_tf = 1.0 / slope
                
                self.results['tully_fisher_exponent'] = float(n_tf)
                self.results['tully_fisher_target'] = SVGConstants.N_TF
                self.results['tully_fisher_error'] = abs(n_tf - SVGConstants.N_TF)
        
        return self.results.get('tully_fisher_exponent')
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'statistics': {
                'n_points': len(self.data.points),
                'phase_mean': float(np.mean(self.data.phase)),
                'phase_std': float(np.std(self.data.phase)),
                'tau_mean': float(np.mean(self.data.tau)),
                'tau_std': float(np.std(self.data.tau)),
            },
            'svg_constants': {
                'delta_deg': SVGConstants.DELTA_DEG,
                'tau': float(SVGConstants.TAU),
                'eta_eff': SVGConstants.ETA_EFF,
                'beta_cmb_target': SVGConstants.BETA_CMB,
                'tau_e_target': SVGConstants.TAU_E,
                'n_tf_target': SVGConstants.N_TF,
            },
            'analysis': self.results
        }
        
        return report
    
    def save_report(self, filename='analysis_report.json'):
        """Save analysis report to JSON"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {filename}")


# ------------------------------
# VISUALIZATION
# ------------------------------
class SVGVisualizer:
    """Create visualizations of SVG data"""
    
    def __init__(self, data, results):
        self.data = data
        self.results = results
        
    def plot_phase_distribution(self, save=True):
        """Plot phase distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Phase histogram
        axes[0, 0].hist(self.data.phase, bins=50, density=True, alpha=0.7)
        axes[0, 0].set_xlabel('Phase φ [rad]')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Phase Distribution')
        axes[0, 0].axvline(np.mean(self.data.phase), color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Phase vs radius
        r = np.linalg.norm(self.data.points, axis=1)
        axes[0, 1].scatter(r[::10], self.data.phase[::10], s=1, alpha=0.5)
        axes[0, 1].set_xlabel('Radius r')
        axes[0, 1].set_ylabel('Phase φ')
        axes[0, 1].set_title('Phase vs Radius')
        
        # Torsion distribution
        axes[1, 0].hist(self.data.tau * 1000, bins=50, density=True, alpha=0.7)
        axes[1, 0].set_xlabel('Torsion τ [mT]')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Torsion Distribution')
        axes[1, 0].axvline(SVGConstants.TAU_E * 1000, color='r', linestyle='--', 
                          label=f'Target: {SVGConstants.TAU_E*1000:.1f} mT')
        axes[1, 0].legend()
        
        # Hub potential
        if self.data.hub_potential is not None:
            axes[1, 1].hist(self.data.hub_potential, bins=50, density=True, alpha=0.7)
            axes[1, 1].set_xlabel('Hub Potential')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Hub Potential Distribution')
            axes[1, 1].axvline(0.8, color='r', linestyle='--', label='Hub Threshold')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig('phase_analysis.png', dpi=150)
        plt.show()
    
    def plot_power_spectrum(self, save=True):
        """Plot power spectrum"""
        if 'power_spectrum' not in self.results:
            return
        
        ps = self.results['power_spectrum']
        k = ps['k']
        power = ps['power']
        
        plt.figure(figsize=(10, 6))
        plt.loglog(k, power, 'b-', linewidth=2, label='Simulation')
        
        if 'power_law_index' in self.results:
            n = self.results['power_law_index']
            k_fit = np.array(k)[power > 0]
            power_fit = power[power > 0]
            plt.loglog(k_fit, power_fit[0] * (k_fit/k_fit[0])**(-n), 
                      'r--', label=f'Fit: P(k) ∝ k^{{-{n:.2f}}}')
        
        plt.xlabel('Wavenumber k')
        plt.ylabel('Power P(k)')
        plt.title('Phase Power Spectrum')
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        
        if save:
            plt.savefig('power_spectrum.png', dpi=150)
        plt.show()
    
    def plot_hub_map(self, save=True):
        """Plot hub locations"""
        if 'hubs' not in self.results:
            return
        
        hubs = self.results['hubs']
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all points (subsampled)
        idx = np.random.choice(len(self.data.points), min(10000, len(self.data.points)), replace=False)
        ax.scatter(self.data.points[idx, 0], 
                  self.data.points[idx, 1], 
                  self.data.points[idx, 2], 
                  c=self.data.phase[idx], cmap='viridis', 
                  s=1, alpha=0.3, label='Background')
        
        # Plot hubs
        hub_pos = np.array([[h['x'], h['y'], h['z']] for h in hubs])
        ax.scatter(hub_pos[:, 0], hub_pos[:, 1], hub_pos[:, 2], 
                  c='red', s=50, marker='*', label='Temporal Hubs')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Temporal Hub Distribution')
        ax.legend()
        
        if save:
            plt.savefig('hub_map.png', dpi=150)
        plt.show()


# ------------------------------
# AI SURROGATE MODELS
# ------------------------------
class TorsionPredictor:
    """AI model for torsion field prediction"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        
    def prepare_data(self, data):
        """Prepare data for training"""
        # Features: coordinates, phase, hub potential, w_coord
        X = np.column_stack([
            data.points,
            data.phase,
            data.w_coord if data.w_coord is not None else np.zeros(len(data.points)),
            data.hub_potential if data.hub_potential is not None else np.zeros(len(data.points))
        ])
        
        # Target: torsion
        y = data.tau
        
        return X, y
    
    def train(self, data, test_size=0.2):
        """Train the model"""
        X, y = self.prepare_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if self.model_type == 'random_forest' and SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                n_jobs=-1,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Random Forest Results:")
            print(f"  MSE: {mse:.6e}")
            print(f"  R²: {r2:.4f}")
            
            # Feature importance
            feature_names = ['x', 'y', 'z', 'phase', 'w', 'hub_potential']
            importance = self.model.feature_importances_
            for name, imp in zip(feature_names, importance):
                print(f"  {name}: {imp:.4f}")
            
            return {'mse': mse, 'r2': r2, 'importance': importance}
        
        elif self.model_type == 'neural_network' and TORCH_AVAILABLE:
            return self._train_neural_network(X_train, y_train, X_test, y_test)
        
        else:
            print(f"Model type {self.model_type} not available")
            return None
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train neural network with PyTorch"""
        # Scale data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        X_test_t = torch.FloatTensor(X_test_scaled)
        y_test_t = torch.FloatTensor(y_test).view(-1, 1)
        
        # Create dataset
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024)
        
        # Define network
        class TorsionNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.model = TorsionNet(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train
        n_epochs = 50
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.6e}")
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_t).numpy().flatten()
            y_test_np = y_test_t.numpy().flatten()
            
            mse = mean_squared_error(y_test_np, y_pred)
            r2 = r2_score(y_test_np, y_pred)
            
            print(f"Neural Network Results:")
            print(f"  MSE: {mse:.6e}")
            print(f"  R²: {r2:.4f}")
        
        return {'mse': mse, 'r2': r2}
    
    def predict(self, X):
        """Make predictions with trained model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if self.model_type == 'random_forest':
            return self.model.predict(X_scaled)
        elif self.model_type == 'neural_network' and TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                return self.model(X_tensor).numpy().flatten()
    
    def save(self, filename):
        """Save model to disk"""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model from disk"""
        import joblib
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        print(f"Model loaded from {filename}")


# ------------------------------
# MAIN POST-PROCESSING
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='SVG Post-Processing Pipeline')
    parser.add_argument('--step', type=int, default=9999, help='Time step to process')
    parser.add_argument('--data-dir', type=str, default='output', help='Data directory')
    parser.add_argument('--train-ai', action='store_true', help='Train AI models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--model-type', type=str, default='random_forest', 
                       choices=['random_forest', 'neural_network'])
    args = parser.parse_args()
    
    print("=" * 60)
    print("SVG Post-Processing Pipeline v3.0")
    print(f"Processing step: {args.step}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 60)
    
    # Load data
    loader = SVGDataLoader(data_dir=args.data_dir)
    data = loader.load_step(args.step)
    
    # Analyze
    analyzer = SVGAnalyzer(data)
    
    print("\n--- Computing residual torsion ---")
    residual_tau = analyzer.compute_residual_torsion()
    print(f"  Mean τ_res: {np.mean(residual_tau):.6e}")
    print(f"  Std τ_res: {np.std(residual_tau):.6e}")
    
    print("\n--- Computing power spectrum ---")
    ps = analyzer.compute_power_spectrum()
    if 'power_law_index' in analyzer.results:
        print(f"  Power law index: {analyzer.results['power_law_index']:.3f}")
    
    print("\n--- Detecting temporal hubs ---")
    hubs = analyzer.detect_hubs_final()
    if hubs:
        print(f"  Found {len(hubs)} hubs")
        print(f"  Hub density: {analyzer.results['hub_density']:.6e}")
    
    print("\n--- Computing CMB birefringence ---")
    beta = analyzer.compute_cmb_birefringence()
    if beta:
        print(f"  β_CMB = {beta:.4f}° (target: {SVGConstants.BETA_CMB}°)")
        print(f"  Error: {abs(beta - SVGConstants.BETA_CMB):.4f}°")
    
    print("\n--- Computing Tully-Fisher relation ---")
    n_tf = analyzer.compute_tully_fisher()
    if n_tf:
        print(f"  n_TF = {n_tf:.3f} (target: {SVGConstants.N_TF})")
        print(f"  Error: {abs(n_tf - SVGConstants.N_TF):.3f}")
    
    # Save report
    analyzer.save_report(f'analysis_step{args.step:06d}.json')
    
    # Visualize
    if args.visualize:
        print("\n--- Generating visualizations ---")
        viz = SVGVisualizer(data, analyzer.results)
        viz.plot_phase_distribution()
        viz.plot_power_spectrum()
        viz.plot_hub_map()
    
    # Train AI models
    if args.train_ai:
        print("\n--- Training AI surrogate models ---")
        
        predictor = TorsionPredictor(model_type=args.model_type)
        results = predictor.train(data)
        
        if results:
            predictor.save(f'torsion_predictor_step{args.step:06d}.pkl')
    
    print("\n" + "=" * 60)
    print("Post-processing completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()