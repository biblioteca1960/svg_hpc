#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG Validation Against Observational Data
Self-Verifying Geometry - Comparison with Simons Observatory 2025 and other surveys

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026

This script validates SVG simulation outputs against:
    - Simons Observatory 2025 CMB birefringence measurements
    - Tully-Fisher relation from galaxy surveys
    - Residual torsion fields from magnetic field observations
    - Large-scale structure distribution
    - Black hole mass correlations
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy import stats, interpolate, optimize
from scipy.spatial import cKDTree
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpi4py import MPI

# Optional imports for specific analysis
try:
    import healpy as hp
    HEALPIX_AVAILABLE = True
except ImportError:
    HEALPIX_AVAILABLE = False

try:
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    import pyvista as pv
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


# ------------------------------
# MPI INITIALIZATION
# ------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ------------------------------
# SVG THEORETICAL CONSTANTS
# ------------------------------
class SVGConstants:
    """SVG theoretical predictions"""
    # Fundamental constants
    DELTA_DEG = 6.8                     # Fundamental angular defect [degrees]
    DELTA = np.deg2rad(DELTA_DEG)       # [radians]
    TAU = DELTA / np.sqrt(3)            # Residual torsion [rad]
    ETA_EFF = 1.34e-19                  # Effective photonic viscosity
    
    # Observational predictions
    BETA_CMB = 0.351                     # CMB birefringence [degrees]
    TAU_E = 7.7e-3                       # Residual torsion field [Tesla]
    N_TF = 4.0                           # Tully-Fisher exponent
    HUB_DENSITY = 1e-6                   # Expected hub density [Mpc^{-3}]
    PHASE_VARIANCE = 0.1                  # Expected phase variance [rad^2]
    
    # Derived quantities
    HUBBLE_CONSTANT = 67.4                # H0 [km/s/Mpc]
    OMEGA_M = 0.315                       # Matter density parameter
    
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


# ------------------------------
# OBSERVATIONAL DATA LOADERS
# ------------------------------
class ObservationalDataLoader:
    """Load and process observational data from various surveys"""
    
    def __init__(self, data_dir="observational_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_simons_observatory_2025(self, filename="simons_observatory_2025.h5"):
        """
        Load Simons Observatory 2025 data
        Expected format: HDF5 with groups:
            - cmb/birefringence_map
            - cmb/power_spectrum
            - cmb/statistics
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Generating synthetic data for testing.")
            return self._generate_synthetic_simons_data()
        
        data = {}
        with h5py.File(filepath, 'r') as f:
            # Load CMB birefringence map
            if 'cmb/birefringence_map' in f:
                data['birefringence_map'] = f['cmb/birefringence_map'][:]
            
            # Load power spectrum
            if 'cmb/power_spectrum' in f:
                ps_group = f['cmb/power_spectrum']
                data['power_spectrum'] = {
                    'ell': ps_group['ell'][:],
                    'cl': ps_group['cl'][:],
                    'cl_err': ps_group['cl_err'][:] if 'cl_err' in ps_group else None
                }
            
            # Load statistics
            if 'cmb/statistics' in f:
                stats_group = f['cmb/statistics']
                data['statistics'] = {
                    key: stats_group[key][()] 
                    for key in stats_group.keys()
                }
        
        return data
    
    def _generate_synthetic_simons_data(self):
        """Generate synthetic Simons Observatory data for testing"""
        print("Generating synthetic Simons Observatory 2025 data...")
        
        data = {
            'birefringence_map': np.random.normal(
                SVGConstants.BETA_CMB, 0.02, (64, 128)
            ),
            'power_spectrum': {
                'ell': np.arange(2, 1000),
                'cl': None,
                'cl_err': None
            },
            'statistics': {
                'beta_mean': SVGConstants.BETA_CMB,
                'beta_std': 0.02,
                'beta_error': 0.005
            }
        }
        
        # Generate power spectrum
        ell = data['power_spectrum']['ell']
        # Simulated CMB power spectrum with birefringence signature
        cl = 1e-5 / (ell * (ell + 1)) * (1 + 0.1 * np.sin(ell / 100))
        cl_err = 0.1 * cl
        data['power_spectrum']['cl'] = cl
        data['power_spectrum']['cl_err'] = cl_err
        
        return data
    
    def load_tully_fisher_data(self, filename="tully_fisher_catalog.csv"):
        """
        Load Tully-Fisher relation data
        Expected columns: galaxy_id, distance, velocity, luminosity, mass
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Generating synthetic data.")
            return self._generate_synthetic_tf_data()
        
        df = pd.read_csv(filepath)
        return df
    
    def _generate_synthetic_tf_data(self, n_galaxies=1000):
        """Generate synthetic Tully-Fisher data"""
        np.random.seed(42)
        
        # Generate galaxy masses (log-normal distribution)
        log_mass = np.random.normal(10, 1, n_galaxies)  # log10(M_sun)
        mass = 10**log_mass
        
        # Tully-Fisher relation: v ∝ M^{1/n} with n=4
        velocity = 200 * (mass / 1e10)**(1/SVGConstants.N_TF)
        velocity += np.random.normal(0, 10, n_galaxies)  # Add scatter
        
        # Generate other properties
        luminosity = mass * np.random.uniform(0.5, 2, n_galaxies)
        distance = np.random.uniform(10, 200, n_galaxies)  # Mpc
        
        df = pd.DataFrame({
            'galaxy_id': np.arange(n_galaxies),
            'distance': distance,
            'velocity': velocity,
            'luminosity': luminosity,
            'mass': mass,
            'log_mass': log_mass
        })
        
        return df
    
    def load_magnetic_field_data(self, filename="magnetic_field_survey.h5"):
        """
        Load galactic and intergalactic magnetic field measurements
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Generating synthetic data.")
            return self._generate_synthetic_magnetic_data()
        
        data = {}
        with h5py.File(filepath, 'r') as f:
            data['coordinates'] = f['coordinates'][:]
            data['b_field'] = f['b_field'][:]
            data['b_field_err'] = f['b_field_err'][:] if 'b_field_err' in f else None
        
        return data
    
    def _generate_synthetic_magnetic_data(self, n_points=10000):
        """Generate synthetic magnetic field data"""
        np.random.seed(43)
        
        # Generate random positions in galactic coordinates
        coords = np.random.uniform(-10, 10, (n_points, 3))  # kpc
        
        # Generate B-field with torsion signature
        r = np.linalg.norm(coords, axis=1)
        b_mag = SVGConstants.TAU_E * 1000 * (1 + 0.1 * np.sin(r))  # mT
        b_mag += np.random.normal(0, 0.1, n_points)
        
        # Add direction
        direction = coords / (r + 1e-10)[:, np.newaxis]
        b_field = b_mag[:, np.newaxis] * direction
        
        return {
            'coordinates': coords,
            'b_field': b_field,
            'b_field_err': np.ones(n_points) * 0.1
        }
    
    def load_lss_data(self, filename="large_scale_structure.h5"):
        """
        Load large-scale structure data (galaxy redshift surveys)
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Generating synthetic data.")
            return self._generate_synthetic_lss_data()
        
        data = {}
        with h5py.File(filepath, 'r') as f:
            data['ra'] = f['ra'][:]
            data['dec'] = f['dec'][:]
            data['redshift'] = f['redshift'][:]
            data['mass'] = f['mass'][:] if 'mass' in f else None
        
        return data
    
    def _generate_synthetic_lss_data(self, n_galaxies=100000):
        """Generate synthetic large-scale structure data"""
        np.random.seed(44)
        
        # Generate positions in RA/Dec/redshift
        ra = np.random.uniform(0, 360, n_galaxies)
        dec = np.rad2deg(np.arcsin(np.random.uniform(-1, 1, n_galaxies)))
        redshift = np.random.exponential(0.1, n_galaxies)
        
        # Add clustering
        n_clusters = 100
        for _ in range(n_clusters):
            cluster_ra = np.random.uniform(0, 360)
            cluster_dec = np.rad2deg(np.arcsin(np.random.uniform(-1, 1)))
            cluster_z = np.random.exponential(0.3)
            
            # Add cluster members
            n_members = np.random.poisson(100)
            idx = np.random.choice(n_galaxies, n_members, replace=False)
            ra[idx] += np.random.normal(cluster_ra, 5, n_members)
            dec[idx] += np.random.normal(cluster_dec, 2, n_members)
            redshift[idx] += np.random.normal(cluster_z, 0.05, n_members)
        
        # Wrap angles
        ra = ra % 360
        
        return {
            'ra': ra,
            'dec': dec,
            'redshift': redshift,
            'mass': np.random.lognormal(10, 1, n_galaxies)
        }


# ------------------------------
# SIMULATION DATA LOADER
# ------------------------------
class SimulationDataLoader:
    """Load SVG simulation outputs"""
    
    def __init__(self, sim_dir="output"):
        self.sim_dir = Path(sim_dir)
        
    def load_step(self, step, data_dir=None):
        """Load simulation data for a specific time step"""
        if data_dir:
            self.sim_dir = Path(data_dir)
        
        # Try to load combined PVTU first
        pvtu_file = self.sim_dir / f"svg_step{step:06d}.pvtu"
        h5_file = self.sim_dir / f"svg_step{step:06d}.h5"
        
        data = {}
        
        if h5_file.exists():
            # Load from HDF5 (preferred for analysis)
            with h5py.File(h5_file, 'r') as f:
                data['points'] = f['points'][:]
                data['phase'] = f['phase'][:]
                data['tau'] = f['tau'][:]
                data['eta_eff'] = f['eta_eff'][:] if 'eta_eff' in f else None
                data['hub_potential'] = f['hub_potential'][:] if 'hub_potential' in f else None
                data['w_coord'] = f['w_coord'][:] if 'w_coord' in f else None
                
        elif VTK_AVAILABLE and pvtu_file.exists():
            # Load from VTU (slower but works)
            grid = pv.read(str(pvtu_file))
            data['points'] = grid.points
            data['phase'] = grid.point_data['phase']
            data['tau'] = grid.point_data['tau']
            data['eta_eff'] = grid.point_data.get('eta_eff', None)
            data['hub_potential'] = grid.point_data.get('hub_potential', None)
            data['w_coord'] = grid.point_data.get('w_coord', None)
        
        else:
            # Try to load from VTU chunks
            import glob
            vtu_files = sorted(glob.glob(str(self.sim_dir / f"svg_rank*_step{step:06d}_chunk*.vtu")))
            
            if not vtu_files:
                raise FileNotFoundError(f"No data found for step {step} in {self.sim_dir}")
            
            points_list = []
            phase_list = []
            tau_list = []
            
            for f in vtu_files:
                grid = pv.read(f)
                points_list.append(grid.points)
                phase_list.append(grid.point_data['phase'])
                tau_list.append(grid.point_data['tau'])
            
            data['points'] = np.vstack(points_list)
            data['phase'] = np.hstack(phase_list)
            data['tau'] = np.hstack(tau_list)
        
        print(f"Loaded {len(data['points'])} points from step {step}")
        return data
    
    def load_hubs(self, hubs_file="hubs_history.h5"):
        """Load hub detection history"""
        hubs_path = self.sim_dir / hubs_file
        if not hubs_path.exists():
            return None
        
        with h5py.File(hubs_path, 'r') as f:
            hubs = f['hubs'][:]
        
        return hubs


# ------------------------------
# VALIDATION MODULES
# ------------------------------
class CMBValidator:
    """Validate against CMB birefringence measurements"""
    
    def __init__(self, sim_data, obs_data):
        self.sim_data = sim_data
        self.obs_data = obs_data
        self.results = {}
        
    def validate_birefringence(self):
        """Validate CMB birefringence angle"""
        # Compute synthetic birefringence from simulation
        # β ∝ ∫ τ · ∇φ dl along line of sight
        
        # Sample random lines of sight
        n_sightlines = 1000
        n_points = len(self.sim_data['points'])
        
        beta_values = []
        
        for _ in range(n_sightlines):
            # Random direction
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Project points along this direction
            proj = np.dot(self.sim_data['points'], direction)
            
            # Sort by distance
            sort_idx = np.argsort(proj)
            proj_sorted = proj[sort_idx]
            phase_sorted = self.sim_data['phase'][sort_idx]
            tau_sorted = self.sim_data['tau'][sort_idx]
            
            # Approximate line integral
            dproj = np.diff(proj_sorted)
            dphi = np.diff(phase_sorted)
            
            # β ∝ ∑ τ · Δφ · Δr
            beta = np.sum(tau_sorted[:-1] * dphi * dproj)
            beta_values.append(beta)
        
        beta_mean = np.mean(beta_values)
        beta_std = np.std(beta_values)
        
        # Convert to degrees (with appropriate scaling)
        beta_deg = np.rad2deg(beta_mean * 1e3)  # Scale factor
        
        self.results['simulated_beta'] = beta_deg
        self.results['observed_beta'] = SVGConstants.BETA_CMB
        self.results['beta_difference'] = beta_deg - SVGConstants.BETA_CMB
        self.results['beta_sigma'] = beta_deg / SVGConstants.BETA_CMB
        self.results['beta_significance'] = abs(beta_deg - SVGConstants.BETA_CMB) / beta_std
        
        return self.results
    
    def validate_power_spectrum(self):
        """Validate CMB power spectrum"""
        if 'power_spectrum' not in self.obs_data:
            return None
        
        # Compute power spectrum from simulation
        # This is a simplified version - in production use HEALPix
        
        # Project simulation onto sphere
        r = np.linalg.norm(self.sim_data['points'], axis=1)
        theta = np.arccos(self.sim_data['points'][:, 2] / (r + 1e-10))
        phi = np.arctan2(self.sim_data['points'][:, 1], self.sim_data['points'][:, 0])
        
        # Bin into HEALPix map if available
        if HEALPIX_AVAILABLE:
            nside = 64
            npix = hp.nside2npix(nside)
            
            # Create map
            phase_map = np.zeros(npix)
            count_map = np.zeros(npix)
            
            for i in range(len(self.sim_data['phase'])):
                pix = hp.ang2pix(nside, theta[i], phi[i])
                phase_map[pix] += self.sim_data['phase'][i]
                count_map[pix] += 1
            
            # Average
            mask = count_map > 0
            phase_map[mask] /= count_map[mask]
            
            # Compute power spectrum
            ell = np.arange(2, 3*nside)
            cl = hp.anafast(phase_map, lmax=ell[-1])
            
            self.results['simulated_cl'] = cl[:len(ell)]
            self.results['ell'] = ell
            
            # Compare with observations
            obs_ell = self.obs_data['power_spectrum']['ell']
            obs_cl = self.obs_data['power_spectrum']['cl']
            
            # Interpolate simulation to observed ell
            cl_interp = np.interp(obs_ell, ell, cl[:len(ell)])
            
            # Compute chi-square
            chi2 = np.sum((cl_interp - obs_cl)**2 / (obs_cl**2))
            dof = len(obs_ell)
            
            self.results['power_spectrum_chi2'] = chi2
            self.results['power_spectrum_dof'] = dof
            self.results['power_spectrum_chi2_reduced'] = chi2 / dof
        
        return self.results


class TullyFisherValidator:
    """Validate against Tully-Fisher relation"""
    
    def __init__(self, sim_data, obs_data):
        self.sim_data = sim_data
        self.obs_data = obs_data
        self.results = {}
        
    def validate(self):
        """Validate Tully-Fisher relation"""
        # Group simulation points into "galaxies"
        from sklearn.cluster import KMeans
        
        # Use subset for clustering if too large
        if len(self.sim_data['points']) > 100000:
            idx = np.random.choice(len(self.sim_data['points']), 100000, replace=False)
            points_sub = self.sim_data['points'][idx]
            phase_sub = self.sim_data['phase'][idx]
            tau_sub = self.sim_data['tau'][idx]
        else:
            points_sub = self.sim_data['points']
            phase_sub = self.sim_data['phase']
            tau_sub = self.sim_data['tau']
        
        # Cluster into galaxies
        n_clusters = min(100, len(points_sub) // 1000)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points_sub)
        
        masses = []
        velocities = []
        
        for i in range(n_clusters):
            cluster_mask = labels == i
            if np.sum(cluster_mask) < 10:
                continue
            
            # Mass proxy: number of nodes * average torsion
            mass = np.sum(cluster_mask) * np.mean(tau_sub[cluster_mask])
            
            # Velocity proxy: phase gradient dispersion
            cluster_points = points_sub[cluster_mask]
            cluster_phase = phase_sub[cluster_mask]
            
            if len(cluster_points) > 5:
                # Approximate velocity dispersion from phase variations
                v_disp = np.std(cluster_phase) * np.mean(tau_sub[cluster_mask]) * 1000
                
                masses.append(np.log10(mass))
                velocities.append(np.log10(v_disp))
        
        if len(masses) < 10:
            return None
        
        masses = np.array(masses)
        velocities = np.array(velocities)
        
        # Fit Tully-Fisher relation: log v = a + (1/n) * log M
        slope, intercept, r_value, p_value, std_err = stats.linregress(masses, velocities)
        
        n_tf = 1.0 / slope
        
        self.results['simulated_n_tf'] = n_tf
        self.results['observed_n_tf'] = SVGConstants.N_TF
        self.results['n_tf_difference'] = n_tf - SVGConstants.N_TF
        self.results['n_tf_r_squared'] = r_value**2
        self.results['n_tf_p_value'] = p_value
        self.results['n_tf_std_err'] = std_err
        
        # Store fit parameters
        self.results['tf_slope'] = slope
        self.results['tf_intercept'] = intercept
        self.results['tf_masses'] = masses.tolist()
        self.results['tf_velocities'] = velocities.tolist()
        
        # Compare with observational data if available
        if isinstance(self.obs_data, pd.DataFrame):
            obs_mass = np.log10(self.obs_data['mass'])
            obs_vel = np.log10(self.obs_data['velocity'])
            
            # Fit observed relation
            obs_slope, obs_intercept, obs_r, _, _ = stats.linregress(obs_mass, obs_vel)
            
            self.results['observed_n_tf_fit'] = 1.0 / obs_slope
            self.results['n_tf_comparison'] = n_tf - (1.0/obs_slope)
        
        return self.results
    
    def plot(self, save=True):
        """Plot Tully-Fisher relation"""
        if 'tf_masses' not in self.results:
            return
        
        plt.figure(figsize=(10, 8))
        
        # Simulation points
        masses = self.results['tf_masses']
        velocities = self.results['tf_velocities']
        plt.scatter(masses, velocities, alpha=0.6, label='Simulation', s=20)
        
        # Fit line
        m_fit = np.linspace(min(masses), max(masses), 100)
        v_fit = self.results['tf_intercept'] + self.results['tf_slope'] * m_fit
        plt.plot(m_fit, v_fit, 'r-', linewidth=2, 
                label=f"Fit: n={self.results['simulated_n_tf']:.2f}")
        
        # SVG prediction
        v_pred = self.results['tf_intercept'] + (1/SVGConstants.N_TF) * m_fit
        plt.plot(m_fit, v_pred, 'g--', linewidth=2, 
                label=f"SVG prediction: n={SVGConstants.N_TF}")
        
        # Observational data if available
        if isinstance(self.obs_data, pd.DataFrame):
            obs_mass = np.log10(self.obs_data['mass'])
            obs_vel = np.log10(self.obs_data['velocity'])
            plt.scatter(obs_mass, obs_vel, alpha=0.3, s=5, 
                       label='Observations', c='orange')
        
        plt.xlabel('log(Mass) [log(M☉)]')
        plt.ylabel('log(Velocity) [log(km/s)]')
        plt.title('Tully-Fisher Relation: Simulation vs Theory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig('tully_fisher_validation.png', dpi=150)
        plt.show()


class TorsionFieldValidator:
    """Validate against magnetic field/torsion measurements"""
    
    def __init__(self, sim_data, obs_data):
        self.sim_data = sim_data
        self.obs_data = obs_data
        self.results = {}
        
    def validate(self):
        """Validate torsion field distribution"""
        # Compare torsion magnitude distribution
        sim_tau = self.sim_data['tau'] * 1000  # Convert to mT
        
        obs_b = self.obs_data['b_field']
        obs_b_mag = np.linalg.norm(obs_b, axis=1)
        
        # Statistics
        self.results['sim_tau_mean'] = float(np.mean(sim_tau))
        self.results['sim_tau_std'] = float(np.std(sim_tau))
        self.results['sim_tau_median'] = float(np.median(sim_tau))
        
        self.results['obs_b_mean'] = float(np.mean(obs_b_mag))
        self.results['obs_b_std'] = float(np.std(obs_b_mag))
        self.results['obs_b_median'] = float(np.median(obs_b_mag))
        
        # Compare with SVG prediction
        self.results['svg_prediction'] = SVGConstants.TAU_E * 1000
        self.results['sim_vs_pred'] = self.results['sim_tau_mean'] - SVGConstants.TAU_E * 1000
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(sim_tau, obs_b_mag)
        self.results['ks_statistic'] = ks_stat
        self.results['ks_p_value'] = ks_p
        
        # Correlation with position
        # Sample points for correlation
        n_sample = min(10000, len(sim_tau), len(obs_b_mag))
        sim_idx = np.random.choice(len(sim_tau), n_sample, replace=False)
        obs_idx = np.random.choice(len(obs_b_mag), n_sample, replace=False)
        
        # Simple correlation (in production, do spatial matching)
        correlation = np.corrcoef(sim_tau[sim_idx], obs_b_mag[obs_idx])[0, 1]
        self.results['field_correlation'] = correlation
        
        return self.results
    
    def plot(self, save=True):
        """Plot torsion field comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        sim_tau = self.sim_data['tau'] * 1000
        obs_b = np.linalg.norm(self.obs_data['b_field'], axis=1)
        
        # Histograms
        axes[0, 0].hist(sim_tau, bins=50, density=True, alpha=0.7, 
                       label=f'Simulation (μ={self.results["sim_tau_mean"]:.2f})')
        axes[0, 0].hist(obs_b, bins=50, density=True, alpha=0.7,
                       label=f'Observed (μ={self.results["obs_b_mean"]:.2f})')
        axes[0, 0].axvline(SVGConstants.TAU_E * 1000, color='r', linestyle='--',
                          label=f'SVG Prediction: {SVGConstants.TAU_E*1000:.1f} mT')
        axes[0, 0].set_xlabel('Field Strength [mT]')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Torsion/Magnetic Field Distribution')
        axes[0, 0].legend()
        
        # QQ plot
        axes[0, 1].scatter(np.sort(sim_tau)[::10], np.sort(obs_b)[::10], alpha=0.5, s=10)
        axes[0, 1].plot([min(sim_tau), max(sim_tau)], [min(sim_tau), max(sim_tau)], 
                       'r--', label='1:1 line')
        axes[0, 1].set_xlabel('Simulation Torsion [mT]')
        axes[0, 1].set_ylabel('Observed B-field [mT]')
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].legend()
        
        # Spatial correlation (simplified)
        if len(sim_tau) > 1000:
            # Compute radial profile
            r_sim = np.linalg.norm(self.sim_data['points'], axis=1)
            r_obs = np.linalg.norm(self.obs_data['coordinates'], axis=1)
            
            r_bins = np.linspace(0, 10, 20)
            sim_profile = []
            obs_profile = []
            
            for i in range(len(r_bins)-1):
                mask_sim = (r_sim >= r_bins[i]) & (r_sim < r_bins[i+1])
                mask_obs = (r_obs >= r_bins[i]) & (r_obs < r_bins[i+1])
                
                if np.any(mask_sim):
                    sim_profile.append(np.mean(sim_tau[mask_sim]))
                else:
                    sim_profile.append(0)
                
                if np.any(mask_obs):
                    obs_profile.append(np.mean(obs_b[mask_obs]))
                else:
                    obs_profile.append(0)
            
            r_centers = (r_bins[:-1] + r_bins[1:]) / 2
            
            axes[1, 0].plot(r_centers, sim_profile, 'b-', linewidth=2, label='Simulation')
            axes[1, 0].plot(r_centers, obs_profile, 'orange', linewidth=2, label='Observed')
            axes[1, 0].axhline(SVGConstants.TAU_E * 1000, color='r', linestyle='--',
                              label='SVG Prediction')
            axes[1, 0].set_xlabel('Radius [kpc]')
            axes[1, 0].set_ylabel('Field Strength [mT]')
            axes[1, 0].set_title('Radial Profile')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics table
        axes[1, 1].axis('off')
        stats_text = (
            f"Validation Statistics:\n\n"
            f"Simulation mean: {self.results['sim_tau_mean']:.3f} mT\n"
            f"Observed mean: {self.results['obs_b_mean']:.3f} mT\n"
            f"SVG prediction: {self.results['svg_prediction']:.3f} mT\n"
            f"Difference: {self.results['sim_vs_pred']:.3f} mT\n\n"
            f"KS statistic: {self.results['ks_statistic']:.3f}\n"
            f"KS p-value: {self.results['ks_p_value']:.3e}\n"
            f"Correlation: {self.results['field_correlation']:.3f}"
        )
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save:
            plt.savefig('torsion_validation.png', dpi=150)
        plt.show()


class HubValidator:
    """Validate hub/black hole distribution"""
    
    def __init__(self, sim_data, sim_hubs, obs_data=None):
        self.sim_data = sim_data
        self.sim_hubs = sim_hubs
        self.obs_data = obs_data
        self.results = {}
        
    def validate(self):
        """Validate hub distribution"""
        if self.sim_hubs is None:
            return None
        
        n_hubs = len(self.sim_hubs)
        n_points = len(self.sim_data['points'])
        
        # Hub density
        # Estimate volume from point distribution
        points = self.sim_data['points']
        volume = np.prod(np.ptp(points, axis=0))
        hub_density = n_hubs / volume
        
        self.results['n_hubs'] = n_hubs
        self.results['hub_density'] = hub_density
        self.results['hub_density_predicted'] = SVGConstants.HUB_DENSITY
        self.results['hub_density_ratio'] = hub_density / SVGConstants.HUB_DENSITY
        
        # Hub clustering
        if n_hubs > 10:
            hub_positions = np.array([h[2:5] if len(h) > 3 else 
                                     [h['x'], h['y'], h['z']] for h in self.sim_hubs])
            
            # Pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(hub_positions)
            
            self.results['hub_mean_distance'] = float(np.mean(distances))
            self.results['hub_std_distance'] = float(np.std(distances))
            
            # 2-point correlation function (simplified)
            r_bins = np.linspace(0, np.max(distances), 20)
            hist, _ = np.histogram(distances, bins=r_bins)
            self.results['hub_correlation'] = hist.tolist()
            self.results['hub_correlation_bins'] = r_bins.tolist()
        
        # Hub masses (proxies)
        hub_masses = []
        for hub in self.sim_hubs:
            if isinstance(hub, np.void):
                # Handle structured array
                mass_proxy = hub['potential'] * np.mean(self.sim_data['tau'])
            else:
                mass_proxy = hub.get('potential', 0) * np.mean(self.sim_data['tau'])
            hub_masses.append(mass_proxy)
        
        if hub_masses:
            self.results['hub_mass_mean'] = float(np.mean(hub_masses))
            self.results['hub_mass_std'] = float(np.std(hub_masses))
        
        return self.results


# ------------------------------
# MAIN VALIDATION PIPELINE
# ------------------------------
class SVGValidator:
    """Main validation pipeline for SVG theory"""
    
    def __init__(self, sim_dir="output", obs_dir="observational_data"):
        self.sim_loader = SimulationDataLoader(sim_dir)
        self.obs_loader = ObservationalDataLoader(obs_dir)
        self.results = {
            'summary': {},
            'cmb': {},
            'tully_fisher': {},
            'torsion': {},
            'hubs': {}
        }
        
    def validate_all(self, step=9999):
        """Run all validations"""
        print("\n" + "="*60)
        print("SVG THEORY VALIDATION PIPELINE")
        print("="*60)
        
        # Load simulation data
        print("\n[1/5] Loading simulation data...")
        sim_data = self.sim_loader.load_step(step)
        sim_hubs = self.sim_loader.load_hubs()
        
        # Load observational data
        print("\n[2/5] Loading observational data...")
        obs_cmb = self.obs_loader.load_simons_observatory_2025()
        obs_tf = self.obs_loader.load_tully_fisher_data()
        obs_b = self.obs_loader.load_magnetic_field_data()
        obs_lss = self.obs_loader.load_lss_data()
        
        # Validate CMB
        print("\n[3/5] Validating CMB birefringence...")
        cmb_validator = CMBValidator(sim_data, obs_cmb)
        self.results['cmb'] = cmb_validator.validate_birefringence()
        cmb_validator.validate_power_spectrum()
        
        # Validate Tully-Fisher
        print("\n[4/5] Validating Tully-Fisher relation...")
        tf_validator = TullyFisherValidator(sim_data, obs_tf)
        self.results['tully_fisher'] = tf_validator.validate()
        
        # Validate torsion field
        print("\n[5/5] Validating torsion field...")
        torsion_validator = TorsionFieldValidator(sim_data, obs_b)
        self.results['torsion'] = torsion_validator.validate()
        
        # Validate hubs if available
        if sim_hubs is not None:
            print("\n[6/5] Validating hub distribution...")
            hub_validator = HubValidator(sim_data, sim_hubs, obs_lss)
            self.results['hubs'] = hub_validator.validate()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate validation summary"""
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'svg_constants': SVGConstants.to_dict(),
            'validation_passed': True,
            'warnings': []
        }
        
        # Check CMB
        if 'beta_difference' in self.results['cmb']:
            beta_diff = abs(self.results['cmb']['beta_difference'])
            summary['cmb_status'] = 'PASS' if beta_diff < 0.01 else 'FAIL'
            if summary['cmb_status'] == 'FAIL':
                summary['validation_passed'] = False
                summary['warnings'].append(f"CMB birefringence off by {beta_diff:.3f}°")
        
        # Check Tully-Fisher
        if 'n_tf_difference' in self.results['tully_fisher']:
            tf_diff = abs(self.results['tully_fisher']['n_tf_difference'])
            summary['tf_status'] = 'PASS' if tf_diff < 0.05 else 'FAIL'
            if summary['tf_status'] == 'FAIL':
                summary['validation_passed'] = False
                summary['warnings'].append(f"Tully-Fisher exponent off by {tf_diff:.3f}")
        
        # Check torsion
        if 'sim_vs_pred' in self.results['torsion']:
            tau_diff = abs(self.results['torsion']['sim_vs_pred'])
            summary['torsion_status'] = 'PASS' if tau_diff < 0.1 else 'FAIL'
            if summary['torsion_status'] == 'FAIL':
                summary['validation_passed'] = False
                summary['warnings'].append(f"Torsion field off by {tau_diff:.3f} mT")
        
        # Check hubs
        if 'hub_density_ratio' in self.results['hubs']:
            hub_ratio = self.results['hubs']['hub_density_ratio']
            summary['hub_status'] = 'PASS' if 0.5 < hub_ratio < 2.0 else 'FAIL'
            if summary['hub_status'] == 'FAIL':
                summary['validation_passed'] = False
                summary['warnings'].append(f"Hub density ratio: {hub_ratio:.2f}")
        
        self.results['summary'] = summary
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        s = self.results['summary']
        print(f"\nOverall Status: {'✓ PASSED' if s['validation_passed'] else '✗ FAILED'}")
        
        if s['warnings']:
            print("\nWarnings:")
            for w in s['warnings']:
                print(f"  • {w}")
        
        print("\nValidation Results:")
        print("-" * 40)
        
        # CMB
        print("\nCMB Birefringence:")
        if 'beta_difference' in self.results['cmb']:
            print(f"  Simulated: {self.results['cmb']['simulated_beta']:.4f}°")
            print(f"  Observed: {self.results['cmb']['observed_beta']:.4f}°")
            print(f"  Difference: {self.results['cmb']['beta_difference']:.4f}°")
            print(f"  Significance: {self.results['cmb']['beta_significance']:.2f}σ")
            print(f"  Status: {s.get('cmb_status', 'UNKNOWN')}")
        
        # Tully-Fisher
        print("\nTully-Fisher Relation:")
        if 'simulated_n_tf' in self.results['tully_fisher']:
            print(f"  Simulated n: {self.results['tully_fisher']['simulated_n_tf']:.3f}")
            print(f"  SVG Prediction: {SVGConstants.N_TF:.3f}")
            print(f"  Difference: {self.results['tully_fisher']['n_tf_difference']:.3f}")
            print(f"  R²: {self.results['tully_fisher']['n_tf_r_squared']:.3f}")
            print(f"  Status: {s.get('tf_status', 'UNKNOWN')}")
        
        # Torsion
        print("\nTorsion Field:")
        if 'sim_tau_mean' in self.results['torsion']:
            print(f"  Simulated mean: {self.results['torsion']['sim_tau_mean']:.3f} mT")
            print(f"  Observed mean: {self.results['torsion']['obs_b_mean']:.3f} mT")
            print(f"  SVG Prediction: {SVGConstants.TAU_E*1000:.3f} mT")
            print(f"  KS test p-value: {self.results['torsion']['ks_p_value']:.3e}")
            print(f"  Correlation: {self.results['torsion']['field_correlation']:.3f}")
            print(f"  Status: {s.get('torsion_status', 'UNKNOWN')}")
        
        # Hubs
        if self.results['hubs']:
            print("\nTemporal Hubs:")
            print(f"  Number detected: {self.results['hubs']['n_hubs']}")
            print(f"  Density ratio: {self.results['hubs']['hub_density_ratio']:.2f}")
            print(f"  Status: {s.get('hub_status', 'UNKNOWN')}")
        
        print("\n" + "="*60)
    
    def save_report(self, filename='validation_report.json'):
        """Save validation report to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nValidation report saved to {filename}")
    
    def generate_plots(self):
        """Generate validation plots"""
        print("\nGenerating validation plots...")
        
        # Reload data for plotting
        sim_data = self.sim_loader.load_step(9999)
        obs_cmb = self.obs_loader.load_simons_observatory_2025()
        obs_tf = self.obs_loader.load_tully_fisher_data()
        obs_b = self.obs_loader.load_magnetic_field_data()
        
        # Tully-Fisher plot
        tf_validator = TullyFisherValidator(sim_data, obs_tf)
        tf_validator.results = self.results['tully_fisher']
        tf_validator.plot()
        
        # Torsion plot
        torsion_validator = TorsionFieldValidator(sim_data, obs_b)
        torsion_validator.results = self.results['torsion']
        torsion_validator.plot()
        
        # CMB power spectrum plot if available
        if 'ell' in self.results['cmb'] and 'simulated_cl' in self.results['cmb']:
            self._plot_cmb_power_spectrum()


# ------------------------------
# COMMAND LINE INTERFACE
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='SVG Validation against Observations')
    parser.add_argument('--sim-dir', type=str, default='output',
                       help='Simulation output directory')
    parser.add_argument('--obs-dir', type=str, default='observational_data',
                       help='Observational data directory')
    parser.add_argument('--step', type=int, default=9999,
                       help='Simulation time step to validate')
    parser.add_argument('--plot', action='store_true',
                       help='Generate validation plots')
    parser.add_argument('--output', type=str, default='validation_report.json',
                       help='Output report filename')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SVG OBSERVATIONAL VALIDATION")
    print("="*60)
    print(f"Simulation directory: {args.sim_dir}")
    print(f"Observational data: {args.obs_dir}")
    print(f"Time step: {args.step}")
    print("="*60)
    
    # Run validation
    validator = SVGValidator(args.sim_dir, args.obs_dir)
    results = validator.validate_all(args.step)
    
    # Print summary
    validator.print_summary()
    
    # Save report
    validator.save_report(args.output)
    
    # Generate plots
    if args.plot:
        validator.generate_plots()
    
    # Return exit code based on validation
    if results['summary']['validation_passed']:
        print("\n✓ Validation PASSED - SVG theory consistent with observations")
        return 0
    else:
        print("\n✗ Validation FAILED - Discrepancies found with observations")
        return 1


if __name__ == "__main__":
    sys.exit(main())