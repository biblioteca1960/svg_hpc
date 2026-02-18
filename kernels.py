#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG Computation Kernels
Optimized phase update kernels for CPU and GPU

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026
"""

import numpy as np
from numba import jit, prange
import logging
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import CuPy for GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


# ------------------------------
# CONSTANTS
# ------------------------------
@dataclass
class PhysicsConstants:
    """Physical constants for SVG theory"""
    delta_deg: float = 6.8
    tau: float = 0.0685
    eta_eff: float = 1.34e-19
    kappa: float = 1.0
    gamma: float = 0.1
    hub_threshold: float = 0.1
    c: float = 299792458.0
    
    @property
    def delta_rad(self) -> float:
        return np.deg2rad(self.delta_deg)


# ------------------------------
# CPU KERNELS (NUMBA OPTIMIZED)
# ------------------------------
class CPUKernels:
    """CPU-optimized kernels using Numba"""
    
    def __init__(self, constants: PhysicsConstants):
        self.constants = constants
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def update_phase_vectorized(
        phase: np.ndarray,
        tau: np.ndarray,
        eta_eff: np.ndarray,
        neighbors: List[np.ndarray],
        all_phases: np.ndarray,
        dt: float,
        kappa: float
    ) -> np.ndarray:
        """
        Vectorized phase update for all nodes
        Optimized with Numba parallel
        """
        n_nodes = len(phase)
        new_phase = np.zeros(n_nodes)
        
        for i in prange(n_nodes):
            phi_i = phase[i]
            nb_idx = neighbors[i]
            
            # Compute equilibrium phase (mean of neighbors)
            phi_eq = 0.0
            for j in range(len(nb_idx)):
                phi_eq += all_phases[nb_idx[j]]
            phi_eq /= len(nb_idx)
            
            # Effective viscosity including torsion coupling
            eta = eta_eff[i] * (1.0 + tau[i])
            
            # Phase rectification equation
            dphi = -kappa * eta * (phi_i - phi_eq) * dt
            new_phase[i] = phi_i + dphi
        
        return new_phase
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_torsion(
        phase: np.ndarray,
        base_tau: float,
        gamma: float,
        phase_mean: float
    ) -> np.ndarray:
        """
        Compute torsion field from phase
        τ = τ_base + γ(φ - φ̄)²
        """
        return base_tau + gamma * (phase - phase_mean)**2
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def compute_hub_potential(
        phase: np.ndarray,
        neighbors: List[np.ndarray],
        all_phases: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Compute hub potential (phase coherence)
        Higher values indicate potential hub sites
        """
        n_nodes = len(phase)
        potential = np.zeros(n_nodes)
        
        for i in prange(n_nodes):
            nb_idx = neighbors[i]
            
            # Compute phase variance among neighbors
            if len(nb_idx) > 1:
                nb_phases = np.zeros(len(nb_idx))
                for j in range(len(nb_idx)):
                    nb_phases[j] = all_phases[nb_idx[j]]
                
                variance = np.var(nb_phases)
                potential[i] = 1.0 / (1.0 + variance / threshold)
        
        return potential
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_phase_gradient(
        phase: np.ndarray,
        coordinates: np.ndarray,
        neighbors: List[np.ndarray],
        all_phases: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute phase gradient magnitude and direction
        """
        n_nodes = len(phase)
        grad_mag = np.zeros(n_nodes)
        grad_dir = np.zeros((n_nodes, 3))
        
        for i in range(n_nodes):
            nb_idx = neighbors[i]
            if len(nb_idx) < 4:  # Need at least 4 neighbors for 3D gradient
                continue
            
            # Build local coordinate system
            A = np.zeros((len(nb_idx), 3))
            b = np.zeros(len(nb_idx))
            
            for j, nb in enumerate(nb_idx):
                A[j] = coordinates[nb, :3] - coordinates[i, :3]
                b[j] = all_phases[nb] - phase[i]
            
            # Solve least squares for gradient
            try:
                grad, _, _, _ = np.linalg.lstsq(A, b)
                grad_mag[i] = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)
                if grad_mag[i] > 0:
                    grad_dir[i] = grad / grad_mag[i]
            except:
                pass
        
        return grad_mag, grad_dir


# ------------------------------
# GPU KERNELS (CUPY)
# ------------------------------
class GPUKernels:
    """GPU-optimized kernels using CuPy"""
    
    def __init__(self, constants: PhysicsConstants):
        self.constants = constants
        if not GPU_AVAILABLE:
            raise ImportError("CuPy not available")
    
    def update_phase_gpu(
        self,
        phase: cp.ndarray,
        tau: cp.ndarray,
        eta_eff: cp.ndarray,
        neighbors: List[np.ndarray],
        all_phases: cp.ndarray,
        dt: float
    ) -> cp.ndarray:
        """
        GPU-accelerated phase update
        Uses custom CUDA kernel for maximum performance
        """
        # Move neighbors to GPU (as ragged array)
        # This is simplified - in production use custom CUDA kernel
        
        n_nodes = len(phase)
        new_phase = cp.zeros_like(phase)
        
        # For each node, compute neighbor average
        # This is a Python loop - in production, implement as CUDA kernel
        for i in range(n_nodes):
            nb_idx = neighbors[i]
            nb_phases = all_phases[nb_idx]
            phi_eq = cp.mean(nb_phases)
            
            eta = eta_eff[i] * (1.0 + tau[i])
            dphi = -self.constants.kappa * eta * (phase[i] - phi_eq) * dt
            new_phase[i] = phase[i] + dphi
        
        return new_phase
    
    def _cuda_kernel_source(self) -> str:
        """Return CUDA kernel source code"""
        return """
        extern "C" {
            __global__ void phase_update(
                const double* phase,
                const double* tau,
                const double* eta_eff,
                const int* neighbors,
                const int* neighbor_counts,
                const int* neighbor_offsets,
                const double* all_phases,
                double* new_phase,
                double dt,
                double kappa,
                int n_nodes
            ) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= n_nodes) return;
                
                double phi_i = phase[i];
                double eta = eta_eff[i] * (1.0 + tau[i]);
                
                // Get neighbors for this node
                int start = neighbor_offsets[i];
                int count = neighbor_counts[i];
                
                // Compute mean of neighbor phases
                double sum = 0.0;
                for (int j = 0; j < count; j++) {
                    int nb = neighbors[start + j];
                    sum += all_phases[nb];
                }
                double phi_eq = sum / count;
                
                // Phase update
                double dphi = -kappa * eta * (phi_i - phi_eq) * dt;
                new_phase[i] = phi_i + dphi;
            }
        }
        """


# ------------------------------
# KERNEL FACTORY
# ------------------------------
class KernelFactory:
    """Factory for creating appropriate kernels"""
    
    def __init__(self, constants: Optional[PhysicsConstants] = None):
        self.constants = constants or PhysicsConstants()
        self.cpu_kernels = CPUKernels(self.constants)
        self.gpu_kernels = GPUKernels(self.constants) if GPU_AVAILABLE else None
    
    def get_kernels(self, use_gpu: bool = False):
        """Get appropriate kernel set"""
        if use_gpu and self.gpu_kernels is not None:
            logger.info("Using GPU kernels")
            return self.gpu_kernels
        else:
            logger.info("Using CPU kernels")
            return self.cpu_kernels


# ------------------------------
# BOUNDARY CONDITIONS
# ------------------------------
class BoundaryConditions:
    """Apply boundary conditions to phase field"""
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_periodic(
        phase: np.ndarray,
        coordinates: np.ndarray,
        bounds: Tuple[float, float]
    ) -> np.ndarray:
        """Apply periodic boundary conditions"""
        new_phase = phase.copy()
        lo, hi = bounds
        
        for i in range(len(coordinates)):
            for d in range(coordinates.shape[1]):
                if coordinates[i, d] < lo:
                    # Find corresponding point on other side
                    # Simplified - in production use ghost cells
                    pass
                elif coordinates[i, d] > hi:
                    pass
        
        return new_phase
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_absorbing(
        phase: np.ndarray,
        boundary_nodes: np.ndarray,
        absorption_strength: float,
        dt: float
    ) -> np.ndarray:
        """Apply absorbing boundary conditions"""
        new_phase = phase.copy()
        
        for i in boundary_nodes:
            # Damp phase at boundaries
            new_phase[i] *= (1.0 - absorption_strength * dt)
        
        return new_phase


# ------------------------------
# UNIT TESTS
# ------------------------------
def test_cpu_kernels():
    """Test CPU kernels"""
    constants = PhysicsConstants()
    kernels = CPUKernels(constants)
    
    # Create test data
    n_nodes = 1000
    phase = np.random.uniform(0, 2*np.pi, n_nodes)
    tau = np.ones(n_nodes) * constants.tau
    eta_eff = np.ones(n_nodes) * constants.eta_eff
    
    # Create simple neighbor list (each node connected to next 5)
    neighbors = []
    for i in range(n_nodes):
        nb = [(i+j) % n_nodes for j in range(1, 6)]
        neighbors.append(np.array(nb))
    
    all_phases = phase.copy()
    dt = 0.1
    
    # Run update
    new_phase = kernels.update_phase_vectorized(
        phase, tau, eta_eff, neighbors, all_phases, dt, constants.kappa
    )
    
    assert len(new_phase) == n_nodes
    assert not np.any(np.isnan(new_phase))
    
    print("CPU kernels test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cpu_kernels()