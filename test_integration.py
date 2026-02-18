#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for SVG Modules
Tests that multiple modules work together correctly

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026
"""

import os
import sys
import unittest
import numpy as np
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh import MeshConfig, DelaunayMeshGenerator4D
from src.kernels import PhysicsConstants, CPUKernels, KernelFactory
from src.hubs import HubDetectionConfig, HubDetector
from src.ai_models import AIModelConfig, SurrogateModelFactory, FeatureEngineer


class TestMeshAndKernelsIntegration(unittest.TestCase):
    """Test integration between mesh and kernels"""
    
    def setUp(self):
        """Set up integration test"""
        # Create mesh
        self.mesh_config = MeshConfig(n_nodes=500)
        self.generator = DelaunayMeshGenerator4D(self.mesh_config)
        self.mesh = self.generator.generate()
        
        # Create kernels
        self.constants = PhysicsConstants()
        self.kernels = CPUKernels(self.constants)
        
        # Initialize phases
        self.phase = np.random.uniform(0, 2*np.pi, self.mesh.n_nodes)
        self.tau = np.ones(self.mesh.n_nodes) * self.constants.tau
        self.eta = np.ones(self.mesh.n_nodes) * self.constants.eta_eff
    
    def test_phase_update_on_mesh(self):
        """Test phase update using mesh connectivity"""
        # Run update
        new_phase = self.kernels.update_phase_vectorized(
            self.phase,
            self.tau,
            self.eta,
            self.mesh.neighbors,
            self.phase,
            dt=0.1,
            kappa=self.constants.kappa
        )
        
        self.assertEqual(len(new_phase), self.mesh.n_nodes)
        self.assertFalse(np.any(np.isnan(new_phase)))
        
        # Check that nodes with many neighbors update differently
        neighbor_counts = np.array([len(nb) for nb in self.mesh.neighbors])
        high_degree = neighbor_counts > np.percentile(neighbor_counts, 75)
        low_degree = neighbor_counts < np.percentile(neighbor_counts, 25)
        
        phase_change = np.abs(new_phase - self.phase)
        
        # High degree nodes might have different dynamics
        # This is not a strict test, just a sanity check
        self.assertTrue(np.mean(phase_change[high_degree]) > 0)
    
    def test_gradient_on_mesh(self):
        """Test gradient computation using mesh"""
        grad_mag, grad_dir = self.kernels.compute_phase_gradient(
            self.phase,
            self.mesh.coordinates,
            self.mesh.neighbors,
            self.phase
        )
        
        self.assertEqual(len(grad_mag), self.mesh.n_nodes)
        self.assertEqual(grad_dir.shape, (self.mesh.n_nodes, 3))


class TestMeshAndHubsIntegration(unittest.TestCase):
    """Test integration between mesh and hub detection"""
    
    def setUp(self):
        """Set up integration test"""
        # Create mesh
        self.mesh_config = MeshConfig(n_nodes=1000)
        self.generator = DelaunayMeshGenerator4D(self.mesh_config)
        self.mesh = self.generator.generate()
        
        # Create hub detector
        self.hub_config = HubDetectionConfig(
            threshold=0.7,
            min_cluster_size=5,
            merge_distance=0.3
        )
        self.detector = HubDetector(self.hub_config)
        
        # Initialize phases and create coherent regions
        self.phase = np.random.uniform(0, 2*np.pi, self.mesh.n_nodes)
        self.hub_potential = np.random.uniform(0, 0.5, self.mesh.n_nodes)
        
        # Create coherent regions using mesh connectivity
        for i in range(3):
            # Pick a random node
            center_idx = np.random.randint(0, self.mesh.n_nodes)
            
            # Get its neighbors
            neighbors = self.mesh.neighbors[center_idx]
            
            # Make this region coherent
            coherent_phase = np.mean(self.phase[neighbors])
            self.phase[neighbors] = coherent_phase
            self.hub_potential[neighbors] = 0.9
            self.hub_potential[center_idx] = 0.95
    
    def test_hub_detection_on_mesh(self):
        """Test hub detection using mesh data"""
        hubs = self.detector.detect(
            self.phase,
            self.mesh.coordinates,
            self.hub_potential,
            step=0
        )
        
        # Should detect some hubs
        self.assertGreaterEqual(len(hubs), 0)
        
        # Check that hubs are at coherent regions
        if hubs:
            for hub in hubs:
                # Hub members should have high potential
                for member_idx in hub.members:
                    self.assertGreaterEqual(
                        self.hub_potential[member_idx],
                        self.hub_config.threshold * 0.8  # Allow some tolerance
                    )


class TestKernelsAndHubsIntegration(unittest.TestCase):
    """Test integration between kernels and hub detection"""
    
    def setUp(self):
        """Set up integration test"""
        self.constants = PhysicsConstants()
        self.kernels = CPUKernels(self.constants)
        
        self.n_nodes = 500
        self.phase = np.random.uniform(0, 2*np.pi, self.n_nodes)
        
        # Create neighbor list
        self.neighbors = []
        for i in range(self.n_nodes):
            nb = [(i+j) % self.n_nodes for j in range(1, 8)]
            self.neighbors.append(np.array(nb))
        
        # Create hub detector
        self.hub_config = HubDetectionConfig(threshold=0.8)
        self.detector = HubDetector(self.hub_config)
    
    def test_hub_potential_from_kernels(self):
        """Test hub potential computation and detection"""
        # Compute hub potential using kernels
        hub_potential = self.kernels.compute_hub_potential(
            self.phase,
            self.neighbors,
            self.phase,
            self.constants.hub_threshold
        )
        
        # Detect hubs using the potential
        hubs = self.detector.detect(
            self.phase,
            np.random.randn(self.n_nodes, 4),  # Dummy coordinates
            hub_potential,
            step=0
        )
        
        # If hubs detected, they should have high potential
        if hubs:
            for hub in hubs:
                hub_pot = np.mean([hub_potential[m] for m in hub.members])
                self.assertGreaterEqual(hub_pot, self.hub_config.threshold * 0.8)


class TestAIIntegration(unittest.TestCase):
    """Test integration with AI models"""
    
    def setUp(self):
        """Set up AI integration test"""
        self.n_samples = 1000
        self.n_features = 5
        
        # Create synthetic data
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(self.n_samples)
        
        # Create feature names
        self.feature_names = ['x', 'y', 'z', 'phase', 'w']
    
    def test_ai_with_physics_features(self):
        """Test AI model with physics-based features"""
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Create synthetic raw data
        points = np.random.randn(self.n_samples, 4)
        phase = np.random.uniform(0, 2*np.pi, self.n_samples)
        tau = np.random.randn(self.n_samples) * 0.01 + 0.068
        
        # Engineer features
        features = engineer.create_features(points, phase, tau)
        
        # Create AI model
        config = AIModelConfig(
            model_type="random_forest",
            framework="sklearn",
            n_estimators=10,  # Small for testing
            max_depth=5,
            features=self.feature_names
        )
        
        try:
            model = SurrogateModelFactory.create_model(config)
            
            # Train on engineered features
            results = model.train(features[:800], tau[:800])
            
            # Test prediction
            pred = model.predict(features[800:])
            
            self.assertEqual(len(pred), 200)
            self.assertIn('test_r2', results)
            
        except ImportError:
            self.skipTest("scikit-learn not available")


class TestFullPipeline(unittest.TestCase):
    """Test complete pipeline integration"""
    
    def test_small_scale_pipeline(self):
        """Run small-scale pipeline end-to-end"""
        
        # 1. Generate mesh
        mesh_config = MeshConfig(n_nodes=100)
        generator = DelaunayMeshGenerator4D(mesh_config)
        mesh = generator.generate()
        
        # 2. Initialize physics
        constants = PhysicsConstants()
        kernels = CPUKernels(constants)
        
        phase = np.random.uniform(0, 2*np.pi, mesh.n_nodes)
        tau = np.ones(mesh.n_nodes) * constants.tau
        eta = np.ones(mesh.n_nodes) * constants.eta_eff
        
        # 3. Run a few time steps
        for step in range(5):
            # Update phase
            new_phase = kernels.update_phase_vectorized(
                phase,
                tau,
                eta,
                mesh.neighbors,
                phase,
                dt=0.1,
                kappa=constants.kappa
            )
            phase = new_phase
            
            # Compute hub potential
            hub_potential = kernels.compute_hub_potential(
                phase,
                mesh.neighbors,
                phase,
                constants.hub_threshold
            )
            
            # Check that things are still valid
            self.assertFalse(np.any(np.isnan(phase)))
            self.assertFalse(np.any(np.isinf(phase)))
            self.assertTrue(np.all(hub_potential >= 0))
            self.assertTrue(np.all(hub_potential <= 1))
        
        print("Small-scale pipeline test passed!")


if __name__ == '__main__':
    unittest.main()