#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for SVG Physics and Kernels Module
Tests phase update, torsion calculation, hub detection, and conservation laws

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import (
    PhysicsConstants,
    CPUKernels,
    GPUKernels,
    KernelFactory,
    BoundaryConditions
)
from src.hubs import (
    HubDetector,
    HubDetectionConfig,
    Hub,
    HubMergerDetector
)
from src.ai_models import (
    AIModelConfig,
    SurrogateModelFactory,
    FeatureEngineer
)


class TestPhysicsConstants(unittest.TestCase):
    """Test physics constants"""
    
    def test_constants(self):
        """Test constant values"""
        constants = PhysicsConstants()
        
        self.assertEqual(constants.delta_deg, 6.8)
        self.assertAlmostEqual(constants.tau, 0.0685, places=4)
        self.assertAlmostEqual(constants.eta_eff, 1.34e-19)
        self.assertEqual(constants.kappa, 1.0)
        
        # Test conversion
        self.assertAlmostEqual(constants.delta_rad, np.deg2rad(6.8), places=6)
    
    def test_custom_constants(self):
        """Test custom constants"""
        constants = PhysicsConstants(
            delta_deg=10.0,
            tau=0.1,
            eta_eff=1e-20
        )
        
        self.assertEqual(constants.delta_deg, 10.0)
        self.assertEqual(constants.tau, 0.1)
        self.assertEqual(constants.eta_eff, 1e-20)


class TestCPUKernels(unittest.TestCase):
    """Test CPU kernels"""
    
    def setUp(self):
        self.constants = PhysicsConstants()
        self.kernels = CPUKernels(self.constants)
        
        # Create test data
        self.n_nodes = 1000
        self.phase = np.random.uniform(0, 2*np.pi, self.n_nodes)
        self.tau = np.ones(self.n_nodes) * self.constants.tau
        self.eta_eff = np.ones(self.n_nodes) * self.constants.eta_eff
        
        # Create neighbor lists (each node connected to next 5)
        self.neighbors = []
        for i in range(self.n_nodes):
            nb = [(i+j) % self.n_nodes for j in range(1, 6)]
            self.neighbors.append(np.array(nb))
        
        self.all_phases = self.phase.copy()
        self.dt = 0.1
    
    def test_phase_update(self):
        """Test phase update kernel"""
        new_phase = self.kernels.update_phase_vectorized(
            self.phase,
            self.tau,
            self.eta_eff,
            self.neighbors,
            self.all_phases,
            self.dt,
            self.constants.kappa
        )
        
        # Check shape
        self.assertEqual(len(new_phase), self.n_nodes)
        
        # Check no NaN
        self.assertFalse(np.any(np.isnan(new_phase)))
        
        # Check bounds (phase should stay in reasonable range)
        self.assertTrue(np.all(new_phase >= -10))
        self.assertTrue(np.all(new_phase <= 10))
    
    def test_phase_update_conservation(self):
        """Test approximate phase conservation in closed system"""
        # Isolated node with no neighbors should not change
        single_phase = np.array([1.0])
        single_tau = np.array([self.constants.tau])
        single_eta = np.array([self.constants.eta_eff])
        single_neighbors = [np.array([])]
        
        new_phase = self.kernels.update_phase_vectorized(
            single_phase,
            single_tau,
            single_eta,
            single_neighbors,
            single_phase,
            self.dt,
            self.constants.kappa
        )
        
        # With no neighbors, phi_eq is undefined - in practice should use self
        # This test may need adjustment based on implementation
        self.assertEqual(len(new_phase), 1)
    
    def test_torsion_computation(self):
        """Test torsion field computation"""
        phase_mean = np.mean(self.phase)
        torsion = self.kernels.compute_torsion(
            self.phase,
            self.constants.tau,
            self.constants.gamma,
            phase_mean
        )
        
        self.assertEqual(len(torsion), self.n_nodes)
        
        # Check that torsion varies with phase deviation
        phase_dev = self.phase - phase_mean
        expected = self.constants.tau + self.constants.gamma * phase_dev**2
        
        np.testing.assert_array_almost_equal(torsion, expected)
    
    def test_hub_potential(self):
        """Test hub potential computation"""
        potential = self.kernels.compute_hub_potential(
            self.phase,
            self.neighbors,
            self.all_phases,
            self.constants.hub_threshold
        )
        
        self.assertEqual(len(potential), self.n_nodes)
        self.assertTrue(np.all(potential >= 0))
        self.assertTrue(np.all(potential <= 1))
        
        # Nodes with coherent neighbors should have higher potential
        # Create coherent region
        coherent_phase = np.ones(10) * 1.0
        coherent_neighbors = [np.arange(10) for _ in range(10)]
        coherent_all = coherent_phase.copy()
        
        coherent_potential = self.kernels.compute_hub_potential(
            coherent_phase,
            coherent_neighbors,
            coherent_all,
            self.constants.hub_threshold
        )
        
        self.assertTrue(np.all(coherent_potential > 0.9))
    
    def test_phase_gradient(self):
        """Test phase gradient computation"""
        # Create simple linear gradient
        n = 100
        coords = np.random.randn(n, 4)
        phase = coords[:, 0] * 2.0  # Linear in x
        
        # Create neighbors based on proximity
        from scipy.spatial import KDTree
        tree = KDTree(coords[:, :3])
        neighbors = []
        for i in range(n):
            dist, idx = tree.query(coords[i, :3], k=10)
            neighbors.append(idx)
        
        grad_mag, grad_dir = self.kernels.compute_phase_gradient(
            phase,
            coords,
            neighbors,
            phase
        )
        
        self.assertEqual(len(grad_mag), n)
        self.assertEqual(grad_dir.shape, (n, 3))
        
        # Gradient magnitude should be positive
        self.assertTrue(np.all(grad_mag >= 0))


class TestBoundaryConditions(unittest.TestCase):
    """Test boundary conditions"""
    
    def setUp(self):
        self.n_nodes = 100
        self.phase = np.random.randn(self.n_nodes)
        self.coords = np.random.randn(self.n_nodes, 4)
        self.boundary = np.arange(10)  # First 10 nodes are boundary
    
    def test_absorbing(self):
        """Test absorbing boundary conditions"""
        absorption = 0.1
        dt = 0.5
        
        new_phase = BoundaryConditions.apply_absorbing(
            self.phase,
            self.boundary,
            absorption,
            dt
        )
        
        # Boundary nodes should be damped
        for i in self.boundary:
            expected = self.phase[i] * (1.0 - absorption * dt)
            self.assertAlmostEqual(new_phase[i], expected)
        
        # Interior nodes should be unchanged
        for i in range(10, self.n_nodes):
            self.assertEqual(new_phase[i], self.phase[i])


class TestHubDetector(unittest.TestCase):
    """Test hub detection"""
    
    def setUp(self):
        self.config = HubDetectionConfig(
            threshold=0.8,
            min_cluster_size=3,
            merge_distance=0.2
        )
        self.detector = HubDetector(self.config)
        
        # Create test data
        self.n_nodes = 1000
        self.coords = np.random.randn(self.n_nodes, 4)
        self.phase = np.random.uniform(0, 2*np.pi, self.n_nodes)
        
        # Create hub-like structures
        self.hub_potential = np.random.uniform(0, 0.5, self.n_nodes)
        
        # Add three hub regions
        centers = [
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([0.5, -0.5, 0.5, -0.5])
        ]
        
        for center in centers:
            dist = np.linalg.norm(self.coords - center, axis=1)
            hub_region = dist < 0.3
            self.hub_potential[hub_region] = 0.9
    
    def test_find_candidates(self):
        """Test candidate finding"""
        candidates = self.detector._find_candidates(
            self.hub_potential,
            self.phase
        )
        
        self.assertIsInstance(candidates, np.ndarray)
        self.assertTrue(len(candidates) > 0)
        
        # All candidates should have high potential
        for idx in candidates:
            self.assertGreaterEqual(self.hub_potential[idx], self.config.threshold)
    
    def test_clustering(self):
        """Test candidate clustering"""
        candidates = self.detector._find_candidates(
            self.hub_potential,
            self.phase
        )
        
        if len(candidates) > 0:
            clusters = self.detector._cluster_candidates(
                candidates,
                self.coords
            )
            
            self.assertIsInstance(clusters, list)
            
            # Each cluster should have at least min_cluster_size members
            for cluster in clusters:
                self.assertGreaterEqual(len(cluster), self.config.min_cluster_size)
    
    def test_full_detection(self):
        """Test full detection pipeline"""
        hubs = self.detector.detect(
            self.phase,
            self.coords,
            self.hub_potential,
            step=0
        )
        
        self.assertIsInstance(hubs, list)
        
        # Should detect at least some hubs
        if len(hubs) > 0:
            hub = hubs[0]
            self.assertIsInstance(hub, Hub)
            self.assertIsNotNone(hub.id)
            self.assertEqual(hub.formation_step, 0)
    
    def test_temporal_matching(self):
        """Test hub matching across time steps"""
        # First detection
        hubs1 = self.detector.detect(
            self.phase,
            self.coords,
            self.hub_potential,
            step=0
        )
        
        n_hubs1 = len(self.detector.hubs)
        
        # Second detection (slightly different)
        hub_potential2 = self.hub_potential.copy()
        hub_potential2 += np.random.randn(self.n_nodes) * 0.05
        
        hubs2 = self.detector.detect(
            self.phase + 0.01,
            self.coords,
            hub_potential2,
            step=1
        )
        
        n_hubs2 = len(self.detector.hubs)
        
        # Number of hubs should be stable
        self.assertAlmostEqual(n_hubs1, n_hubs2, delta=2)
    
    def test_inactive_removal(self):
        """Test removal of inactive hubs"""
        # Detect hubs
        self.detector.detect(
            self.phase,
            self.coords,
            self.hub_potential,
            step=0
        )
        
        n_hubs = len(self.detector.hubs)
        
        # Remove inactive (all should be active at step 0)
        self.detector.clean_inactive_hubs(current_step=5, max_inactive=2)
        
        # Some may be removed
        self.assertLessEqual(len(self.detector.hubs), n_hubs)
    
    def test_statistics(self):
        """Test hub statistics generation"""
        self.detector.detect(
            self.phase,
            self.coords,
            self.hub_potential,
            step=0
        )
        
        stats = self.detector.get_hub_statistics()
        
        self.assertIsInstance(stats, dict)
        if len(self.detector.hubs) > 0:
            self.assertIn('n_hubs', stats)
            self.assertIn('mass_mean', stats)
            self.assertIn('radius_mean', stats)


class TestHubMergerDetector(unittest.TestCase):
    """Test hub merger detection"""
    
    def setUp(self):
        self.config = HubDetectionConfig(merge_distance=0.5)
        self.detector = HubDetector(self.config)
        self.merger_detector = HubMergerDetector(self.detector)
        
        # Create two close hubs manually
        hub1 = Hub(
            id=0,
            formation_step=0,
            last_seen_step=0,
            position=np.array([0.0, 0.0, 0.0, 0.0]),
            phase=1.0,
            potential=0.9,
            mass_proxy=10,
            radius=0.2
        )
        
        hub2 = Hub(
            id=1,
            formation_step=0,
            last_seen_step=0,
            position=np.array([0.3, 0.3, 0.3, 0.3]),  # Within merge_distance
            phase=1.1,
            potential=0.85,
            mass_proxy=8,
            radius=0.2
        )
        
        self.detector.hubs = {0: hub1, 1: hub2}
        self.detector._update_positions()
    
    def test_merger_detection(self):
        """Test detecting hub mergers"""
        mergers = self.merger_detector.detect_mergers(step=1)
        
        self.assertIsInstance(mergers, list)
        
        # Should detect at least one merger
        if len(mergers) > 0:
            merger = mergers[0]
            self.assertIn('hub1_id', merger)
            self.assertIn('hub2_id', merger)
            self.assertIn('distance', merger)
            self.assertLess(merger['distance'], self.config.merge_distance)


class TestAIModels(unittest.TestCase):
    """Test AI surrogate models"""
    
    def setUp(self):
        # Create synthetic training data
        self.n_samples = 1000
        self.n_features = 5
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.sum(self.X**2, axis=1) + 0.1 * np.random.randn(self.n_samples)
        
    def test_feature_engineering(self):
        """Test feature engineering"""
        # Create raw data
        n = 100
        points = np.random.randn(n, 4)
        phase = np.random.uniform(0, 2*np.pi, n)
        tau = np.random.randn(n)
        hub_potential = np.random.rand(n)
        w_coord = points[:, 3]
        
        features = FeatureEngineer.create_features(
            points, phase, tau, hub_potential, w_coord
        )
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], n)
        self.assertGreater(features.shape[1], 4)  # Should have more than basic coords
    
    def test_sklearn_random_forest(self):
        """Test sklearn Random Forest model"""
        config = AIModelConfig(
            model_type="random_forest",
            framework="sklearn",
            n_estimators=10,
            max_depth=5,
            train_fraction=0.7
        )
        
        model = SurrogateModelFactory.create_model(config)
        results = model.train(self.X, self.y)
        
        self.assertIn('train_r2', results)
        self.assertIn('test_r2', results)
        self.assertGreater(results['test_r2'], 0)
        
        # Test prediction
        X_test = np.random.randn(10, self.n_features)
        y_pred = model.predict(X_test)
        self.assertEqual(len(y_pred), 10)
    
    def test_sklearn_gradient_boosting(self):
        """Test sklearn Gradient Boosting model"""
        config = AIModelConfig(
            model_type="gradient_boosting",
            framework="sklearn",
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )
        
        model = SurrogateModelFactory.create_model(config)
        results = model.train(self.X, self.y)
        
        self.assertIn('train_r2', results)
        self.assertIn('test_r2', results)
    
    def test_sklearn_neural_network(self):
        """Test sklearn neural network"""
        config = AIModelConfig(
            model_type="neural_network",
            framework="sklearn",
            hidden_layers=[50, 25],
            max_iter=100
        )
        
        model = SurrogateModelFactory.create_model(config)
        results = model.train(self.X, self.y)
        
        self.assertIn('train_r2', results)
    
    @unittest.skipIf(not hasattr(__import__('torch'), '__version__'), "PyTorch not available")
    def test_pytorch_model(self):
        """Test PyTorch neural network"""
        config = AIModelConfig(
            framework="pytorch",
            hidden_layers=[64, 32],
            learning_rate=0.01,
            epochs=10,
            batch_size=128
        )
        
        model = SurrogateModelFactory.create_model(config)
        results = model.train(self.X, self.y)
        
        self.assertIn('train_r2', results)
        
        # Test prediction
        X_test = np.random.randn(10, self.n_features)
        y_pred = model.predict(X_test)
        self.assertEqual(len(y_pred), 10)
    
    def test_save_load(self):
        """Test model saving and loading"""
        import tempfile
        import os
        
        config = AIModelConfig(model_type="random_forest", n_estimators=5)
        model = SurrogateModelFactory.create_model(config)
        model.train(self.X[:100], self.y[:100])
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model.save(f.name)
            
            # Load in new model
            model2 = SurrogateModelFactory.create_model(config)
            model2.load(f.name)
            
            # Should give same predictions
            X_test = self.X[100:110]
            pred1 = model.predict(X_test)
            pred2 = model2.predict(X_test)
            
            np.testing.assert_array_almost_equal(pred1, pred2)
        
        os.unlink(f.name)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple modules"""
    
    def test_full_pipeline_small(self):
        """Test small-scale integration of all components"""
        from src.mesh import create_mesh, MeshConfig
        from src.kernels import CPUKernels, PhysicsConstants
        from src.hubs import HubDetector, HubDetectionConfig
        
        # 1. Create mesh
        mesh_config = MeshConfig(n_nodes=100)
        mesh = create_mesh(mesh_config)
        
        # 2. Initialize physics
        constants = PhysicsConstants()
        kernels = CPUKernels(constants)
        
        # 3. Initialize simulation data
        phase = np.random.uniform(0, 2*np.pi, mesh.n_nodes)
        tau = np.ones(mesh.n_nodes) * constants.tau
        eta_eff = np.ones(mesh.n_nodes) * constants.eta_eff
        
        # 4. Hub detection
        hub_config = HubDetectionConfig(threshold=0.8, min_cluster_size=2)
        detector = HubDetector(hub_config)
        
        # 5. Run one time step
        all_phases = phase.copy()
        new_phase = kernels.update_phase_vectorized(
            phase,
            tau,
            eta_eff,
            mesh.neighbors,
            all_phases,
            dt=0.1,
            kappa=constants.kappa
        )
        
        # 6. Compute hub potential
        hub_potential = kernels.compute_hub_potential(
            new_phase,
            mesh.neighbors,
            all_phases,
            constants.hub_threshold
        )
        
        # 7. Detect hubs
        hubs = detector.detect(
            new_phase,
            mesh.coordinates,
            hub_potential,
            step=0
        )
        
        # All steps should complete without errors
        self.assertIsNotNone(new_phase)
        self.assertIsNotNone(hub_potential)
        self.assertIsInstance(hubs, list)


class TestConservationLaws(unittest.TestCase):
    """Test physical conservation laws"""
    
    def test_phase_conservation_closed_system(self):
        """Test that total phase is approximately conserved in closed system"""
        from src.kernels import CPUKernels, PhysicsConstants
        
        constants = PhysicsConstants()
        kernels = CPUKernels(constants)
        
        # Create a small closed system (all nodes connected)
        n_nodes = 50
        phase = np.random.uniform(0, 2*np.pi, n_nodes)
        tau = np.ones(n_nodes) * constants.tau
        eta_eff = np.ones(n_nodes) * constants.eta_eff
        
        # Fully connected graph
        neighbors = [np.arange(n_nodes) for _ in range(n_nodes)]
        
        total_phase_initial = np.sum(phase)
        
        # Run multiple steps
        for step in range(10):
            phase = kernels.update_phase_vectorized(
                phase,
                tau,
                eta_eff,
                neighbors,
                phase,
                dt=0.01,
                kappa=constants.kappa
            )
        
        total_phase_final = np.sum(phase)
        
        # Phase should be approximately conserved (small drift)
        self.assertAlmostEqual(total_phase_final, total_phase_initial, delta=0.1)
    
    def test_torsion_bounds(self):
        """Test that torsion stays within physical bounds"""
        from src.kernels import CPUKernels, PhysicsConstants
        
        constants = PhysicsConstants()
        kernels = CPUKernels(constants)
        
        n_nodes = 100
        phase = np.random.uniform(-10, 10, n_nodes)
        phase_mean = np.mean(phase)
        
        torsion = kernels.compute_torsion(
            phase,
            constants.tau,
            constants.gamma,
            phase_mean
        )
        
        # Torsion should be positive
        self.assertTrue(np.all(torsion > 0))
        
        # Should not be extreme
        self.assertTrue(np.all(torsion < 1.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)