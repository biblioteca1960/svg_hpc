#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for SVG Mesh Generation Module
Tests the 4D tetrahedral mesh generation and partitioning

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh import (
    MeshConfig,
    MeshData,
    DelaunayMeshGenerator4D,
    MeshPartitioner,
    create_mesh
)


class TestMeshConfig(unittest.TestCase):
    """Test MeshConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = MeshConfig()
        self.assertEqual(config.dimension, 4)
        self.assertEqual(config.n_nodes, 1000000)
        self.assertEqual(config.distribution, "uniform")
        self.assertEqual(config.seed, 42)
        self.assertTrue(config.periodic)
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = MeshConfig(
            dimension=3,
            n_nodes=5000,
            distribution="radial",
            seed=123,
            periodic=False,
            refine_regions=[{"center": [0,0,0,0], "radius": 1.0, "factor": 2.0}]
        )
        self.assertEqual(config.dimension, 3)
        self.assertEqual(config.n_nodes, 5000)
        self.assertEqual(config.distribution, "radial")
        self.assertEqual(config.seed, 123)
        self.assertFalse(config.periodic)
        self.assertEqual(len(config.refine_regions), 1)


class TestMeshData(unittest.TestCase):
    """Test MeshData container"""
    
    def setUp(self):
        """Create test mesh data"""
        self.n_nodes = 100
        self.coordinates = np.random.randn(self.n_nodes, 4)
        self.connectivity = np.random.randint(0, self.n_nodes, (200, 5))
        self.neighbors = [np.array([1,2,3]), np.array([0,2]), np.array([0,1])]
        
        self.mesh = MeshData(
            coordinates=self.coordinates,
            connectivity=self.connectivity,
            neighbors=self.neighbors
        )
    
    def test_properties(self):
        """Test mesh properties"""
        self.assertEqual(self.mesh.n_nodes, self.n_nodes)
        self.assertEqual(self.mesh.n_simplices, 200)
        
    def test_summary(self):
        """Test summary generation"""
        summary = self.mesh.summary()
        self.assertIn('n_nodes', summary)
        self.assertIn('neighbors_mean', summary)
        self.assertEqual(summary['n_nodes'], self.n_nodes)
        
    def test_empty_mesh(self):
        """Test empty mesh handling"""
        empty_mesh = MeshData(
            coordinates=np.array([]),
            connectivity=np.array([]),
            neighbors=[]
        )
        self.assertEqual(empty_mesh.n_nodes, 0)
        self.assertEqual(empty_mesh.n_simplices, 0)


class TestDelaunayMeshGenerator(unittest.TestCase):
    """Test 4D Delaunay mesh generator"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = MeshConfig(
            dimension=4,
            n_nodes=500,  # Small for testing
            distribution="uniform",
            seed=42,
            save_mesh=False
        )
        self.generator = DelaunayMeshGenerator4D(self.config)
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_generate_uniform(self):
        """Test uniform distribution generation"""
        mesh = self.generator.generate()
        
        self.assertEqual(mesh.n_nodes, 500)
        self.assertEqual(mesh.coordinates.shape[1], 4)
        self.assertIsNotNone(mesh.connectivity)
        self.assertIsNotNone(mesh.neighbors)
        
        # Check bounds
        self.assertTrue(np.all(mesh.coordinates >= -1))
        self.assertTrue(np.all(mesh.coordinates <= 1))
    
    def test_generate_radial(self):
        """Test radial distribution generation"""
        self.config.distribution = "radial"
        generator = DelaunayMeshGenerator4D(self.config)
        mesh = generator.generate()
        
        self.assertEqual(mesh.n_nodes, 500)
        
        # Check radial distribution
        radii = np.linalg.norm(mesh.coordinates, axis=1)
        self.assertTrue(np.all(radii >= 0))
        self.assertTrue(np.any(radii > 1))  # Some points beyond unit sphere
    
    def test_generate_clustered(self):
        """Test clustered distribution generation"""
        self.config.distribution = "clustered"
        generator = DelaunayMeshGenerator4D(self.config)
        mesh = generator.generate()
        
        self.assertEqual(mesh.n_nodes, 500)
        
        # Should have multiple clusters
        # Check via simple variance ratio
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(mesh.coordinates)
        
        # At least some variation in cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        self.assertTrue(len(unique) > 1)
    
    def test_neighbor_consistency(self):
        """Test neighbor list consistency"""
        mesh = self.generator.generate()
        
        # Check that neighbor relations are symmetric
        for i, neighbors in enumerate(mesh.neighbors):
            for j in neighbors:
                self.assertIn(i, mesh.neighbors[j])
        
        # Check minimum neighbors
        neighbor_counts = [len(nb) for nb in mesh.neighbors]
        self.assertGreater(np.min(neighbor_counts), 0)
    
    def test_connectivity_validity(self):
        """Test that connectivity indices are valid"""
        mesh = self.generator.generate()
        
        # All indices should be within range
        max_idx = np.max(mesh.connectivity)
        min_idx = np.min(mesh.connectivity)
        
        self.assertLessEqual(max_idx, mesh.n_nodes - 1)
        self.assertGreaterEqual(min_idx, 0)
    
    def test_save_load(self):
        """Test saving and loading mesh"""
        mesh = self.generator.generate()
        
        # Save
        filename = os.path.join(self.test_dir, "test_mesh.h5")
        self.generator.save(filename, mesh)
        self.assertTrue(os.path.exists(filename))
        
        # Load
        loaded_mesh = self.generator.load(filename)
        
        # Compare
        np.testing.assert_array_equal(mesh.coordinates, loaded_mesh.coordinates)
        np.testing.assert_array_equal(mesh.connectivity, loaded_mesh.connectivity)
        self.assertEqual(len(mesh.neighbors), len(loaded_mesh.neighbors))


class TestMeshPartitioner(unittest.TestCase):
    """Test mesh partitioning for distributed computing"""
    
    def setUp(self):
        """Set up test mesh and partitioner"""
        self.config = MeshConfig(n_nodes=1000)
        self.generator = DelaunayMeshGenerator4D(self.config)
        self.mesh = self.generator.generate()
        self.partitioner = MeshPartitioner(self.mesh, n_partitions=4)
    
    def test_simple_partition(self):
        """Test simple geometric partitioning"""
        partition = self.partitioner._partition_simple()
        
        self.assertEqual(len(partition), self.mesh.n_nodes)
        self.assertTrue(np.all(partition >= 0))
        self.assertTrue(np.all(partition < 4))
        
        # Check sizes
        unique, counts = np.unique(partition, return_counts=True)
        self.assertEqual(len(unique), 4)
        
        # Should be roughly equal
        mean_size = self.mesh.n_nodes / 4
        for count in counts:
            self.assertAlmostEqual(count, mean_size, delta=mean_size*0.2)
    
    def test_get_local_nodes(self):
        """Test getting local nodes for a rank"""
        local, ghost = self.partitioner.get_local_nodes(0)
        
        self.assertIsInstance(local, np.ndarray)
        self.assertIsInstance(ghost, np.ndarray)
        
        # Local nodes should be unique
        self.assertEqual(len(local), len(np.unique(local)))
        
        # Ghost nodes should not be in local
        for g in ghost:
            self.assertNotIn(g, local)
    
    def test_communication_pattern(self):
        """Test communication pattern generation"""
        pattern = self.partitioner.create_communication_pattern()
        
        self.assertIn(0, pattern)
        self.assertIsInstance(pattern[0], dict)


class TestMeshFactory(unittest.TestCase):
    """Test mesh factory function"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_create_new_mesh(self):
        """Test creating new mesh"""
        config = MeshConfig(
            n_nodes=100,
            save_mesh=False
        )
        
        mesh = create_mesh(config)
        self.assertEqual(mesh.n_nodes, 100)
    
    def test_load_existing_mesh(self):
        """Test loading existing mesh"""
        # First create and save mesh
        config1 = MeshConfig(
            n_nodes=100,
            save_mesh=True,
            mesh_file=os.path.join(self.test_dir, "mesh.h5")
        )
        mesh1 = create_mesh(config1)
        
        # Then load it
        config2 = MeshConfig(
            n_nodes=100,
            mesh_file=os.path.join(self.test_dir, "mesh.h5")
        )
        mesh2 = create_mesh(config2)
        
        # Should be the same
        np.testing.assert_array_equal(mesh1.coordinates, mesh2.coordinates)


class TestMeshScaling(unittest.TestCase):
    """Test mesh scaling behavior (performance tests)"""
    
    def test_scaling_small(self):
        """Test small mesh generation time"""
        import time
        
        config = MeshConfig(n_nodes=100)
        start = time.time()
        mesh = create_mesh(config)
        elapsed = time.time() - start
        
        # Should be fast (< 1 second)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(mesh.n_nodes, 100)
    
    @unittest.skip("Slow test - enable manually")
    def test_scaling_medium(self):
        """Test medium mesh generation (100k nodes)"""
        import time
        
        config = MeshConfig(n_nodes=100000)
        start = time.time()
        mesh = create_mesh(config)
        elapsed = time.time() - start
        
        print(f"100k nodes generated in {elapsed:.2f} seconds")
        self.assertLess(elapsed, 30.0)  # Should be under 30 seconds


if __name__ == '__main__':
    unittest.main(verbosity=2)