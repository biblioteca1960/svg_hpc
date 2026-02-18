#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG Mesh Generation Module
4D Tetrahedral Mesh with Proper Connectivity

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026
"""

import numpy as np
from scipy.spatial import Delaunay, KDTree
import h5py
import logging
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
from mpi4py import MPI

# ------------------------------
# LOGGING
# ------------------------------
logger = logging.getLogger(__name__)


# ------------------------------
# DATA CLASSES
# ------------------------------
@dataclass
class MeshConfig:
    """Configuration for mesh generation"""
    dimension: int = 4
    n_nodes: int = 1000000
    distribution: str = "uniform"  # uniform, radial, clustered
    seed: int = 42
    periodic: bool = True
    refine_regions: List[Dict] = field(default_factory=list)
    min_neighbors: int = 10
    max_neighbors: int = 50
    save_mesh: bool = True
    mesh_file: Optional[str] = None


@dataclass
class MeshData:
    """Container for mesh data"""
    coordinates: np.ndarray
    connectivity: np.ndarray
    neighbors: List[np.ndarray]
    edges: Optional[np.ndarray] = None
    volumes: Optional[np.ndarray] = None
    boundary_nodes: Optional[np.ndarray] = None
    
    @property
    def n_nodes(self) -> int:
        return len(self.coordinates)
    
    @property
    def n_simplices(self) -> int:
        return len(self.connectivity)
    
    def summary(self) -> Dict:
        """Generate mesh summary statistics"""
        neighbor_counts = [len(nb) for nb in self.neighbors]
        return {
            'n_nodes': self.n_nodes,
            'n_simplices': self.n_simplices,
            'dimension': self.coordinates.shape[1],
            'neighbors_mean': float(np.mean(neighbor_counts)),
            'neighbors_std': float(np.std(neighbor_counts)),
            'neighbors_min': int(np.min(neighbor_counts)),
            'neighbors_max': int(np.max(neighbor_counts)),
            'bounds': [float(b) for b in np.ptp(self.coordinates, axis=0)],
            'volume_estimate': float(np.prod(np.ptp(self.coordinates, axis=0)))
        }


# ------------------------------
# MESH GENERATOR BASE CLASS
# ------------------------------
class MeshGenerator:
    """Base class for mesh generation"""
    
    def __init__(self, config: MeshConfig):
        self.config = config
        self.mesh = None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
    def generate(self) -> MeshData:
        """Generate mesh - to be overridden"""
        raise NotImplementedError
    
    def save(self, filename: str, mesh: MeshData):
        """Save mesh to HDF5 file"""
        with h5py.File(filename, 'w') as f:
            # Coordinates
            f.create_dataset('coordinates', data=mesh.coordinates, 
                           compression='gzip', compression_opts=9)
            
            # Connectivity
            f.create_dataset('connectivity', data=mesh.connectivity,
                           compression='gzip', compression_opts=9)
            
            # Neighbors (variable length)
            dt = h5py.vlen_dtype(np.dtype('int32'))
            nb_dataset = f.create_dataset('neighbors', (len(mesh.neighbors),), dtype=dt)
            for i, nb in enumerate(mesh.neighbors):
                nb_dataset[i] = nb.astype(np.int32)
            
            # Edges if present
            if mesh.edges is not None:
                f.create_dataset('edges', data=mesh.edges,
                               compression='gzip', compression_opts=9)
            
            # Volumes if present
            if mesh.volumes is not None:
                f.create_dataset('volumes', data=mesh.volumes,
                               compression='gzip', compression_opts=9)
            
            # Attributes
            f.attrs['dimension'] = mesh.coordinates.shape[1]
            f.attrs['n_nodes'] = mesh.n_nodes
            f.attrs['n_simplices'] = mesh.n_simplices
            f.attrs['distribution'] = self.config.distribution
            f.attrs['seed'] = self.config.seed
            
        logger.info(f"Mesh saved to {filename}")
    
    def load(self, filename: str) -> MeshData:
        """Load mesh from HDF5 file"""
        with h5py.File(filename, 'r') as f:
            coordinates = f['coordinates'][:]
            connectivity = f['connectivity'][:]
            
            # Load neighbors (variable length)
            neighbors = []
            for i in range(len(f['neighbors'])):
                neighbors.append(f['neighbors'][i][:])
            
            # Optional data
            edges = f['edges'][:] if 'edges' in f else None
            volumes = f['volumes'][:] if 'volumes' in f else None
        
        mesh = MeshData(
            coordinates=coordinates,
            connectivity=connectivity,
            neighbors=neighbors,
            edges=edges,
            volumes=volumes
        )
        
        logger.info(f"Mesh loaded from {filename}")
        return mesh


# ------------------------------
# DELAUNAY MESH GENERATOR (4D)
# ------------------------------
class DelaunayMeshGenerator4D(MeshGenerator):
    """4D Delaunay tetrahedral mesh generator"""
    
    def generate(self) -> MeshData:
        """Generate 4D Delaunay mesh"""
        np.random.seed(self.config.seed + self.rank)
        
        logger.info(f"Generating {self.config.dimension}D mesh with {self.config.n_nodes} nodes")
        
        # Generate points based on distribution
        if self.config.distribution == "uniform":
            points = self._generate_uniform()
        elif self.config.distribution == "radial":
            points = self._generate_radial()
        elif self.config.distribution == "clustered":
            points = self._generate_clustered()
        else:
            raise ValueError(f"Unknown distribution: {self.config.distribution}")
        
        # Apply refinements
        for region in self.config.refine_regions:
            points = self._refine_region(points, region)
        
        # Delaunay triangulation in 4D
        logger.info("Computing Delaunay triangulation...")
        tri = Delaunay(points)
        
        # Build neighbor lists
        logger.info("Building neighbor lists...")
        neighbors = self._build_neighbors(tri.simplices, len(points))
        
        # Compute edges
        logger.info("Computing edges...")
        edges = self._compute_edges(tri.simplices)
        
        # Compute simplex volumes (4D content)
        logger.info("Computing simplex volumes...")
        volumes = self._compute_volumes(points, tri.simplices)
        
        # Find boundary nodes
        boundary_nodes = self._find_boundary(points)
        
        mesh = MeshData(
            coordinates=points,
            connectivity=tri.simplices,
            neighbors=neighbors,
            edges=edges,
            volumes=volumes,
            boundary_nodes=boundary_nodes
        )
        
        # Log summary
        summary = mesh.summary()
        logger.info(f"Mesh generation complete: {summary}")
        
        return mesh
    
    def _generate_uniform(self) -> np.ndarray:
        """Generate uniformly distributed points in 4D hypercube"""
        return np.random.uniform(-1, 1, (self.config.n_nodes, self.config.dimension))
    
    def _generate_radial(self) -> np.ndarray:
        """Generate points with radial distribution"""
        # Generate directions uniformly on sphere
        points = np.random.randn(self.config.n_nodes, self.config.dimension)
        radii = np.random.power(2, self.config.n_nodes)  # r^2 distribution
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = points * radii[:, np.newaxis] * 2
        return points
    
    def _generate_clustered(self) -> np.ndarray:
        """Generate points with clustered distribution"""
        n_clusters = max(1, self.config.n_nodes // 10000)
        points = []
        
        for i in range(n_clusters):
            center = np.random.randn(self.config.dimension) * 0.5
            n_cluster = self.config.n_nodes // n_clusters
            if i == n_clusters - 1:
                n_cluster += self.config.n_nodes % n_clusters
            
            cluster_points = np.random.randn(n_cluster, self.config.dimension) * 0.2 + center
            points.append(cluster_points)
        
        return np.vstack(points)
    
    def _refine_region(self, points: np.ndarray, region: Dict) -> np.ndarray:
        """Add refined points in specific region"""
        center = np.array(region['center'])
        radius = region['radius']
        factor = region.get('factor', 2.0)
        
        # Find points in region
        distances = np.linalg.norm(points - center, axis=1)
        in_region = distances < radius
        
        if not np.any(in_region):
            return points
        
        # Add refined points
        n_refine = int(np.sum(in_region) * factor)
        refined = points[in_region] + np.random.randn(n_refine, self.config.dimension) * 0.05
        
        return np.vstack([points, refined])
    
    def _build_neighbors(self, simplices: np.ndarray, n_nodes: int) -> List[np.ndarray]:
        """Build neighbor lists from simplices"""
        neighbor_sets = [set() for _ in range(n_nodes)]
        
        for simplex in simplices:
            for i in range(len(simplex)):
                for j in range(len(simplex)):
                    if i != j:
                        neighbor_sets[simplex[i]].add(simplex[j])
        
        # Convert to sorted arrays for consistency
        neighbors = [np.array(sorted(list(nb)), dtype=np.int32) for nb in neighbor_sets]
        
        return neighbors
    
    def _compute_edges(self, simplices: np.ndarray) -> np.ndarray:
        """Compute all edges from simplices"""
        edge_set = set()
        
        for simplex in simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    if a < b:
                        edge_set.add((a, b))
                    else:
                        edge_set.add((b, a))
        
        edges = np.array(list(edge_set), dtype=np.int32)
        return edges
    
    def _compute_volumes(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        """Compute 4D simplex volumes"""
        volumes = np.zeros(len(simplices))
        
        for i, simplex in enumerate(simplices):
            # Get vertices
            v = points[simplex]
            
            # Compute edge vectors
            edges = v[1:] - v[0]
            
            # Volume = |det(edges)| / 4!
            det = np.linalg.det(edges)
            volumes[i] = abs(det) / 24.0  # 4! = 24
        
        return volumes
    
    def _find_boundary(self, points: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Find boundary nodes (simplified)"""
        # Use convex hull approximation
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(points[:, :3])  # Use 3D projection
            boundary = np.unique(hull.vertices)
            return boundary
        except:
            # Fallback: nodes with extreme coordinates
            bounds = np.percentile(points, [5, 95], axis=0)
            boundary = []
            for i, p in enumerate(points):
                if np.any(p < bounds[0]) or np.any(p > bounds[1]):
                    boundary.append(i)
            return np.array(boundary)


# ------------------------------
# MESH PARTITIONER
# ------------------------------
class MeshPartitioner:
    """Partition mesh for distributed computing"""
    
    def __init__(self, mesh: MeshData, n_partitions: int):
        self.mesh = mesh
        self.n_partitions = n_partitions
        self.partition = None
        self.ghost_nodes = None
        
    def partition_metis(self) -> np.ndarray:
        """Partition using METIS (if available)"""
        try:
            import pymetis
            
            # Build adjacency list
            adj_list = [nb.tolist() for nb in self.mesh.neighbors]
            
            # Call METIS
            n_cuts, partition = pymetis.part_graph(self.n_partitions, adjacency=adj_list)
            
            self.partition = np.array(partition)
            logger.info(f"METIS partition complete: {n_cuts} edge cuts")
            
        except ImportError:
            logger.warning("METIS not available, using simple partitioning")
            self.partition = self._partition_simple()
        
        return self.partition
    
    def _partition_simple(self) -> np.ndarray:
        """Simple geometric partitioning"""
        # Sort by first coordinate and split
        sort_idx = np.argsort(self.mesh.coordinates[:, 0])
        partition = np.zeros(len(sort_idx), dtype=np.int32)
        
        nodes_per_part = len(sort_idx) // self.n_partitions
        for i in range(self.n_partitions):
            start = i * nodes_per_part
            end = (i + 1) * nodes_per_part if i < self.n_partitions - 1 else len(sort_idx)
            partition[sort_idx[start:end]] = i
        
        return partition
    
    def get_local_nodes(self, rank: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get local nodes and ghost nodes for a rank"""
        if self.partition is None:
            self.partition_metis()
        
        # Local nodes
        local_mask = self.partition == rank
        local_indices = np.where(local_mask)[0]
        
        # Ghost nodes (neighbors of local nodes that are not local)
        ghost_set = set()
        for idx in local_indices:
            for nb in self.mesh.neighbors[idx]:
                if not local_mask[nb]:
                    ghost_set.add(nb)
        
        ghost_indices = np.array(list(ghost_set), dtype=np.int32)
        
        return local_indices, ghost_indices
    
    def create_communication_pattern(self) -> Dict[int, List[int]]:
        """Create communication pattern for MPI"""
        send_pattern = {}
        
        for rank in range(self.n_partitions):
            local, ghost = self.get_local_nodes(rank)
            
            # Find which ranks own the ghost nodes
            owner_ranks = {}
            for g in ghost:
                owner = self.partition[g]
                if owner not in owner_ranks:
                    owner_ranks[owner] = []
                owner_ranks[owner].append(g)
            
            send_pattern[rank] = owner_ranks
        
        return send_pattern


# ------------------------------
# FACTORY FUNCTION
# ------------------------------
def create_mesh(config: MeshConfig) -> MeshData:
    """Factory function to create mesh"""
    
    if config.mesh_file and os.path.exists(config.mesh_file):
        # Load existing mesh
        generator = DelaunayMeshGenerator4D(config)
        mesh = generator.load(config.mesh_file)
    else:
        # Generate new mesh
        generator = DelaunayMeshGenerator4D(config)
        mesh = generator.generate()
        
        if config.save_mesh:
            filename = config.mesh_file or f"mesh_{config.dimension}d_{config.n_nodes}.h5"
            generator.save(filename, mesh)
    
    return mesh


# ------------------------------
# UNIT TESTS
# ------------------------------
def test_mesh_generation():
    """Test mesh generation"""
    config = MeshConfig(
        dimension=4,
        n_nodes=1000,
        distribution="uniform",
        seed=42
    )
    
    mesh = create_mesh(config)
    
    assert mesh.n_nodes == 1000
    assert mesh.coordinates.shape[1] == 4
    assert len(mesh.neighbors) == 1000
    
    # Check neighbor counts
    neighbor_counts = [len(nb) for nb in mesh.neighbors]
    assert np.mean(neighbor_counts) > 10
    
    print("Mesh generation test passed!")
    return mesh


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mesh_generation()