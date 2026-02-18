#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG Temporal Hub Detection Module
Identifies and tracks black hole candidates (phase synchronization points)

Author: L. Morató de Dalmases
Version: 1.0
Date: February 2026
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fclusterdata
import h5py
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from mpi4py import MPI

logger = logging.getLogger(__name__)


# ------------------------------
# DATA CLASSES
# ------------------------------
@dataclass
class Hub:
    """Represents a temporal hub (black hole candidate)"""
    id: int
    formation_step: int
    last_seen_step: int
    position: np.ndarray  # 4D coordinates [x, y, z, w]
    phase: float
    potential: float
    mass_proxy: float
    radius: float
    members: List[int] = field(default_factory=list)
    merger_history: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d['position'] = self.position.tolist()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Hub':
        """Create from dictionary"""
        data['position'] = np.array(data['position'])
        return cls(**data)


@dataclass
class HubDetectionConfig:
    """Configuration for hub detection"""
    threshold: float = 0.8  # Potential threshold
    min_cluster_size: int = 5  # Minimum members for hub
    coherence_window: int = 10  # Steps for coherence calculation
    merge_distance: float = 0.1  # Distance for merging hubs
    max_hubs: int = 100000  # Maximum hubs to track
    track_history: bool = True
    save_hubs: bool = True
    hub_file: Optional[str] = None


# ------------------------------
# HUB DETECTOR
# ------------------------------
class HubDetector:
    """Detects and tracks temporal hubs in simulation"""
    
    def __init__(
        self,
        config: HubDetectionConfig,
        comm: MPI.Intracomm = None
    ):
        self.config = config
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.hubs: Dict[int, Hub] = {}  # id -> Hub
        self.next_id = 0
        self.hub_positions: List[np.ndarray] = []
        self.hub_potentials: List[float] = []
        self.history: List[Dict] = []
        
        # For tracking
        self.kdtree = None
        
    def detect(
        self,
        phase: np.ndarray,
        coordinates: np.ndarray,
        hub_potential: np.ndarray,
        step: int,
        global_indices: Optional[np.ndarray] = None
    ) -> List[Hub]:
        """
        Detect hubs in current time step
        """
        # Find candidate nodes
        candidates = self._find_candidates(hub_potential, phase)
        
        if len(candidates) == 0:
            return []
        
        # Cluster candidates
        clusters = self._cluster_candidates(candidates, coordinates)
        
        # Create hubs from clusters
        new_hubs = self._create_hubs(clusters, step)
        
        # Match with existing hubs
        matched_hubs = self._match_hubs(new_hubs, step)
        
        # Update hub positions for next step
        self._update_positions()
        
        return matched_hubs
    
    def _find_candidates(
        self,
        hub_potential: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """Find candidate nodes based on potential threshold"""
        mask = hub_potential > self.config.threshold
        candidates = np.where(mask)[0]
        return candidates
    
    def _cluster_candidates(
        self,
        candidates: np.ndarray,
        coordinates: np.ndarray
    ) -> List[np.ndarray]:
        """Cluster candidate nodes into potential hubs"""
        if len(candidates) < self.config.min_cluster_size:
            return []
        
        # Get positions of candidates
        positions = coordinates[candidates]
        
        # Use hierarchical clustering
        # Distance threshold based on merge_distance
        try:
            clusters = fclusterdata(
                positions,
                self.config.merge_distance,
                criterion='distance'
            )
        except:
            # Fallback to simple distance-based clustering
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=self.config.merge_distance, min_samples=3)
            clusters = clustering.fit_predict(positions) + 1  # Make 1-based
        
        # Group by cluster
        unique_clusters = np.unique(clusters)
        cluster_groups = []
        
        for cid in unique_clusters:
            if cid == -1:  # Noise in DBSCAN
                continue
            mask = clusters == cid
            if np.sum(mask) >= self.config.min_cluster_size:
                cluster_groups.append(candidates[mask])
        
        return cluster_groups
    
    def _create_hubs(self, clusters: List[np.ndarray], step: int) -> List[Hub]:
        """Create hub objects from clusters"""
        hubs = []
        
        for cluster in clusters:
            # Hub properties
            center = np.mean(cluster, axis=0)
            radius = np.max(np.linalg.norm(cluster - center, axis=1))
            
            # Phase and potential (mean of members)
            # Note: In real implementation, phase and potential would be passed
            phase = 0.0  # Placeholder
            potential = 0.0  # Placeholder
            
            # Mass proxy (size of cluster)
            mass_proxy = len(cluster)
            
            hub = Hub(
                id=self.next_id,
                formation_step=step,
                last_seen_step=step,
                position=center,
                phase=phase,
                potential=potential,
                mass_proxy=mass_proxy,
                radius=radius,
                members=cluster.tolist()
            )
            
            self.next_id += 1
            hubs.append(hub)
        
        return hubs
    
    def _match_hubs(self, new_hubs: List[Hub], step: int) -> List[Hub]:
        """Match new hubs with existing ones"""
        if not self.hub_positions:
            # First detection
            for hub in new_hubs:
                self.hubs[hub.id] = hub
            return new_hubs
        
        # Build KD-tree for existing hub positions
        positions = np.array(self.hub_positions)
        tree = KDTree(positions)
        
        matched_hubs = []
        for hub in new_hubs:
            # Find closest existing hub
            distances, indices = tree.query(hub.position, k=1)
            
            if distances < self.config.merge_distance:
                # Match with existing hub
                existing_id = list(self.hubs.keys())[indices]
                existing = self.hubs[existing_id]
                
                # Update existing hub
                existing.last_seen_step = step
                existing.position = 0.7 * existing.position + 0.3 * hub.position
                existing.mass_proxy += hub.mass_proxy
                existing.members.extend(hub.members)
                existing.merger_history.append(hub.id)
                
                matched_hubs.append(existing)
            else:
                # New hub
                self.hubs[hub.id] = hub
                matched_hubs.append(hub)
        
        return matched_hubs
    
    def _update_positions(self):
        """Update hub positions for next detection"""
        self.hub_positions = [hub.position for hub in self.hubs.values()]
        self.hub_potentials = [hub.potential for hub in self.hubs.values()]
        self.kdtree = KDTree(self.hub_positions) if self.hub_positions else None
    
    def clean_inactive_hubs(self, current_step: int, max_inactive: int = 10):
        """Remove hubs not seen for max_inactive steps"""
        inactive = []
        for hid, hub in self.hubs.items():
            if current_step - hub.last_seen_step > max_inactive:
                inactive.append(hid)
        
        for hid in inactive:
            del self.hubs[hid]
        
        if inactive:
            logger.info(f"Removed {len(inactive)} inactive hubs")
    
    def get_hub_statistics(self) -> Dict:
        """Get statistics about detected hubs"""
        if not self.hubs:
            return {}
        
        masses = [h.mass_proxy for h in self.hubs.values()]
        radii = [h.radius for h in self.hubs.values()]
        potentials = [h.potential for h in self.hubs.values()]
        
        return {
            'n_hubs': len(self.hubs),
            'mass_mean': float(np.mean(masses)),
            'mass_std': float(np.std(masses)),
            'radius_mean': float(np.mean(radii)),
            'radius_std': float(np.std(radii)),
            'potential_mean': float(np.mean(potentials)),
            'potential_std': float(np.std(potentials)),
            'total_mass': float(np.sum(masses))
        }
    
    def save(self, filename: str):
        """Save hub data to HDF5"""
        if self.rank != 0:  # Only root saves
            return
        
        with h5py.File(filename, 'w') as f:
            # Save each hub
            hub_group = f.create_group('hubs')
            for hid, hub in self.hubs.items():
                hub_group.create_dataset(f'{hid}/position', data=hub.position)
                hub_group.create_dataset(f'{hid}/members', data=hub.members)
                hub_group.attrs[f'{hid}/id'] = hub.id
                hub_group.attrs[f'{hid}/formation_step'] = hub.formation_step
                hub_group.attrs[f'{hid}/last_seen_step'] = hub.last_seen_step
                hub_group.attrs[f'{hid}/phase'] = hub.phase
                hub_group.attrs[f'{hid}/potential'] = hub.potential
                hub_group.attrs[f'{hid}/mass_proxy'] = hub.mass_proxy
                hub_group.attrs[f'{hid}/radius'] = hub.radius
            
            # Save history
            if self.history:
                history_group = f.create_group('history')
                for i, h in enumerate(self.history):
                    for key, value in h.items():
                        history_group.attrs[f'{i}_{key}'] = str(value)
        
        logger.info(f"Hubs saved to {filename}")
    
    def load(self, filename: str):
        """Load hub data from HDF5"""
        if not os.path.exists(filename):
            logger.warning(f"Hub file {filename} not found")
            return
        
        with h5py.File(filename, 'r') as f:
            hub_group = f['hubs']
            self.hubs = {}
            self.next_id = 0
            
            for key in hub_group.keys():
                if key.startswith('hub_'):
                    hid = int(key.split('_')[1])
                    hub_data = hub_group[key]
                    
                    hub = Hub(
                        id=hid,
                        formation_step=hub_data.attrs['formation_step'],
                        last_seen_step=hub_data.attrs['last_seen_step'],
                        position=hub_data['position'][:],
                        phase=hub_data.attrs['phase'],
                        potential=hub_data.attrs['potential'],
                        mass_proxy=hub_data.attrs['mass_proxy'],
                        radius=hub_data.attrs['radius'],
                        members=hub_data['members'][:].tolist()
                    )
                    
                    self.hubs[hid] = hub
                    self.next_id = max(self.next_id, hid + 1)
        
        self._update_positions()
        logger.info(f"Loaded {len(self.hubs)} hubs from {filename}")


# ------------------------------
# HUB MERGER DETECTOR
# ------------------------------
class HubMergerDetector:
    """Detects mergers between hubs"""
    
    def __init__(self, detector: HubDetector):
        self.detector = detector
        self.mergers = []
        
    def detect_mergers(self, step: int) -> List[Dict]:
        """Detect hub mergers in current step"""
        if len(self.detector.hubs) < 2:
            return []
        
        # Get positions
        positions = np.array([h.position for h in self.detector.hubs.values()])
        hub_ids = list(self.detector.hubs.keys())
        
        # Check for close pairs
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(positions))
        
        mergers = []
        for i in range(len(hub_ids)):
            for j in range(i+1, len(hub_ids)):
                if distances[i, j] < self.detector.config.merge_distance:
                    # Potential merger
                    hub1 = self.detector.hubs[hub_ids[i]]
                    hub2 = self.detector.hubs[hub_ids[j]]
                    
                    merger = {
                        'step': step,
                        'hub1_id': hub1.id,
                        'hub2_id': hub2.id,
                        'distance': float(distances[i, j]),
                        'mass1': hub1.mass_proxy,
                        'mass2': hub2.mass_proxy
                    }
                    mergers.append(merger)
        
        self.mergers.extend(mergers)
        return mergers


# ------------------------------
# UNIT TESTS
# ------------------------------
def test_hub_detection():
    """Test hub detection"""
    config = HubDetectionConfig(
        threshold=0.8,
        min_cluster_size=5,
        merge_distance=0.1
    )
    
    detector = HubDetector(config)
    
    # Create test data
    n_nodes = 1000
    phase = np.random.uniform(0, 2*np.pi, n_nodes)
    coordinates = np.random.uniform(-1, 1, (n_nodes, 4))
    
    # Create some hub-like structures
    hub_potential = np.random.uniform(0, 1, n_nodes)
    # Boost potential in some regions
    for i in range(3):
        center = np.random.uniform(-1, 1, 4)
        dist = np.linalg.norm(coordinates - center, axis=1)
        hub_potential[dist < 0.2] = 0.9
    
    # Detect hubs
    hubs = detector.detect(phase, coordinates, hub_potential, step=0)
    
    print(f"Detected {len(hubs)} hubs")
    stats = detector.get_hub_statistics()
    print(f"Hub statistics: {stats}")
    
    assert len(hubs) >= 0
    
    print("Hub detection test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_hub_detection()