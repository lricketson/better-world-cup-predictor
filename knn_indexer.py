import torch
from typing import Dict, List, Any, Tuple


class TacticalKNNIndexer:
    """
    Executes a GPU-accelerated Euclidean distance calculation with torch.cdist() against a database
    of M >= 10,000 historical match feature vectors to find the K-most similar neighbours.
    """

    def __init__(
        self,
        historical_database: Dict[int, torch.Tensor] = None,
        k_neighbours: int = 50,
    ):
        """
        historical_database: a dictionary mapping minute timestamps (which are the keys, e.g.
        10, 20, 30...) to a 2D PyTorch tensor of shape (M=num_matches, 4) containing normalised
        historical vectors.
        """

        self.use_pinned = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_pinned else "cpu")
        self.k = k_neighbours

        # if no historical db passed, initialise an empty registry
        self.db = historical_database or {}

        def register_historical_slice(
            self, minute_timestamp: int, vectors_matrix: torch.Tensor
        ):
            """
            Loads a matrix of shape (M, 4) into memory for a specific match minute. Automatically
            pushes the database to the active hardware device (GPU VRAM or CPU RAM).
            """
            self.db[minute_timestamp] = vectors_matrix.to(
                self.device, dtype=torch.float32, non_blocking=True
            )
