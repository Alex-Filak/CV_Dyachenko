import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, List, Dict, Any, Optional, Tuple, Union, Sequence
import warnings

class PointsCloud:
    """
    Unified N-dimentional points cloud representation

    Data storage: Symple numpy array, where firs k are spatial coordinates
                  remaining - values of scholar/vector field 
      
    """

    # Init method
    def __init__(self, np.ndarray,
                 spatial_dims: int=3,
                 field_names: Optional[List[str]]=None,
                 field_dimensions: Optional[List[float]]=None):

        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")

        self.n_points, self.total_dims = data.shape
        self.spatial_dims = spatial_dims

        if self.spatial_dims > self.total_dims:
            raise ValueError(f"Spatial dimensions ({spatial_dims}) cannot exceed total dimensions ({self.total_dims})")

        # Hendling fields metadata 
        self.field_names = field_names or []
        self.field_dimensions = field_dimensions or [1] * (self.total_dims - self.spatial_dims)
        
        # Validate field metadata
        if field_names:
            if len(field_names) != len(self.field_dimensions):
                raise ValueError("Length of field_names must match field_dimensions")
            expected_dims = sum(self.field_dimensions)
            actual_dims = self.total_dims - self.spatial_dims
            if expected_dims != actual_dims:
                raise ValueError(
                    f"Field dimensions mismatch: expected {expected_dims} attribute columns, "
                    f"but found {actual_dims} (total dims: {self.total_dims}, spatial dims: {self.spatial_dims})"
                )
        
        self.data = data.copy()        



    @classmethod

    # __________ FILE METHODS __________

    def from_file(cls, 
                    filepath: str, 
                    spatial_dims: int = 3,
                    field_names: Optional[List[str]] = None,
                    field_dimensions: Optional[List[int]] = None) -> 'PointsCloud':

        try:
            data = np.loadtxt(file_path)

        except Exception as e:
            raise IOError(f"Failed to load file {file_path}: {str(e)}")


        return cls(data, spatial_dims, field_names,field_dimensions) 

    def to_file(self, file_path: str):
        np.savetxt(file_path, sefl.data, fmt='%.6f')

    def copy(self):
        return PointsCloud(
            self.data.copy(),
            spatial_dims = self.spatial_dims,
            field_names = self.field_names,
            field_dimensions = self.field_dimension
            )
        


    # __________ ACCESS METHODS __________

    def get_spatial(self) -> np.ndarray:
        return self.data[:, :self.spatial_dims]

    def get_field(self, field_name: str) -> np.ndarray:
        if field_name not in self.field_names:
            raise ValueError(f"Field '{field_name}' not found. Available fields: {self.field_names}")

        idx = self.field_names.index(field_name)
        start_col = self.spatial_dims + sum(self.field_dimensions[:idx])
        end_col = start_col + self.field_dimensions[idx]
        return self.data[:, start_col:end_col]
    

    def add_field(self, name: str, value: np.ndarray, field_dim: int = 1):
        # Check given values
        if name in self.field_names:
            raise ValueError(f"Field '{name}' already exists")

        if len(value.shape) == 1:
            values = values.reshape(-1, 1)

        if values.shape[0] != self.n_points:
            raise ValueError(f"Values must have {self.n_points} points, got {values.shape[0]}")
        
        if values.shape[1] != field_dim:
            raise ValueError(f"Expected field dimension {field_dim}, got {values.shape[1]}")       

        # Add to data array
        self.data = np.hstack([self.data, values])
        self.total_dims += field_dim

        # Update metadata
        self.field_names.append(name)
        self.field_dimensions.append(field_dim)

    # __________ PRUNING __________

    def prune_random(self, fraction: float) -> 'PointsCloud':
        if not 0 < fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")

        n_keep = int(self.n_points * fraction)
        idx = np.random.choise(self.n_points, n_keep, replace=False)
        return PointsCloud(
            self.data[idx],
            spatial_dims = self.spatial_dims,
            field_names = self.field_names.copy(),
            field_dimensions = self.field_dimensions.copy()
            )


    #def prune_voxel() - 

    
    # __________ FILTERING SYSTEM __________

    #def filter() - filter points using custom condition function

    #def fileter_bbox() - Axes-aligned boundig box filter for spatial dimentions

    
    # __________ FIELD OPERATIONS __________

    #def constant_field() - add constant scolar/vector field

    #def scale_field() - multiply fields by scalar

    #def offset_field() - add offset to field value

    #def gaussian_smooth() - apply gaussian smooth ot scalar/vector field

    #def compute_gradient() - compute gradient of scalar field

    # __________ VISUALIZATION __________

    #def visualize() 




# __________ UTILITY FUNCTIONS __________

#def merge_clouds()

#def create_grid_points()
    

    
