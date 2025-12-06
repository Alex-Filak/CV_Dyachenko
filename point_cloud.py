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
    def __init__(self,
                 data: np.ndarray,
                 spatial_dims: int=3,
                 field_names: Optional[List[str]]=None,
                 field_dimensions: Optional[List[int]]=None):

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
            data = np.loadtxt(filepath)

        except Exception as e:
            raise IOError(f"Failed to load file {filepath}: {str(e)}")


        return cls(data, spatial_dims, field_names, field_dimensions) 

    def to_file(self, file_path: str):
        np.savetxt(file_path, self.data, fmt='%.6f')

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

    def subsample_random(self, fraction: float) -> 'PointsCloud':
        """
        Random Point Sampling

        From the given point cloud randomly select given percent of
        points.

        Parametrs:
        fraction: float
          Percent of points, which would be selected.
        
        Returns:
          Subsampled point cloud
        """

        if not 0 < fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")

        # Select given fraction of points
        n_keep = int(self.n_points * fraction)
        idx = np.random.choice(self.n_points, n_keep, replace=False)

        return PointsCloud(
            self.data[idx],
            spatial_dims = self.spatial_dims,
            field_names = self.field_names.copy(),
            field_dimensions = self.field_dimensions.copy()
            )


    def subsample_voxel(self, voxel_size: Union[float, Sequence[float]]) -> 'PointsCloud':
        """
        Voxel Point Sampling

        Separate spaces on voxels with size voxel_size and 
        select one point from each voxel, which is the
        most close to voxel's center.

        Parametrs:
        voxel_size: float
          Size of voxels which would be used

        Returns:
          Subsampled point cloud
        """
        spatial = self.get_spatial()
        
        if isinstance(voxel_size, (int, float)):
            voxel_size = [voxel_size] * self.spatial_dims
        
        if len(voxel_size) != self.spatial_dims:
            raise ValueError(f"Voxel size must have {self.spatial_dims} dimensions")
        
        # Compute voxel indices
        voxel_indices = np.floor(spatial / voxel_size).astype(int)
        
        # Create unique voxel keys
        voxel_keys = [tuple(idx) for idx in voxel_indices]
        unique_voxels = np.unique(voxel_keys, axis=0)
        
        selected_indices = []

        # for each voxel select centroid point
        for voxel in unique_voxels:
            mask = np.all(voxel_indices == voxel, axis=1)
            if np.any(mask):
                # Get centroid of points in this voxel
                points_in_voxel = spatial[mask]
                centroid = np.mean(points_in_voxel, axis=0)
                
                # Find closest point to centroid
                distances = np.linalg.norm(points_in_voxel - centroid, axis=1)
                closest_idx = np.where(mask)[0][np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return PointsCloud(
            self.data[selected_indices],
            spatial_dims=self.spatial_dims,
            field_names=self.field_names.copy(),
            field_dimensions=self.field_dimensions.copy()
        )

    def subsample_fps(self, k: int) -> "PointsCloud":
        """
        Farthest Point Sampling

        Select k points such as each new point as far as posible from the
        already selected points.

        Parametrs:
        k: int
          Number of points which will be selected

        Returns:
          Subsampled point cloud with k points
        """

        if not (1 <= k <= self.n_points):
            raise ValueError(f"Num of sampled points must be between 1 and {self.n_points}, got {k}")

        if k == self.n_points:
            return self.copy()

        spatial  = self.get_spatial()
        n_points = spatial.shape[0]

        selected_indices = []
        min_distances =np.full(n_points, np.inf)

        # Choose first point with idx 0
        first_idx = 0
        selected_indices[0] = first_idx
        min_distances[first_idx] = 0.0

        # Calculate distances from the first point
        dists = np.linalg.norm(spatial - spatial[first_idx], axis=1)
        min_distances = np.minimum(min_distance, dists)

        # Choose other k-1 points
        for i in range(1, k):
            next_idx = np.argmax(min_distances)
            selected_indices[i] = next_idx

            new_dists = np.linalg.norm(coords - coords[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_dists)

        return PointsCloud(
            self.data[],
            spatial_dims = self.spatial_dims,
            field_names  = self.field_names.copy(),
            field_dimensions = self_field_dimensions.copy()
            )

        

        
    
    # __________ FILTERING SYSTEM __________

    def filter(self, condition: Callable[[np.ndarray], bool]) -> 'PointsCloud':

        mask = np.apply_along_axis(condition, 1, self.data)

        return PointsCloud(
            self.data[mask],
            spatial_dims = self.spatial_dims,
            field_names = self.field_names.copy(),
            field_dimensions =self.field_dimensions.copy()
            )


    def fileter_bbox(self, min_bounds: Sequence[float], max_bounds: Sequence[float]) -> 'PointsCloud':

        if len(min_bounds) != self.spatial_dims or len(max_bounds) != self.spatial_dims:
            raise ValueError(f"Bounds must have {self.spatial_dims} dimensions")

        spatial = self.get_spatial()
        mask = np.all((spatial >= min_bounds) & (spatial <= max_bounds), axis=1)

        return PointsCloud(
            self.data[mask],
            spatial_dims = self.spatial_dims,
            field_names = self.field_names,
            field_dimensions = self.field_dimensions
            )

    # __________ FIELD OPERATIONS __________

    #def constant_field() - add constant scolar/vector field

    #def scale_field() - multiply fields by scalar

    #def offset_field() - add offset to field value

    #def gaussian_smooth() - apply gaussian smooth ot scalar/vector field

    #def compute_gradient() - compute gradient of scalar field

    # __________ VISUALIZATION __________

    def visualize(self,
                    spatial_dim_indices: Optional[List[int]] = None,
                    color: Union[str, Tuple[float, float, float], np.ndarray, None] = None,
                    color_field: Optional[str] = None,
                    color_map: Optional[str] = None,
                    size: Union[float, str, np.ndarray] = 1.0,
                    backend: str = 'matplotlib',
                    title: str = "Point Cloud",
                    figsize: Tuple[int, int] = (10, 8),
                    **kwargs):

        if spatial_dim_indices is None:
            n_plot_dims = min(3, self.spatial_dims)
            spatial_dim_indices = list(range(n_plot_dims))

        else:
            n_plot_dims = len(spatial_dim_indices)

        if n_plot_dims not in (2, 3):
            raise ValueError("Can only visualize 2D or 3D spatial data")
        
        coords = self.data[:, spatial_dim_indices]

        final_colors = None
        color_is_rgb = False
        
        if color is not None:

            if isinstance(color, str):

                final_colors = np.tile(matplotlib.colors.to_rgb(color), (self.n_points, 1))
                color_is_rgb = True
            elif isinstance(color, tuple):

                rgb = color[:3]
                final_colors = np.tile(rgb, (self.n_points, 1))
                color_is_rgb = True

            elif isinstance(color, np.ndarray):

                if color.shape == (self.n_points, 3) or color.shape == (self.n_points, 4):
                    final_colors = color[:, :3]  # Ensure RGB
                    color_is_rgb = True

                else:
                    raise ValueError("Color array must be (N, 3) or (N, 4)")
        elif color_field is not None:

            if color_field not in self.field_names:
                raise ValueError(f"Color field '{color_field}' not found")
            
            field_vals = self.get_field(color_field)
            field_dim = field_vals.shape[1]
            field_type = self.field_types[self.field_names.index(color_field)]
            
            if field_type == 'rgb' or field_dim == 3:

                final_colors = field_vals[:, :3]  # Ensure only RGB

                if np.max(final_colors) > 1.0:
                    final_colors = final_colors / 255.0

                color_is_rgb = True
            else:
                norm_vals = field_vals.flatten()

                if np.ptp(norm_vals) > 0:
                    norm_vals = (norm_vals - np.min(norm_vals)) / np.ptp(norm_vals)

                cmap = plt.get_cmap(color_map or 'viridis')
                final_colors = cmap(norm_vals)[:, :3]
                color_is_rgb = True
        
        final_sizes = None

        if isinstance(size, str):

            if size not in self.field_names:
                raise ValueError(f"Size field '{size}' not found")
            
            size_vals = self.get_field(size)
            if size_vals.shape[1] != 1:
                raise ValueError("Size field must be scalar")
            
            vals = size_vals.flatten()

            if np.ptp(vals) > 0:
                norm_vals = (vals - np.min(vals)) / np.ptp(vals)
                final_sizes = 5 + 45 * norm_vals

            else:
                final_sizes = np.full(self.n_points, 25)

        elif isinstance(size, (int, float)):
            final_sizes = np.full(self.n_points, float(size))

        elif isinstance(size, np.ndarray):

            if size.shape == (self.n_points,):
                final_sizes = size

            else:
                raise ValueError("Size array must be (N,)")
        else:
            final_sizes = np.full(self.n_points, 1.0)

        # Dispatch to backend
        if backend == 'matplotlib':
            self._vis_matplotlib(coords, final_colors, color_is_rgb, final_sizes, 
                                n_plot_dims, title, figsize, **kwargs)
        elif backend == 'plotly':
            self._vis_plotly(coords, final_colors, color_is_rgb, final_sizes, 
                            n_plot_dims, title, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _vis_matplotlib(self, coords, colors, color_is_rgb, sizes,
                       n_dims, title, figsize, **kwargs):
        fig = plt.figure(figsize=figsize)
        
        scatter_args = {
            's': sizes,
            'alpha': 0.6,
            'edgecolors': 'none'
        }
        
        if colors is not None:
            scatter_args['c'] = colors
        
        if n_dims == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter_args.update({
                'xs': coords[:, 0],
                'ys': coords[:, 1],
                'zs': coords[:, 2],
            })
            if not color_is_rgb and colors is not None:
                scatter_args['cmap'] = kwargs.get('cmap', 'viridis')
            
            sc = ax.scatter(**scatter_args)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        else:  # 2D
            ax = fig.add_subplot(111)
            scatter_args.update({
                'x': coords[:, 0],
                'y': coords[:, 1],
            })

        if not color_is_rgb and colors is not None:
            
            sc = ax.scatter(**scatter_args)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
        
        if colors is not None and not color_is_rgb:
            plt.colorbar(sc, ax=ax, label='Field Value')
        
        plt.title(title)
        plt.tight_layout()
        plt.show()


# __________ UTILITY FUNCTIONS __________

#def merge_clouds()

#def create_grid_points()
    

    
