"""
Plateau Search (v4.6)

Find stable parameter regions using Connected Components.
Avoids selecting isolated peaks (overfitting risk).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import ndimage


@dataclass
class PlateauResult:
    """Result of plateau search."""

    indices: tuple[int, ...]  # Grid indices (sl_idx, pt_idx, time_idx)
    component_size: int  # Size of the selected component
    n_components: int  # Total number of components found
    threshold_used: float  # Threshold percentile used
    center_of_mass: tuple[float, ...]  # Exact center of mass


def find_plateau_center(
    scores: np.ndarray,
    grid_shape: tuple[int, ...],
    threshold_percentile: float = 90,
    fallback_percentile: float = 80,
    min_component_size: int = 2,
) -> PlateauResult:
    """
    Find the center of the largest high-score plateau.

    Algorithm:
    1. Threshold scores to create binary mask (top N%)
    2. Find connected components in the mask
    3. Select the largest component
    4. Return center of mass of largest component

    Args:
        scores: 1D array of composite scores (flattened grid)
        grid_shape: Original N-D grid shape, e.g., (3, 5, 5)
        threshold_percentile: Initial threshold (90 = top 10%)
        fallback_percentile: Fallback if no components found
        min_component_size: Minimum component size to consider

    Returns:
        PlateauResult with grid indices and metadata
    """
    scores_nd = scores.reshape(grid_shape)

    # Primary attempt with threshold_percentile
    result = _find_components(
        scores_nd,
        scores,
        threshold_percentile,
        min_component_size,
    )

    if result is not None:
        return result

    # Fallback with lower threshold
    result = _find_components(
        scores_nd,
        scores,
        fallback_percentile,
        min_component_size,
    )

    if result is not None:
        return result

    # Last resort: return argmax (isolated peak)
    flat_idx = int(np.argmax(scores))
    indices = np.unravel_index(flat_idx, grid_shape)

    return PlateauResult(
        indices=tuple(int(i) for i in indices),
        component_size=1,
        n_components=0,
        threshold_used=0.0,
        center_of_mass=tuple(float(i) for i in indices),
    )


def _find_components(
    scores_nd: np.ndarray,
    scores_flat: np.ndarray,
    percentile: float,
    min_size: int,
) -> Optional[PlateauResult]:
    """
    Internal function to find connected components at given threshold.

    Returns None if no valid components found.
    """
    threshold = np.percentile(scores_flat, percentile)
    binary_mask = (scores_nd >= threshold).astype(np.int32)

    # Find connected components
    # structure=None uses default connectivity (face-connected in N-D)
    labeled, n_components = ndimage.label(binary_mask)

    if n_components == 0:
        return None

    # Calculate component sizes
    component_sizes = ndimage.sum(
        binary_mask,
        labeled,
        range(1, n_components + 1),
    )

    # Filter by minimum size
    valid_components = [
        (i + 1, size)
        for i, size in enumerate(component_sizes)
        if size >= min_size
    ]

    if not valid_components:
        return None

    # Select largest component
    largest_label, largest_size = max(valid_components, key=lambda x: x[1])

    # Calculate center of mass
    center = ndimage.center_of_mass(binary_mask, labeled, largest_label)

    # Round to nearest integer indices
    indices = tuple(int(round(c)) for c in center)

    # Clamp to valid range
    indices = tuple(
        min(max(0, idx), dim - 1)
        for idx, dim in zip(indices, scores_nd.shape)
    )

    return PlateauResult(
        indices=indices,
        component_size=int(largest_size),
        n_components=n_components,
        threshold_used=percentile,
        center_of_mass=tuple(float(c) for c in center),
    )


def visualize_plateau(
    scores: np.ndarray,
    grid_shape: tuple[int, ...],
    result: PlateauResult,
    axis_names: Optional[list[str]] = None,
) -> str:
    """
    Create ASCII visualization of plateau search result.

    Only works for 2D or 3D grids.

    Args:
        scores: Flattened scores array
        grid_shape: Grid shape
        result: PlateauResult from find_plateau_center
        axis_names: Names for each axis

    Returns:
        ASCII string visualization
    """
    if len(grid_shape) not in (2, 3):
        return f"Visualization not supported for {len(grid_shape)}D grid"

    if axis_names is None:
        axis_names = [f"dim{i}" for i in range(len(grid_shape))]

    scores_nd = scores.reshape(grid_shape)
    threshold = np.percentile(scores, result.threshold_used)

    lines = []
    lines.append(f"Plateau Search Result (threshold={result.threshold_used}%)")
    lines.append(f"Selected: {result.indices}")
    lines.append(f"Component size: {result.component_size}")
    lines.append(f"Total components: {result.n_components}")
    lines.append("")

    if len(grid_shape) == 2:
        # 2D visualization
        lines.append(f"  {axis_names[1]} →")
        for i in range(grid_shape[0]):
            row = ""
            for j in range(grid_shape[1]):
                if (i, j) == result.indices:
                    row += " ★"
                elif scores_nd[i, j] >= threshold:
                    row += " ●"
                else:
                    row += " ○"
            prefix = f"{i}" if i == grid_shape[0] // 2 else " "
            lines.append(f"{prefix} {row}")
        lines.append(f"  ↓ {axis_names[0]}")

    elif len(grid_shape) == 3:
        # 3D: show slice at selected index
        sel_idx = result.indices[0]
        slice_2d = scores_nd[sel_idx, :, :]

        lines.append(f"Slice at {axis_names[0]}={sel_idx}")
        lines.append(f"  {axis_names[2]} →")
        for i in range(grid_shape[1]):
            row = ""
            for j in range(grid_shape[2]):
                if result.indices == (sel_idx, i, j):
                    row += " ★"
                elif slice_2d[i, j] >= threshold:
                    row += " ●"
                else:
                    row += " ○"
            lines.append(f"  {row}")
        lines.append(f"  ↓ {axis_names[1]}")

    lines.append("")
    lines.append("Legend: ★=selected, ●=plateau, ○=below threshold")

    return "\n".join(lines)


def grid_idx_to_params(
    idx: tuple[int, ...],
    param_ranges: dict[str, list],
) -> dict:
    """
    Convert grid indices to actual parameter values.

    Args:
        idx: Grid indices (e.g., (1, 2, 3))
        param_ranges: Dict of param_name -> list of values

    Returns:
        Dict of param_name -> selected value
    """
    param_names = list(param_ranges.keys())

    if len(idx) != len(param_names):
        raise ValueError(
            f"Index length {len(idx)} != param count {len(param_names)}"
        )

    result = {}
    for i, name in enumerate(param_names):
        values = param_ranges[name]
        result[name] = values[idx[i]]

    return result
