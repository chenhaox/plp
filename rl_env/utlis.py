"""
Created on 2025/8/6 
Author: Hao Chen (chen960216@gmail.com)
"""
import numpy as np


def generate_object_classes(pallet_dims,
                            num_classes=5,
                            random_hdl=None):
    """
    Generates a list of random object classes for a pallet loading problem.

    Each object class has an ID, dimensions (dims), and a count.
    The dimensions are generated to be smaller than the pallet dimensions.

    Args:
        pallet_dims (tuple): A tuple of 3 integers representing the
                                (length, width, height) of the pallet.
        num_classes (int): The number of different object classes to generate.

    Returns:
        list: A list of dictionaries, where each dictionary represents an
              object class. For example:
              [{'id': 1, 'dims': (4, 2, 4), 'count': 2}, ...]
    """
    if not isinstance(pallet_dims, (list, tuple)) or len(pallet_dims) != 3:
        raise ValueError("pallet_dims must be a list or tuple of 3 integers.")
    if not all(isinstance(dim, int) and dim > 0 for dim in pallet_dims):
        raise ValueError("All pallet dimensions must be positive integers.")
    if random_hdl is None:
        randint = np.random.randint
    else:
        if hasattr(random_hdl, 'integers'):
            # If the random handler has an 'integers' method, use it.
            # This is useful for libraries like numpy or random.
            randint = random_hdl.integers
        else:
            # Otherwise, fall back to the standard randint method.
            randint = random_hdl.randint
    object_classes = []
    pallet_l, pallet_w, pallet_h = pallet_dims
    used_wd = []
    idx = 1
    while len(object_classes) < num_classes:
        obj_l = randint(max(1, pallet_l // 6), max(1, pallet_l // 2))
        obj_w = randint(max(1, pallet_l // 6), max(1, pallet_w // 2))
        if (obj_w, obj_l) in used_wd or (obj_l, obj_w) in used_wd:
            # already have this pair; skip to avoid duplicates
            continue
        obj_h = randint(max(1, pallet_l // 6), max(1, pallet_h // 2))
        # Generate a random count for this object class.
        count = randint(1, 10)
        used_wd.append((obj_w, obj_l))
        object_classes.append({
            'id': idx,
            'dims': (obj_l, obj_w, obj_h),
            'count': count
        })
        idx += 1
    return object_classes


def generate_grid(dimensions):
    """
    Generate all (x, y, z) grid points within a 3D rectangular boundary
    and sort them in lexicographic order: x -> y -> z.

    Parameters
    ----------
    dimensions : tuple of int
        The size of the grid in (x, y, z) form.

    Returns
    -------
    points_sorted : np.ndarray
        Array of shape (N, 3) containing all points sorted by x, y, z.
    """
    x_vals = np.arange(dimensions[0])
    y_vals = np.arange(dimensions[1])
    z_vals = np.arange(dimensions[2])

    # Here x increases fastest, then y, then z
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='xy')
    points = np.stack([X.ravel(order='F'),
                       Y.ravel(order='F'),
                       Z.ravel(order='F')], axis=-1)
    return points
