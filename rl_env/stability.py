from typing import Dict

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rasterio.features import rasterize


# --- CORE LOGIC ---

class Item:
    """A simple class to represent a cuboid item."""

    def __init__(self, dimensions, position=(0, 0, 0), color='blue', name=''):
        self.w, self.d, self.h = dimensions
        self.x, self.y, self.z = position
        self.color = color
        self.name = name

    def copy(self):
        """Returns a copy of the item."""
        return Item((self.w, self.d, self.h), (self.x, self.y, self.z), self.color, self.name)


def calculate_stability(new_item,
                        height_map,
                        feasibility_map,
                        cog_uncertainty_ratio=0.1,
                        toggle_viz=False) -> Dict:
    """
    Calculates the stability of a new item placement, now using cell corners for an accurate hull.
    """
    x_start, y_start = int(new_item.x), int(new_item.y)
    x_end, y_end = x_start + int(new_item.w), y_start + int(new_item.d)
    if x_end > height_map.shape[0] or y_end > height_map.shape[1]:
        return {"is_stable": False, "support_polygon": None, "reason": "Item out of bounds."}
    footprint_h_map = height_map[x_start:x_end, y_start:y_end]
    support_height = np.max(footprint_h_map)
    new_item.z = support_height
    footprint_f_map = feasibility_map[x_start:x_end, y_start:y_end, int(support_height)]
    # A point provides valid support if it's at the correct height AND it's a stable LBCP surface.
    contact_mask = (footprint_h_map == support_height)
    valid_support_mask = contact_mask & footprint_f_map
    valid_cells_local = np.argwhere(valid_support_mask)
    if valid_cells_local.shape[0] == 0:
        return {"is_stable": False,
                "support_height": support_height,
                "support_polygon": None,
                "reason": "No support points."}
    # For accurate convex hull, use all four corners of each valid grid cell
    hull_points = []
    for c_local, r_local in valid_cells_local:
        gx, gy = c_local + x_start, r_local + y_start
        hull_points.extend([(gx, gy), (gx + 1, gy), (gx, gy + 1), (gx + 1, gy + 1)])  # assume a grid
        # hull_points.append((gx, gy))
    unique_hull_points = np.unique(np.array(hull_points), axis=0)
    if unique_hull_points.shape[0] < 3:
        return {"is_stable": False,
                "support_height": support_height,
                "support_polygon": None,
                "reason": "Not enough unique support corners."}
    # Calculate the convex hull of the unique support points
    hull = ConvexHull(unique_hull_points)
    support_polygon = Polygon(hull.points[hull.vertices])
    center_x = new_item.x + new_item.w / 2
    center_y = new_item.y + new_item.d / 2
    dx = (new_item.w / 2) * cog_uncertainty_ratio
    dy = (new_item.d / 2) * cog_uncertainty_ratio
    cog_test_points = [
        Point(center_x - dx, center_y - dy), Point(center_x + dx, center_y - dy),
        Point(center_x - dx, center_y + dy), Point(center_x + dx, center_y + dy)
    ]
    # Check if the center of gravity (CoG) test points are within the support polygon
    is_stable = all(support_polygon.contains(p) for p in cog_test_points)
    results = {
        "is_stable": is_stable,
        "support_height": support_height,
        "support_polygon": support_polygon,
        "cog_test_points": cog_test_points,
        "reason": "CoG is within support polygon." if is_stable else "CoG is outside support polygon."
    }
    # For visualization, get the center points of the valid grid cells
    if toggle_viz:
        support_points_for_viz = valid_cells_local + np.array([x_start, y_start])
        support_points_for_viz = support_points_for_viz[:, [1, 0]]
        results["support_points"] = support_points_for_viz
    return results


def update_maps(item, support_height, support_polygon, height_map, feasibility_map):
    """Updates the height and feasibility maps after placing an item."""
    x_start, y_start = int(item.x), int(item.y)
    x_end, y_end = x_start + int(item.w), y_start + int(item.d)
    height_map = height_map.copy()
    feasibility_map = feasibility_map.copy()
    height_map[x_start:x_end, y_start:y_end] = item.z + item.h
    if support_polygon:
        height_s = int(support_height) + item.h
        if height_s >= feasibility_map.shape[2]:
            # print("Support height exceeds feasibility map height dimension.")
            return height_map, feasibility_map
        # TODO it seems the polygon shape may 1 pixel smaller than expected?
        bool_array = rasterize(
            [(support_polygon, 1)],
            out_shape=feasibility_map.shape[:2],
            fill=0,
            dtype=np.int8,
            all_touched=False  # Set to True if you want partially covered pixels too
        )
        feasibility_map[..., height_s] = feasibility_map[..., height_s] | bool_array.astype(bool).T
    return height_map, feasibility_map


# --- VISUALIZATION LOGIC ---

def draw_cuboid(ax, item):
    """Draws a 3D cuboid for an item on the given matplotlib axes."""
    x, y, z = item.x, item.y, item.z
    w, d, h = item.w, item.d, item.h

    v = np.array([[x, y, z], [x + w, y, z], [x + w, y + d, z], [x, y + d, z], [x, y, z + h], [x + w, y, z + h],
                  [x + w, y + d, z + h], [x, y + d, z + h]])
    faces = [[v[0], v[1], v[5], v[4]], [v[7], v[6], v[2], v[3]], [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]],
             [v[4], v[5], v[6], v[7]], [v[0], v[1], v[2], v[3]]]

    poly3d = Poly3DCollection(faces, facecolors=item.color, linewidths=1, edgecolors='k', alpha=0.8)
    ax.add_collection3d(poly3d)
    ax.text(x + w / 2, y + d / 2, z + h, item.name, color='black', ha='center')


def plot_scene(step_title, items, feasibility_map, bin_dims, analysis_item=None, analysis_result=None):
    """Creates and displays a 3-panel plot for the current step of the simulation."""
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(step_title, fontsize=16, y=0.98)

    # --- 1. 3D Bin View ---
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.set_title("3D Bin View")
    for item in items:
        draw_cuboid(ax1, item)
    if analysis_item:  # Draw the item being tested with transparency
        # FIX: Correctly instantiate the Item class for the temporary visualization object.
        test_item_viz = Item(
            dimensions=[analysis_item.w, analysis_item.d, analysis_item.h],
            position=(analysis_item.x, analysis_item.y, analysis_item.z),
            color='gray',
            name=f"Test {analysis_item.name}"
        )
        draw_cuboid(ax1, test_item_viz)

    ax1.set_xlim([0, bin_dims['w']]);
    ax1.set_ylim([0, bin_dims['d']]);
    ax1.set_zlim([0, bin_dims['h']])
    ax1.set_xlabel('X');
    ax1.set_ylabel('Y');
    ax1.set_zlabel('Z')

    # --- 2. Feasibility Map ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Feasibility Height Map")

    # --- START REVISION ---
    # Create an array of Z-indices [0, 1, 2, ..., H-1] and broadcast it to the shape of the feasibility map.
    # We add 1 to the heights so that the floor (z=0) is represented by 1, and 0 can be used for "infeasible".
    z_indices = np.arange(feasibility_map.shape[2]) + 1
    broadcasted_z = z_indices.reshape(1, 1, -1)

    # Multiply the boolean feasibility map by the Z-indices.
    # This results in an array where feasible points have value z+1 and infeasible points have value 0.
    height_values = feasibility_map * broadcasted_z

    # Project to 2D by taking the maximum height at each (y, x) position.
    # The result is a 2D map where each pixel's value is the max feasible height (or 0 if none).
    projected_height_map = np.max(height_values, axis=2)

    # Create a masked array where all 0 values are masked. This is how we'll make them white.
    masked_projected_map = np.ma.masked_where(projected_height_map == 0, projected_height_map)

    # Get a colormap. 'viridis' is a good choice. We discretize it into a number of colors
    # equal to the max height for clear steps.
    cmap = plt.cm.get_cmap('viridis', bin_dims['h'])
    # Set the color for the masked values (where our map was 0) to white.
    cmap.set_bad(color='white')

    # Display the image using the masked array and custom colormap.
    # We subtract 1 in the vmin/vmax to map back to original z-values [0, h-1] for the color scale.
    im = ax2.imshow(masked_projected_map - 1, cmap=cmap, origin='lower',
                    interpolation='nearest', vmin=-1, vmax=bin_dims['h'] - 1)

    # Add a colorbar to show what height each color represents.
    cbar = fig.colorbar(im, ax=ax2, ticks=np.arange(0, bin_dims['h'], 5))
    cbar.set_label('Feasible Surface Height (z)')
    # --- END REVISION ---

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xticks(np.arange(0, bin_dims['w'], 10))
    ax2.set_yticks(np.arange(0, bin_dims['d'], 10))
    ax2.grid(True, which='both', color='white', linewidth=0.5)

    # --- 3. Analysis Plot ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Analysis: Support & CoG")
    ax3.set_xlim(0, bin_dims['w'])
    ax3.set_ylim(0, bin_dims['d'])
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, linestyle=':')

    if analysis_item and analysis_result:
        footprint_poly = Polygon(
            [(analysis_item.x, analysis_item.y), (analysis_item.x + analysis_item.w, analysis_item.y),
             (analysis_item.x + analysis_item.w, analysis_item.y + analysis_item.d),
             (analysis_item.x, analysis_item.y + analysis_item.d)])
        x, y = footprint_poly.exterior.xy
        ax3.plot(x, y, color='blue', alpha=0.7, linewidth=2, solid_capstyle='round', label='Item Footprint')

        if analysis_result.get("support_points") is not None:
            points = analysis_result["support_points"]
            ax3.scatter(points[:, 0] + 0.5, points[:, 1] + 0.5, c='green', s=20, alpha=0.5, label='Feasible Support')

        if analysis_result.get("support_polygon"):
            poly = analysis_result["support_polygon"]
            x, y = poly.exterior.xy
            ax3.fill(x, y, alpha=0.3, fc='orange', ec='none', label='Support Polygon')
            ax3.plot(x, y, color='orange', linewidth=2)

        if analysis_result.get("cog_test_points"):
            cog_points = analysis_result["cog_test_points"]
            for p in cog_points:
                ax3.plot(p.x, p.y, 'rx', markersize=10, markeredgewidth=2)
            ax3.plot([], [], 'rx', markersize=10, markeredgewidth=2, label='CoG Uncertainty')

    ax3.legend(loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# --- MAIN SCRIPT ---
if __name__ == '__main__':
    BIN_DIMS = {'w': 20, 'd': 20, 'h': 30}

    print("--- Starting Stability Simulation ---")
    print("NOTE: Close each plot window to proceed to the next step.")

    # --- Step 1: Initialize ---
    print("\nStep 1: The bin is initialized. The floor is the first LBCP.")
    height_map = np.zeros((BIN_DIMS['d'], BIN_DIMS['w']))
    feasibility_map = np.full((BIN_DIMS['d'], BIN_DIMS['w'], BIN_DIMS['h']), False)
    feasibility_map[:, :, 0] = True  # Floor is feasible at height 0
    items = []
    # --- Step 2: Place Base Item ---
    base_item = Item(dimensions=[10, 10, 10], position=[0, 0, 0], color='lightblue', name='A')
    print(f"\nStep 2: Analyzing placement of Item '{base_item.name}' on the floor.")
    result_base = calculate_stability(base_item, height_map, feasibility_map, 0.1)
    height_map, feasibility_map = update_maps(base_item,
                                              result_base['support_height'],
                                              result_base['support_polygon'],
                                              height_map, feasibility_map)
    plot_scene(f"Analysis for Item '{base_item.name}' - Result: {result_base['is_stable']}", items, feasibility_map,
               BIN_DIMS,
               analysis_item=base_item, analysis_result=result_base)
    items.append(base_item)

    # --- Step 3: Place Second Item ---
    second_item = Item(dimensions=[10, 10, 10], position=[0, 4, 10], color='lightgreen', name='B')
    print(f"\nStep 3: Analyzing placement of Item '{second_item.name}' on top of Item 'A'.")
    result_second = calculate_stability(second_item, height_map, feasibility_map, 0.1)
    height_map, feasibility_map = update_maps(second_item,
                                              result_second['support_height'],
                                              result_second['support_polygon'], height_map,
                                              feasibility_map)
    plot_scene(f"Analysis for Item '{second_item.name}' - Result: {result_second['is_stable']}", items, feasibility_map,
               BIN_DIMS,
               analysis_item=second_item, analysis_result=result_second)
    items.append(second_item)

    # --- Step 4: Test a STABLE placement ---
    test_item_stable = Item(dimensions=[10, 10, 10], position=[0, 8, 20], color='orange', name='C')
    print(f"\nStep 4: Testing a stable placement for Item '{test_item_stable.name}'.")
    result = calculate_stability(test_item_stable, height_map, feasibility_map, 0.1)
    plot_scene(f"Test Stable Item '{test_item_stable.name}' - Result: {result['is_stable']}", items, feasibility_map,
               BIN_DIMS,
               analysis_item=test_item_stable, analysis_result=result)
    #
    # # --- Step 5: Test an UNSTABLE placement ---
    # test_item_unstable = Item(dimensions=[10, 10, 5], position=[20, 20, 0], color='purple', name='D')
    # print(f"\nStep 5: Testing an unstable placement for Item '{test_item_unstable.name}'.")
    # result = calculate_stability(test_item_unstable, height_map, feasibility_map, 0.1)
    # plot_scene(f"Test Unstable Item '{test_item_unstable.name}' - Result: {result['is_stable']}", items,
    #            feasibility_map, BIN_DIMS,
    #            analysis_item=test_item_unstable, analysis_result=result)

    print("\nSimulation finished.")
