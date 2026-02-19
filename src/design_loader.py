import json
import numpy as np
import os

class DesignLoader:
    def __init__(self, grid_size=16):
        self.N = grid_size
        
    def load_from_json(self, json_path, roi_bounds=None):
        """
        Parses a user design and rasterizes it to the Voxel Grid.
        roi_bounds: (x_min, y_min, x_max, y_max) in um. 
                    If None, fits to Die Size.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Design file {json_path} not found.")
            
        with open(json_path, 'r') as f:
            design = json.load(f)
            
        # 1. Determine Coordinate System (Viewport)
        if roi_bounds:
            xmin, ymin, xmax, ymax = roi_bounds
        else:
            # Default to full die
            xmin, ymin = 0, 0
            xmax, ymax = design.get("die_width_um", 1000), design.get("die_height_um", 1000)
            
        width = xmax - xmin
        height = ymax - ymin
        
        # Pixel Pitch for this Viewport
        dx = width / self.N
        dy = height / self.N
        
        print(f"📐 Viewport: {width:.0f}x{height:.0f} um. Resolution: {dx:.1f} um/pixel")
        
        # 2. Rasterize Blocks into Power Grid
        power_grid = np.zeros((self.N, self.N))
        
        for block in design["blocks"]:
            b_name = block["name"]
            b_x = block["x"]
            b_y = block["y"]
            b_w = block["w"]
            b_h = block["h"]
            b_p = block["power_mw"]
            
            # Check overlap with Viewport
            if (b_x + b_w < xmin) or (b_x > xmax) or (b_y + b_h < ymin) or (b_y > ymax):
                continue # Skip blocks outside ROI
                
            # Rasterization Logic (Simple Center Sampling or Area Overlap)
            # For simplicity in this demo: Map Center to Voxel
            # Professional version would use Area-Weighted Anti-Aliasing
            
            # Start/End Indices in Grid
            # Clip to ROI relative coords
            rel_x = max(0, b_x - xmin)
            rel_y = max(0, b_y - ymin)
            
            # Convert to Grid Indices
            col_start = int(rel_x / dx)
            row_start = int(rel_y / dy)
            col_end = int(min(width, (b_x + b_w - xmin)) / dx)
            row_end = int(min(height, (b_y + b_h - ymin)) / dy)
            
            # Clamp
            col_start = max(0, min(self.N-1, col_start))
            row_start = max(0, min(self.N-1, row_start))
            col_end = max(0, min(self.N, col_end + 1))
            row_end = max(0, min(self.N, row_end + 1))
            
            # Distribute Power
            num_voxels = (col_end - col_start) * (row_end - row_start)
            if num_voxels > 0:
                p_per_voxel = b_p / num_voxels
                power_grid[row_start:row_end, col_start:col_end] += p_per_voxel
                
        return power_grid

if __name__ == "__main__":
    # Test stub
    pass
