import json
import numpy as np
import os
from src.tech_loader import TechLoader

class DesignLoader:
    def __init__(self, grid_size=16):
        self.N = grid_size
        self.tech_loader = TechLoader()
        
    def collapse_stack(self, raw_stack):
        """Compresses N-layer stack into 5 Canonical Layers using Thermal Resistance Rule."""
        if not raw_stack: return [150.0, 400.0, 60.0, 10.0, 0.5]
        buckets = [[], [], [], [], []]
        for layer in raw_stack:
            l_type = layer.get("type", "pkg")
            idx = 3 # Default pkg
            if l_type == "die": idx = 0
            elif l_type == "metal": idx = 1
            elif l_type == "bump": idx = 2
            elif l_type == "pkg": idx = 3
            elif l_type == "board": idx = 4
            buckets[idx].append(layer)
        k_canonical = []
        for i in range(5):
            layers = buckets[i]
            if not layers:
                k_canonical.append([150.0, 200.0, 50.0, 5.0, 0.5][i])
                continue
            t_total = sum([l["thickness"] for l in layers])
            r_total = sum([l["thickness"] / l["k"] for l in layers])
            k_canonical.append(t_total / r_total)
        return k_canonical

    def _rasterize_recursive(self, blocks, power_grid, xmin, ymin, dx, dy, parent_x=0, parent_y=0):
        """Recursively parses blocks and sub-blocks into the grid."""
        for block in blocks:
            # Absolute coordinates
            abs_x = parent_x + block["x"]
            abs_y = parent_y + block["y"]
            w, h = block["w"], block["h"]
            p = block.get("power_mw", 0)

            # Process Sub-blocks if they exist
            if "sub_blocks" in block:
                self._rasterize_recursive(block["sub_blocks"], power_grid, xmin, ymin, dx, dy, abs_x, abs_y)
            
            # Rasterize current block power
            if p > 0:
                rel_x, rel_y = max(0, abs_x - xmin), max(0, abs_y - ymin)
                col_start, row_start = int(rel_x / dx), int(rel_y / dy)
                col_end = int(min(self.N * dx, (abs_x + w - xmin)) / dx)
                row_end = int(min(self.N * dy, (abs_y + h - ymin)) / dy)
                
                # Bounds check
                if col_start >= self.N or row_start >= self.N or col_end < 0 or row_end < 0: continue
                
                col_start, row_start = max(0, col_start), max(0, row_start)
                col_end, row_end = min(self.N, col_end + 1), min(self.N, row_end + 1)
                
                num_voxels = (col_end - col_start) * (row_end - row_start)
                if num_voxels > 0:
                    power_grid[row_start:row_end, col_start:col_end] += p / num_voxels

    def load_from_json(self, json_path, roi_bounds=None):
        with open(json_path, 'r') as f:
            design = json.load(f)
        
        if "tech_file" in design:
            raw_stack = self.tech_loader.load_itf(design["tech_file"])
        else:
            raw_stack = design.get("stackup", [])
        k_layers = self.collapse_stack(raw_stack)

        xmin, ymin = (roi_bounds[0], roi_bounds[1]) if roi_bounds else (0, 0)
        xmax, ymax = (roi_bounds[2], roi_bounds[3]) if roi_bounds else (design.get("die_width_um", 1000), design.get("die_height_um", 1000))
        
        dx, dy = (xmax - xmin) / self.N, (ymax - ymin) / self.N
        power_grid = np.zeros((self.N, self.N))
        
        self._rasterize_recursive(design["blocks"], power_grid, xmin, ymin, dx, dy)
        return power_grid, k_layers
