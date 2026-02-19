import json
import numpy as np
import os

class DesignLoader:
    def __init__(self, grid_size=16):
        self.N = grid_size
        
    def collapse_stack(self, raw_stack):
        """
        Compresses an N-layer physical stack into the 5-Layer Canonical Model.
        Canonical Layers: [Die, BEOL, Interconnect, Package, Board]
        Uses Series Thermal Resistance rule: K_eff = Sum(t) / Sum(t/k)
        """
        if not raw_stack:
            # Default 3nm Stack
            return [150.0, 400.0, 60.0, 10.0, 0.5]
            
        # Buckets for the 5 canonical layers
        # Mapping strategy:
        # 0 -> Die (Silicon)
        # 1 -> BEOL (Metals M1-Mx)
        # 2 -> C4/Bumps (Underfill)
        # 3 -> Package (Substrate)
        # 4 -> Board (PCB/TIM)
        
        buckets = [[], [], [], [], []]
        
        for layer in raw_stack:
            # User must tag layer type, or we infer by order?
            # Let's assume user tags: "type": "die" | "metal" | "bump" | "pkg" | "board"
            l_type = layer.get("type", "pkg")
            idx = 3 # Default to package
            
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
                # Fallback defaults if missing
                defaults = [150.0, 200.0, 50.0, 5.0, 0.5]
                k_canonical.append(defaults[i])
                continue
                
            # Calculate Effective Vertical K
            # R_total = Sum(R_i) = Sum(t_i / k_i)
            # K_eff = T_total / R_total
            t_total = sum([l["thickness"] for l in layers])
            r_total = sum([l["thickness"] / l["k"] for l in layers])
            
            k_eff = t_total / r_total
            k_canonical.append(k_eff)
            
        print(f"🧱 Stack Collapsed: {len(raw_stack)} Layers -> 5 Canonical Layers")
        print(f"   K_eff: {k_canonical}")
        return k_canonical

    def load_from_json(self, json_path, roi_bounds=None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Design file {json_path} not found.")
            
        with open(json_path, 'r') as f:
            design = json.load(f)
            
        # 1. Collapse Stack
        k_layers = self.collapse_stack(design.get("stackup", []))
            
        # 2. Determine Viewport
        if roi_bounds:
            xmin, ymin, xmax, ymax = roi_bounds
        else:
            xmin, ymin = 0, 0
            xmax, ymax = design.get("die_width_um", 1000), design.get("die_height_um", 1000)
            
        width = xmax - xmin
        height = ymax - ymin
        dx = width / self.N
        dy = height / self.N
        
        # 3. Rasterize
        power_grid = np.zeros((self.N, self.N))
        
        for block in design["blocks"]:
            b_x, b_y, b_w, b_h = block["x"], block["y"], block["w"], block["h"]
            b_p = block["power_mw"]
            
            if (b_x + b_w < xmin) or (b_x > xmax) or (b_y + b_h < ymin) or (b_y > ymax):
                continue 
                
            rel_x = max(0, b_x - xmin)
            rel_y = max(0, b_y - ymin)
            
            col_start = int(rel_x / dx)
            row_start = int(rel_y / dy)
            col_end = int(min(width, (b_x + b_w - xmin)) / dx)
            row_end = int(min(height, (b_y + b_h - ymin)) / dy)
            
            col_start = max(0, min(self.N-1, col_start))
            row_start = max(0, min(self.N-1, row_start))
            col_end = max(0, min(self.N, col_end + 1))
            row_end = max(0, min(self.N, row_end + 1))
            
            num_voxels = (col_end - col_start) * (row_end - row_start)
            if num_voxels > 0:
                p_per_voxel = b_p / num_voxels
                power_grid[row_start:row_end, col_start:col_end] += p_per_voxel
                
        return power_grid, k_layers # Return both

if __name__ == "__main__":
    pass