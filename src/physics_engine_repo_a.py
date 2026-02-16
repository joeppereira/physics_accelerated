import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

# Add serdes_architect to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../serdes_architect')))
from src.itf_parser import ITFParser
from src.lib_parser import LibertyParser # Updated class name
from src.thermal import ThermalAuditor    # Updated import module

class SerDesPhysicsEngine:
    """
    The 'Physical Twin' Engine (Repo A).
    Simulates actual hardware behavior for the 3nm PCIe 7.0 macro.
    """
    def __init__(self, itf_path="config/foundry_3nm.itf"):
        # Initialize Repo A Parsers
        # Using local config/foundry_3nm.itf as the shared ground truth
        self.itf = ITFParser(itf_path)
        
        lib_path = os.path.abspath(os.path.join(os.getcwd(), '../serdes_architect/data/technology/liberty_3nm.lib'))
        self.lib = LibertyParser(lib_path) # Updated initialization
        self.auditor = ThermalAuditor()
        
        # 3nm Node Constants derived from ITF/Lib
        self.node = "3nm_TSMC"
        self.dfe_limit = 35.0  # mV
        self.thermal_limit = 105.0 # C
        
        # Copper resistivity from ITF
        cu_props = self.itf.get_material_properties("Copper")
        self.res_factor = cu_props.get("res", 1.68e-8) / 1.68e-8

        # Thermal RC Network (Simplified 3D Spread)
        self.theta_ia = np.array([120, 110, 105, 100, 95, 90, 85]) / 1000.0 # C/mW
        self.coupling_matrix = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                self.coupling_matrix[i, j] = 0.05 / (abs(i - j) + 1)

    def simulate_thermal_spatial(self, v_pp_mv, vga_gain, ambient_temp):
        """
        Simulates heat distribution using LibertyParser and ThermalAuditor.
        """
        # Calculate power per stage using LibertyParser
        powers = np.zeros(7)
        # Stage 6 (Driver)
        driver_stats = self.lib.get_power_stats("driver", v_pp_mv=v_pp_mv)
        powers[6] = driver_stats["total_mw"]
        
        # Stage 0-5 (AFE)
        afe_stats = self.lib.get_power_stats("afe", gain=vga_gain)
        powers[0:6] = afe_stats["total_mw"] / 6.0
        
        # Calculate local temperatures using coupling logic (informed by Auditor)
        self.auditor.ambient_temp = ambient_temp
        
        # Create a power map for the auditor
        power_map = {f"stage_{i}": powers[i] for i in range(7)}
        
        # Get hotspots from auditor
        hotspots, _ = self.auditor.calculate_hotspots(power_map)
        
        temps = np.zeros(7)
        for i in range(7):
            # Assuming auditor gives absolute temps for each component:
            temps[i] = hotspots.get(f"stage_{i}", ambient_temp)
            
        return temps, np.sum(powers)

    def calculate_eye_margins(self, ffe_taps, channel_loss_db, temps):
        """
        Propagates signal through 7 stages using simplified Peak Distortion.
        """
        # Start with full swing
        current_eye = 800.0 
        
        # Apply Channel Loss (e.g. -36dB)
        channel_factor = 10**(channel_loss_db / 20.0)
        current_eye *= channel_factor
        
        # FFE Gain/Loss
        ffe_boost = np.sum(np.abs(ffe_taps))
        current_eye *= ffe_boost
        
        margins = []
        for i in range(7):
            # Thermal Noise Degradation (kT/C)
            # Higher temp in a stage reduces signal-to-noise
            thermal_penalty = (temps[i] - 25.0) * 0.4
            
            # Stage nonlinearity
            current_eye = current_eye * 0.95 - thermal_penalty
            
            # Store [Vert_mv, Horiz_ui]
            # Eye width is roughly proportional to height in PAM4
            vert = max(current_eye * (0.8 + i/20.0), 0.0)
            horiz = min(vert / 200.0, 0.8)
            margins.append((vert, horiz))
            
        return margins

    def generate_dataset(self, n_samples=50000, output_path="data/samples_50k.parquet"):
        print(f"🏭 Repo A: Characterizing {self.node} Physics ({n_samples} iterations)...")
        
        results = []
        for _ in range(n_samples):
            # Sample Inputs
            ffe = np.random.uniform(-0.1, 0.1, 4)
            ffe[1] = 1.0 - np.sum(np.abs([ffe[0], ffe[2], ffe[3]])) # Normalize
            
            loss = np.random.uniform(-45.0, -10.0)
            tamb = np.random.uniform(25.0, 85.0)
            vpp  = np.random.uniform(700.0, 1100.0)
            vga  = np.random.uniform(1.0, 10.0)
            
            # Run Physics
            temps, total_power = self.simulate_thermal_spatial(vpp, vga, tamb)
            margins = self.calculate_eye_margins(ffe, loss, temps)
            
            # DFE Estimate (Simplified: Needs to cover residual ISI)
            dfe1 = abs(loss) * 0.9 + np.random.normal(0, 1)
            
            # Format row matching SCHEMA_DEFINITION
            row = {
                "ffe_c_minus_1": ffe[0], "ffe_c_0": ffe[1], "ffe_c_plus_1": ffe[2], "ffe_c_plus_2": ffe[3],
                "channel_loss_db": loss, "ambient_temp_c": tamb, "v_pp_mv": vpp, "vga_gain": vga,
                "max_junction_temp_c": np.max(temps),
                "total_power_mw": total_power,
                "dfe_1_mv": dfe1,
                "temp_stage_0_c": temps[0],
                "temp_stage_6_c": temps[6],
                "temp_diff_pair_delta_c": abs(np.random.normal(0, 0.5) * (vpp/1000))
            }
            # Add stage margins
            for i, (v, h) in enumerate(margins):
                row[f"stage_{i}_eye_height_mv"] = v
                row[f"stage_{i}_eye_width_ui"] = h
                
            results.append(row)
            
        df = pd.DataFrame(results)
        
        # Save with Hardware Rules in Metadata
        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata({
            b'technology_node': self.node.encode(),
            b'dfe_tap1_limit_mv': str(self.dfe_limit).encode(),
            b'thermal_ceiling_c': str(self.thermal_limit).encode()
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pq.write_table(table, output_path)
        print(f"📦 Physics-Grounded Knowledge Base created: {output_path}")

if __name__ == "__main__":
    engine = SerDesPhysicsEngine()
    engine.generate_dataset(n_samples=100) # Small batch for verification
