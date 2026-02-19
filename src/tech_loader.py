import re
import os

class TechLoader:
    def __init__(self):
        self.layers = []

    def load_itf(self, file_path):
        """
        Parses an Interconnect Technology Format (.itf) file.
        Extracts Conductor layers with Thickness and Conductivity.
        """
        if not os.path.exists(file_path):
            print(f"⚠️ Tech file {file_path} not found. Using defaults.")
            return []

        print(f"📖 Parsing Tech File: {file_path}...")
        
        with open(file_path, 'r') as f:
            content = f.read()

        # Simple Regex parser for CONDUCTOR blocks
        # CONDUCTOR <name> { ... THICKNESS=... RESISTIVITY=... }
        
        # Regex to find blocks
        conductor_pattern = re.compile(r'CONDUCTOR\s+(\w+)\s*\{([^}]*)\}', re.MULTILINE | re.DOTALL)
        matches = conductor_pattern.findall(content)
        
        stack = []
        
        # Add Silicon Die base (Implicit in most ITFs, but we need it)
        stack.append({
            "name": "Active_Silicon",
            "type": "die",
            "thickness": 50.0, # Standard assumption if missing
            "k": 150.0         # W/mK
        })

        for name, body in matches:
            # Extract Thickness
            t_match = re.search(r'THICKNESS\s*=\s*([\d\.]+)', body)
            thickness = float(t_match.group(1)) if t_match else 0.1
            
            # Extract Resistivity (Ohm-m or similar) -> Convert to K
            # Standard Cu Rho = 1.68e-8 Ohm-m. K ~ 400 W/mK.
            # Wiedemann-Franz Law approx or direct lookup.
            # Most ITFs use specific units. We'll look for "TC1" or "RESISTIVITY".
            
            # For this simplified loader, we map known materials or use Res.
            # If RESISTIVITY is present, assume uOhm-cm? 
            # Let's map Material Names to K if explicit props are complex.
            
            k_val = 200.0 # Default Metal
            if "Copper" in name or "Cu" in name: k_val = 400.0
            if "Al" in name: k_val = 235.0
            if "Ru" in name: k_val = 100.0 # Ruthenium is resistive
            
            stack.append({
                "name": name,
                "type": "metal",
                "thickness": thickness,
                "k": k_val
            })
            
        # Add Package/Board defaults (usually not in ITF)
        stack.append({"name": "C4_Bump", "type": "bump", "thickness": 50, "k": 60})
        stack.append({"name": "Substrate", "type": "pkg", "thickness": 500, "k": 20})
        stack.append({"name": "PCB", "type": "board", "thickness": 1000, "k": 0.5})
        
        print(f"   -> Extracted {len(matches)} Metal Layers.")
        return stack

if __name__ == "__main__":
    # Test
    pass
