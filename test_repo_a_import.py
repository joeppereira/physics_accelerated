import sys
import os

# Add serdes_architect to path
repo_a_path = os.path.abspath(os.path.join(os.getcwd(), '../serdes_architect'))
sys.path.append(repo_a_path)

print(f"Attempting imports from: {repo_a_path}")

try:
    from src.itf_parser import ITFParser
    print("✅ Successfully imported ITFParser")
except ImportError as e:
    print(f"❌ Failed to import ITFParser: {e}")

try:
    from src.lib_parser import LibParser
    print("✅ Successfully imported LibParser")
except ImportError as e:
    print(f"❌ Failed to import LibParser: {e}")

try:
    from src.thermal import ThermalAuditor
    print("✅ Successfully imported ThermalAuditor from src.thermal")
except ImportError as e:
    print(f"❌ Failed to import ThermalAuditor from src.thermal: {e}")
    # Try alternate location if any
    try:
        from src.thermal_auditor import ThermalAuditor
        print("✅ Successfully imported ThermalAuditor from src.thermal_auditor")
    except ImportError as e:
        print(f"❌ Failed to import ThermalAuditor from src.thermal_auditor: {e}")

