import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def generate():
    samples = 10000
    df = pd.DataFrame({
        "ffe_m1": np.random.uniform(-0.1, 0, samples),
        "ffe_0": np.random.uniform(0.7, 0.9, samples),
        "ffe_p1": np.random.uniform(-0.1, 0, samples),
        "ffe_p2": np.random.uniform(-0.05, 0, samples),
        "loss_db": np.random.uniform(-40, -20, samples),
        "temp_c": np.random.uniform(25, 85, samples),
        "vpp": np.random.uniform(300, 600, samples),
        "pwr": np.random.uniform(40, 80, samples)
    })
    # Physics correlations
    df["tj_c"] = df["temp_c"] + (df["pwr"] * 0.42)
    df["eye_height_mv"] = (df["vpp"] * 0.1) - (df["loss_db"] * -0.5)
    df["eye_width_ui"] = 0.6 - (df["tj_c"] * 0.001)
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "data/samples_50k.parquet")
    print("✅ Dummy data created in data/")

if __name__ == "__main__": generate()
