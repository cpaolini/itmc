import pandas as pd

# Define filenames
file1_path = "D:/Jetson Project Sample Videos/vehicle_rectangles_4200sec2.csv"
file2_path = "D:/Jetson Project Sample Videos/vehicle_rectangles_2460sec4.csv"
output_path = "D:/Jetson Project Sample Videos/vehicle_rectangles_4200sec3.csv"

# Load both CSV files with headers
csv1 = pd.read_csv(file1_path)
csv2 = pd.read_csv(file2_path)

# Convert timestamp column to float
csv1["timestamp"] = csv1["timestamp"].astype(float)
csv2["timestamp"] = csv2["timestamp"].astype(float)

# Remove overlapping row(s) from csv2 (timestamp ≤ 59.900)
csv2_filtered = csv2[csv2["timestamp"] > 1963.100].copy()

# Shift timestamp by 1740.0
csv2_filtered["timestamp"] += 1740.0

# Combine the two datasets
combined = pd.concat([csv1, csv2_filtered], ignore_index=True)

# Write to a new CSV file
combined.to_csv(output_path, index=False, header=True)

print(f"✅ Combined CSV written to: {output_path}")
