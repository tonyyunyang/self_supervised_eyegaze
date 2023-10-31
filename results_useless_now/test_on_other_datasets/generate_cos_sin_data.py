import os
import numpy as np

# Parameters
sampling_rate = 60  # 60 Hz, chosen as a moderate sampling rate
time_duration = 1000 # seconds
time_points = np.linspace(0, time_duration, int(sampling_rate * time_duration))

# Frequencies for 8 categories
frequencies = [i / 4 for i in range(1, 9)]

# Amplitudes for 6 files in each category
amplitudes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Create a directory named "generated_files" if it doesn't exist
if not os.path.exists("generated_files"):
    os.makedirs("generated_files")

# Generate data and save to files
for i, freq in enumerate(frequencies):
    for j, amp in enumerate(amplitudes):
        sin_data = amp * np.sin(2 * np.pi * freq * time_points)
        cos_data = amp * np.cos(2 * np.pi * freq * time_points)

        # File naming
        amp_str = str(amp).replace('.', ':')  # Replace decimal point with underscore
        filename = f"P0{i + 1}_{amp_str}sincos.csv"
        filepath = os.path.join("generated_files", filename)

        # Save to CSV
        with open(filepath, 'w') as file:
            for s, c in zip(sin_data, cos_data):
                file.write(f"{s},{c}\n")

print("Files have been generated and saved in the 'generated_files' folder.")
