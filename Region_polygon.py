import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load full image
img = cv2.imread("final_thermal.png")
if img is None:
    raise FileNotFoundError("Image not found")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

# Load temperature CSV as matrix
temp = pd.read_csv("thermal_20251007_182341.csv", header=None).to_numpy()
temp_rot = np.rot90(temp, 2)  # rotate 180°

# Load JSON
data = json.load(open("project-5-at-2026-01-07-11-00-fe9b9b72.json"))

region_stats = {}
count = 0

plt.figure()
plt.imshow(img_rgb)
plt.axis("off")

for task in data:
    for ann in task.get("annotations", []):
        for r in ann.get("result", []):
            labels = r["value"].get("polygonlabels")
            if not labels:
                continue
            label = labels[0]
            pts = r["value"].get("points", [])
            if not pts:
                continue

            # Convert % → absolute full-image coords
            poly = np.array([[p[0]*W/100, p[1]*H/100] for p in pts], np.int32)

            # Create mask on full image
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [poly], 1)

            # Rotate mask 180° to match temp_rot
            mask_rot = np.rot90(mask, 2)

            # Extract region temperatures safely
            region_vals = temp_rot[mask_rot == 1]

            # Compute stats
            region_stats[label] = {
                "index": count + 1,
                "pixel_count": int(mask_rot.sum()),
                "mean_temp": float(region_vals.mean()),
                "max_temp": float(region_vals.max()),
                "min_temp": float(region_vals.min()),
                "std_temp": float(region_vals.std()),
                "temp_range": float(region_vals.max() - region_vals.min())
            }

            # Draw overlays
            plt.plot(*(poly.T), linewidth=1)
            ys, xs = np.where(mask_rot == 1)
            plt.scatter(xs, ys, s=2)

            count += 1

plt.show()

# Save final CSV
df_final = pd.DataFrame.from_dict(region_stats, orient="index")
df_final.index.name = "region"
df_final.reset_index(inplace=True)
df_final.to_csv("final_regionwise_thermal_180.csv", index=False)

print("\nTotal regions:", count)
print("Saved CSV: final_regionwise_thermal_180.csv")
