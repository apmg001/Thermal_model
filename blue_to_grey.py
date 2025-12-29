import cv2

# Load your current color thermal image
img = cv2.imread("thermal_image2.png")

# Convert to grayscale (visual only)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save output
cv2.imwrite("output_gray.png", gray)
print("Grayscale saved as output_gray.png")
