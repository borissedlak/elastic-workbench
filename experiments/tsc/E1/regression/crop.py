from PIL import Image

# Define image file paths
image_paths = ['QR.png', 'CV.png', 'PC.png']
output_paths = ['E1_QR_cropped.png', 'E1_CV_cropped.png', 'E1_PC_cropped.png']

# Define crop area (left, upper, right, lower)
# For example: crop a 200x200 square from (50, 50)
crop_box = (175, 105, 705, 600)

# Crop and save
for in_path, out_path in zip(image_paths, output_paths):
    with Image.open(in_path) as img:
        cropped = img.crop(crop_box)
        cropped.save(out_path)

print("Done cropping.")
