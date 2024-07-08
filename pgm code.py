from PIL import Image

# Load the image
image_path = '/home/siddhant/Camera Images/depth map merged.png'
image = Image.open(image_path)

# Convert the image to grayscale (if not already in grayscale)
image = image.convert('L')

# Resize the image to 512x512
image_resized = image.resize((512, 512))

# Save the image as .pgm file
output_path = '/home/siddhant/Camera Images/Depth_Map_Resized.pgm'
image_resized.save(output_path)

print(f'Image saved at: {output_path}')
