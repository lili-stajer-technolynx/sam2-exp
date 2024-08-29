import numpy as np
from numpy import asarray
from PIL import Image
import os

frames_path =  os.path.join(os.getcwd(),"segment-anything-2", "video_frames_points", "video1")
output_folder = os.path.join(os.getcwd(),"segment-anything-2", "video_frames_np", "frames")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(frames_path):
    if filename.endswith(('.jpg', '.jpeg')):
        img_path = os.path.join(frames_path, filename)
        
        img = Image.open(img_path)
        numpydata = np.array(img.convert("RGB"))
        
        output_filename = os.path.splitext(filename)[0] + '.npy'
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, numpydata)
        print(f"Converted and saved: {filename} -> {output_filename}")

print("All images processed and saved")