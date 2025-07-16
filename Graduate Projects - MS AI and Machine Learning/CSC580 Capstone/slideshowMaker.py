#!/usr/bin/env python3
"""
GAN Training Progress Slideshow Generator
Creates an animated GIF from GAN output images showing training progression
"""

import os
import glob
from PIL import Image
import re

def create_gan_slideshow(images_dir, output_path="gan_training_slideshow.gif", duration=200):
    """
    Create an animated GIF slideshow from GAN training images
    
    Args:
        images_dir (str): Directory containing the epoch images
        output_path (str): Path for the output GIF file
        duration (int): Duration per frame in milliseconds
    """
    # Find all PNG files in the directory
    image_pattern = os.path.join(images_dir, "*.png")
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"No PNG images found in {images_dir}")
        return
    
    # Sort files by epoch number if they follow the epoch naming pattern
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    # Check if files follow epoch pattern, otherwise sort alphabetically
    if any(re.search(r'epoch_\d+', f) for f in image_files):
        image_files.sort(key=extract_epoch)
    else:
        image_files.sort()
    
    print(f"Found {len(image_files)} images")
    if image_files:
        print(f"First image: {os.path.basename(image_files[0])}")
        print(f"Last image: {os.path.basename(image_files[-1])}")
    
    # Load and process images
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            # Convert to RGB if needed (removes alpha channel)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            print(f"Loaded: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if not images:
        print("No valid images could be loaded!")
        return
    
    # Create the GIF
    print(f"Creating GIF with {len(images)} frames...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 0 means infinite loop
    )
    
    print(f"GIF saved as: {output_path}")
    print(f"Duration per frame: {duration}ms")
    print(f"Total animation time: {len(images) * duration / 1000:.1f} seconds")

def main():
    print("GAN Slideshow Maker")
    
    # Prompt user for image directory path
    images_dir = input("Enter the path to the directory containing your images: ").strip()
    
    # Handle relative paths
    if not os.path.isabs(images_dir):
        images_dir = os.path.join(os.getcwd(), images_dir)
    
    if not os.path.exists(images_dir):
        print(f"Directory '{images_dir}' does not exist!")
        return
    
    if not os.path.isdir(images_dir):
        print(f"'{images_dir}' is not a directory!")
        return
    
    # Get output filename
    default_output = "slideshow.gif"
    output_filename = input(f"Enter output filename (default: {default_output}): ").strip()
    if not output_filename:
        output_filename = default_output
    
    # Add .gif extension if not present
    if not output_filename.lower().endswith('.gif'):
        output_filename += '.gif'
    
    # Get frame duration
    try:
        duration_input = input("Enter frame duration in milliseconds (default: 200): ").strip()
        duration = int(duration_input) if duration_input else 200
    except ValueError:
        print("Invalid duration, using default of 200ms")
        duration = 200
    
    # Create the slideshow
    create_gan_slideshow(images_dir, output_filename, duration)

if __name__ == "__main__":
    main()