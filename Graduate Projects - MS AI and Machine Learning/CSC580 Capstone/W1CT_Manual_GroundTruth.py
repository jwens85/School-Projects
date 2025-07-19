import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

class ManualGroundTruthAssigner:
    def __init__(self, tolerance=0.55, model='hog'):
        self.tolerance = tolerance
        self.model = model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.ground_truth_mapping = {}
        self.available_scientists = []
        
    def get_available_scientists(self):
        """Get list of scientists from reference image folders"""
        reference_dir = Path("/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/reference_images")
        
        scientists = []
        for scientist_folder in reference_dir.iterdir():
            if scientist_folder.is_dir():
                scientist_name = scientist_folder.name.replace("_", " ")
                scientists.append(scientist_name)
        
        # Add "Unknown" option
        scientists.append("Unknown")
        scientists.sort()
        return scientists

    def preprocess_image(self, image_path):
        """Preprocess image using OpenCV for better face detection"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        gray_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return rgb_img, gray_blur

    def detect_faces(self, image_path):
        """Detect faces in the Solvay photo"""
        rgb_img, gray_img = self.preprocess_image(image_path)
        
        # Use face_recognition for detection
        face_locations = face_recognition.face_locations(rgb_img, model=self.model, number_of_times_to_upsample=1)
        
        # Filter faces by size and aspect ratio
        filtered_face_locations = []
        img_height, img_width = rgb_img.shape[:2]
        
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_width = right - left
            face_height = bottom - top
            
            min_face_size = int(img_width * 0.02)
            max_face_size = int(img_width * 0.12)
            aspect_ratio = face_width / face_height if face_height > 0 else 0
            
            if (min_face_size < face_width < max_face_size and
                min_face_size < face_height < max_face_size and
                0.7 < aspect_ratio < 1.3):
                filtered_face_locations.append(face_location)
        
        # Sort faces by rows first, then left to right within each row
        sorted_faces = self.sort_faces_by_rows(filtered_face_locations, img_height)
        
        return rgb_img, sorted_faces

    def estimate_body_position(self, face_location):
        """Estimate body center position from face location"""
        top, right, bottom, left = face_location
        
        # Estimate body center as the bottom-center of the face
        # This assumes the body is directly below the face
        body_center_x = (left + right) / 2
        body_center_y = bottom  # Use bottom of face as proxy for body position
        
        return body_center_x, body_center_y

    def sort_faces_by_rows(self, face_locations, img_height):
        """Sort faces by estimated body positions (back, middle, front) then left to right"""
        if not face_locations:
            return face_locations
            
        # Group faces into rows based on estimated body positions
        back_row = []    # Standing people in back 
        middle_row = []  # Sitting people in middle
        front_row = []   # Sitting people in front
        
        for face in face_locations:
            body_x, body_y = self.estimate_body_position(face)
            
            # Define row boundaries based on estimated body position
            if body_y < img_height * 0.5:  # Top 50% - back row (standing)
                back_row.append(face)
            elif body_y < img_height * 0.75:  # Middle 25% - middle row  
                middle_row.append(face)
            else:  # Bottom 25% - front row
                front_row.append(face)
        
        # Sort each row left to right by estimated body X position
        back_row.sort(key=lambda face: self.estimate_body_position(face)[0])
        middle_row.sort(key=lambda face: self.estimate_body_position(face)[0])  
        front_row.sort(key=lambda face: self.estimate_body_position(face)[0])
        
        # Combine rows in order: back, middle, front
        return back_row + middle_row + front_row

    def show_face_for_assignment(self, rgb_img, face_location, face_number, total_faces):
        """Show individual face with red box for manual assignment"""
        # Close any existing matplotlib windows
        plt.close('all')
        
        # Create a copy of the image
        pil_image = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        top, right, bottom, left = face_location
        
        # Draw red rectangle around the face being assigned
        draw.rectangle([left, top, right, bottom], outline="red", width=5)
        
        # Add face number label
        label_text = f"Face {face_number}/{total_faces}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([left, top - text_height - 10, left + text_width + 10, top],
                      fill="red", outline="red")
        draw.text((left + 5, top - text_height - 5), label_text, fill="white", font=font)
        
        # Display using matplotlib for better control
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(pil_image)
            plt.title(f"Face {face_number}/{total_faces} - Select the scientist", fontsize=14)
            plt.axis('off')
            plt.show(block=False)  # Non-blocking so user can interact with terminal
            plt.draw()
        except:
            # Fallback to saving file
            temp_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/current_face_assignment.png"
            pil_image.save(temp_path)
            print(f"Image saved to: {temp_path}")
            print("Please open this file to see the highlighted face.")
        
        return pil_image

    def get_user_choice(self, face_number, total_faces):
        """Get user's choice for ground truth assignment"""
        print(f"\n{'='*60}")
        print(f"ASSIGNING FACE {face_number}/{total_faces}")
        print(f"{'='*60}")
        print("Available scientists:")
        
        for i, scientist in enumerate(self.available_scientists, 1):
            print(f"{i:2d}. {scientist}")
        
        while True:
            try:
                choice = input(f"\nEnter number (1-{len(self.available_scientists)}) or 'skip' to skip: ").strip()
                
                if choice.lower() == 'skip':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.available_scientists):
                    selected_scientist = self.available_scientists[choice_num - 1]
                    
                    # Remove selected scientist from available list (except "Unknown")
                    if selected_scientist != "Unknown":
                        self.available_scientists.remove(selected_scientist)
                    
                    return selected_scientist
                else:
                    print(f"Please enter a number between 1 and {len(self.available_scientists)}")
                    
            except ValueError:
                print("Please enter a valid number or 'skip'")

    def run_manual_assignment(self):
        """Main function to run manual ground truth assignment"""
        print("MANUAL GROUND TRUTH ASSIGNMENT TOOL")
        print("=" * 50)
        
        # Get available scientists
        self.available_scientists = self.get_available_scientists()
        print(f"Found {len(self.available_scientists)} scientists (including Unknown)")
        
        # Load and detect faces
        solvay_photo_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/solvay.jpg"
        
        print("Detecting faces in Solvay conference photo...")
        rgb_img, face_locations = self.detect_faces(solvay_photo_path)
        
        print(f"Detected {len(face_locations)} faces")
        
        # Manual assignment for each face
        for i, face_location in enumerate(face_locations, 1):
            # Show face image
            self.show_face_for_assignment(rgb_img, face_location, i, len(face_locations))
            
            # Get user assignment
            assigned_scientist = self.get_user_choice(i, len(face_locations))
            
            if assigned_scientist:
                self.ground_truth_mapping[i] = {
                    "face_number": i,
                    "location": [int(x) for x in face_location],  # [top, right, bottom, left]
                    "ground_truth": assigned_scientist
                }
                print(f"Face {i} assigned to: {assigned_scientist}")
            else:
                print(f"Face {i} skipped")
        
        # Clean up any remaining windows
        plt.close('all')
        
        # Save results
        self.save_ground_truth_mapping()
        self.display_summary()

    def save_ground_truth_mapping(self):
        """Save the ground truth mapping to a JSON file"""
        output_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/manual_ground_truth_mapping.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.ground_truth_mapping, f, indent=2)
        
        print(f"\nGround truth mapping saved to: {output_path}")

    def display_summary(self):
        """Display summary of assignments"""
        print(f"\nASSIGNMENT SUMMARY")
        print("=" * 50)
        
        assigned_count = len(self.ground_truth_mapping)
        unknown_count = sum(1 for assignment in self.ground_truth_mapping.values() 
                           if assignment["ground_truth"] == "Unknown")
        identified_count = assigned_count - unknown_count
        
        print(f"Total faces processed: {assigned_count}")
        print(f"Successfully identified: {identified_count}")
        print(f"Marked as Unknown: {unknown_count}")
        print(f"Success rate: {(identified_count/assigned_count)*100:.1f}%" if assigned_count > 0 else "N/A")
        
        print(f"\nAssignments:")
        for face_num, assignment in sorted(self.ground_truth_mapping.items()):
            print(f"  Face {face_num}: {assignment['ground_truth']}")
        
        remaining_scientists = [s for s in self.available_scientists if s != "Unknown"]
        if remaining_scientists:
            print(f"\nUnassigned scientists: {', '.join(remaining_scientists)}")

def main():
    """Main function"""
    print("Starting Manual Ground Truth Assignment Tool...")
    
    assigner = ManualGroundTruthAssigner(tolerance=0.60, model='hog')
    assigner.run_manual_assignment()
    
    print("\nManual assignment complete!")
    print("You can now use the ground truth mapping in other programs.")

if __name__ == "__main__":
    main()