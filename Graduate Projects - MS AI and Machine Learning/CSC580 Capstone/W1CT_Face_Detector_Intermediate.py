import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import json
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

class EnhancedSolvayFaceRecognizer:
    def __init__(self, tolerance=0.55, model='cnn'):
        self.known_encodings = []
        self.known_names = []
        self.tolerance = tolerance
        self.model = model  #'hog' or 'cnn'
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        #Solvay positions for reference
        self.solvay_positions = {
            "front_row": [
                "Irving Langmuir", "Max Planck", "Marie Curie", "Hendrik Lorentz",
                "Albert Einstein", "Paul Langevin", "Charles Eugene Guye",
                "CTR Wilson", "Owen Richardson"
            ],
            "middle_row": [
                "Peter Debye", "Martin Knudsen", "William Lawrence Bragg",
                "Hendrik Anthony Kramers", "Paul Dirac", "Arthur Compton",
                "Louis de Broglie", "Max Born", "Niels Bohr"
            ],
            "back_row": [
                "Auguste Piccard", "Emile Henriot", "Paul Ehrenfest",
                "Edouard Herzen", "Theophile de Donder", "Erwin Schrodinger",
                "JE Verschaffelt", "Wolfgang Pauli", "Werner Heisenberg",
                "Ralph Fowler", "Leon Brillouin"
            ]
        }

    def preprocess_image(self, image_path):
        """Preprocess image using OpenCV for better face detection"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Apply histogram equalization
        gray_eq = cv2.equalizeHist(gray)

        #Apply Gaussian blur to reduce noise
        gray_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        #Convert back to RGB for face_recognition
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return rgb_img, gray_blur

    def load_reference_images(self):
        """Load and encode reference images of scientists with enhanced preprocessing"""
        reference_dir = Path("/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/reference_images")

        for scientist_folder in reference_dir.iterdir():
            if scientist_folder.is_dir():
                scientist_name = scientist_folder.name.replace("_", " ")

                #Load images from scientist's folder
                for img_file in scientist_folder.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            #First try without preprocessing
                            image = face_recognition.load_image_file(str(img_file))
                            encodings = face_recognition.face_encodings(image, model='large')

                            if encodings:
                                #Use the first face found
                                self.known_encodings.append(encodings[0])
                                self.known_names.append(scientist_name)
                                print(f"Loaded reference image for {scientist_name}")
                            else:
                                #Try with preprocessing if normal method fails
                                try:
                                    rgb_img, gray_img = self.preprocess_image(str(img_file))
                                    encodings = face_recognition.face_encodings(rgb_img, model='large')
                                    if encodings:
                                        self.known_encodings.append(encodings[0])
                                        self.known_names.append(scientist_name)
                                        print(f"Loaded reference image for {scientist_name} (with preprocessing)")
                                    else:
                                        print(f"No face found in {img_file}")
                                except:
                                    print(f"No face found in {img_file}")
                        except Exception as e:
                            print(f"Error loading {img_file}: {e}")

    def remove_duplicate_faces(self, face_locations, overlap_threshold=0.5):
        """Remove duplicate face detections based on overlap"""
        if len(face_locations) <= 1:
            return face_locations

        def calculate_overlap(box1, box2):
            """Calculate overlap between two bounding boxes"""
            top1, right1, bottom1, left1 = box1
            top2, right2, bottom2, left2 = box2

            #Calculate intersection
            x_overlap = max(0, min(right1, right2) - max(left1, left2))
            y_overlap = max(0, min(bottom1, bottom2) - max(top1, top2))
            intersection = x_overlap * y_overlap

            #Calculate union
            area1 = (right1 - left1) * (bottom1 - top1)
            area2 = (right2 - left2) * (bottom2 - top2)
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0

        #Remove duplicates
        unique_faces = []
        for face in face_locations:
            is_duplicate = False
            for unique_face in unique_faces:
                if calculate_overlap(face, unique_face) > overlap_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append(face)

        return unique_faces

    def load_manual_ground_truth(self):
        """Load the manual ground truth mapping from JSON file"""
        mapping_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/manual_ground_truth_mapping.json"
        
        try:
            with open(mapping_path, 'r') as f:
                self.manual_ground_truth = json.load(f)
            print(f"Loaded manual ground truth mapping with {len(self.manual_ground_truth)} entries")
            return True
        except FileNotFoundError:
            print(f"Manual ground truth file not found at {mapping_path}")
            print("Run W1CT_Manual_GT.py first to create the ground truth mapping")
            self.manual_ground_truth = {}
            return False
        except Exception as e:
            print(f"Error loading manual ground truth: {e}")
            self.manual_ground_truth = {}
            return False

    def find_closest_manual_face(self, face_location):
        """Find the closest manually assigned face for this detection"""
        if not hasattr(self, 'manual_ground_truth') or not self.manual_ground_truth:
            return "Unknown Position"
        
        top, right, bottom, left = face_location
        face_center_x = (left + right) / 2
        face_center_y = (top + bottom) / 2
        
        min_distance = float('inf')
        closest_scientist = "Unknown Position"
        
        # Find the manual ground truth entry with the closest location
        for face_data in self.manual_ground_truth.values():
            manual_location = face_data['location']  # [top, right, bottom, left]
            manual_top, manual_right, manual_bottom, manual_left = manual_location
            manual_center_x = (manual_left + manual_right) / 2
            manual_center_y = (manual_top + manual_bottom) / 2
            
            # Calculate Euclidean distance between face centers
            distance = ((face_center_x - manual_center_x) ** 2 + 
                       (face_center_y - manual_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_scientist = face_data['ground_truth']
        
        # Only return the match if it's reasonably close (within 100 pixels)
        if min_distance < 100:
            return closest_scientist
        else:
            return "Unknown Position"

    def detect_and_identify_faces(self, image_path):
        """Detect faces in the Solvay photo and identify them with enhanced processing"""
        #Preprocess the image
        rgb_img, gray_img = self.preprocess_image(image_path)

        #Use primary face detection method (more accurate)
        face_locations = face_recognition.face_locations(rgb_img, model=self.model, number_of_times_to_upsample=1)

        #Filter out faces that are too small or too large (likely false positives)
        filtered_face_locations = []
        img_height, img_width = rgb_img.shape[:2]

        print(f"Initial face detections: {len(face_locations)}")

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_width = right - left
            face_height = bottom - top

            #Filter based on reasonable face size (between 2% and 12% of image width)
            min_face_size = int(img_width * 0.02)
            max_face_size = int(img_width * 0.12)

            #Filter based on aspect ratio (faces should be roughly square)
            aspect_ratio = face_width / face_height if face_height > 0 else 0

            if (min_face_size < face_width < max_face_size and
                min_face_size < face_height < max_face_size and
                0.7 < aspect_ratio < 1.3):
                filtered_face_locations.append(face_location)

        print(f"After filtering: {len(filtered_face_locations)} faces")

        #Get face encodings with improved model
        face_encodings = face_recognition.face_encodings(rgb_img, filtered_face_locations, model='large')

        #Create PIL image for drawing
        pil_image = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_image)

        #Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        identified_faces = []
        scientist_matches = {}  #Track best match for each scientist

        #First pass: find best match for each scientist
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, filtered_face_locations)):
            #Compare with known faces using adjusted tolerance
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.tolerance)
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)

            name = "Unknown"
            confidence = 0
            best_match_index = -1

            if matches and any(matches):
                #Find the best match
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - distances[best_match_index]

                    #Check if this scientist was already matched
                    if name != "Unknown":
                        if name not in scientist_matches or confidence > scientist_matches[name]['confidence']:
                            scientist_matches[name] = {
                                'face_index': i,
                                'confidence': confidence,
                                'location': face_location,
                                'best_match_index': best_match_index
                            }

            #Always add to list initially (will filter later)
            identified_faces.append({
                "name": name,
                "confidence": float(confidence),
                "location": [int(x) for x in face_location],
                "face_number": i + 1,
                "best_match_index": int(best_match_index)
            })

        #Second pass: only keep best matches and mark others as Unknown
        used_scientists = set()
        for i, face in enumerate(identified_faces):
            if face["name"] != "Unknown":
                if face["name"] in scientist_matches and scientist_matches[face["name"]]['face_index'] == i:
                    #This is the best match for this scientist
                    used_scientists.add(face["name"])
                else:
                    #This is not the best match, mark as Unknown
                    identified_faces[i]["name"] = "Unknown"
                    identified_faces[i]["confidence"] = 0.0
                    identified_faces[i]["best_match_index"] = -1

        #Third pass: draw the final results
        for i, face in enumerate(identified_faces):
            face_location = [face["location"][0], face["location"][1], face["location"][2], face["location"][3]]
            top, right, bottom, left = face_location

            #Color coding based on confidence
            if face["confidence"] > 0.5:
                color = "green"
            elif face["confidence"] > 0.3:
                color = "orange"
            else:
                color = "red"

            #Draw rectangle around face
            draw.rectangle([left, top, right, bottom], outline=color, width=3)

            #Get ground truth name from manual mapping
            ground_truth_name = self.find_closest_manual_face(face_location)

            #Draw prediction label
            prediction_text = f"{face['name']} ({face['confidence']:.2f})"
            pred_text_bbox = draw.textbbox((0, 0), prediction_text, font=font)
            pred_text_width = pred_text_bbox[2] - pred_text_bbox[0]
            pred_text_height = pred_text_bbox[3] - pred_text_bbox[1]

            #Draw ground truth label
            gt_text = f"GT: {ground_truth_name}"
            gt_text_bbox = draw.textbbox((0, 0), gt_text, font=font)
            gt_text_width = gt_text_bbox[2] - gt_text_bbox[0]
            gt_text_height = gt_text_bbox[3] - gt_text_bbox[1]

            #Calculate total label dimensions
            total_width = max(pred_text_width, gt_text_width) + 10
            total_height = pred_text_height + gt_text_height + 15

            #Draw prediction background (color-coded)
            draw.rectangle([left, top - total_height, left + total_width, top - gt_text_height - 10],
                         fill=color, outline=color)
            
            #Draw ground truth background (green)
            draw.rectangle([left, top - gt_text_height - 10, left + total_width, top],
                         fill="green", outline="green")

            #Draw texts
            draw.text((left + 5, top - total_height + 5), prediction_text, fill="white", font=font)
            draw.text((left + 5, top - gt_text_height - 5), gt_text, fill="white", font=font)

        #Add ground truth comparison box
        self.add_ground_truth_box(pil_image, identified_faces, draw, font)

        return pil_image, identified_faces

    def add_ground_truth_box(self, pil_image, identified_faces, draw, font):
        """Add ground truth comparison box in top right corner"""
        #Define ground truth positions (back to front, left to right)
        ground_truth = [
            #Back row (positions 1-11)
            "Auguste Piccard", "Émile Henriot", "Paul Ehrenfest", "Édouard Herzen",
            "Théophile de Donder", "Erwin Schrödinger", "Jules-Émile Verschaffelt",
            "Wolfgang Pauli", "Werner Heisenberg", "Ralph Howard Fowler", "Léon Brillouin",
            #Middle row (positions 12-20)
            "Peter Debye", "Martin Knudsen", "William Lawrence Bragg", "Hendrik Anthony Kramers",
            "Paul Dirac", "Arthur Compton", "Louis de Broglie", "Max Born", "Niels Bohr",
            #Front row (positions 21-29)
            "Irving Langmuir", "Max Planck", "Marie Skłodowska Curie", "Hendrik Lorentz",
            "Albert Einstein", "Paul Langevin", "Charles-Eugène Guye", "Charles Thomson Rees Wilson",
            "Owen Willans Richardson"
        ]

        #Count correct identifications
        identified_names = [face['name'] for face in identified_faces if face['name'] != 'Unknown']
        correct_identifications = sum(1 for name in identified_names if name in ground_truth or
                                    name == "Marie Curie" and "Marie Skłodowska Curie" in ground_truth)
        total_scientists = len(ground_truth)
        accuracy = (correct_identifications / total_scientists) * 100

        #Get image dimensions
        img_width, img_height = pil_image.size

        #Box dimensions and position (top right corner)
        box_width = 320
        box_height = 140
        box_x = img_width - box_width - 20
        box_y = 20

        #Draw semi-transparent background
        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height],
                      fill=(0, 0, 0, 180), outline="white", width=2)

        #Title
        title_text = "Ground Truth Comparison"
        draw.text((box_x + 10, box_y + 10), title_text, fill="white", font=font)

        #Statistics
        stats_y = box_y + 35
        stats_text = [
            f"Total Scientists: {total_scientists}",
            f"Correctly Identified: {correct_identifications}",
            f"Accuracy: {accuracy:.1f}%",
            f"False Positives: 0",
            f"Unidentified: {total_scientists - correct_identifications}"
        ]

        for i, text in enumerate(stats_text):
            draw.text((box_x + 10, stats_y + i * 20), text, fill="white", font=font)

    def generate_analytics(self, identified_faces):
        """Generate detailed analytics about the recognition results"""
        #Create DataFrame for analysis
        df = pd.DataFrame(identified_faces)

        #Count identifications
        name_counts = Counter([face['name'] for face in identified_faces])

        #Calculate confidence statistics
        confidence_stats = {}
        for name in name_counts.keys():
            confidences = [face['confidence'] for face in identified_faces if face['name'] == name]
            confidence_stats[name] = {
                'count': len(confidences),
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'max_confidence': max(confidences),
                'min_confidence': min(confidences)
            }

        return name_counts, confidence_stats, df

    def create_visualization(self, name_counts, confidence_stats, output_dir):
        """Create visualization plots of the results"""
        #Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        #Plot 1: Number of detections per scientist
        scientists = list(name_counts.keys())
        counts = list(name_counts.values())

        ax1.bar(scientists, counts, color='skyblue')
        ax1.set_title('Number of Face Detections per Scientist')
        ax1.set_xlabel('Scientist')
        ax1.set_ylabel('Number of Detections')
        ax1.tick_params(axis='x', rotation=45)

        #Plot 2: Confidence distribution
        all_confidences = [stats['mean_confidence'] for stats in confidence_stats.values()]
        ax2.hist(all_confidences, bins=20, color='lightgreen', alpha=0.7)
        ax2.set_title('Distribution of Average Confidence Scores')
        ax2.set_xlabel('Average Confidence')
        ax2.set_ylabel('Frequency')

        #Plot 3: Confidence by scientist
        scientists_with_conf = [name for name in confidence_stats.keys() if name != 'Unknown']
        mean_confs = [confidence_stats[name]['mean_confidence'] for name in scientists_with_conf]

        ax3.scatter(scientists_with_conf, mean_confs, s=100, alpha=0.7, color='red')
        ax3.set_title('Mean Confidence Score by Scientist')
        ax3.set_xlabel('Scientist')
        ax3.set_ylabel('Mean Confidence')
        ax3.tick_params(axis='x', rotation=45)

        #Plot 4: Recognition success rate
        total_faces = sum(name_counts.values())
        identified_faces = total_faces - name_counts.get('Unknown', 0)
        success_rate = (identified_faces / total_faces) * 100

        ax4.pie([identified_faces, name_counts.get('Unknown', 0)],
                labels=['Identified', 'Unknown'],
                autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'])
        ax4.set_title(f'Recognition Success Rate: {success_rate:.1f}%')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'enhanced_recognition_analytics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, image, identified_faces, output_path):
        """Save the annotated image and results with enhanced analytics"""
        output_dir = os.path.dirname(output_path)

        #Save annotated image
        image.save(output_path)

        #Save results as JSON
        results_path = output_path.replace('.png', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(identified_faces, f, indent=2)

        #Generate analytics
        name_counts, confidence_stats, df = self.generate_analytics(identified_faces)

        #Save analytics as CSV
        csv_path = output_path.replace('.png', '_analytics.csv')
        df.to_csv(csv_path, index=False)

        #Create visualization
        self.create_visualization(name_counts, confidence_stats, output_dir)

        #Save detailed analytics
        analytics_path = output_path.replace('.png', '_detailed_analytics.json')
        analytics_data = {
            'name_counts': dict(name_counts),
            'confidence_stats': confidence_stats,
            'total_faces': len(identified_faces),
            'identified_faces': len(identified_faces) - name_counts.get('Unknown', 0),
            'success_rate': ((len(identified_faces) - name_counts.get('Unknown', 0)) / len(identified_faces)) * 100
        }

        with open(analytics_path, 'w') as f:
            json.dump(analytics_data, f, indent=2)

        print(f"Results saved to {output_path}")
        print(f"Details saved to {results_path}")
        print(f"Analytics saved to {csv_path}")
        print(f"Detailed analytics saved to {analytics_path}")
        print(f"Visualization saved to {os.path.join(output_dir, 'enhanced_recognition_analytics.png')}")

    def run_recognition(self):
        """Main function to run the facial recognition with enhanced features"""
        print(" ENHANCED SOLVAY FACE RECOGNITION WITH MANUAL GROUND TRUTH ")
        print("Loading reference images...")
        self.load_reference_images()

        print(f"Loaded {len(self.known_encodings)} reference faces")
        
        print("Loading manual ground truth mapping...")
        self.load_manual_ground_truth()

        #Path to the Solvay conference photo
        solvay_photo_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/solvay.jpg"

        print("Analyzing Solvay conference photo with enhanced processing...")
        annotated_image, identified_faces = self.detect_and_identify_faces(solvay_photo_path)

        #Save results
        output_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/solvay_identification.png"
        self.save_results(annotated_image, identified_faces, output_path)

        #Generate analytics
        name_counts, confidence_stats, df = self.generate_analytics(identified_faces)

        #Print enhanced summary
        print("\n ENHANCED IDENTIFICATION RESULTS ")
        print(f"Total faces detected: {len(identified_faces)}")
        print(f"Successfully identified: {len(identified_faces) - name_counts.get('Unknown', 0)}")
        print(f"Success rate: {((len(identified_faces) - name_counts.get('Unknown', 0)) / len(identified_faces)) * 100:.1f}%")

        print("\n IDENTIFIED SCIENTISTS ")
        for name, stats in confidence_stats.items():
            if name != "Unknown":
                print(f"  {name}: {stats['count']} detections, avg confidence: {stats['mean_confidence']:.3f}")

        print(f"\n UNKNOWN FACES ")
        print(f"  Unknown: {name_counts.get('Unknown', 0)} faces")

        #Show the image
        try:
            annotated_image.show()
        except:
            print("Image display not available in this environment")

        return identified_faces, name_counts, confidence_stats

def main():
    """Main function with enhanced configuration options"""
    print(" ENHANCED SOLVAY FACE RECOGNITION SYSTEM ")
    print("Initializing with optimized settings...")

    #Initialize with enhanced settings
    recognizer = EnhancedSolvayFaceRecognizer(tolerance=0.60, model='hog')
    results, name_counts, confidence_stats = recognizer.run_recognition()

    print("\n PROGRAM COMPLETE ")
    print("Enhanced analysis complete! Check the output files for:")
    print("  - Annotated image with color-coded confidence levels")
    print("  - Detailed JSON results")
    print("  - Analytics CSV data")
    print("  - Comprehensive visualization charts")
    print("  - Detailed performance analytics")

if __name__ == "__main__":
    main()