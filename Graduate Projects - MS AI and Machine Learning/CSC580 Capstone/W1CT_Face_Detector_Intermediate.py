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
                loaded_count = 0

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
                                loaded_count += 1
                            else:
                                #Try with preprocessing if normal method fails
                                try:
                                    rgb_img, gray_img = self.preprocess_image(str(img_file))
                                    encodings = face_recognition.face_encodings(rgb_img, model='large')
                                    if encodings:
                                        self.known_encodings.append(encodings[0])
                                        self.known_names.append(scientist_name)
                                        loaded_count += 1
                                    else:
                                        print(f"No face found in {img_file}")
                                except:
                                    print(f"No face found in {img_file}")
                        except Exception as e:
                            print(f"Error loading {img_file}: {e}")

                #Print summary for this scientist
                if loaded_count > 0:
                    print(f"Loaded {loaded_count} reference images for {scientist_name}")

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

            #Color coding based on confidence (round to 2 decimal places for consistent display)
            rounded_confidence = round(face["confidence"], 2)
            if rounded_confidence > 0.5:
                color = "green"
            elif rounded_confidence >= 0.4:
                color = "orange"
            else:
                color = "red"

            #Draw rectangle around face
            draw.rectangle([left, top, right, bottom], outline=color, width=3)

            #Get ground truth name from manual mapping
            ground_truth_name = self.find_closest_manual_face(face_location)

            #Check if this is a misidentification
            is_misidentified = (face['name'] != "Unknown" and 
                              ground_truth_name != "Unknown Position" and 
                              face['name'] != ground_truth_name)

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
            
            #Draw ground truth background (green, with red border if misidentified)
            gt_fill_color = "green"
            gt_outline_color = "red" if is_misidentified else "green"
            gt_outline_width = 3 if is_misidentified else 1
            
            draw.rectangle([left, top - gt_text_height - 10, left + total_width, top],
                         fill=gt_fill_color, outline=gt_outline_color, width=gt_outline_width)

            #Draw texts
            draw.text((left + 5, top - total_height + 5), prediction_text, fill="white", font=font)
            draw.text((left + 5, top - gt_text_height - 5), gt_text, fill="white", font=font)

        #Add ground truth comparison box
        self.add_ground_truth_box(pil_image, identified_faces, draw, font)

        return pil_image, identified_faces

    def add_ground_truth_box(self, pil_image, identified_faces, draw, font):
        """Add ground truth comparison box in top right corner"""
        #Count correct vs incorrect identifications using manual ground truth
        correct_identifications = 0
        incorrect_identifications = 0
        unknown_faces = 0
        total_faces = len(identified_faces)
        
        for face in identified_faces:
            predicted_name = face['name']
            face_location = [face["location"][0], face["location"][1], face["location"][2], face["location"][3]]
            ground_truth_name = self.find_closest_manual_face(face_location)
            
            if predicted_name == "Unknown":
                unknown_faces += 1
            elif ground_truth_name != "Unknown Position" and predicted_name == ground_truth_name:
                correct_identifications += 1
            else:
                incorrect_identifications += 1
        
        # Calculate accuracy based on faces that were actually identified (not Unknown)
        identified_faces_count = total_faces - unknown_faces
        accuracy = (correct_identifications / identified_faces_count * 100) if identified_faces_count > 0 else 0
        unidentified_count = unknown_faces + incorrect_identifications

        #Get image dimensions
        img_width, img_height = pil_image.size

        #Box dimensions and position (top right corner) - much larger
        box_width = 800
        box_height = 320
        box_x = img_width - box_width - 30
        box_y = 30

        #Create larger font for the box
        try:
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except:
            large_font = font
            title_font = font

        #Draw semi-transparent background
        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height],
                      fill=(0, 0, 0, 180), outline="white", width=3)

        #Title - larger font
        title_text = "Ground Truth Comparison"
        draw.text((box_x + 25, box_y + 25), title_text, fill="white", font=title_font)

        #Statistics - with more spacing and larger font
        stats_y = box_y + 70
        stats_text = [
            f"Total Faces Detected: {total_faces}",
            f"Correctly Identified: {correct_identifications}",
            f"Incorrectly Identified: {incorrect_identifications}",
            f"Unidentified: {unidentified_count}",
            f"Accuracy (of identified): {accuracy:.1f}%",
            f"Overall Success Rate: {(correct_identifications/total_faces*100):.1f}%"
        ]

        for i, text in enumerate(stats_text):
            draw.text((box_x + 25, stats_y + i * 35), text, fill="white", font=large_font)

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
        """Create enhanced visualization with meaningful analytics"""
        plt.style.use('default')
        
        # Create figure with better layout (removed historical context for more space)
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3, height_ratios=[1, 2])
        
        # Color scheme
        primary_color = '#2E86AB'
        success_color = '#A23B72'
        warning_color = '#F18F01'
        error_color = '#C73E1D'
        
        # 1. System Performance Dashboard (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._create_performance_dashboard(ax1, name_counts, confidence_stats)
        
        # 2. Confidence Analysis (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._create_confidence_analysis(ax2, confidence_stats, primary_color, success_color)
        
        # 3. Recognition Accuracy by Scientist (bottom row, compressed slightly to avoid overlap)
        ax3 = fig.add_subplot(gs[1, :2])
        self._create_accuracy_breakdown(ax3, confidence_stats, primary_color, success_color, error_color)
        
        # 4. Confusion Analysis (bottom right, more space)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Calculate confidence breakdown for the donut chart (round confidence for consistent display)
        confidence_breakdown = {'high': 0, 'medium': 0, 'low': 0}
        for name, stats in confidence_stats.items():
            if name != 'Unknown':
                rounded_conf = round(stats['mean_confidence'], 2)
                if rounded_conf > 0.5:
                    confidence_breakdown['high'] += stats['count']
                elif rounded_conf >= 0.4:
                    confidence_breakdown['medium'] += stats['count']
                else:
                    confidence_breakdown['low'] += stats['count']
        
        self._temp_confidence_breakdown = confidence_breakdown
        self._create_confusion_insights(ax4, name_counts, warning_color, error_color)
        
        plt.suptitle('Solvay Conference 1927 - Face Recognition Analytics Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        plt.savefig(os.path.join(output_dir, 'recognition_analytics.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_performance_dashboard(self, ax, name_counts, confidence_stats):
        """Create a clean performance metrics dashboard with confidence breakdown"""
        ax.axis('off')
        
        total_faces = sum(name_counts.values())
        unknown_faces = name_counts.get('Unknown', 0)
        
        # Calculate confidence breakdowns using rounded values for consistency
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0
        
        for name, stats in confidence_stats.items():
            if name != 'Unknown':
                rounded_conf = round(stats['mean_confidence'], 2)
                if rounded_conf > 0.5:
                    high_confidence += stats['count']
                elif rounded_conf >= 0.4:
                    medium_confidence += stats['count']
                else:
                    low_confidence += stats['count']
        
        metrics = [
            ('Total Faces Detected', total_faces, '#2E86AB'),
            ('High Confidence IDs', high_confidence, '#2E8B57'),  # Green
            ('Medium Confidence IDs', medium_confidence, '#F18F01'),  # Orange
            ('Low Confidence IDs', low_confidence, '#C73E1D'),  # Red
            ('Unknown/Unidentified', unknown_faces, '#C73E1D'),  # Red
            ('Overall Success Rate', f"{((high_confidence + medium_confidence + low_confidence)/total_faces*100):.1f}%", '#2E8B57')
        ]
        
        # Adjust layout for more metrics
        y_positions = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
        
        for i, (label, value, color) in enumerate(metrics):
            # Metric box
            rect = plt.Rectangle((0.05, y_positions[i] - 0.04), 0.9, 0.08, 
                               facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Label and value
            ax.text(0.1, y_positions[i], label, fontsize=11, fontweight='bold')
            ax.text(0.9, y_positions[i], str(value), fontsize=12, fontweight='bold', 
                   ha='right', color=color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('System Performance Breakdown', fontsize=14, fontweight='bold', pad=20)
    
    def _create_confidence_analysis(self, ax, confidence_stats, primary_color, success_color):
        """Create confidence distribution analysis"""
        # Get confidence data for identified scientists only
        identified_scientists = [name for name in confidence_stats.keys() if name != 'Unknown']
        confidences = [confidence_stats[name]['mean_confidence'] for name in identified_scientists]
        
        if not confidences:
            ax.text(0.5, 0.5, 'No identified scientists', ha='center', va='center', fontsize=12)
            ax.set_title('Confidence Analysis', fontsize=14, fontweight='bold')
            return
        
        # Create histogram with better bins
        n_bins = min(10, len(confidences))
        n, bins, patches = ax.hist(confidences, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Color bars based on confidence level (round for consistency)
        for i, patch in enumerate(patches):
            rounded_bin = round(bins[i], 2)
            if rounded_bin < 0.4:
                patch.set_facecolor(error_color := '#C73E1D')
            elif rounded_bin <= 0.5:
                patch.set_facecolor('#F18F01')
            else:
                patch.set_facecolor('#2E8B57')
        
        ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(confidences):.2f}')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Number of Scientists', fontsize=12)
        ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_accuracy_breakdown(self, ax, confidence_stats, primary_color, success_color, error_color):
        """Create scientist-by-scientist accuracy breakdown including unidentified faces"""
        identified_scientists = [name for name in confidence_stats.keys() if name != 'Unknown']
        
        # Add identified scientists to the data
        scientist_data = [(name, confidence_stats[name]['mean_confidence'], 'identified') 
                         for name in identified_scientists]
        
        # Get unidentified scientists from ground truth
        if hasattr(self, 'manual_ground_truth') and self.manual_ground_truth:
            # Get all scientists that should be in the photo from ground truth
            all_gt_scientists = set()
            for face_data in self.manual_ground_truth.values():
                gt_name = face_data['ground_truth']
                if gt_name != "Unknown Position":
                    all_gt_scientists.add(gt_name)
            
            # Find scientists in ground truth but not identified
            identified_names = set(identified_scientists)
            unidentified_scientists = all_gt_scientists - identified_names
            
            # Add unidentified scientists with negative value to make them visible as red bars
            for scientist in unidentified_scientists:
                scientist_data.append((scientist, -0.12, 'unidentified'))
        
        if not scientist_data:
            ax.text(0.5, 0.5, 'No faces detected', ha='center', va='center', fontsize=12)
            ax.set_title('Recognition Accuracy by Scientist', fontsize=14, fontweight='bold')
            return
        
        # Sort by confidence (identified scientists first, then unidentified)
        scientist_data.sort(key=lambda x: x[1], reverse=True)
        
        names = [item[0] for item in scientist_data]
        confidences = [item[1] for item in scientist_data]
        status = [item[2] for item in scientist_data]
        
        # Create horizontal bar chart with colors (round confidence for consistent display)
        colors = []
        for i, conf in enumerate(confidences):
            rounded_conf = round(conf, 2)
            if status[i] == 'unidentified' or conf < 0:
                colors.append(error_color)  # Red for unidentified (negative values)
            elif rounded_conf > 0.5:
                colors.append('#2E8B57')    # Green for high confidence
            elif rounded_conf >= 0.4:
                colors.append('#F18F01')    # Orange for medium confidence (includes 0.5)
            else:
                colors.append(error_color)  # Red for low confidence
        
        bars = ax.barh(range(len(names)), confidences, color=colors, alpha=0.8, edgecolor='black')
        
        # Add confidence values on bars
        for i, (bar, conf, stat) in enumerate(zip(bars, confidences, status)):
            if stat == 'identified':  # Show confidence for identified scientists
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{conf:.2f}', va='center', fontsize=10, fontweight='bold')
            elif stat == 'unidentified':  # Show "Unknown" inside negative bars
                ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height()/2, 
                       'Unknown', va='center', ha='center', fontsize=9, fontweight='bold', color='white')
        
        ax.set_yticks(range(len(names)))
        # Truncate long names for better display
        display_names = []
        for name in names:
            display_names.append(name.split()[-1] if len(name.split()) > 1 else name)
        
        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_title('Recognition Results: All Scientists (Sorted by Confidence)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(-0.15, 1)  # Extended more to show negative bars for unidentified scientists
        
        # Add confidence thresholds
        ax.axvline(0.5, color='green', linestyle='--', alpha=0.7, label='High Confidence')
        ax.axvline(0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
        ax.legend(loc='upper right')
    
    def _create_confusion_insights(self, ax, name_counts, warning_color, error_color):
        """Create insights about recognition challenges"""
        ax.axis('off')
        
        total_faces = sum(name_counts.values())
        unknown_faces = name_counts.get('Unknown', 0)
        identified_faces = total_faces - unknown_faces
        
        # Break down identified faces by confidence level
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0
        
        # Access confidence stats from parent scope (we need to pass this data)
        # For now, we'll approximate based on the identified vs unknown split
        # This is a simplified version - ideally we'd pass detailed confidence breakdown
        
        if hasattr(self, '_temp_confidence_breakdown'):
            high_confidence = self._temp_confidence_breakdown.get('high', 0)
            medium_confidence = self._temp_confidence_breakdown.get('medium', 0) 
            low_confidence = self._temp_confidence_breakdown.get('low', 0)
        else:
            # Fallback: assume all identified are high confidence for now
            high_confidence = identified_faces
        
        # Create pie chart with confidence breakdown
        if total_faces > 0:
            sizes = []
            labels = []
            colors = []
            
            if high_confidence > 0:
                sizes.append(high_confidence)
                labels.append('High Confidence')
                colors.append('#2E8B57')  # Green
                
            if medium_confidence > 0:
                sizes.append(medium_confidence)
                labels.append('Medium Confidence') 
                colors.append('#F18F01')  # Orange/Yellow
                
            if low_confidence > 0:
                sizes.append(low_confidence)
                labels.append('Low Confidence')
                colors.append(error_color)  # Red
                
            if unknown_faces > 0:
                sizes.append(unknown_faces)
                labels.append('Unidentified')
                colors.append(error_color)  # Red
            
            wedges, texts, autotexts = ax.pie(sizes,
                                            labels=labels,
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90,
                                            textprops={'fontsize': 10, 'fontweight': 'bold'})
            
            # Add center circle for donut effect
            centre_circle = plt.Circle((0,0), 0.50, fc='white')
            ax.add_artist(centre_circle)
            
            # Add center text
            ax.text(0, 0, f'{total_faces}\nFaces\nDetected', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        ax.set_title('Recognition Quality Breakdown', fontsize=14, fontweight='bold')
    

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
        print(f"Visualization saved to {os.path.join(output_dir, 'recognition_analytics.png')}")

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