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
import wikipedia
from typing import Optional, Dict
import time
import textwrap
import warnings
from urllib.parse import unquote
import requests
from bs4 import BeautifulSoup
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available - using fallback summarization")

#Suppress warnings from various libraries
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message="Your max_length is set to")
warnings.filterwarnings("ignore", message="Asking to truncate")

#Suppress device messages and all transformers logging
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable transformers logging completely
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class WikipediaLLMSummarizer:
    """Class to fetch Wikipedia content and summarize scientist contributions using LLM"""
    
    def __init__(self):
        """Initialize the summarizer with HuggingFace transformers"""
        self.cache = {}  #Cache summaries to avoid repeat API calls
        self.summarizer = None
        self.verified_urls = self.load_verified_url_mapping()
        
        #Initialize HuggingFace summarization pipeline
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading summarization model... (this may take a moment on first run)")
                # Check if CUDA is available
                import torch
                if torch.cuda.is_available():
                    device = 0  # Use first GPU
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"Device set to CUDA GPU: {gpu_name}")
                else:
                    device = -1  # Fallback to CPU
                    print("Device set to CPU (CUDA not available)")
                
                # Temporarily redirect stderr to suppress device messages
                import sys
                old_stderr = sys.stderr
                sys.stderr = open(os.devnull, 'w')
                
                try:
                    self.summarizer = pipeline(
                        "summarization", 
                        model="facebook/bart-large-cnn",
                        device=device
                    )
                finally:
                    sys.stderr.close()
                    sys.stderr = old_stderr
                    
                print("Summarization model loaded successfully!")
            except Exception as e:
                print(f"Error loading summarization model: {e}")
                print("Falling back to simple text extraction")
                self.summarizer = None
    
    def load_verified_url_mapping(self):
        """Load the verified Wikipedia URL mapping from JSON file"""
        try:
            mapping_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/wikipedia_url_mapping.json"
            with open(mapping_path, 'r') as f:
                url_mapping = json.load(f)
            print(f"Loaded verified URLs for {len(url_mapping)} scientists")
            return url_mapping
        except Exception as e:
            print(f"Could not load verified URL mapping: {e}")
            return {}
        
    def fetch_wikipedia_content(self, scientist_name: str) -> Optional[str]:
        """Fetch Wikipedia page content for a scientist using verified URLs"""
        try:
            #Set Wikipedia language to English explicitly
            wikipedia.set_lang("en")
            
            # First priority: Use verified URLs from WikiFinder.py with direct HTTP request
            if scientist_name in self.verified_urls:
                verified_url = self.verified_urls[scientist_name]
                try:
                    # Use direct HTTP request since the URLs work when clicked
                    response = requests.get(verified_url, timeout=10)
                    response.raise_for_status()
                    
                    # Parse the HTML to extract text content
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'sup', 'table']):
                        element.decompose()
                    
                    # Extract main content from Wikipedia page
                    content_div = soup.find('div', {'id': 'mw-content-text'})
                    if content_div:
                        # Get all paragraphs from the main content
                        paragraphs = content_div.find_all('p')
                        text_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        
                        if text_content:
                            print(f"Successfully loaded {scientist_name}'s page from Wikipedia")
                            return text_content
                    
                    # Fallback: get all text if structured extraction fails
                    text_content = soup.get_text()
                    if text_content and len(text_content) > 500:
                        print(f"Successfully loaded {scientist_name}'s page from Wikipedia (fallback)")
                        return text_content
                        
                except Exception as e:
                    print(f"Error with direct HTTP request for {scientist_name}: {e}")
                    # Fall through to backup methods
            
            # Backup: Direct Wikipedia page mappings for 1927 Solvay Conference participants
            # NOTE: These are now backup only - verified URLs from WikiFinder.py take priority
            direct_page_mappings = {
                # Front Row
                "Irving Langmuir": "Irving_Langmuir",
                "Max Planck": "Max_Planck",
                "Marie Curie": "Marie_Curie",
                "Hendrik Lorentz": "Hendrik_Lorentz",
                "Albert Einstein": "Albert_Einstein",
                "Paul Langevin": "Paul_Langevin",
                "Charles Eugene Guye": "Charles-Eugène_Guye",
                "Charles Wilson": "Charles_Thomson_Rees_Wilson",
                "CTR Wilson": "Charles_Thomson_Rees_Wilson",
                "Owen Richardson": "Owen_Willans_Richardson",
                
                # Middle Row
                "Peter Debye": "Peter_Debye",
                "Martin Knudsen": "Martin_Knudsen",
                "William Lawrence Bragg": "Lawrence_Bragg",
                "Hendrik Anthony Kramers": "Hendrik_Kramers",
                "Paul Dirac": "Paul_Dirac",
                "Arthur Compton": "Arthur_Compton",
                "Louis de Broglie": "Louis_de_Broglie",
                "Max Born": "Max_Born",
                "Niels Bohr": "Niels_Bohr",
                
                # Back Row
                "Auguste Piccard": "Auguste_Piccard",
                "Emile Henriot": "Émile_Henriot",
                "Paul Ehrenfest": "Paul_Ehrenfest",
                "Edouard Herzen": "Édouard_Herzen",
                "Theophile de Donder": "Théophile_de_Donder",
                "Erwin Schrodinger": "Erwin_Schrödinger",
                "Jules Emile Verschaffelt": "Jules-Émile_Verschaffelt",
                "JE Verschaffelt": "Jules-Émile_Verschaffelt",
                "Wolfgang Pauli": "Wolfgang_Pauli",
                "Werner Heisenberg": "Werner_Heisenberg",
                "Ralph Fowler": "Ralph_Fowler",
                "Leon Brillouin": "Léon_Brillouin"
            }
            
            #Check if we have a direct mapping
            if scientist_name in direct_page_mappings:
                page_title = direct_page_mappings[scientist_name]
                try:
                    page = wikipedia.page(page_title)
                    return page.content
                except Exception as e:
                    print(f"Error fetching direct page {page_title}: {e}")
                    # Fall through to original logic
            
            #Fallback to original name mappings for backwards compatibility
            name_mappings = {
                "Charles Wilson": "Charles Thomson Rees Wilson",
                "CTR Wilson": "Charles Thomson Rees Wilson", 
                "Marie Curie": "Marie Curie",
                "Albert Einstein": "Albert Einstein",
                "Max Planck": "Max Planck",
                "Niels Bohr": "Niels Bohr",
                "Werner Heisenberg": "Werner Heisenberg",
                "Erwin Schrodinger": "Erwin Schrödinger",
                "Wolfgang Pauli": "Wolfgang Pauli",
                "Paul Dirac": "Paul Dirac",
                "Louis de Broglie": "Louis de Broglie",
                "Max Born": "Max Born",
                "Arthur Compton": "Arthur Compton",
                "Auguste Piccard": "Auguste Piccard",
                "Hendrik Lorentz": "Hendrik Lorentz",
                "Paul Langevin": "Paul Langevin",
                "Hendrik Anthony Kramers": "Hendrik Kramers",
                "Martin Knudsen": "Martin Knudsen",
                "William Lawrence Bragg": "William Lawrence Bragg",
                "Peter Debye": "Peter Debye",
                "Irving Langmuir": "Irving Langmuir",
                "Owen Richardson": "Owen Richardson",
                "Paul Ehrenfest": "Paul Ehrenfest",
                "Emile Henriot": "Émile Henriot (chemist)",
                "Edouard Herzen": "Édouard Herzen",
                "Theophile de Donder": "Théophile de Donder",
                "Jules Emile Verschaffelt": "Jozef Emile Verschaffelt",
                "Leon Brillouin": "Léon Brillouin",
                "Ralph Fowler": "Ralph Fowler"
            }
            
            #Use mapped name if available
            search_name = name_mappings.get(scientist_name, scientist_name)
            
            #Try multiple approaches for specific scientists
            if scientist_name == "Auguste Piccard":
                search_attempts = ["Auguste Piccard", "Auguste Antoine Piccard"]
                for attempt in search_attempts:
                    try:
                        page = wikipedia.page(attempt)
                        return page.content
                    except wikipedia.DisambiguationError as e:
                        if e.options:
                            try:
                                page = wikipedia.page(e.options[0])
                                return page.content
                            except:
                                continue
                    except wikipedia.PageError:
                        continue
                    except Exception:
                        continue
                
                #Try search if direct lookup fails
                try:
                    search_results = wikipedia.search("Auguste Piccard", results=5)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                        return page.content
                except Exception:
                    pass
            
            elif scientist_name == "Emile Henriot":
                search_attempts = ["Émile Henriot (chemist)", "Émile Henriot", "Emile Henriot"]
                for attempt in search_attempts:
                    try:
                        page = wikipedia.page(attempt)
                        return page.content
                    except wikipedia.DisambiguationError as e:
                        if e.options:
                            try:
                                page = wikipedia.page(e.options[0])
                                return page.content
                            except:
                                continue
                    except wikipedia.PageError:
                        continue
                    except Exception:
                        continue
                
                #Try search if direct lookup fails
                try:
                    search_results = wikipedia.search("Émile Henriot chemist", results=5)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                        return page.content
                except Exception:
                    pass
            
            elif scientist_name == "Wolfgang Pauli":
                search_attempts = ["Wolfgang Pauli", "Wolfgang Ernst Pauli"]
                for attempt in search_attempts:
                    try:
                        page = wikipedia.page(attempt)
                        return page.content
                    except wikipedia.DisambiguationError as e:
                        if e.options:
                            try:
                                page = wikipedia.page(e.options[0])
                                return page.content
                            except:
                                continue
                    except wikipedia.PageError:
                        continue
                    except Exception:
                        continue
                
                #Try search if direct lookup fails
                try:
                    search_results = wikipedia.search("Wolfgang Pauli physicist", results=5)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                        return page.content
                except Exception:
                    pass
            
            elif scientist_name == "Hendrik Lorentz":
                search_attempts = ["Hendrik Lorentz", "Hendrik Antoon Lorentz"]
                for attempt in search_attempts:
                    try:
                        page = wikipedia.page(attempt)
                        return page.content
                    except wikipedia.DisambiguationError as e:
                        if e.options:
                            try:
                                page = wikipedia.page(e.options[0])
                                return page.content
                            except:
                                continue
                    except wikipedia.PageError:
                        continue
                    except Exception:
                        continue
                
                #Try search if direct lookup fails
                try:
                    search_results = wikipedia.search("Hendrik Lorentz physicist", results=5)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                        return page.content
                except Exception:
                    pass
            
            #Standard approach for other scientists
            else:
                try:
                    page = wikipedia.page(search_name)
                    return page.content
                except wikipedia.DisambiguationError as e:
                    if e.options:
                        page = wikipedia.page(e.options[0])
                        return page.content
                except wikipedia.PageError:
                    search_results = wikipedia.search(search_name, results=5)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                        return page.content
                    
            return None
                    
        except Exception as e:
            print(f"Error fetching Wikipedia content for {scientist_name}: {e}")
            return None
            
    def extract_scientific_content(self, wikipedia_content: str, scientist_name: str) -> str:
        """Extract relevant scientific content from Wikipedia text"""
        #Split into paragraphs and filter for scientific content
        paragraphs = wikipedia_content.split('\n\n')
        scientific_keywords = [
            'theory', 'discovery', 'equation', 'principle', 'research', 'experiment',
            'physics', 'quantum', 'relativity', 'nobel', 'conference', 'contribution',
            'developed', 'invented', 'proposed', 'demonstrated', 'proved', 'formulated'
        ]
        
        relevant_paragraphs = []
        for para in paragraphs[:10]:  #Check first 10 paragraphs
            if any(keyword.lower() in para.lower() for keyword in scientific_keywords):
                relevant_paragraphs.append(para)
        
        #Combine and limit length
        scientific_text = ' '.join(relevant_paragraphs)[:2000]
        return scientific_text if scientific_text else wikipedia_content[:1500]
    
    def summarize_contributions(self, scientist_name: str, wikipedia_content: str) -> str:
        """Use HuggingFace transformers to summarize scientist's contributions"""
        
        #Check cache first
        if scientist_name in self.cache:
            return self.cache[scientist_name]
        
        try:
            #Extract scientific content
            scientific_content = self.extract_scientific_content(wikipedia_content, scientist_name)
            
            if self.summarizer and len(scientific_content) > 100:
                #Use HuggingFace BART model for summarization
                #Calculate appropriate lengths based on input
                input_length = len(scientific_content.split())
                
                #Suppress warnings temporarily
                import logging
                logging.getLogger("transformers").setLevel(logging.ERROR)
                
                try:
                    #Set appropriate length parameters for a concise paragraph
                    if input_length < 150:
                        #For short content, use the content as-is or with minimal summarization
                        max_len = min(input_length + 20, 200)
                        min_len = min(80, input_length - 10)
                    else:
                        #For longer content, create a proper summary
                        max_len = 200
                        min_len = 100
                    
                    summary_result = self.summarizer(
                        scientific_content,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False,
                        truncation=True,
                        clean_up_tokenization_spaces=True
                    )
                    summary = summary_result[0]['summary_text']
                finally:
                    #Restore logging level
                    logging.getLogger("transformers").setLevel(logging.WARNING)
                
                #Ensure the summary ends with proper punctuation
                if summary and not summary.endswith(('.', '!', '?')):
                    #Find the last complete sentence
                    last_period = summary.rfind('.')
                    if last_period > 50:  #Keep at least some content
                        summary = summary[:last_period + 1]
                    else:
                        summary += '.'
                
                #Add context about the scientist
                enhanced_summary = f"{scientist_name}: {summary}"
                
            else:
                #Fallback: Extract meaningful content and ensure complete sentences
                sentences = []
                
                #Split content into sentences more carefully
                import re
                sentence_endings = re.split(r'(?<=[.!?])\s+', scientific_content)
                
                #Look for key biographical and scientific information
                keywords = ['discovered', 'developed', 'theory', 'known for', 'invented', 
                           'research', 'physicist', 'chemist', 'scientist', 'professor', 
                           'born', 'studied', 'contribution', 'work', 'pioneer', 'nobel',
                           'award', 'founded', 'principle', 'equation', 'effect']
                
                for sentence in sentence_endings[:20]:  #Check more sentences
                    sentence = sentence.strip()
                    if len(sentence) > 30 and any(word in sentence.lower() for word in keywords):
                        #Ensure sentence ends properly
                        if not sentence.endswith(('.', '!', '?')):
                            sentence += '.'
                        sentences.append(sentence)
                        
                        #Stop when we have enough content for a paragraph
                        if len(' '.join(sentences)) > 150:
                            break
                
                if sentences:
                    #Create a coherent paragraph from the best sentences
                    summary_text = ' '.join(sentences[:5])  #Up to 5 sentences
                    
                    #Ensure it's not too long and ends properly
                    if len(summary_text) > 400:
                        #Truncate at last complete sentence within 400 chars
                        last_sentence_end = max(
                            summary_text[:400].rfind('.'),
                            summary_text[:400].rfind('!'),
                            summary_text[:400].rfind('?')
                        )
                        if last_sentence_end > 100:
                            summary_text = summary_text[:last_sentence_end + 1]
                    
                    enhanced_summary = f"{scientist_name}: {summary_text}"
                else:
                    #Last resort: use first paragraph of content
                    paragraphs = scientific_content.split('\n\n')
                    for para in paragraphs:
                        if len(para.strip()) > 100:
                            para = para.strip()
                            #Ensure complete sentences
                            if not para.endswith(('.', '!', '?')):
                                last_punct = max(para.rfind('.'), para.rfind('!'), para.rfind('?'))
                                if last_punct > 50:
                                    para = para[:last_punct + 1]
                            enhanced_summary = f"{scientist_name}: {para[:350]}"
                            break
                    else:
                        enhanced_summary = f"{scientist_name}: A prominent physicist and participant in the 1927 Solvay Conference on Electrons and Photons, which brought together the world's leading quantum physicists."
            
            #Cache the result
            self.cache[scientist_name] = enhanced_summary
            return enhanced_summary
            
        except Exception as e:
            error_msg = f"Error generating summary for {scientist_name}: {e}"
            print(error_msg)
            fallback = f"{scientist_name}: Unable to generate detailed summary. Featured at the 1927 Solvay Conference."
            self.cache[scientist_name] = fallback
            return fallback
            
    def get_scientist_summary(self, scientist_name: str) -> Dict[str, str]:
        """Get complete summary including Wikipedia content and LLM analysis"""
        
        print(f"Fetching information for {scientist_name}...")
        
        #Fetch Wikipedia content
        wikipedia_content = self.fetch_wikipedia_content(scientist_name)
        
        if not wikipedia_content:
            return {
                "scientist": scientist_name,
                "status": "error",
                "message": f"Could not find Wikipedia page for {scientist_name}",
                "summary": "",
                "wikipedia_available": False
            }
        
        #Generate LLM summary
        ai_summary = self.summarize_contributions(scientist_name, wikipedia_content)
        
        return {
            "scientist": scientist_name,
            "status": "success", 
            "summary": ai_summary,
            "wikipedia_available": True,
            "wikipedia_length": len(wikipedia_content),
            "cached": scientist_name in self.cache
        }


class EnhancedSolvayFaceRecognizer:
    def __init__(self, tolerance=0.55, model='cnn'):
        self.known_encodings = []
        self.known_names = []
        self.tolerance = tolerance
        self.model = model  #'hog' or 'cnn'
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        #Initialize Wikipedia summarizer for interactive features
        self.wikipedia_summarizer = WikipediaLLMSummarizer()
        self.identified_faces_data = []  #Store face data for click detection

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
        reference_dir = Path(
            "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/reference_images")

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

        #Find the manual ground truth entry with the closest location
        for face_data in self.manual_ground_truth.values():
            manual_location = face_data['location']  #[top, right, bottom, left]
            manual_top, manual_right, manual_bottom, manual_left = manual_location
            manual_center_x = (manual_left + manual_right) / 2
            manual_center_y = (manual_top + manual_bottom) / 2

            #Calculate Euclidean distance between face centers
            distance = ((face_center_x - manual_center_x) ** 2 +
                        (face_center_y - manual_center_y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_scientist = face_data['ground_truth']

        #Only return the match if it's reasonably close (within 100 pixels)
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

        #Calculate accuracy based on faces that were actually identified (not Unknown)
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
            f"Overall Success Rate: {(correct_identifications / total_faces * 100):.1f}%"
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

        #Create figure with better layout (removed historical context for more space)
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3, height_ratios=[1, 2])

        #Color scheme
        primary_color = '#2E86AB'
        success_color = '#A23B72'
        warning_color = '#F18F01'
        error_color = '#C73E1D'

        #1. System Performance Dashboard (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._create_performance_dashboard(ax1, name_counts, confidence_stats)

        #2. Confidence Analysis (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._create_confidence_analysis(ax2, confidence_stats, primary_color, success_color)

        #3. Recognition Accuracy by Scientist (bottom row, compressed slightly to avoid overlap)
        ax3 = fig.add_subplot(gs[1, :2])
        self._create_accuracy_breakdown(ax3, confidence_stats, primary_color, success_color, error_color)

        #4. Confusion Analysis (bottom right, more space)
        ax4 = fig.add_subplot(gs[1, 2:])

        #Calculate confidence breakdown for the donut chart (round confidence for consistent display)
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

        plt.savefig(os.path.join(output_dir, 'advanced_recognition_analytics.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_performance_dashboard(self, ax, name_counts, confidence_stats):
        """Create a clean performance metrics dashboard with confidence breakdown"""
        ax.axis('off')

        total_faces = sum(name_counts.values())
        unknown_faces = name_counts.get('Unknown', 0)

        #Calculate confidence breakdowns using rounded values for consistency
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
            ('High Confidence IDs', high_confidence, '#2E8B57'),  #Green
            ('Medium Confidence IDs', medium_confidence, '#F18F01'),  #Orange
            ('Low Confidence IDs', low_confidence, '#C73E1D'),  #Red
            ('Unknown/Unidentified', unknown_faces, '#C73E1D'),  #Red
            ('Overall Success Rate',
             f"{((high_confidence + medium_confidence + low_confidence) / total_faces * 100):.1f}%", '#2E8B57')
        ]

        #Adjust layout for more metrics
        y_positions = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]

        for i, (label, value, color) in enumerate(metrics):
            #Metric box
            rect = plt.Rectangle((0.05, y_positions[i] - 0.04), 0.9, 0.08,
                                 facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax.add_patch(rect)

            #Label and value
            ax.text(0.1, y_positions[i], label, fontsize=11, fontweight='bold')
            ax.text(0.9, y_positions[i], str(value), fontsize=12, fontweight='bold',
                    ha='right', color=color)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('System Performance Breakdown', fontsize=14, fontweight='bold', pad=20)

    def _create_confidence_analysis(self, ax, confidence_stats, primary_color, success_color):
        """Create confidence distribution analysis"""
        #Get confidence data for identified scientists only
        identified_scientists = [name for name in confidence_stats.keys() if name != 'Unknown']
        confidences = [confidence_stats[name]['mean_confidence'] for name in identified_scientists]

        if not confidences:
            ax.text(0.5, 0.5, 'No identified scientists', ha='center', va='center', fontsize=12)
            ax.set_title('Confidence Analysis', fontsize=14, fontweight='bold')
            return

        #Create histogram with better bins
        n_bins = min(10, len(confidences))
        n, bins, patches = ax.hist(confidences, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=1)

        #Color bars based on confidence level (round for consistency)
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

        #Add identified scientists to the data
        scientist_data = [(name, confidence_stats[name]['mean_confidence'], 'identified')
                          for name in identified_scientists]

        #Get unidentified scientists from ground truth
        if hasattr(self, 'manual_ground_truth') and self.manual_ground_truth:
            #Get all scientists that should be in the photo from ground truth
            all_gt_scientists = set()
            for face_data in self.manual_ground_truth.values():
                gt_name = face_data['ground_truth']
                if gt_name != "Unknown Position":
                    all_gt_scientists.add(gt_name)

            #Find scientists in ground truth but not identified
            identified_names = set(identified_scientists)
            unidentified_scientists = all_gt_scientists - identified_names

            #Add unidentified scientists with negative value to make them visible as red bars
            for scientist in unidentified_scientists:
                scientist_data.append((scientist, -0.12, 'unidentified'))

        if not scientist_data:
            ax.text(0.5, 0.5, 'No faces detected', ha='center', va='center', fontsize=12)
            ax.set_title('Recognition Accuracy by Scientist', fontsize=14, fontweight='bold')
            return

        #Sort by confidence (identified scientists first, then unidentified)
        scientist_data.sort(key=lambda x: x[1], reverse=True)

        names = [item[0] for item in scientist_data]
        confidences = [item[1] for item in scientist_data]
        status = [item[2] for item in scientist_data]

        #Create horizontal bar chart with colors (round confidence for consistent display)
        colors = []
        for i, conf in enumerate(confidences):
            rounded_conf = round(conf, 2)
            if status[i] == 'unidentified' or conf < 0:
                colors.append(error_color)  #Red for unidentified (negative values)
            elif rounded_conf > 0.5:
                colors.append('#2E8B57')  #Green for high confidence
            elif rounded_conf >= 0.4:
                colors.append('#F18F01')  #Orange for medium confidence (includes 0.5)
            else:
                colors.append(error_color)  #Red for low confidence

        bars = ax.barh(range(len(names)), confidences, color=colors, alpha=0.8, edgecolor='black')

        #Add confidence values on bars
        for i, (bar, conf, stat) in enumerate(zip(bars, confidences, status)):
            if stat == 'identified':  #Show confidence for identified scientists
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{conf:.2f}', va='center', fontsize=10, fontweight='bold')
            elif stat == 'unidentified':  #Show "Unknown" inside negative bars
                ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        'Unknown', va='center', ha='center', fontsize=9, fontweight='bold', color='white')

        ax.set_yticks(range(len(names)))
        #Truncate long names for better display
        display_names = []
        for name in names:
            display_names.append(name.split()[-1] if len(name.split()) > 1 else name)

        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_title('Recognition Results: All Scientists (Sorted by Confidence)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(-0.15, 1)  #Extended more to show negative bars for unidentified scientists

        #Add confidence thresholds
        ax.axvline(0.5, color='green', linestyle='--', alpha=0.7, label='High Confidence')
        ax.axvline(0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
        ax.legend(loc='upper right')

    def _create_confusion_insights(self, ax, name_counts, warning_color, error_color):
        """Create insights about recognition challenges"""
        ax.axis('off')

        total_faces = sum(name_counts.values())
        unknown_faces = name_counts.get('Unknown', 0)
        identified_faces = total_faces - unknown_faces

        #Break down identified faces by confidence level
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0

        #Access confidence stats from parent scope (we need to pass this data)
        #For now, we'll approximate based on the identified vs unknown split
        #This is a simplified version - ideally we'd pass detailed confidence breakdown

        if hasattr(self, '_temp_confidence_breakdown'):
            high_confidence = self._temp_confidence_breakdown.get('high', 0)
            medium_confidence = self._temp_confidence_breakdown.get('medium', 0)
            low_confidence = self._temp_confidence_breakdown.get('low', 0)
        else:
            #Fallback: assume all identified are high confidence for now
            high_confidence = identified_faces

        #Create pie chart with confidence breakdown
        if total_faces > 0:
            sizes = []
            labels = []
            colors = []

            if high_confidence > 0:
                sizes.append(high_confidence)
                labels.append('High Confidence')
                colors.append('#2E8B57')  #Green

            if medium_confidence > 0:
                sizes.append(medium_confidence)
                labels.append('Medium Confidence')
                colors.append('#F18F01')  #Orange/Yellow

            if low_confidence > 0:
                sizes.append(low_confidence)
                labels.append('Low Confidence')
                colors.append(error_color)  #Red

            if unknown_faces > 0:
                sizes.append(unknown_faces)
                labels.append('Unidentified')
                colors.append(error_color)  #Red

            wedges, texts, autotexts = ax.pie(sizes,
                                              labels=labels,
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              startangle=90,
                                              textprops={'fontsize': 10, 'fontweight': 'bold'})

            #Add center circle for donut effect
            centre_circle = plt.Circle((0, 0), 0.50, fc='white')
            ax.add_artist(centre_circle)

            #Add center text
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
        print(f"Visualization saved to {os.path.join(output_dir, 'advanced_recognition_analytics.png')}")

    def run_recognition(self):
        """Main function to run the facial recognition with enhanced features"""
        print(" ADVANCED INTERACTIVE SOLVAY FACE RECOGNITION ")
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
        output_path = "/home/jwens/PycharmProjects/School-Projects/Graduate Projects - MS AI and Machine Learning/CSC580 Capstone/data/W1CT_Faces/solvay_advanced_identification.png"
        self.save_results(annotated_image, identified_faces, output_path)

        #Generate analytics
        name_counts, confidence_stats, df = self.generate_analytics(identified_faces)

        #Print enhanced summary
        print("\n ADVANCED IDENTIFICATION RESULTS ")
        print(f"Total faces detected: {len(identified_faces)}")
        print(f"Successfully identified: {len(identified_faces) - name_counts.get('Unknown', 0)}")
        print(
            f"Success rate: {((len(identified_faces) - name_counts.get('Unknown', 0)) / len(identified_faces)) * 100:.1f}%")

        print("\n IDENTIFIED SCIENTISTS ")
        for name, stats in confidence_stats.items():
            if name != "Unknown":
                print(f"  {name}: {stats['count']} detections, avg confidence: {stats['mean_confidence']:.3f}")

        print(f"\n UNKNOWN FACES ")
        print(f"  Unknown: {name_counts.get('Unknown', 0)} faces")

        #Launch interactive mode
        try:
            print("\n PROGRAM COMPLETE ")
            print("Advanced analysis complete! Check the output files for:")
            print("  - Interactive annotated image with clickable scientist names")
            print("  - Detailed JSON results")
            print("  - Analytics CSV data")
            print("  - Comprehensive visualization charts")
            print("  - Wikipedia-powered scientist summaries")
            print("\nLaunching interactive mode...")
            
            self.display_interactive_image(output_path, identified_faces)
        except Exception as e:
            print(f"Interactive display not available: {e}")
            print("Falling back to static image display...")
            try:
                annotated_image.show()
            except:
                print("Image display not available in this environment")

        return identified_faces, name_counts, confidence_stats
    
    def display_interactive_image(self, image_path: str, identified_faces: list):
        """Display the annotated image with OpenCV click functionality"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE ACTIVATED")
        print("="*60)
        print("Click on any scientist's face to get their HuggingFace Transformers summary!")
        print("Press 'q' or ESC to quit.")
        print("="*60)
        
        #Store face data for click detection
        self.identified_faces_data = identified_faces
        
        #Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return
            
        #Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #Store the image for the callback
        self.display_image = img_rgb.copy()
        
        #Get original image dimensions
        img_height, img_width = img_rgb.shape[:2]
        print(f"Original image size: {img_width} x {img_height}")
        
        #Calculate appropriate window size (scale down if too large for screen)
        max_width = 1400
        max_height = 900
        
        if img_width > max_width or img_height > max_height:
            #Scale down proportionally
            scale_factor = min(max_width / img_width, max_height / img_height)
            display_width = int(img_width * scale_factor)
            display_height = int(img_height * scale_factor)
        else:
            #Use original size if it fits
            display_width = img_width
            display_height = img_height
            scale_factor = 1.0
        
        print(f"Display window size: {display_width} x {display_height} (scale: {scale_factor:.2f})")
        
        #Create window and set it to the proper size immediately
        window_name = "Interactive Solvay Conference 1927 - Click on Scientists!"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        #Resize the image for display if needed
        if scale_factor != 1.0:
            display_img = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (display_width, display_height))
        else:
            display_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        #Store scaling info for coordinate conversion
        self.img_width = img_width
        self.img_height = img_height
        self.scale_factor = scale_factor
        
        print(f"Using {scale_factor:.2f} scaling factor for coordinate mapping")
        
        #Set mouse callback
        cv2.setMouseCallback(window_name, self.opencv_mouse_callback)
        
        print(f"\nOpenCV window '{window_name}' is now open!")
        print("Click anywhere on a scientist's face or name to get their HuggingFace Transformers summary.")
        print("Press 'q' or ESC to close the window and end the program.")
        
        #Display loop
        while True:
            cv2.imshow(window_name, display_img)
            
            #Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  #'q' or ESC key
                break
                
        #Clean up
        cv2.destroyAllWindows()
        print("\nInteractive session ended.")
        print("Note: For full interactive Wikipedia summaries, run:")
        print("python Interactive_Wikipedia_Tester.py")
    
    def opencv_mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback function - much more reliable than matplotlib"""
        if event == cv2.EVENT_LBUTTONDOWN:  #Left mouse button click
            
            #Convert display coordinates back to original image coordinates
            original_x = int(x / self.scale_factor)
            original_y = int(y / self.scale_factor)
            
            #Check if click is within original image bounds
            if original_x < 0 or original_x >= self.img_width or original_y < 0 or original_y >= self.img_height:
                print(f"Click outside image bounds! Original image size: {self.img_width} x {self.img_height}")
                return
            
            #Find the scientist that was clicked using original coordinates
            clicked_scientist = self.find_scientist_at_coordinates(original_x, original_y)
            
            if clicked_scientist:
                print(f"You clicked on: {clicked_scientist}")
                print("Fetching HuggingFace Transformers summary...")
                
                #Get AI summary
                summary_data = self.wikipedia_summarizer.get_scientist_summary(clicked_scientist)
                
                #Display the summary
                self.display_scientist_summary(summary_data)
            else:
                print("No face found at this location. Try clicking directly on any face box!")
    
    def find_scientist_at_coordinates(self, click_x: float, click_y: float) -> Optional[str]:
        """Find which scientist was clicked based on coordinates and return ground truth name"""
        
        #First, check if click is directly within any face box (no margin)
        for face_data in self.identified_faces_data:
            top, right, bottom, left = face_data['location']
            
            #Check if click is directly within the face box
            if (left <= click_x <= right and top <= click_y <= bottom):
                #Get ground truth name for this face location
                ground_truth_name = self.find_closest_manual_face([top, right, bottom, left])
                if ground_truth_name and ground_truth_name != "Unknown Position":
                    print(f"Direct hit on face box: {ground_truth_name}")
                    return ground_truth_name
                else:
                    print(f"Direct hit on face box: {face_data['name']}")
                    return face_data['name']
        
        #If no direct hit, check with margins but be more selective
        potential_matches = []
        
        for face_data in self.identified_faces_data:
            #Get face location [top, right, bottom, left]
            top, right, bottom, left = face_data['location']
            
            #Create clickable area with smaller margin to reduce overlap
            margin_x = 30  #Reduced horizontal margin
            margin_y = 50  #Vertical margin for labels
            clickable_left = left - margin_x
            clickable_right = right + margin_x  
            clickable_top = top - 80  #Space above for text labels
            clickable_bottom = bottom + margin_y
            
            #Check if click is within this scientist's area
            if (clickable_left <= click_x <= clickable_right and 
                clickable_top <= click_y <= clickable_bottom):
                
                #Get ground truth name for this face
                ground_truth_name = self.find_closest_manual_face([top, right, bottom, left])
                display_name = ground_truth_name if (ground_truth_name and ground_truth_name != "Unknown Position") else face_data['name']
                
                #Calculate distance from click to face center
                face_center_x = (left + right) / 2
                face_center_y = (top + bottom) / 2
                distance = ((click_x - face_center_x) ** 2 + (click_y - face_center_y) ** 2) ** 0.5
                
                #Also calculate if click is above or below face (for label clicks)
                is_above_face = click_y < top
                
                potential_matches.append((display_name, distance, face_data, is_above_face))
        
        #If multiple matches, use smarter logic
        if potential_matches:
            #If click is above faces (in label area), prefer the face whose horizontal center is closest
            above_face_matches = [m for m in potential_matches if m[3]]
            if above_face_matches:
                #For label clicks, use horizontal distance only
                best_match = min(above_face_matches, key=lambda m: abs(click_x - ((m[2]['location'][3] + m[2]['location'][1]) / 2)))
                return best_match[0]
            
            #Otherwise, return the closest match by euclidean distance
            potential_matches.sort(key=lambda x: x[1])
            closest_match = potential_matches[0]
            return closest_match[0]
        
        return None
    
    def display_scientist_summary(self, summary_data: Dict):
        """Display the scientist summary in a formatted way with text wrapping"""
        #Set text width for comfortable reading (80 characters is standard terminal width)
        text_width = 80
        
        print("\n" + "="*text_width)
        scientist_name = summary_data['scientist']
        print(f"SCIENTIST PROFILE: {scientist_name}")
        print("="*text_width)
        
        if summary_data['status'] == 'success':
            print(f"Wikipedia: {'Found' if summary_data['wikipedia_available'] else 'Not found'}")
            if summary_data['wikipedia_available']:
                print(f"Content Length: {summary_data['wikipedia_length']:,} characters")
            
            print(f"\nHUGGINGFACE TRANSFORMERS SUMMARY:")
            print("-" * text_width)
            
            #Wrap the summary text to avoid horizontal scrolling
            summary_text = summary_data['summary']
            wrapped_lines = textwrap.fill(summary_text, width=text_width)
            print(wrapped_lines)
            
            print("-" * text_width)
            
        else:
            error_message = summary_data.get('message', 'Unknown error')
            wrapped_error = textwrap.fill(f"Error: {error_message}", width=text_width)
            print(wrapped_error)
        
        print("="*text_width)
        print("Click on another scientist or press 'q' to close!")
        print("="*text_width + "\n")


def main():
    """Main function with enhanced configuration options"""
    print(" ADVANCED INTERACTIVE SOLVAY FACE RECOGNITION SYSTEM ")
    print("Initializing with interactive capabilities...")

    #Initialize with enhanced settings
    recognizer = EnhancedSolvayFaceRecognizer(tolerance=0.60, model='hog')
    results, name_counts, confidence_stats = recognizer.run_recognition()



if __name__ == "__main__":
    main()