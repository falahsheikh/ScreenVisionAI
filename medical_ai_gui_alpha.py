import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import openai
import threading
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tf_keras_vis
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ===========================================
# CONFIGURATION & CONSTANTS
# ===========================================

# Configuration
OPENAI_API_KEY = "openai-api-key"
openai.api_key = OPENAI_API_KEY

# Urgency level definitions
URGENCY_LEVELS = {
    "routine": {
        "label": "Routine",
        "description": "No immediate concern - Regular follow-up recommended",
        "color_hex": "#28a745",  # Green
        "color_rgb": (40, 167, 69),
        "color_reportlab": colors.HexColor('#28a745'),
        "priority": 1
    },
    "monitor": {
        "label": "Monitor",
        "description": "Worth monitoring - Schedule follow-up within reasonable timeframe",
        "color_hex": "#90c648",  # Yellow-Green
        "color_rgb": (144, 198, 72),
        "color_reportlab": colors.HexColor('#90c648'),
        "priority": 2
    },
    "attention": {
        "label": "Needs Attention",
        "description": "Requires clinical attention - Schedule appointment soon",
        "color_hex": "#ffc107",  # Yellow
        "color_rgb": (255, 193, 7),
        "color_reportlab": colors.HexColor('#ffc107'),
        "priority": 3
    },
    "urgent": {
        "label": "Urgent",
        "description": "Urgent care needed - Prompt medical evaluation required",
        "color_hex": "#ff9800",  # Orange
        "color_rgb": (255, 152, 0),
        "color_reportlab": colors.HexColor('#ff9800'),
        "priority": 4
    },
    "very_urgent": {
        "label": "Very Urgent",
        "description": "Very urgent - Immediate medical consultation required",
        "color_hex": "#ff5722",  # Deep Orange
        "color_rgb": (255, 87, 34),
        "color_reportlab": colors.HexColor('#ff5722'),
        "priority": 5
    },
    "emergency": {
        "label": "Emergency",
        "description": "Medical emergency - Seek immediate emergency care",
        "color_hex": "#dc3545",  # Red
        "color_rgb": (220, 53, 69),
        "color_reportlab": colors.HexColor('#dc3545'),
        "priority": 6
    }
}

# Model configurations with urgency mappings.
MODEL_CONFIGS = {
    "brain_mri_alzheimers": {
        "path": "./models/Early_Alzheimers_DenseNet121_Augmented.keras",
        "classes": ["CN", "EMCI", "LMCI"],
        "input_size": (224, 224),
        "description": "Brain MRI Alzheimer's detection",
        "keywords": ["alzheimer", "brain", "mri", "dementia", "cognitive", "emci", "lmci", "mild cognitive impairment"],
        "constraints": {
            "modality": ["MRI", "Brain MRI"],
            "view": ["Coronal"],
            "preprocessing": ["Skull Stripped", "Standard"]
        },
        "constraint_descriptions": {
            "modality": "Image modality type",
            "view": "Image view/slice type",
            "preprocessing": "Preprocessing applied"
        },
        "gradcam_layer": -1,
        "clinical_questions": [
            {"key": "age", "question": "Patient's age (in years):", "type": "number", "required": True},
            {"key": "symptoms", "question": "Primary symptoms (e.g., memory loss, confusion):", "type": "text", "required": True},
            {"key": "duration", "question": "Duration of symptoms (in months):", "type": "number", "required": False},
            {"key": "family_history", "question": "Family history of dementia (yes/no):", "type": "text", "required": False}
        ],
        # Urgency mapping for each classification
        "urgency_mapping": {
            "CN": "routine",  # Cognitively Normal - routine follow-up
            "EMCI": "monitor",  # Early MCI - monitor closely
            "LMCI": "attention"  # Late MCI - needs attention and intervention
        }
    },
    
    "skin_cancer_isic": {
        "path": "./models/Skin_Cancer_DenseNet201_Augmented.keras",
        "classes": [
            "actinic keratosis",
            "basal cell carcinoma",
            "dermatofibroma",
            "melanoma",
            "nevus",
            "pigmented benign keratosis",
            "seborrheic keratosis",
            "squamous cell carcinoma",
            "vascular lesion"
        ],
        "input_size": (75, 100),
        "description": "Skin cancer classification (ISIC dataset)",
        "keywords": ["skin", "cancer", "melanoma", "lesion", "dermatology", "mole"],
        "constraints": {
            "modality": ["Dermoscopy", "Clinical Photography"],
            "view": ["Close-up", "Surface"],
            "preprocessing": ["None", "Standard"]
        },
        "constraint_descriptions": {
            "modality": "Image capture type",
            "view": "Image perspective",
            "preprocessing": "Preprocessing applied"
        },
        "gradcam_layer": -1,
        "clinical_questions": [
            {"key": "age", "question": "Patient's age (in years):", "type": "number", "required": True},
            {"key": "lesion_duration", "question": "How long has this lesion been present (in months):", "type": "number", "required": True},
            {"key": "changes", "question": "Recent changes in size, color, or shape (yes/no):", "type": "text", "required": True},
            {"key": "symptoms", "question": "Associated symptoms (bleeding, itching, etc.):", "type": "text", "required": False},
            {"key": "sun_exposure", "question": "History of significant sun exposure (yes/no):", "type": "text", "required": False}
        ],
        # Urgency mapping for each skin lesion type
        "urgency_mapping": {
            "actinic keratosis": "attention",  # Pre-cancerous, needs treatment
            "basal cell carcinoma": "urgent",  # Cancer but slow growing
            "dermatofibroma": "routine",  # Benign
            "melanoma": "emergency",  # Aggressive cancer - immediate attention
            "nevus": "routine",  # Benign mole
            "pigmented benign keratosis": "routine",  # Benign
            "seborrheic keratosis": "routine",  # Benign
            "squamous cell carcinoma": "very_urgent",  # Cancer, can metastasize
            "vascular lesion": "monitor"  # Usually benign but monitor
        }
    }, 

    "eye_cataract_detection": {
        "path": "./models/Cataracts_VGG16_Augmented.keras",
        "classes": ["cataract", "normal"],
        "input_size": (224, 224),
        "description": "Eye cataract detection from fundus images",
        "keywords": ["cataract", "eye", "ophthalmology", "fundus", "lens", "vision", "clouding", "visual impairment"],
        "constraints": {
            "modality": ["Fundus Photography", "Slit-lamp Photography"],
            "view": ["Anterior", "Lens"],
            "preprocessing": ["Standard", "Color Normalized"]
        },
        "constraint_descriptions": {
            "modality": "Image capture type",
            "view": "Anatomical view",
            "preprocessing": "Preprocessing applied"
        },
        "gradcam_layer": -1,
        "clinical_questions": [
            {"key": "age", "question": "Patient's age (in years):", "type": "number", "required": True},
            {"key": "visual_symptoms", "question": "Visual symptoms (e.g., blurred vision, glare, difficulty seeing at night):", "type": "text", "required": True},
            {"key": "duration", "question": "Duration of symptoms (in months):", "type": "number", "required": True},
            {"key": "visual_acuity", "question": "Current visual acuity if available (e.g., 20/40):", "type": "text", "required": False},
            {"key": "previous_eye_surgery", "question": "History of eye surgery (yes/no):", "type": "text", "required": False},
            {"key": "diabetes", "question": "History of diabetes (yes/no):", "type": "text", "required": False}
        ],
        # Urgency mapping for each classification
        "urgency_mapping": {
            "cataract": "attention",  # Cataracts require ophthalmologic evaluation and potential surgical intervention
            "normal": "routine"  # Normal - routine eye care recommended
        }
    },

    "chest_xray_pneumonia": {
        "path": "./models/Pneumonia_ResNet50_best.keras",
        "classes": ["NORMAL", "PNEUMONIA"],
        "input_size": (224, 224),
        "description": "Chest X-ray pneumonia detection using ResNet50",
        "keywords": ["pneumonia", "chest", "xray", "x-ray", "lung", "respiratory", "infection", "bacterial", "viral"],
        "constraints": {
            "modality": ["Chest X-ray", "Radiography"],
            "view": ["PA", "AP", "Lateral"],
            "preprocessing": ["Standard", "DICOM"]
        },
        "constraint_descriptions": {
            "modality": "Imaging modality",
            "view": "X-ray projection view",
            "preprocessing": "Image preprocessing applied"
        },
        "gradcam_layer": -1,  # Last conv layer in ResNet50 base model
        "clinical_questions": [
            {"key": "age", "question": "Patient's age (in years):", "type": "number", "required": True},
            {"key": "symptoms", "question": "Primary symptoms (e.g., cough, fever, difficulty breathing):", "type": "text", "required": True},
            {"key": "duration", "question": "Duration of symptoms (in days):", "type": "number", "required": True},
            {"key": "fever", "question": "Fever present (yes/no):", "type": "text", "required": True},
            {"key": "temperature", "question": "Temperature if measured (°F):", "type": "number", "required": False},
            {"key": "oxygen_saturation", "question": "Oxygen saturation (SpO2 %):", "type": "number", "required": False},
            {"key": "comorbidities", "question": "Pre-existing conditions (e.g., COPD, asthma, diabetes):", "type": "text", "required": False}
        ],
        "urgency_mapping": {
            "NORMAL": "routine",  # Normal chest X-ray - routine follow-up
            "PNEUMONIA": "urgent"  # Pneumonia requires prompt treatment with antibiotics
        }
    }
}

# ===========================================
# THEME & STYLING
# ===========================================

class Win95Style:
    BG_GRAY = "#c0c0c0"
    DARK_BLUE = "#000080"
    LIGHT_BLUE = "#0000aa"
    LIGHT_GRAY = "#dfdfdf"
    DARK_GRAY = "#808080"
    DARKER_GRAY = "#404040"
    WHITE = "#ffffff"
    CREAM = "#fffbf0"
    BLACK = "#000000"
    BUTTON_HIGHLIGHT = "#ffffff"
    BUTTON_SHADOW = "#000000"
    ACTIVE_TITLE = "#000080"
    WINDOW_FRAME = "#c0c0c0"
    SCROLLBAR_THUMB = "#000080"
    
# ===========================================
# CLINICAL DATA MODEL
# ===========================================

class ClinicalData:
    """Stores clinical information for a single analysis session"""
    def __init__(self):
        self.timestamp = datetime.now()
        self.patient_info = {}
        self.image_info = None
        self.analysis_result = None
        self.explanation = None
        self.model_used = None
        self.heatmap_path = None
        self.urgency_level = None
        self.urgency_info = None
        
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "patient_info": self.patient_info,
            "image_info": self.image_info,
            "analysis_result": self.analysis_result,
            "explanation": self.explanation,
            "model_used": self.model_used,
            "urgency_level": self.urgency_level,
            "urgency_info": self.urgency_info
        }

# ===========================================
# ALGORITHM & MODEL COMPONENTS
# ===========================================

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        for name, config in MODEL_CONFIGS.items():
            try:
                if os.path.exists(config["path"]):
                    model = keras.models.load_model(config["path"])
                    self.models[name] = {
                        "model": model,
                        "config": config
                    }
                    print(f"Loaded model: {name}")
                else:
                    print(f"Model file not found: {config['path']}")
                    self.create_dummy_model(name, config)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                self.create_dummy_model(name, config)
    
    def create_dummy_model(self, name, config):
        try:
            model = keras.Sequential([
                keras.layers.Input(shape=(*config["input_size"], 3)),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(len(config["classes"]), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            self.models[name] = {
                "model": model,
                "config": config
            }
            print(f"Created dummy model for: {name}")
        except Exception as e:
            print(f"Failed to create dummy model for {name}: {e}")
    
    def find_suitable_model_with_details(self, query, constraint_responses):
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        constraint_mismatches = {}
        
        for model_name, model_data in self.models.items():
            keywords = model_data["config"]["keywords"]
            keyword_score = sum(1 for keyword in keywords if keyword in query_lower)
            
            constraints = model_data["config"]["constraints"]
            constraint_descriptions = model_data["config"]["constraint_descriptions"]
            mismatches = []
            
            for constraint_key, allowed_values in constraints.items():
                if not allowed_values:
                    continue
                
                user_value = constraint_responses.get(constraint_key, "")
                
                if not any(allowed.lower() == user_value.lower() for allowed in allowed_values):
                    mismatches.append({
                        "constraint": constraint_key,
                        "description": constraint_descriptions.get(constraint_key, constraint_key),
                        "user_selected": user_value,
                        "required": allowed_values,
                        "model": model_data["config"]["description"]
                    })
            
            constraint_mismatches[model_name] = {
                "description": model_data["config"]["description"],
                "mismatches": mismatches,
                "keyword_score": keyword_score
            }
            
            if len(mismatches) == 0 and keyword_score > best_score:
                best_score = keyword_score
                best_match = (model_name, model_data)
        
        return best_match, constraint_mismatches
    
    def find_suitable_model(self, query, constraint_responses):
        result, _ = self.find_suitable_model_with_details(query, constraint_responses)
        return result
    
    def check_constraints_match(self, model_name, user_responses):
        if model_name not in self.models:
            return False
        
        constraints = self.models[model_name]["config"]["constraints"]
        
        for constraint_key, allowed_values in constraints.items():
            if not allowed_values:
                continue
            
            user_value = user_responses.get(constraint_key, "")
            
            if not any(allowed.lower() == user_value.lower() for allowed in allowed_values):
                return False
        
        return True
    
    def get_all_constraint_options(self):
        all_constraints = {}
        
        for model_data in self.models.values():
            constraints = model_data["config"]["constraints"]
            descriptions = model_data["config"]["constraint_descriptions"]
            
            for constraint_key, allowed_values in constraints.items():
                if constraint_key not in all_constraints:
                    all_constraints[constraint_key] = {
                        "description": descriptions.get(constraint_key, constraint_key),
                        "options": set()
                    }
                
                all_constraints[constraint_key]["options"].update(allowed_values)
        
        for constraint_key in all_constraints:
            all_constraints[constraint_key]["options"] = sorted(
                list(all_constraints[constraint_key]["options"])
            )
        
        return all_constraints
    
    def get_models_for_constraints(self, constraint_responses):
        matching_models = []
        
        for model_name, model_data in self.models.items():
            if self.check_constraints_match(model_name, constraint_responses):
                matching_models.append({
                    "name": model_name,
                    "description": model_data["config"]["description"]
                })
        
        return matching_models
    
    def generate_detailed_error_message(self, query, constraint_responses):
        best_match, constraint_mismatches = self.find_suitable_model_with_details(query, constraint_responses)
        
        if not constraint_mismatches:
            return "No models available that match your query. Please rephrase your question."
        
        sorted_models = sorted(
            constraint_mismatches.items(),
            key=lambda x: x[1]["keyword_score"],
            reverse=True
        )
        
        has_keyword_match = any(info["keyword_score"] > 0 for _, info in sorted_models)
        
        if not has_keyword_match:
            error_msg = "I couldn't determine which analysis to perform based on your question.\n\n"
            error_msg += "Available analyses:\n"
            for model_name, info in sorted_models:
                error_msg += f"  - {info['description']}\n"
            error_msg += "\nPlease specify which type of analysis you would like to perform."
            return error_msg
        
        error_msg = "Cannot perform the requested analysis. Based on your question, here's what's needed:\n\n"
        
        for model_name, info in sorted_models:
            if info["keyword_score"] > 0:
                error_msg += f"For {info['description']}:\n"
                
                if len(info["mismatches"]) == 0:
                    error_msg += "  - All requirements met!\n"
                else:
                    error_msg += "  Requirements not met:\n"
                    for mismatch in info["mismatches"]:
                        error_msg += f"    * {mismatch['description']}: "
                        error_msg += f"You selected '{mismatch['user_selected']}', "
                        error_msg += f"but this model requires: {' or '.join(mismatch['required'])}\n"
                
                error_msg += "\n"
        
        best_model = sorted_models[0]
        if len(best_model[1]["mismatches"]) > 0:
            error_msg += "To proceed with the analysis:\n"
            error_msg += f"1. Change your image requirements to match {best_model[1]['description']}\n"
            error_msg += "2. Upload a new image that meets these requirements\n"
            error_msg += "3. Or ask a different question that matches your current image selections"
        
        return error_msg
    
    def preprocess_image(self, image, target_size):
        img = cv2.resize(image, (target_size[1], target_size[0]))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def find_conv_layer(self, model, layer_index=-1):
        """Find the best convolutional layer for Grad-CAM"""
        # If specific layer index is provided and valid, use it
        if layer_index != -1 and 0 <= layer_index < len(model.layers):
            layer = model.layers[layer_index]
            if any('conv' in layer.name.lower() for layer_type in ['conv', 'conv2d', 'convolution']):
                return layer_index
        
        # Otherwise, find the last convolutional layer
        conv_layers = []
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__.lower()
            if any(conv_type in layer_type for conv_type in ['conv2d', 'convolutional']):
                conv_layers.append(i)
        
        if conv_layers:
            return conv_layers[-1]  # Return last conv layer
        
        # If no conv layers found, try to find any feature extraction layer
        for i, layer in enumerate(model.layers):
            if any(name in layer.name.lower() for name in ['mixed', 'block', 'conv', 'features']):
                return i
        
        # Fallback: use the layer before the final dense layers
        for i in range(len(model.layers)-1, -1, -1):
            layer = model.layers[i]
            if 'dense' not in layer.name.lower() and 'flatten' not in layer.name.lower():
                return i
        
        return -1  # Use default if nothing found
    
    def generate_gradcam(self, model, image, pred_class, layer_index=-1):
        """Generate Grad-CAM++ heatmap with automatic layer detection"""
        try:
            # Find the best convolutional layer
            actual_layer_index = self.find_conv_layer(model, layer_index)
            print(f"Using layer index {actual_layer_index} for Grad-CAM: {model.layers[actual_layer_index].name}")
            
            replace2linear = ReplaceToLinear()
            gradcam_plusplus = GradcamPlusPlus(model, model_modifier=replace2linear, clone=True)
            
            # Generate heatmap
            cam = gradcam_plusplus(
                CategoricalScore(pred_class),
                image,
                penultimate_layer=actual_layer_index
            )
            
            heatmap = cam[0]
            
            # Ensure heatmap is 2D
            if heatmap.ndim > 2:
                heatmap = np.mean(heatmap, axis=-1)
            
            return heatmap
            
        except Exception as e:
            print(f"Grad-CAM++ error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: try with default layer
            try:
                print("Trying fallback Grad-CAM with default layer...")
                replace2linear = ReplaceToLinear()
                gradcam_plusplus = GradcamPlusPlus(model, model_modifier=replace2linear, clone=True)
                
                cam = gradcam_plusplus(
                    CategoricalScore(pred_class),
                    image,
                    penultimate_layer=-1  # Let tf-keras-vis choose
                )
                
                heatmap = cam[0]
                if heatmap.ndim > 2:
                    heatmap = np.mean(heatmap, axis=-1)
                
                return heatmap
                
            except Exception as e2:
                print(f"Fallback Grad-CAM also failed: {e2}")
                return None

    def create_overlay(self, original_image, heatmap, target_size):
        """Create heatmap overlay on original image with proper normalization"""
        if heatmap is None:
            return original_image
        
        try:
            # Resize original image to target size
            img_resized = cv2.resize(original_image, (target_size[1], target_size[0]))
            
            # Ensure heatmap is 2D and resize to match image
            if heatmap.ndim > 2:
                heatmap = np.mean(heatmap, axis=-1)
            
            heatmap_resized = cv2.resize(heatmap, (target_size[1], target_size[0]))
            
            # Normalize heatmap to 0-1 range
            heatmap_min = np.min(heatmap_resized)
            heatmap_max = np.max(heatmap_resized)
            
            if heatmap_max - heatmap_min > 1e-10:
                heatmap_normalized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
            else:
                heatmap_normalized = heatmap_resized
            
            # Apply colormap (DO NOT INVERT - high values should be red/hot)
            heatmap_uint8 = np.uint8(255 * heatmap_normalized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Convert heatmap to RGB if needed
            if heatmap_colored.shape[-1] == 3:
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay - areas with high activation will be red
            overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
            
            return overlay
            
        except Exception as e:
            print(f"Error creating overlay: {e}")
            import traceback
            traceback.print_exc()
            return original_image

    def predict_with_xai(self, model_name, image_array):
        """Make prediction and generate Grad-CAM++ XAI visualization with urgency assessment"""
        model_data = self.models[model_name]
        model = model_data["model"]
        config = model_data["config"]
        
        # Preprocess
        processed = self.preprocess_image(image_array, config["input_size"])
        
        # Predict
        predictions = model.predict(processed, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        predicted_class_name = config["classes"][pred_class]
        
        # Get urgency level for this classification
        urgency_mapping = config.get("urgency_mapping", {})
        urgency_level = urgency_mapping.get(predicted_class_name, "monitor")  # Default to monitor
        urgency_info = URGENCY_LEVELS.get(urgency_level, URGENCY_LEVELS["monitor"])
        
        # Get the layer index for this model's Grad-CAM
        gradcam_layer = config.get("gradcam_layer", -1)
        
        # Generate Grad-CAM with model-specific layer
        heatmap = self.generate_gradcam(model, processed, pred_class, gradcam_layer)
        
        # Create overlay
        overlay = self.create_overlay(image_array, heatmap, config["input_size"])
        
        result = {
            "class": predicted_class_name,
            "confidence": confidence,
            "all_probabilities": {config["classes"][i]: float(predictions[0][i]) 
                                 for i in range(len(config["classes"]))},
            "heatmap": overlay,
            "model_used": config["description"],
            "raw_image": image_array,
            "heatmap_only": heatmap,
            "urgency_level": urgency_level,
            "urgency_info": urgency_info
        }
        
        return result

# ===========================================
# DATA MODELS
# ===========================================

class ChatMessage:
    def __init__(self, role, content, image=None, image_info=None, heatmap=None, timestamp=None, clinical_data=None):
        self.role = role
        self.content = content
        self.image = image
        self.image_info = image_info
        self.heatmap = heatmap
        self.timestamp = timestamp or datetime.now()
        self.clinical_data = clinical_data

# ===========================================
# GUI COMPONENTS - CUSTOM WIDGETS
# ===========================================

class Win95Button(tk.Frame):
    def __init__(self, parent, text="Button", command=None, width=None, height=None, **kwargs):
        super().__init__(parent, bg=Win95Style.BG_GRAY, **kwargs)
        
        self.command = command
        self.is_pressed = False
        self.is_enabled = True
        
        if width:
            self.configure(width=width)
            self.pack_propagate(False)
        if height:
            self.configure(height=height)
        
        self.create_button(text)
        
    def create_button(self, text):
        self.button_container = tk.Frame(self, bg=Win95Style.BG_GRAY)
        self.button_container.pack(fill=tk.BOTH, expand=True)
        
        self.top_highlight = tk.Frame(
            self.button_container,
            bg=Win95Style.BUTTON_HIGHLIGHT,
            height=2
        )
        self.top_highlight.pack(side=tk.TOP, fill=tk.X)
        
        self.left_highlight = tk.Frame(
            self.button_container,
            bg=Win95Style.BUTTON_HIGHLIGHT,
            width=2
        )
        self.left_highlight.pack(side=tk.LEFT, fill=tk.Y)
        
        self.bottom_shadow = tk.Frame(
            self.button_container,
            bg=Win95Style.BUTTON_SHADOW,
            height=2
        )
        self.bottom_shadow.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.right_shadow = tk.Frame(
            self.button_container,
            bg=Win95Style.BUTTON_SHADOW,
            width=2
        )
        self.right_shadow.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.button_face = tk.Label(
            self.button_container,
            text=text,
            bg=Win95Style.BG_GRAY,
            fg=Win95Style.BLACK,
            font=("MS Sans Serif", 8, "bold"),
            cursor="hand2"
        )
        self.button_face.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.button_face.bind("<ButtonPress-1>", self.on_press)
        self.button_face.bind("<ButtonRelease-1>", self.on_release)
        self.button_face.bind("<Leave>", self.on_leave)
        
    def on_press(self, event):
        if not self.is_enabled:
            return
        self.is_pressed = True
        self.top_highlight.configure(bg=Win95Style.BUTTON_SHADOW)
        self.left_highlight.configure(bg=Win95Style.BUTTON_SHADOW)
        self.bottom_shadow.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.right_shadow.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.button_face.configure(padx=5, pady=5)
        
    def on_release(self, event):
        if self.is_pressed and self.is_enabled:
            self.reset_appearance()
            if self.command:
                self.command()
        
    def on_leave(self, event):
        if self.is_pressed:
            self.reset_appearance()
    
    def reset_appearance(self):
        self.is_pressed = False
        self.top_highlight.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.left_highlight.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.bottom_shadow.configure(bg=Win95Style.BUTTON_SHADOW)
        self.right_shadow.configure(bg=Win95Style.BUTTON_SHADOW)
        self.button_face.configure(padx=4, pady=4)
    
    def config_state(self, state):
        if state == tk.DISABLED:
            self.is_enabled = False
            self.button_face.configure(fg=Win95Style.DARK_GRAY, cursor="")
        else:
            self.is_enabled = True
            self.button_face.configure(fg=Win95Style.BLACK, cursor="hand2")

class Win95Scrollbar(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=18, highlightthickness=0, **kwargs)
        self.configure(bg=Win95Style.BG_GRAY)
        
        self.thumb_color = Win95Style.SCROLLBAR_THUMB
        self.track_color = Win95Style.WHITE
        self.border_color = Win95Style.DARK_GRAY
        
        self.thumb_pos = 0.0
        self.thumb_size = 0.3
        self.dragging = False
        self.drag_start_y = 0
        
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        
        self.command = None
        self.draw()
    
    def set_command(self, command):
        self.command = command
    
    def set(self, first, last):
        self.thumb_pos = float(first)
        self.thumb_size = float(last) - float(first)
        self.draw()
    
    def draw(self):
        self.delete("all")
        width = self.winfo_width() or 18
        height = self.winfo_height() or 100
        
        self.create_rectangle(1, 1, width-1, height-1, 
                            fill=self.track_color, outline=self.border_color)
        
        thumb_height = max(20, int(height * self.thumb_size))
        thumb_y = int((height - thumb_height) * (self.thumb_pos / (1 - self.thumb_size)) if self.thumb_size < 1 else 0)
        
        self.create_rectangle(2, thumb_y+2, width-2, thumb_y+thumb_height-2,
                            fill=self.thumb_color, outline="")
        
        self.create_line(2, thumb_y+2, width-2, thumb_y+2, 
                        fill=Win95Style.LIGHT_GRAY, width=2)
        self.create_line(2, thumb_y+2, 2, thumb_y+thumb_height-2,
                        fill=Win95Style.LIGHT_GRAY, width=2)
        
        self.create_line(2, thumb_y+thumb_height-2, width-2, thumb_y+thumb_height-2,
                        fill=Win95Style.BLACK, width=2)
        self.create_line(width-2, thumb_y+2, width-2, thumb_y+thumb_height-2,
                        fill=Win95Style.BLACK, width=2)
        
        self.thumb_y = thumb_y
        self.thumb_height = thumb_height
    
    def on_click(self, event):
        if self.thumb_y <= event.y <= self.thumb_y + self.thumb_height:
            self.dragging = True
            self.drag_start_y = event.y - self.thumb_y
        else:
            height = self.winfo_height()
            ratio = event.y / height
            if self.command:
                self.command("moveto", ratio)
    
    def on_drag(self, event):
        if self.dragging and self.command:
            height = self.winfo_height()
            new_y = event.y - self.drag_start_y
            ratio = new_y / (height - self.thumb_height)
            ratio = max(0, min(1, ratio))
            self.command("moveto", ratio)
    
    def on_release(self, event):
        self.dragging = False

# ===========================================
# GUI COMPONENTS - MAIN APPLICATION
# ===========================================

class MedicalAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TriageVision AI")
        self.root.geometry("1000x700")
        self.root.configure(bg=Win95Style.WINDOW_FRAME)
        
        # Initialize components
        self.model_manager = ModelManager()
        self.conversation_history = []
        self.current_image = None
        self.current_image_path = None
        self.current_image_thumbnail = None
        self.constraint_responses = {}
        self.image_references = []
        self.constraint_widgets = {}
        
        # Clinical triaging
        self.pending_questions = []
        self.current_question_index = 0
        self.clinical_responses = {}
        self.awaiting_clinical_response = False
        self.selected_model_for_triage = None
        
        # Clinical data storage for PDF generation
        self.clinical_sessions = []
        self.current_clinical_data = None
        
        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        outer_frame = tk.Frame(
            self.root, 
            bg=Win95Style.BUTTON_HIGHLIGHT,
            relief=tk.RAISED,
            bd=2
        )
        outer_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        inner_border = tk.Frame(
            outer_frame,
            bg=Win95Style.DARK_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        inner_border.pack(fill=tk.BOTH, expand=True)
        
        title_frame = tk.Frame(
            inner_border, 
            bg=Win95Style.ACTIVE_TITLE, 
            height=28
        )
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)
        
        title_container = tk.Frame(title_frame, bg=Win95Style.ACTIVE_TITLE)
        title_container.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        
        title_label = tk.Label(
            title_container,
            text="TriageVision AI with Urgency Classification",
            bg=Win95Style.ACTIVE_TITLE,
            fg=Win95Style.WHITE,
            font=("MS Sans Serif", 9, "bold"),
            anchor="w"
        )
        title_label.pack(side=tk.LEFT)
        
        main_container = tk.Frame(inner_border, bg=Win95Style.BG_GRAY)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        self.setup_chat_area(main_container)
        self.setup_input_area(main_container)
        
        self.add_message("assistant", 
            "Welcome to TriageVision AI with Clinical Triaging and Urgency Classification!\n\n"
            "I can help you with:\n"
            "  • General medical questions and information\n"
            "  • Analysis of medical images with explainable AI\n"
            "  • Clinical triaging with relevant patient questions\n"
            "  • Urgency assessment with color-coded classifications\n"
            "  • Generation of detailed clinical analysis reports\n\n"
            "Each diagnosis is assigned an urgency level from Routine (green) to Emergency (red).\n"
            "Ask me anything or upload a medical image for analysis!")
    
    def setup_chat_area(self, parent):
        chat_outer = tk.Frame(parent, bg=Win95Style.BG_GRAY)
        chat_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8, side=tk.TOP)
        
        chat_frame = tk.Frame(
            chat_outer,
            bg=Win95Style.DARKER_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        chat_inner = tk.Frame(chat_frame, bg=Win95Style.WHITE, bd=0)
        chat_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        self.chat_canvas = tk.Canvas(
            chat_inner,
            bg=Win95Style.WHITE,
            highlightthickness=0,
            bd=0
        )
        
        self.custom_scrollbar = Win95Scrollbar(chat_inner)
        self.custom_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.custom_scrollbar.set_command(self.on_scrollbar)
        
        self.chat_canvas.configure(yscrollcommand=self.custom_scrollbar.set)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.chat_display = tk.Frame(self.chat_canvas, bg=Win95Style.WHITE)
        self.chat_window = self.chat_canvas.create_window(
            (0, 0),
            window=self.chat_display,
            anchor="nw",
            tags="chat_window"
        )
        
        self.chat_display.bind("<Configure>", self.on_frame_configure)
        self.chat_canvas.bind("<Configure>", self.on_canvas_configure)
        
        self.chat_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.chat_canvas.bind("<Button-4>", self.on_mousewheel)
        self.chat_canvas.bind("<Button-5>", self.on_mousewheel)
        
        self.bind_mousewheel_recursively(self.chat_display)

    def on_scrollbar(self, *args):
        self.chat_canvas.yview(*args)
    
    def bind_mousewheel_recursively(self, widget):
        widget.bind("<MouseWheel>", self.on_mousewheel)
        widget.bind("<Button-4>", self.on_mousewheel)
        widget.bind("<Button-5>", self.on_mousewheel)
        for child in widget.winfo_children():
            self.bind_mousewheel_recursively(child)
    
    def on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.chat_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.chat_canvas.yview_scroll(1, "units")
        return "break"
    
    def setup_input_area(self, parent):
        input_outer = tk.Frame(parent, bg=Win95Style.BG_GRAY)
        input_outer.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 8))
        
        input_frame = tk.Frame(
            input_outer,
            bg=Win95Style.DARKER_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        input_inner = tk.Frame(input_frame, bg=Win95Style.WHITE, bd=0)
        input_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        input_container = tk.Frame(input_inner, bg=Win95Style.WHITE)
        input_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        
        self.image_preview_frame = tk.Frame(input_container, bg=Win95Style.WHITE)
        self.constraint_frame = tk.Frame(input_container, bg=Win95Style.WHITE)
        
        text_container = tk.Frame(input_container, bg=Win95Style.WHITE)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.input_text = tk.Text(
            text_container,
            height=3,
            font=("MS Sans Serif", 9),
            bg=Win95Style.WHITE,
            fg=Win95Style.BLACK,
            wrap=tk.WORD,
            relief=tk.FLAT,
            insertbackground=Win95Style.BLACK,
            selectbackground=Win95Style.DARK_BLUE,
            selectforeground=Win95Style.WHITE
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.input_text.bind("<Return>", self.on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        
        buttons_container = tk.Frame(text_container, bg=Win95Style.WHITE)
        buttons_container.pack(side=tk.RIGHT)
        
        self.upload_btn = Win95Button(
            buttons_container,
            text="Upload Image",
            command=self.upload_image,
            width=100,
            height=32
        )
        self.upload_btn.pack(pady=(0, 6))
        
        self.download_btn = Win95Button(
            buttons_container,
            text="Download PDF",
            command=self.download_clinical_pdf,
            width=100,
            height=32
        )
        self.download_btn.pack(pady=(0, 6))
        
        self.send_btn = Win95Button(
            buttons_container,
            text="Send",
            command=self.send_message,
            width=100,
            height=32
        )
        self.send_btn.pack()
    
    def on_frame_configure(self, event=None):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        self.custom_scrollbar.draw()
    
    def on_canvas_configure(self, event):
        canvas_width = event.width
        self.chat_canvas.itemconfig(self.chat_window, width=canvas_width)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                self.show_error("Failed to load image. Please try another file.")
                return
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image
            self.current_image_path = file_path
            
            self.create_image_preview(image, os.path.basename(file_path))
            self.show_constraint_inputs()
    
    def create_image_preview(self, image, filename):
        for widget in self.image_preview_frame.winfo_children():
            widget.destroy()
        
        self.image_preview_frame.pack(fill=tk.X, pady=(0, 6), side=tk.TOP, anchor='n')
        
        thumbnail = self.resize_for_display(image, max_width=60)
        img_pil = Image.fromarray(thumbnail)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.current_image_thumbnail = img_tk
        
        preview_container = tk.Frame(
            self.image_preview_frame,
            bg=Win95Style.DARKER_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        preview_container.pack(fill=tk.X)
        
        preview_inner = tk.Frame(preview_container, bg=Win95Style.CREAM)
        preview_inner.pack(fill=tk.X, padx=1, pady=1)
        
        content_frame = tk.Frame(preview_inner, bg=Win95Style.CREAM)
        content_frame.pack(fill=tk.X, padx=6, pady=6)
        
        img_label = tk.Label(content_frame, image=img_tk, bg=Win95Style.CREAM)
        img_label.pack(side=tk.LEFT, padx=(0, 8))
        
        text_frame = tk.Frame(content_frame, bg=Win95Style.CREAM)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        filename_label = tk.Label(
            text_frame,
            text=filename,
            bg=Win95Style.CREAM,
            fg=Win95Style.BLACK,
            font=("MS Sans Serif", 9, "bold"),
            anchor="w"
        )
        filename_label.pack(fill=tk.X)
        
        status_label = tk.Label(
            text_frame,
            text="Image uploaded - Complete requirements below",
            bg=Win95Style.CREAM,
            fg=Win95Style.DARK_BLUE,
            font=("MS Sans Serif", 8),
            anchor="w"
        )
        status_label.pack(fill=tk.X)
        
        remove_btn = Win95Button(
            content_frame,
            text="Remove",
            command=self.remove_image,
            width=70,
            height=50
        )
        remove_btn.pack(side=tk.RIGHT)
    
    def show_constraint_inputs(self):
        for widget in self.constraint_frame.winfo_children():
            widget.destroy()
        
        all_constraints = self.model_manager.get_all_constraint_options()
        
        if not all_constraints:
            return
        
        self.constraint_frame.pack(fill=tk.X, pady=(0, 6), side=tk.TOP, anchor='n')
        
        constraint_container = tk.Frame(
            self.constraint_frame,
            bg=Win95Style.DARKER_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        constraint_container.pack(fill=tk.X)
        
        constraint_inner = tk.Frame(constraint_container, bg=Win95Style.LIGHT_GRAY)
        constraint_inner.pack(fill=tk.X, padx=1, pady=1)
        
        title_label = tk.Label(
            constraint_inner,
            text="Image Requirements (select appropriate options):",
            bg=Win95Style.LIGHT_GRAY,
            fg=Win95Style.BLACK,
            font=("MS Sans Serif", 9, "bold"),
            anchor="w"
        )
        title_label.pack(fill=tk.X, padx=6, pady=(6, 4))
        
        self.constraint_widgets = {}
        
        for constraint_key, constraint_data in all_constraints.items():
            row_frame = tk.Frame(constraint_inner, bg=Win95Style.LIGHT_GRAY)
            row_frame.pack(fill=tk.X, padx=6, pady=3)
            
            label_text = constraint_data["description"]
            label = tk.Label(
                row_frame,
                text=label_text + ":",
                bg=Win95Style.LIGHT_GRAY,
                fg=Win95Style.BLACK,
                font=("MS Sans Serif", 9),
                width=20,
                anchor="w"
            )
            label.pack(side=tk.LEFT, padx=(0, 8))
            
            var = tk.StringVar()
            options = constraint_data["options"]
            dropdown = ttk.Combobox(
                row_frame,
                textvariable=var,
                values=options,
                state="readonly",
                font=("MS Sans Serif", 9),
                width=25
            )
            dropdown.pack(side=tk.LEFT)
            dropdown.set(options[0] if options else "")
            
            self.constraint_widgets[constraint_key] = var
        
        self.constraint_frame.update_idletasks()
    
    def remove_image(self):
        self.current_image = None
        self.current_image_path = None
        self.current_image_thumbnail = None
        self.constraint_responses = {}
        
        self.image_preview_frame.pack_forget()
        self.constraint_frame.pack_forget()
        
        for widget in self.image_preview_frame.winfo_children():
            widget.destroy()
        for widget in self.constraint_frame.winfo_children():
            widget.destroy()
    
    def on_enter(self, event):
        if not event.state & 0x1:
            self.send_message()
            return "break"

    def send_message(self):
        message = self.input_text.get("1.0", tk.END).strip()
        
        if not message:
            return
        
        self.input_text.delete("1.0", tk.END)
        
        if self.awaiting_clinical_response:
            self.process_clinical_response(message)
            return
        
        sent_image = None
        sent_image_info = None
        if self.current_image is not None:
            if hasattr(self, 'constraint_widgets') and self.constraint_widgets:
                sent_image = self.current_image.copy()
                sent_image_info = {
                    "filename": os.path.basename(self.current_image_path),
                    "constraints": {k: v.get() for k, v in self.constraint_widgets.items()}
                }
            else:
                sent_image = self.current_image.copy()
                sent_image_info = {
                    "filename": os.path.basename(self.current_image_path),
                    "constraints": {}
                }
        
        self.add_message_with_image("user", message, sent_image, sent_image_info)
        
        if self.current_image is not None:
            image_to_process = self.current_image
            constraints_to_process = {}
            if hasattr(self, 'constraint_widgets') and self.constraint_widgets:
                constraints_to_process = {k: v.get() for k, v in self.constraint_widgets.items()}
            
            self.remove_image()
            self.set_input_state(False)
            
            thread = threading.Thread(
                target=self.process_message_with_image, 
                args=(message, image_to_process, constraints_to_process)
            )
            thread.daemon = True
            thread.start()
        else:
            self.set_input_state(False)
            
            thread = threading.Thread(target=self.process_chat_message, args=(message,))
            thread.daemon = True
            thread.start()
    
    def process_message_with_image(self, message, image, constraints):
        try:
            best_match, constraint_mismatches = self.model_manager.find_suitable_model_with_details(
                message, constraints
            )
            
            if best_match is None or best_match[0] is None:
                error_msg = self.model_manager.generate_detailed_error_message(message, constraints)
                self.root.after(0, self.show_error, error_msg)
                return
            
            model_name, model_data = best_match
            
            self.current_clinical_data = ClinicalData()
            self.current_clinical_data.image_info = {
                "constraints": constraints,
                "description": message
            }
            self.current_clinical_data.model_used = model_data['config']['description']
            
            clinical_questions = model_data['config'].get('clinical_questions', [])
            
            if clinical_questions:
                self.selected_model_for_triage = model_name
                self.pending_questions = clinical_questions
                self.current_question_index = 0
                self.clinical_responses = {}
                
                self.pending_image_data = {
                    "model_name": model_name,
                    "message": message,
                    "image": image
                }
                
                self.root.after(0, self.ask_next_clinical_question)
            else:
                self.process_image_analysis(model_name, message, image)
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in process_message_with_image: {error_details}")
            self.root.after(0, self.show_error, f"An error occurred during processing: {str(e)}")
        finally:
            self.root.after(0, lambda: self.set_input_state(True))
    
    def ask_next_clinical_question(self):
        if self.current_question_index < len(self.pending_questions):
            question_data = self.pending_questions[self.current_question_index]
            required_text = " (Required)" if question_data.get("required", False) else " (Optional - type 'skip' to skip)"
            
            question_text = f"Clinical Information Request:\n\n{question_data['question']}{required_text}\n\nYou may also type 'cancel' to skip all remaining questions and proceed with analysis."
            
            self.add_message("assistant", question_text)
            self.awaiting_clinical_response = True
            self.set_input_state(True)
        else:
            self.awaiting_clinical_response = False
            
            self.current_clinical_data.patient_info = self.clinical_responses.copy()
            
            summary = "Clinical Information Summary:\n\n"
            for key, value in self.clinical_responses.items():
                summary += f"• {key.replace('_', ' ').title()}: {value}\n"
            summary += "\nProceeding with image analysis..."
            
            self.add_message("assistant", summary)
            
            self.set_input_state(False)
            thread = threading.Thread(
                target=self.process_image_analysis,
                args=(
                    self.pending_image_data["model_name"],
                    self.pending_image_data["message"],
                    self.pending_image_data["image"]
                )
            )
            thread.daemon = True
            thread.start()
    
    def process_clinical_response(self, response):
        response_lower = response.lower().strip()
        
        if response_lower == 'cancel':
            self.add_message("assistant", "Clinical triaging cancelled. Proceeding with image analysis using available information...")
            
            self.current_clinical_data.patient_info = self.clinical_responses.copy()
            
            self.awaiting_clinical_response = False
            self.set_input_state(False)
            
            thread = threading.Thread(
                target=self.process_image_analysis,
                args=(
                    self.pending_image_data["model_name"],
                    self.pending_image_data["message"],
                    self.pending_image_data["image"]
                )
            )
            thread.daemon = True
            thread.start()
            return
        
        question_data = self.pending_questions[self.current_question_index]
        question_key = question_data["key"]
        is_required = question_data.get("required", False)
        
        if response_lower == 'skip':
            if is_required:
                self.add_message("assistant", "This question is required. Please provide an answer or type 'cancel' to skip all questions.")
                return
            else:
                self.current_question_index += 1
                self.ask_next_clinical_question()
                return
        
        if not response.strip():
            if is_required:
                self.add_message("assistant", "This question is required. Please provide an answer or type 'cancel' to skip all questions.")
                return
            else:
                self.current_question_index += 1
                self.ask_next_clinical_question()
                return
        
        self.clinical_responses[question_key] = response
        
        self.current_question_index += 1
        self.ask_next_clinical_question()
    
    def process_image_analysis(self, model_name, description, image):
        try:
            model_data = self.model_manager.models[model_name]
            
            self.root.after(0, self.add_message, "assistant",
                f"Analyzing with {model_data['config']['description']}...\n"
                "Running deep learning model and generating explainable AI visualizations...")
            
            result = self.model_manager.predict_with_xai(model_name, image)
            
            if self.current_clinical_data:
                self.current_clinical_data.analysis_result = result
                self.current_clinical_data.urgency_level = result['urgency_level']
                self.current_clinical_data.urgency_info = result['urgency_info']
            
            explanation = self.get_detailed_gpt_explanation(result, description, self.clinical_responses)
            
            if self.current_clinical_data:
                self.current_clinical_data.explanation = explanation
                self.clinical_sessions.append(self.current_clinical_data)
            
            self.root.after(0, self.display_analysis_results, result, explanation)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in process_image_analysis: {error_details}")
            self.root.after(0, self.show_error, f"Analysis error: {str(e)}")
        finally:
            self.root.after(0, lambda: self.set_input_state(True))
    
    def get_detailed_gpt_explanation(self, result, context, clinical_data):
        """Get detailed narrative explanation with urgency information"""
        
        clinical_context = ""
        if clinical_data:
            clinical_context = "\n\nClinical Information:\n"
            for key, value in clinical_data.items():
                clinical_context += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        urgency_info = result['urgency_info']
        urgency_context = f"\n\nUrgency Assessment:\nLevel: {urgency_info['label']}\nDescription: {urgency_info['description']}"
        
        prompt = f"""As a TriageVision AI, provide a detailed, narrative explanation of this diagnostic analysis. Write in flowing paragraphs, NOT numbered lists or bullet points.

Model: {result['model_used']}
Primary Diagnosis: {result['class']}
Confidence Level: {result['confidence']*100:.1f}%
All Classification Probabilities: {result['all_probabilities']}
Clinical Context: {context}{clinical_context}{urgency_context}

Please provide a comprehensive narrative explanation that includes:

First, explain what this diagnosis means in clear medical terms, describing the condition and its characteristics. Then discuss the confidence level and what it indicates about the certainty of this diagnosis, considering the probability distribution across all possible classifications.

Next, describe the clinical significance of this finding, including what it means for the patient and any important considerations. Explain why certain regions in the image (highlighted by the heat map) were particularly important for reaching this diagnosis.

CRITICALLY IMPORTANT: Explain the urgency level ({urgency_info['label']}) and why this classification warrants this level of urgency. Discuss the timeframe for follow-up or intervention and any immediate actions that should be taken.

Finally, provide detailed recommendations for next steps, including what additional tests or evaluations might be warranted, when follow-up should occur, and what clinical actions should be considered based on this finding{' and the provided clinical information' if clinical_data else ''}.

Write in a professional medical tone using complete paragraphs. Do not use numbered lists, bullet points, or emojis. Focus on creating a flowing, narrative explanation that a healthcare professional would appreciate."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            fallback = f"Diagnostic Analysis Results:\n\n"
            fallback += f"The analysis has identified {result['class']} with a confidence level of {result['confidence']*100:.1f}%. "
            fallback += f"This assessment is based on the deep learning model's evaluation of the provided medical image. "
            
            fallback += f"\n\nUrgency Assessment: This finding has been classified as {urgency_info['label']}. {urgency_info['description']} "
            
            if clinical_data:
                fallback += f"\n\nClinical information provided includes: "
                fallback += ", ".join([f"{k.replace('_', ' ')}: {v}" for k, v in clinical_data.items()])
                fallback += ". "
            
            fallback += f"\n\nThe explainable AI heat map visualization highlights the regions of the image that were most influential in reaching this diagnosis. "
            fallback += f"Healthcare professionals should review these findings in the context of the complete clinical picture and take appropriate action based on the urgency level.\n\n"
            fallback += f"(Note: Unable to generate detailed AI explanation due to API error: {str(e)})"
            
            return fallback
    
    def process_chat_message(self, message):
        try:
            messages = [
                {"role": "system", "content": 
                 "You are a helpful TriageVision AI. Provide accurate, detailed, "
                 "professional medical information in narrative form with flowing paragraphs. "
                 "Do not use numbered lists or bullet points unless specifically requested. "
                 "Always remind users to consult healthcare professionals for medical advice. "
                 "Do not use emojis or special symbols. Write in complete, well-structured paragraphs."}
            ]
            
            for msg in self.conversation_history[-10:]:
                if msg.image is None:
                    messages.append({"role": msg.role, "content": msg.content})
            
            messages.append({"role": "user", "content": message})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content
            self.root.after(0, self.add_message, "assistant", reply)
            
        except Exception as e:
            self.root.after(0, self.show_error, 
                f"ChatGPT Error: {str(e)}\n\nPlease check your API key and try again.")
        finally:
            self.root.after(0, lambda: self.set_input_state(True))
    
    def add_message_with_image(self, role, content, image, image_info):
        msg = ChatMessage(role, content, image=image, image_info=image_info)
        self.conversation_history.append(msg)
        
        msg_frame = tk.Frame(self.chat_display, bg=Win95Style.WHITE, padx=10, pady=5)
        msg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        role_color = Win95Style.DARK_BLUE if role == "user" else "#008000"
        role_text = "You" if role == "user" else "Medical AI"
        
        header = tk.Label(
            msg_frame,
            text=f"{role_text} - {msg.timestamp.strftime('%H:%M')}",
            bg=Win95Style.WHITE,
            fg=role_color,
            font=("MS Sans Serif", 8, "bold"),
            anchor="w"
        )
        header.pack(fill=tk.X)
        
        if image is not None and image_info is not None:
            image_container = tk.Frame(msg_frame, bg=Win95Style.WHITE)
            image_container.pack(fill=tk.X, pady=(2, 4))
            
            image_box = tk.Frame(
                image_container,
                bg=Win95Style.DARKER_GRAY,
                relief=tk.SUNKEN,
                bd=2
            )
            image_box.pack(fill=tk.X)
            
            image_inner = tk.Frame(image_box, bg=Win95Style.CREAM)
            image_inner.pack(fill=tk.X, padx=1, pady=1)
            
            image_content = tk.Frame(image_inner, bg=Win95Style.CREAM)
            image_content.pack(fill=tk.X, padx=6, pady=6)
            
            thumbnail = self.resize_for_display(image, max_width=60)
            img_pil = Image.fromarray(thumbnail)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_references.append(img_tk)
            
            thumb_label = tk.Label(image_content, image=img_tk, bg=Win95Style.CREAM)
            thumb_label.pack(side=tk.LEFT, padx=(0, 8))
            
            info_frame = tk.Frame(image_content, bg=Win95Style.CREAM)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            filename_label = tk.Label(
                info_frame,
                text=image_info["filename"],
                bg=Win95Style.CREAM,
                fg=Win95Style.BLACK,
                font=("MS Sans Serif", 9, "bold"),
                anchor="w"
            )
            filename_label.pack(fill=tk.X)
            
            constraints_text = " | ".join([f"{k}: {v}" for k, v in image_info["constraints"].items()])
            constraints_label = tk.Label(
                info_frame,
                text=constraints_text,
                bg=Win95Style.CREAM,
                fg=Win95Style.DARK_GRAY,
                font=("MS Sans Serif", 8),
                anchor="w"
            )
            constraints_label.pack(fill=tk.X)
        
        content_outer = tk.Frame(msg_frame, bg=Win95Style.DARKER_GRAY, relief=tk.SUNKEN, bd=2)
        content_outer.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        
        msg_bg = Win95Style.CREAM if role == "user" else Win95Style.LIGHT_GRAY
        
        content_frame = tk.Frame(content_outer, bg=msg_bg)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        content_label = tk.Label(
            content_frame,
            text=content,
            bg=msg_bg,
            fg=Win95Style.BLACK,
            font=("MS Sans Serif", 9),
            justify=tk.LEFT,
            wraplength=750,
            anchor="w"
        )
        content_label.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.bind_mousewheel_recursively(msg_frame)
        self.root.after(100, self.scroll_to_bottom)
    
    def display_analysis_results(self, result, explanation):
        """Display analysis results with urgency-based color coding"""
        msg_frame = tk.Frame(self.chat_display, bg=Win95Style.WHITE, padx=10, pady=5)
        msg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Get urgency color
        urgency_info = result['urgency_info']
        urgency_color = urgency_info['color_hex']
        
        # Header
        header = tk.Label(
            msg_frame,
            text=f"Medical AI - {datetime.now().strftime('%H:%M')}",
            bg=Win95Style.WHITE,
            fg="#008000",
            font=("MS Sans Serif", 8, "bold"),
            anchor="w"
        )
        header.pack(fill=tk.X)
        
        # Urgency banner
        urgency_banner = tk.Frame(msg_frame, bg=urgency_color, height=40)
        urgency_banner.pack(fill=tk.X, pady=(2, 4))
        urgency_banner.pack_propagate(False)
        
        urgency_label = tk.Label(
            urgency_banner,
            text=f"⚠ URGENCY LEVEL: {urgency_info['label'].upper()} ⚠",
            bg=urgency_color,
            fg=Win95Style.WHITE,
            font=("MS Sans Serif", 11, "bold")
        )
        urgency_label.pack(expand=True)
        
        urgency_desc = tk.Label(
            msg_frame,
            text=urgency_info['description'],
            bg=Win95Style.WHITE,
            fg=urgency_color,
            font=("MS Sans Serif", 9, "bold"),
            wraplength=750,
            justify=tk.CENTER
        )
        urgency_desc.pack(fill=tk.X, pady=(0, 4))
        
        # Main content frame with urgency-colored border
        content_outer = tk.Frame(msg_frame, bg=urgency_color, relief=tk.RAISED, bd=3)
        content_outer.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        
        content_frame = tk.Frame(content_outer, bg=Win95Style.LIGHT_GRAY)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Text explanation
        text_label = tk.Label(
            content_frame,
            text=f"ANALYSIS COMPLETE\n\n{explanation}",
            bg=Win95Style.LIGHT_GRAY,
            fg=Win95Style.BLACK,
            font=("MS Sans Serif", 9),
            justify=tk.LEFT,
            wraplength=750,
            anchor="w"
        )
        text_label.pack(fill=tk.X, padx=10, pady=10)
        
        # Visual results container
        visual_container = tk.Frame(content_frame, bg=Win95Style.LIGHT_GRAY)
        visual_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Heatmap section
        if result['heatmap'] is not None:
            heatmap_frame = tk.Frame(visual_container, bg=Win95Style.LIGHT_GRAY)
            heatmap_frame.pack(side=tk.LEFT, padx=(0, 10))
            
            heatmap_label = tk.Label(
                heatmap_frame,
                text="Explainable AI Heat Map",
                bg=Win95Style.LIGHT_GRAY,
                fg=Win95Style.DARK_BLUE,
                font=("MS Sans Serif", 8, "bold")
            )
            heatmap_label.pack(pady=(0, 4))
            
            display_heatmap = self.resize_for_display(result['heatmap'], max_width=300)
            img_pil = Image.fromarray(display_heatmap)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_references.append(img_tk)
            
            heatmap_img = tk.Label(heatmap_frame, image=img_tk, bg=Win95Style.LIGHT_GRAY, relief=tk.SUNKEN, bd=2)
            heatmap_img.pack()
            
            if self.current_clinical_data:
                temp_path = f"temp_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(temp_path, cv2.cvtColor(result['heatmap'], cv2.COLOR_RGB2BGR))
                self.current_clinical_data.heatmap_path = temp_path
        
        # Graph section
        graph_frame = tk.Frame(visual_container, bg=Win95Style.LIGHT_GRAY)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        graph_label = tk.Label(
            graph_frame,
            text="Classification Probabilities",
            bg=Win95Style.LIGHT_GRAY,
            fg=Win95Style.DARK_BLUE,
            font=("MS Sans Serif", 8, "bold")
        )
        graph_label.pack(pady=(0, 4))
        
        self.create_probability_graph(graph_frame, result['all_probabilities'])
        
        self.bind_mousewheel_recursively(msg_frame)
        self.root.after(100, self.scroll_to_bottom)
    
    def create_probability_graph(self, parent, probabilities):
        fig = Figure(figsize=(4, 3), dpi=80, facecolor=Win95Style.LIGHT_GRAY)
        ax = fig.add_subplot(111)
        
        classes = list(probabilities.keys())
        probs = [probabilities[c] * 100 for c in classes]
        
        colors_list = ['#000080' if p == max(probs) else '#808080' for p in probs]
        bars = ax.barh(classes, probs, color=colors_list)
        
        ax.set_xlabel('Probability (%)', fontsize=9, fontname='MS Sans Serif')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor(Win95Style.WHITE)
        fig.patch.set_facecolor(Win95Style.LIGHT_GRAY)
        
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=8, fontname='MS Sans Serif')
        
        fig.tight_layout()
        
        canvas_frame = tk.Frame(parent, bg=Win95Style.DARKER_GRAY, relief=tk.SUNKEN, bd=2)
        canvas_frame.pack()
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=1, pady=1)

    def add_message(self, role, content):
        msg = ChatMessage(role, content)
        self.conversation_history.append(msg)
        
        msg_frame = tk.Frame(self.chat_display, bg=Win95Style.WHITE, padx=10, pady=5)
        msg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        role_color = Win95Style.DARK_BLUE if role == "user" else "#008000"
        role_text = "You" if role == "user" else "Medical AI"
        
        header = tk.Label(
            msg_frame,
            text=f"{role_text} - {msg.timestamp.strftime('%H:%M')}",
            bg=Win95Style.WHITE,
            fg=role_color,
            font=("MS Sans Serif", 8, "bold"),
            anchor="w"
        )
        header.pack(fill=tk.X)
        
        content_outer = tk.Frame(msg_frame, bg=Win95Style.DARKER_GRAY, relief=tk.SUNKEN, bd=2)
        content_outer.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        
        msg_bg = Win95Style.CREAM if role == "user" else Win95Style.LIGHT_GRAY
        
        content_frame = tk.Frame(content_outer, bg=msg_bg)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        content_label = tk.Label(
            content_frame,
            text=content,
            bg=msg_bg,
            fg=Win95Style.BLACK,
            font=("MS Sans Serif", 9),
            justify=tk.LEFT,
            wraplength=750,
            anchor="w"
        )
        content_label.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.bind_mousewheel_recursively(msg_frame)
        self.root.after(100, self.scroll_to_bottom)

    def resize_for_display(self, image, max_width=400):
        h, w = image.shape[:2]
        if w > max_width:
            ratio = max_width / w
            new_w = max_width
            new_h = int(h * ratio)
            image = cv2.resize(image, (new_w, new_h))
        return image
    
    def scroll_to_bottom(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)
        self.custom_scrollbar.draw()
    
    def set_input_state(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.input_text.config(state=state)
        
        if enabled:
            self.send_btn.config_state(tk.NORMAL)
            self.upload_btn.config_state(tk.NORMAL)
            self.download_btn.config_state(tk.NORMAL)
        else:
            self.send_btn.config_state(tk.DISABLED)
            self.upload_btn.config_state(tk.DISABLED)
            self.download_btn.config_state(tk.DISABLED)
    
    def show_error(self, message):
        self.add_message("assistant", f"Error: {message}")
    
    # === PDF GENERATION WITH URGENCY ===
    
    def download_clinical_pdf(self):
        if not self.clinical_sessions:
            self.show_error("No clinical analysis data available. Please perform an image analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"Clinical_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if not filename:
            return
        
        try:
            self.add_message("assistant", "Generating clinical analysis PDF report with urgency classifications...")
            self.generate_clinical_pdf(filename)
            self.add_message("assistant", f"PDF report successfully saved to: {filename}")
        except Exception as e:
            self.show_error(f"Failed to generate PDF: {str(e)}")
    
    def generate_clinical_pdf(self, filename):
        """Generate PDF with urgency color coding"""
        doc = SimpleDocTemplate(filename, pagesize=letter,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#000080'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#000080'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        
        # Title
        story.append(Paragraph("Clinical Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 
                              ParagraphStyle('Subtitle', parent=body_style, alignment=TA_CENTER, fontSize=10)))
        story.append(Spacer(1, 0.3*inch))
        
        for idx, session in enumerate(self.clinical_sessions):
            # Session header
            story.append(Paragraph(f"Analysis Session {idx + 1}", heading_style))
            story.append(Paragraph(f"Timestamp: {session.timestamp.strftime('%B %d, %Y at %H:%M:%S')}", body_style))
            story.append(Spacer(1, 0.15*inch))
            
            # URGENCY ASSESSMENT (prominently displayed)
            if session.urgency_info:
                urgency_info = session.urgency_info
                urgency_color = urgency_info['color_reportlab']
                
                urgency_style = ParagraphStyle(
                    'UrgencyStyle',
                    parent=heading_style,
                    fontSize=16,
                    textColor=urgency_color,
                    alignment=TA_CENTER,
                    spaceAfter=6,
                    spaceBefore=6
                )
                
                story.append(Paragraph(f"⚠ URGENCY LEVEL: {urgency_info['label'].upper()} ⚠", urgency_style))
                
                # Urgency description box
                urgency_data = [[urgency_info['description']]]
                urgency_table = Table(urgency_data, colWidths=[6.5*inch])
                urgency_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), urgency_color),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ]))
                story.append(urgency_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Patient Information
            if session.patient_info:
                story.append(Paragraph("Patient Information", heading_style))
                
                patient_data = [[key.replace('_', ' ').title(), value] 
                               for key, value in session.patient_info.items()]
                
                patient_table = Table(patient_data, colWidths=[2.5*inch, 4*inch])
                patient_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E0E0E0')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                
                story.append(patient_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Image Information
            if session.image_info:
                story.append(Paragraph("Image Details", heading_style))
                
                if 'description' in session.image_info:
                    story.append(Paragraph(f"<b>Clinical Question:</b> {session.image_info['description']}", body_style))
                
                if 'constraints' in session.image_info and session.image_info['constraints']:
                    constraints_text = ", ".join([f"{k.title()}: {v}" 
                                                 for k, v in session.image_info['constraints'].items()])
                    story.append(Paragraph(f"<b>Image Specifications:</b> {constraints_text}", body_style))
                
                story.append(Spacer(1, 0.15*inch))
            
            # Analysis Results
            if session.analysis_result:
                story.append(Paragraph("Diagnostic Analysis", heading_style))
                
                result = session.analysis_result
                
                story.append(Paragraph(f"<b>Model Used:</b> {session.model_used}", body_style))
                story.append(Paragraph(f"<b>Primary Diagnosis:</b> {result['class']}", body_style))
                story.append(Paragraph(f"<b>Confidence Level:</b> {result['confidence']*100:.2f}%", body_style))
                
                story.append(Spacer(1, 0.1*inch))
                
                # Probability table with urgency color highlight
                story.append(Paragraph("Classification Probabilities:", body_style))
                prob_data = [["Classification", "Probability", "Urgency Level"]]
                
                # Get urgency mapping for this model
                model_config = None
                for config in MODEL_CONFIGS.values():
                    if config['description'] == session.model_used:
                        model_config = config
                        break
                
                for class_name, prob in sorted(result['all_probabilities'].items(), 
                                              key=lambda x: x[1], reverse=True):
                    # Get urgency for this class
                    if model_config and 'urgency_mapping' in model_config:
                        class_urgency = model_config['urgency_mapping'].get(class_name, 'monitor')
                        urgency_label = URGENCY_LEVELS[class_urgency]['label']
                    else:
                        urgency_label = "Monitor"
                    
                    prob_data.append([class_name, f"{prob*100:.2f}%", urgency_label])
                
                prob_table = Table(prob_data, colWidths=[3*inch, 1.5*inch, 2*inch])
                
                # Highlight the predicted class row with urgency color
                predicted_class = result['class']
                predicted_row = next((i for i, row in enumerate(prob_data[1:], 1) if row[0] == predicted_class), None)
                
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#000080')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')])
                ]
                
                # Highlight predicted class with urgency color
                if predicted_row and session.urgency_info:
                    urgency_color = session.urgency_info['color_reportlab']
                    table_style.append(('BACKGROUND', (0, predicted_row), (-1, predicted_row), urgency_color))
                    table_style.append(('TEXTCOLOR', (0, predicted_row), (-1, predicted_row), colors.white))
                    table_style.append(('FONTNAME', (0, predicted_row), (-1, predicted_row), 'Helvetica-Bold'))
                
                prob_table.setStyle(TableStyle(table_style))
                
                story.append(prob_table)
                story.append(Spacer(1, 0.2*inch))
                
                # Heat map image
                if session.heatmap_path and os.path.exists(session.heatmap_path):
                    story.append(Paragraph("Explainable AI Visualization", heading_style))
                    story.append(Paragraph("The heat map below highlights the regions of the image that were most significant in determining the diagnosis:", body_style))
                    
                    img = RLImage(session.heatmap_path, width=4*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
            
            # Clinical Explanation
            if session.explanation:
                story.append(Paragraph("Clinical Interpretation", heading_style))
                
                explanation_paragraphs = session.explanation.split('\n\n')
                for para in explanation_paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), body_style))
                        story.append(Spacer(1, 0.1*inch))
            
            if idx < len(self.clinical_sessions) - 1:
                story.append(PageBreak())
        
        # Footer disclaimer
        story.append(Spacer(1, 0.3*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=body_style,
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            "IMPORTANT: This report is generated by an AI system and should be reviewed by qualified healthcare professionals. "
            "It is not a substitute for professional medical judgment and diagnosis. The urgency classifications are algorithmically "
            "determined and should be validated by clinical assessment.",
            disclaimer_style
        ))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary heatmap files
        for session in self.clinical_sessions:
            if session.heatmap_path and os.path.exists(session.heatmap_path):
                try:
                    os.remove(session.heatmap_path)
                except:
                    pass

# ===========================================
# APPLICATION ENTRY POINT
# ===========================================

def main():
    root = tk.Tk()
    app = MedicalAIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()