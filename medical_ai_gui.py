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
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import os

# ===========================================
# CONFIGURATION & CONSTANTS
# ===========================================

# Configuration
OPENAI_API_KEY = "your-api-key-here" # <-- Replace with your actual API key
openai.api_key = OPENAI_API_KEY

# Model configurations with constraints
MODEL_CONFIGS = {
        "brain_mri_alzheimers": {
        "path": "models/Early_Alzheimers_DenseNet121_Augmented.keras",
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
        }
    },


    "skin_cancer_isic": {
    "path": "models/Skin_Cancer_DenseNet201_Augmented.keras",
    "classes": [
        "actinic keratosis",
        "basal cell carcinoma",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "pigmented benign keratosis",
        "seborrheic keratosis",
        "squamous cell carcinoma",
        "vascular lesion"],

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
        }
    }
}

# ===========================================
# THEME & STYLING
# ===========================================

# Theme
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
# ALGORITHM & MODEL COMPONENTS
# ===========================================

class ModelManager:
    # Manages loading and inference of medical models
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        for name, config in MODEL_CONFIGS.items():
            try:
                # Check if model file exists
                if os.path.exists(config["path"]):
                    model = keras.models.load_model(config["path"])
                    self.models[name] = {
                        "model": model,
                        "config": config
                    }
                    print(f"Loaded model: {name}")
                else:
                    print(f"Model file not found: {config['path']}")
                    # Create a dummy model for testing
                    self.create_dummy_model(name, config)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                # Create a dummy model for testing
                self.create_dummy_model(name, config)
    
    def create_dummy_model(self, name, config):
        # Dummy model for testing when real models are not available
        try:
            # Create a simple dummy model
            model = keras.Sequential([
                keras.layers.Input(shape=(*config["input_size"], 3)),
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
            # Check keyword match first
            keywords = model_data["config"]["keywords"]
            keyword_score = sum(1 for keyword in keywords if keyword in query_lower)
            
            # Check constraints and record mismatches
            constraints = model_data["config"]["constraints"]
            constraint_descriptions = model_data["config"]["constraint_descriptions"]
            mismatches = []
            
            for constraint_key, allowed_values in constraints.items():
                if not allowed_values:
                    continue
                
                user_value = constraint_responses.get(constraint_key, "")
                
                # Check if user value matches any allowed value
                if not any(allowed.lower() == user_value.lower() for allowed in allowed_values):
                    mismatches.append({
                        "constraint": constraint_key,
                        "description": constraint_descriptions.get(constraint_key, constraint_key),
                        "user_selected": user_value,
                        "required": allowed_values,
                        "model": model_data["config"]["description"]
                    })
            
            # Store mismatch info for this model
            constraint_mismatches[model_name] = {
                "description": model_data["config"]["description"],
                "mismatches": mismatches,
                "keyword_score": keyword_score
            }
            
            # If all constraints match and keyword score is best, this is our model
            if len(mismatches) == 0 and keyword_score > best_score:
                best_score = keyword_score
                best_match = (model_name, model_data)
        
        return best_match, constraint_mismatches
    
    def find_suitable_model(self, query, constraint_responses):
        # Find the most suitable model based on query and constraints
        result, _ = self.find_suitable_model_with_details(query, constraint_responses)
        return result
    
    def check_constraints_match(self, model_name, user_responses):
        # Check if user responses match model constraints
        if model_name not in self.models:
            return False
        
        constraints = self.models[model_name]["config"]["constraints"]
        
        for constraint_key, allowed_values in constraints.items():
            if not allowed_values:
                continue
            
            user_value = user_responses.get(constraint_key, "")
            
            # Check if user value matches any allowed value
            if not any(allowed.lower() == user_value.lower() for allowed in allowed_values):
                return False
        
        return True
    
    def get_all_constraint_options(self):
        # Get all unique constraint options across all models
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
        
        # Convert sets to sorted lists
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
        
        # Sort models by keyword score (most relevant first)
        sorted_models = sorted(
            constraint_mismatches.items(),
            key=lambda x: x[1]["keyword_score"],
            reverse=True
        )
        
        # Check if any models matched the query at all
        has_keyword_match = any(info["keyword_score"] > 0 for _, info in sorted_models)
        
        if not has_keyword_match:
            # No models match the query keywords
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
        
        # Find the best matching model (highest keyword score)
        best_model = sorted_models[0]
        if len(best_model[1]["mismatches"]) > 0:
            error_msg += "To proceed with the analysis:\n"
            error_msg += f"1. Change your image requirements to match {best_model[1]['description']}\n"
            error_msg += "2. Upload a new image that meets these requirements\n"
            error_msg += "3. Or ask a different question that matches your current image selections"
        
        return error_msg
    
    def preprocess_image(self, image, target_size):
        # Preprocess image for model input
        img = cv2.resize(image, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def generate_gradcam(self, model, image, pred_class):
        # Generate Grad-CAM heatmap
        try:
            replace2linear = ReplaceToLinear()
            gradcam = Gradcam(model, model_modifier=replace2linear, clone=True)
            cam = gradcam(
                CategoricalScore(pred_class),
                image,
                penultimate_layer=-1
            )
            return cam[0]
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return None

# Will need to have more than one option that the user can multi-select
    def predict_with_xai(self, model_name, image_array):
        # Make prediction and generate XAI visualization
        model_data = self.models[model_name]
        model = model_data["model"]
        config = model_data["config"]
        
        # Preprocess
        processed = self.preprocess_image(image_array, config["input_size"])
        
        # Predict
        predictions = model.predict(processed, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        
        # Generate Grad-CAM
        heatmap = self.generate_gradcam(model, processed, pred_class)
        
        # Create overlay
        overlay = self.create_overlay(image_array, heatmap, config["input_size"])
        
        result = {
            "class": config["classes"][pred_class],
            "confidence": confidence,
            "all_probabilities": {config["classes"][i]: float(predictions[0][i]) 
                                 for i in range(len(config["classes"]))},
            "heatmap": overlay,
            "model_used": config["description"]
        }
        
        return result

# Currently not working as intended
    def create_overlay(self, original_image, heatmap, target_size):
        # Create heatmap overlay on original image
        if heatmap is None:
            return original_image
        
        img = cv2.resize(original_image, target_size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, target_size)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        
        return overlay

# ===========================================
# DATA MODELS
# ===========================================

class ChatMessage:
    # This class represents a chat message
    def __init__(self, role, content, image=None, image_info=None, heatmap=None, timestamp=None):
        self.role = role
        self.content = content
        self.image = image
        self.image_info = image_info
        self.heatmap = heatmap
        self.timestamp = timestamp or datetime.now()

# ===========================================
# GUI COMPONENTS - CUSTOM WIDGETS
# ===========================================

class Win95Button(tk.Frame):
    # My custom Windows 95 style button widget
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
        # Create the 3D button appearance
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
        # Handle button press
        if not self.is_enabled:
            return
        self.is_pressed = True
        self.top_highlight.configure(bg=Win95Style.BUTTON_SHADOW)
        self.left_highlight.configure(bg=Win95Style.BUTTON_SHADOW)
        self.bottom_shadow.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.right_shadow.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.button_face.configure(padx=5, pady=5)
        
    def on_release(self, event):
        # Handle button release
        if self.is_pressed and self.is_enabled:
            self.reset_appearance()
            if self.command:
                self.command()
        
    def on_leave(self, event):
        # Handle mouse leave
        if self.is_pressed:
            self.reset_appearance()
    
    def reset_appearance(self):
        # Reset button to unpressed state
        self.is_pressed = False
        self.top_highlight.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.left_highlight.configure(bg=Win95Style.BUTTON_HIGHLIGHT)
        self.bottom_shadow.configure(bg=Win95Style.BUTTON_SHADOW)
        self.right_shadow.configure(bg=Win95Style.BUTTON_SHADOW)
        self.button_face.configure(padx=4, pady=4)
    
    def config_state(self, state):
        # Configure button state (NORMAL or DISABLED)
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
        # Set the command to call when scrolling
        self.command = command
    
    def set(self, first, last):
        # Set scrollbar position (called by scrolled widget)
        self.thumb_pos = float(first)
        self.thumb_size = float(last) - float(first)
        self.draw()
    
    def draw(self):
        # Draw the scrollbar
        self.delete("all")
        width = self.winfo_width() or 18
        height = self.winfo_height() or 100
        
        # Draw track
        self.create_rectangle(1, 1, width-1, height-1, 
                            fill=self.track_color, outline=self.border_color)
        
        # Calculate thumb position and size
        thumb_height = max(20, int(height * self.thumb_size))
        thumb_y = int((height - thumb_height) * (self.thumb_pos / (1 - self.thumb_size)) if self.thumb_size < 1 else 0)
        
        # Draw thumb with 3D effect
        # Main thumb
        self.create_rectangle(2, thumb_y+2, width-2, thumb_y+thumb_height-2,
                            fill=self.thumb_color, outline="")
        
        # Highlight (top and left)
        self.create_line(2, thumb_y+2, width-2, thumb_y+2, 
                        fill=Win95Style.LIGHT_GRAY, width=2)
        self.create_line(2, thumb_y+2, 2, thumb_y+thumb_height-2,
                        fill=Win95Style.LIGHT_GRAY, width=2)
        
        # Shadow (bottom and right)
        self.create_line(2, thumb_y+thumb_height-2, width-2, thumb_y+thumb_height-2,
                        fill=Win95Style.BLACK, width=2)
        self.create_line(width-2, thumb_y+2, width-2, thumb_y+thumb_height-2,
                        fill=Win95Style.BLACK, width=2)
        
        self.thumb_y = thumb_y
        self.thumb_height = thumb_height
    
    def on_click(self, event):
        # Handle click on scrollbar
        if self.thumb_y <= event.y <= self.thumb_y + self.thumb_height:
            self.dragging = True
            self.drag_start_y = event.y - self.thumb_y
        else:
            # Click on track - jump to position
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
        self.root.title("Medical AI Assistant")
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
        self.constraint_widgets = {}  # Initialize constraint_widgets here
        
        # Setup GUI
        self.setup_gui()

        # === GUI Setup Methods ===

    def setup_gui(self):
        # Setup the main GUI
        # Outer frame for window border
        outer_frame = tk.Frame(
            self.root, 
            bg=Win95Style.BUTTON_HIGHLIGHT,
            relief=tk.RAISED,
            bd=2
        )
        outer_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Inner shadow
        inner_border = tk.Frame(
            outer_frame,
            bg=Win95Style.DARK_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        inner_border.pack(fill=tk.BOTH, expand=True)
        
        # Title bar
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
            text="Medical AI Assistant",
            bg=Win95Style.ACTIVE_TITLE,
            fg=Win95Style.WHITE,
            font=("MS Sans Serif", 9, "bold"),
            anchor="w"
        )
        title_label.pack(side=tk.LEFT)
        
        # Main container
        main_container = tk.Frame(inner_border, bg=Win95Style.BG_GRAY)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Chat display area
        self.setup_chat_area(main_container)
        
        # Input area
        self.setup_input_area(main_container)
        
        # Welcome message
        self.add_message("assistant", 
            "Welcome to Medical AI Assistant!\n\n"
            "I can help you with:\n"
            "  - General medical questions and information\n"
            "  - Analysis of medical images with explainable AI\n\n"
            "Ask me anything or upload a medical image for analysis!")
    
    def setup_chat_area(self, parent):
        # Setup the chat display area
        chat_outer = tk.Frame(parent, bg=Win95Style.BG_GRAY)
        chat_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8, side=tk.TOP)
        
        # Create "pressed-in" effect of frame for chat area
        chat_frame = tk.Frame(
            chat_outer,
            bg=Win95Style.DARKER_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Inner frame
        chat_inner = tk.Frame(chat_frame, bg=Win95Style.WHITE, bd=0)
        chat_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Canvas for scrolling
        self.chat_canvas = tk.Canvas(
            chat_inner,
            bg=Win95Style.WHITE,
            highlightthickness=0,
            bd=0
        )
        
        # Custom scrollbar
        self.custom_scrollbar = Win95Scrollbar(chat_inner)
        self.custom_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.custom_scrollbar.set_command(self.on_scrollbar)
        
        self.chat_canvas.configure(yscrollcommand=self.custom_scrollbar.set)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas
        self.chat_display = tk.Frame(self.chat_canvas, bg=Win95Style.WHITE)
        self.chat_window = self.chat_canvas.create_window(
            (0, 0),
            window=self.chat_display,
            anchor="nw",
            tags="chat_window"
        )
        
        # Bind events
        self.chat_display.bind("<Configure>", self.on_frame_configure)
        self.chat_canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Mouse wheel scrolling
        self.chat_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.chat_canvas.bind("<Button-4>", self.on_mousewheel)
        self.chat_canvas.bind("<Button-5>", self.on_mousewheel)
        
        self.bind_mousewheel_recursively(self.chat_display)

        # === Event Handlers ===

    def on_scrollbar(self, *args):
        # Handles custom scrollbar movement
        self.chat_canvas.yview(*args)
    
    def bind_mousewheel_recursively(self, widget):
        # Bind mousewheel to widget and all its children
        widget.bind("<MouseWheel>", self.on_mousewheel)
        widget.bind("<Button-4>", self.on_mousewheel)
        widget.bind("<Button-5>", self.on_mousewheel)
        for child in widget.winfo_children():
            self.bind_mousewheel_recursively(child)
    
    def on_mousewheel(self, event):
        # Handle mouse wheel scrolling
        if event.num == 4 or event.delta > 0:
            self.chat_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.chat_canvas.yview_scroll(1, "units")
        return "break"
    
    def setup_input_area(self, parent):
        # Setup the input area
        input_outer = tk.Frame(parent, bg=Win95Style.BG_GRAY)
        input_outer.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 8))
        
        # Text input with frame
        input_frame = tk.Frame(
            input_outer,
            bg=Win95Style.DARKER_GRAY,
            relief=tk.SUNKEN,
            bd=2
        )
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        input_inner = tk.Frame(input_frame, bg=Win95Style.WHITE, bd=0)
        input_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Container for all input elements
        input_container = tk.Frame(input_inner, bg=Win95Style.WHITE)
        input_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        
        # Image preview area (initially hidden)
        self.image_preview_frame = tk.Frame(input_container, bg=Win95Style.WHITE)
        
        # Constraint area (initially hidden)
        self.constraint_frame = tk.Frame(input_container, bg=Win95Style.WHITE)
        
        # Text input area
        text_container = tk.Frame(input_container, bg=Win95Style.WHITE)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        # Text input
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
        
        # Buttons container
        buttons_container = tk.Frame(text_container, bg=Win95Style.WHITE)
        buttons_container.pack(side=tk.RIGHT)
        
        # Upload button
        self.upload_btn = Win95Button(
            buttons_container,
            text="Upload Image",
            command=self.upload_image,
            width=100,
            height=32
        )
        self.upload_btn.pack(pady=(0, 6))
        
        # Send button
        self.send_btn = Win95Button(
            buttons_container,
            text="Send",
            command=self.send_message,
            width=100,
            height=32
        )
        self.send_btn.pack()
    
    def on_frame_configure(self, event=None):
        # Updates scroll region
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        self.custom_scrollbar.draw()
    
    def on_canvas_configure(self, event):
        # Adjusts window width when canvas is resized
        canvas_width = event.width
        self.chat_canvas.itemconfig(self.chat_window, width=canvas_width)

        # === Image Handling ===

    def upload_image(self):
        # Handle image upload
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
            
            # Create thumbnail
            self.create_image_preview(image, os.path.basename(file_path))
            
            # Show constraint inputs
            self.show_constraint_inputs()
    
    def create_image_preview(self, image, filename):
        # Clear existing preview first
        for widget in self.image_preview_frame.winfo_children():
            widget.destroy()
        
        # Pack the preview frame
        self.image_preview_frame.pack(fill=tk.X, pady=(0, 6), side=tk.TOP, anchor='n')
        
        # Create thumbnail
        thumbnail = self.resize_for_display(image, max_width=60)
        img_pil = Image.fromarray(thumbnail)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.current_image_thumbnail = img_tk
        
        # Preview container
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
        
        # Thumbnail
        img_label = tk.Label(content_frame, image=img_tk, bg=Win95Style.CREAM)
        img_label.pack(side=tk.LEFT, padx=(0, 8))
        
        # Filename
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
        
        # Remove button
        remove_btn = Win95Button(
            content_frame,
            text="Remove",
            command=self.remove_image,
            width=70,
            height=50
        )
        remove_btn.pack(side=tk.RIGHT)
    
    def show_constraint_inputs(self):
        # Show constraint input fields with all options from all models
        # Clear existing constraints first
        for widget in self.constraint_frame.winfo_children():
            widget.destroy()
        
        all_constraints = self.model_manager.get_all_constraint_options()
        
        if not all_constraints:
            return
        
        # Pack the constraint frame
        self.constraint_frame.pack(fill=tk.X, pady=(0, 6), side=tk.TOP, anchor='n')
        
        # Constraint container
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
        
        # Create dropdowns for each constraint
        self.constraint_widgets = {}  # Re-initialize when showing constraints
        
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
        
        # FORCE update to ensure proper display
        self.constraint_frame.update_idletasks()
    
    def remove_image(self):
        self.current_image = None
        self.current_image_path = None
        self.current_image_thumbnail = None
        self.constraint_responses = {}
        
        # Hide both frames
        self.image_preview_frame.pack_forget()
        self.constraint_frame.pack_forget()
        
        # Clear widgets
        for widget in self.image_preview_frame.winfo_children():
            widget.destroy()
        for widget in self.constraint_frame.winfo_children():
            widget.destroy()
    
    def on_enter(self, event):
        # Handle Enter key
        if not event.state & 0x1:
            self.send_message()
            return "break"

        # === Message Processing (Algorithm Integration) ===

    def send_message(self):
        # Send user message
        message = self.input_text.get("1.0", tk.END).strip()
        
        if not message:
            return
        
        # Clear input
        self.input_text.delete("1.0", tk.END)
        
        # Store image data if present
        sent_image = None
        sent_image_info = None
        if self.current_image is not None:
            # Check if constraint_widgets exists and has values
            if hasattr(self, 'constraint_widgets') and self.constraint_widgets:
                sent_image = self.current_image.copy()
                sent_image_info = {
                    "filename": os.path.basename(self.current_image_path),
                    "constraints": {k: v.get() for k, v in self.constraint_widgets.items()}
                }
            else:
                # If no constraint widgets, use empty constraints
                sent_image = self.current_image.copy()
                sent_image_info = {
                    "filename": os.path.basename(self.current_image_path),
                    "constraints": {}
                }
        
        # Add user message with image
        self.add_message_with_image("user", message, sent_image, sent_image_info)
        
        # Remove image from input area after sending
        if self.current_image is not None:
            # Store for processing
            image_to_process = self.current_image
            constraints_to_process = {}
            if hasattr(self, 'constraint_widgets') and self.constraint_widgets:
                constraints_to_process = {k: v.get() for k, v in self.constraint_widgets.items()}
            
            # Clear input area
            self.remove_image()
            
            # Disable input while processing
            self.set_input_state(False)
            
            # Process in thread
            thread = threading.Thread(
                target=self.process_message_with_image, 
                args=(message, image_to_process, constraints_to_process)
            )
            thread.daemon = True
            thread.start()
        else:
            # Disable input while processing
            self.set_input_state(False)
            
            # Regular chat - process in thread
            thread = threading.Thread(target=self.process_chat_message, args=(message,))
            thread.daemon = True
            thread.start()
    
    def process_message_with_image(self, message, image, constraints):
        try:
            # Find suitable model and get detailed mismatch info
            best_match, constraint_mismatches = self.model_manager.find_suitable_model_with_details(
                message, constraints
            )
            
            # Check if we found a matching model
            if best_match is None or best_match[0] is None:
                # Generate detailed error message
                error_msg = self.model_manager.generate_detailed_error_message(message, constraints)
                self.root.after(0, self.show_error, error_msg)
                return
            
            model_name, model_data = best_match
            
            # Perform analysis
            self.process_image_analysis(model_name, message, image)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in process_message_with_image: {error_details}")
            self.root.after(0, self.show_error, f"An error occurred during processing: {str(e)}")
        finally:
            self.root.after(0, lambda: self.set_input_state(True))
    
    def process_image_analysis(self, model_name, description, image):
        # Process image with description
        try:
            model_data = self.model_manager.models[model_name]
            
            # Perform analysis
            self.root.after(0, self.add_message, "assistant",
                f"Analyzing with {model_data['config']['description']}...\n"
                "Generating diagnosis and explanation...")
            
            # Run model
            result = self.model_manager.predict_with_xai(model_name, image)
            
            # Get GPT explanation
            explanation = self.get_gpt_explanation(result, description)
            
            # Display results
            self.root.after(0, self.display_analysis_results, result, explanation)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in process_image_analysis: {error_details}")
            self.root.after(0, self.show_error, f"Analysis error: {str(e)}")
    
    def get_gpt_explanation(self, result, context):
        prompt = f"""As a medical AI assistant, explain this diagnosis clearly and professionally:

Model: {result['model_used']}
Diagnosis: {result['class']}
Confidence: {result['confidence']*100:.1f}%
All Probabilities: {result['all_probabilities']}
Context: {context}

Provide a clear explanation including:
1. What the diagnosis means
2. The confidence level interpretation
3. Clinical significance
4. Recommended next steps

Keep it concise but informative. Do not use emojis or special symbols."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Diagnosis: {result['class']}\nConfidence: {result['confidence']*100:.1f}%\n\n(Note: Unable to generate detailed explanation)"
    
    def process_chat_message(self, message):
        """Process regular chat message with ChatGPT"""
        try:
            messages = [
                {"role": "system", "content": 
                 "You are a helpful medical AI assistant. Provide accurate, "
                 "professional medical information. Always remind users to consult "
                 "healthcare professionals for medical advice. Do not use emojis or special symbols."}
            ]
            
            for msg in self.conversation_history[-10:]:
                if msg.image is None:
                    messages.append({"role": msg.role, "content": msg.content})
            
            messages.append({"role": "user", "content": message})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
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
        """Add a message with an attached image to the chat"""
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
        
        # Image section if present
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
            
            # Thumbnail
            thumbnail = self.resize_for_display(image, max_width=60)
            img_pil = Image.fromarray(thumbnail)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_references.append(img_tk)
            
            thumb_label = tk.Label(image_content, image=img_tk, bg=Win95Style.CREAM)
            thumb_label.pack(side=tk.LEFT, padx=(0, 8))
            
            # Image info
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
            
            # Constraints
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
        
        # Message content
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
    
# possibly have different type of outputs for given models
    def display_analysis_results(self, result, explanation):
        # Display analysis results with heatmap and graph
        msg_frame = tk.Frame(self.chat_display, bg=Win95Style.WHITE, padx=10, pady=5)
        msg_frame.pack(fill=tk.X, padx=10, pady=5)
        
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
        
        # Main content frame
        content_outer = tk.Frame(msg_frame, bg=Win95Style.DARKER_GRAY, relief=tk.SUNKEN, bd=2)
        content_outer.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        
        content_frame = tk.Frame(content_outer, bg=Win95Style.LIGHT_GRAY)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
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
        # Bar graph of classification probabilities
        fig = Figure(figsize=(4, 3), dpi=80, facecolor=Win95Style.LIGHT_GRAY)
        ax = fig.add_subplot(111)
        
        classes = list(probabilities.keys())
        probs = [probabilities[c] * 100 for c in classes]
        
        colors = ['#000080' if p == max(probs) else '#808080' for p in probs]
        bars = ax.barh(classes, probs, color=colors)
        
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

        # === Display Methods ===

    def add_message(self, role, content):
        # Add a message to the chat
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

        # === Utility Methods ===

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
        # Enable/disable input controls
        state = tk.NORMAL if enabled else tk.DISABLED
        self.input_text.config(state=state)
        
        if enabled:
            self.send_btn.config_state(tk.NORMAL)
            self.upload_btn.config_state(tk.NORMAL)
        else:
            self.send_btn.config_state(tk.DISABLED)
            self.upload_btn.config_state(tk.DISABLED)
    
    def show_error(self, message):
        self.add_message("assistant", f"Error: {message}")

# ===========================================
# APPLICATION ENTRY POINT
# ===========================================

def main():
    root = tk.Tk()
    app = MedicalAIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()