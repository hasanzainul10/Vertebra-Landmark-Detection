import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import torch
import sys
from models import spinal_net
import decoder
import cobb_evaluate
import threading
from pathlib import Path
import json

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ZoomableCanvas(tk.Canvas):
    """A canvas widget with zoom and pan capabilities"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<Button-4>", self.zoom)  # Linux scroll up
        self.bind("<Button-5>", self.zoom)  # Linux scroll down
        
        self.scale_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.image_id = None
        self.original_image = None
        
    def start_pan(self, event):
        self.scan_mark(event.x, event.y)
        
    def pan(self, event):
        self.scan_dragto(event.x, event.y, gain=1)
        
    def zoom(self, event):
        if event.num == 4 or event.delta > 0:  # Zoom in
            self.scale_factor *= 1.1
        elif event.num == 5 or event.delta < 0:  # Zoom out
            self.scale_factor /= 1.1
            
        self.scale_factor = max(0.1, min(5.0, self.scale_factor))
        self.update_display()
        
    def set_image(self, image):
        """Set the image to display"""
        self.original_image = image
        self.update_display()
        
    def update_display(self):
        """Update the display with current zoom and pan"""
        if self.original_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Resize image based on scale factor
        img_width, img_height = self.original_image.size
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display new image
        self.delete("all")
        self.image_id = self.create_image(
            canvas_width//2 + self.pan_x, 
            canvas_height//2 + self.pan_y, 
            image=self.photo_image, 
            anchor="center"
        )
        
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.scale_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()

class CobbAngleGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Cobb Angle Analysis Tool")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.current_image_index = 0
        self.image_files = []
        self.current_image = None
        self.landmarks = None
        self.cobb_angles = None
        self.vertebrae_info = None
        
        # GUI settings
        self.show_primary = tk.BooleanVar(value=True)
        self.show_secondary = tk.BooleanVar(value=False)
        self.show_tertiary = tk.BooleanVar(value=False)
        
        # Customization settings
        self.line_color = "#FF0000"  # Red
        self.line_width = tk.IntVar(value=3)
        self.text_size = tk.IntVar(value=12)
        self.text_color = "#FFFFFF"  # White
        
        # Initialize model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top control panel
        self.setup_control_panel(main_frame)
        
        # Main content area
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # Left panel for settings
        self.setup_settings_panel(content_frame)
        
        # Right panel for image display
        self.setup_display_panel(content_frame)
        
    def setup_control_panel(self, parent):
        """Setup the top control panel"""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", pady=(0, 10))
        
        # File selection
        file_frame = ctk.CTkFrame(control_frame)
        file_frame.pack(side="left", padx=10, pady=10)
        
        ctk.CTkButton(file_frame, text="Select Image", command=self.select_image).pack(side="left", padx=5)
        ctk.CTkButton(file_frame, text="Select Folder", command=self.select_folder).pack(side="left", padx=5)
        
        # Navigation
        nav_frame = ctk.CTkFrame(control_frame)
        nav_frame.pack(side="left", padx=10, pady=10)
        
        ctk.CTkButton(nav_frame, text="◀ Previous", command=self.previous_image).pack(side="left", padx=5)
        ctk.CTkButton(nav_frame, text="Next ▶", command=self.next_image).pack(side="left", padx=5)
        
        # Image info
        self.image_info_label = ctk.CTkLabel(control_frame, text="No image selected")
        self.image_info_label.pack(side="left", padx=10, pady=10)
        
        # Detection button
        self.detect_button = ctk.CTkButton(control_frame, text="Detect Cobb Angles", command=self.detect_angles)
        self.detect_button.pack(side="right", padx=10, pady=10)
        
    def setup_settings_panel(self, parent):
        """Setup the left settings panel"""
        settings_frame = ctk.CTkFrame(parent, width=300)
        settings_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # Angle visibility settings
        angle_frame = ctk.CTkFrame(settings_frame)
        angle_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(angle_frame, text="Angle Visibility", font=("Arial", 14, "bold")).pack(pady=5)
        
        ctk.CTkCheckBox(angle_frame, text="Primary Angle", variable=self.show_primary, 
                       command=self.update_display).pack(anchor="w", padx=10, pady=2)
        ctk.CTkCheckBox(angle_frame, text="Secondary Angle", variable=self.show_secondary, 
                       command=self.update_display).pack(anchor="w", padx=10, pady=2)
        ctk.CTkCheckBox(angle_frame, text="Tertiary Angle", variable=self.show_tertiary, 
                       command=self.update_display).pack(anchor="w", padx=10, pady=2)
        
        # Customization settings
        custom_frame = ctk.CTkFrame(settings_frame)
        custom_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(custom_frame, text="Customization", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Line width
        ctk.CTkLabel(custom_frame, text="Line Width:").pack(anchor="w", padx=10, pady=2)
        line_width_slider = ctk.CTkSlider(custom_frame, from_=1, to=10, variable=self.line_width, 
                                         command=self.update_display)
        line_width_slider.pack(fill="x", padx=10, pady=5)
        
        # Text size
        ctk.CTkLabel(custom_frame, text="Text Size:").pack(anchor="w", padx=10, pady=2)
        text_size_slider = ctk.CTkSlider(custom_frame, from_=8, to=24, variable=self.text_size, 
                                        command=self.update_display)
        text_size_slider.pack(fill="x", padx=10, pady=5)
        
        # Color selection
        color_frame = ctk.CTkFrame(custom_frame)
        color_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(color_frame, text="Colors:").pack(anchor="w", pady=2)
        
        # Line color
        ctk.CTkButton(color_frame, text="Line Color", command=self.choose_line_color, 
                     fg_color=self.line_color).pack(fill="x", padx=5, pady=2)
        
        # Text color
        ctk.CTkButton(color_frame, text="Text Color", command=self.choose_text_color, 
                     fg_color=self.text_color).pack(fill="x", padx=5, pady=2)
        
        # Results display
        results_frame = ctk.CTkFrame(settings_frame)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(results_frame, text="Analysis Results", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Reset view button
        ctk.CTkButton(settings_frame, text="Reset View", command=self.reset_view).pack(pady=10)
        
    def setup_display_panel(self, parent):
        """Setup the right display panel"""
        display_frame = ctk.CTkFrame(parent)
        display_frame.pack(side="right", fill="both", expand=True)
        
        # Canvas for image display
        self.canvas = ZoomableCanvas(display_frame, bg="black")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
    
    def load_model(self):
        """Load the pretrained model"""
        try:
            model_path = "pretrained_model/model_last.pth"
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
                
            # Model configuration
            heads = {'hm': 1, 'reg': 2*1, 'wh': 2*4}
            
            # Initialize model
            self.model = spinal_net.SpineNet(heads=heads, pretrained=False, 
                                           down_ratio=4, final_kernel=1, head_conv=256)
            
            # Load pretrained weights
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            state_dict_ = checkpoint['state_dict']
            state_dict = {}
            
            for k in state_dict_:
                if k.startswith('module') and not k.startswith('module_list'):
                    state_dict[k[7:]] = state_dict_[k]
                else:
                    state_dict[k] = state_dict_[k]
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize decoder
            self.decoder = decoder.DecDecoder(K=100, conf_thresh=0.2)
            
            print("Model loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def select_image(self):
        """Select a single image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_files = [file_path]
            self.current_image_index = 0
            self.load_current_image()
            
    def select_folder(self):
        """Select a folder containing images"""
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            # Get all image files in the folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            self.image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(image_extensions)
            ]
            self.image_files.sort()
            
            if self.image_files:
                self.current_image_index = 0
                self.load_current_image()
            else:
                messagebox.showwarning("Warning", "No image files found in the selected folder")
                
    def load_current_image(self):
        """Load the current image"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
            
        try:
            image_path = self.image_files[self.current_image_index]
            self.current_image = cv2.imread(image_path)
            
            if self.current_image is None:
                messagebox.showerror("Error", f"Failed to load image: {image_path}")
                return
                
            # Convert BGR to RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Update display
            self.update_image_display()
            
            # Update image info
            filename = os.path.basename(image_path)
            self.image_info_label.configure(
                text=f"Image {self.current_image_index + 1}/{len(self.image_files)}: {filename}"
            )
            
            # Clear previous results
            self.landmarks = None
            self.cobb_angles = None
            self.vertebrae_info = None
            self.results_text.delete("1.0", "end")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def update_image_display(self):
        """Update the image display"""
        if self.current_image is None:
            return
            
        # Convert to PIL Image
        pil_image = Image.fromarray(self.current_image)
        
        # Set image in canvas
        self.canvas.set_image(pil_image)
        
    def previous_image(self):
        """Go to previous image"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
    def next_image(self):
        """Go to next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
    
    def detect_angles(self):
        """Detect Cobb angles in the current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
            
        # Run detection in a separate thread
        self.detect_button.configure(state="disabled", text="Detecting...")
        thread = threading.Thread(target=self._detect_angles_thread)
        thread.daemon = True
        thread.start()
        
    def _detect_angles_thread(self):
        """Thread function for angle detection"""
        try:
            # Preprocess image
            image_tensor, resized_image = self.preprocess_image()
            
            # Detect landmarks
            pts2 = self.detect_landmarks(image_tensor)
            
            if len(pts2) == 0:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No landmarks detected"))
                return
                
            # Process landmarks for Cobb calculation
            landmarks, pts0 = self.process_landmarks_for_cobb(pts2, resized_image)
            
            # Calculate Cobb angles
            cobb_angles = self.calculate_cobb_angles(landmarks)
            
            # Get vertebral information
            vertebrae_info = self.get_vertebrae_info(landmarks)
            
            # Update results in main thread
            self.root.after(0, lambda: self._update_results(landmarks, cobb_angles, vertebrae_info))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.detect_button.configure(state="normal", text="Detect Cobb Angles"))
            
    def preprocess_image(self):
        """Preprocess image for inference"""
        # Resize image
        input_h, input_w = 1024, 512
        resized_image = cv2.resize(self.current_image, (input_w, input_h))
        
        # Normalize
        out_image = resized_image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        
        # Convert to tensor format
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        
        return out_image, resized_image
        
    def detect_landmarks(self, image_tensor):
        """Detect landmarks in the image"""
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
            hm = output['hm']
            wh = output['wh']
            reg = output['reg']
        
        # Decode landmarks
        pts2 = self.decoder.ctdet_decode(hm, wh, reg)
        return pts2
        
    def process_landmarks_for_cobb(self, pts2, resized_image):
        """Process detected landmarks and prepare for Cobb angle calculation"""
        pts0 = pts2.copy()
        pts0[:,:10] *= 4  # down_ratio
        
        # Sort landmarks by y-coordinate (top to bottom)
        sort_ind = np.argsort(pts0[:,1])
        pts0 = pts0[sort_ind]
        
        # Scale landmarks back to original image dimensions
        original_h, original_w = self.current_image.shape[:2]
        x_index = range(0, 10, 2)
        y_index = range(1, 10, 2)
        pts0[:, x_index] = pts0[:, x_index] / 512 * original_w
        pts0[:, y_index] = pts0[:, y_index] / 1024 * original_h
        
        # Convert to the format expected by Cobb angle calculation
        pr_landmarks = []
        for i, pt in enumerate(pts0):
            pr_landmarks.append(pt[2:4])  # Top-left
            pr_landmarks.append(pt[4:6])  # Top-right
            pr_landmarks.append(pt[6:8])  # Bottom-left
            pr_landmarks.append(pt[8:10]) # Bottom-right
        
        pr_landmarks = np.asarray(pr_landmarks, np.float32)
        return pr_landmarks, pts0
        
    def calculate_cobb_angles(self, landmarks):
        """Calculate Cobb angles using the landmarks"""
        try:
            # Calculate Cobb angles
            cobb_angles = cobb_evaluate.cobb_angle_calc(landmarks, self.current_image.copy())
            return cobb_angles
        except Exception as e:
            print(f"Error calculating Cobb angles: {e}")
            return None
            
    def get_vertebrae_info(self, landmarks):
        """Get vertebral information for each angle"""
        try:
            pts = np.asarray(landmarks, np.float32)
            num_pts = pts.shape[0]
            vnum = num_pts//4-1

            # Calculate midpoints
            mid_p_v = (pts[0::2,:]+pts[1::2,:])/2
            mid_p = []
            for i in range(0, num_pts, 4):
                pt1 = (pts[i,:]+pts[i+2,:])/2
                pt2 = (pts[i+1,:]+pts[i+3,:])/2
                mid_p.append(pt1)
                mid_p.append(pt2)
            mid_p = np.asarray(mid_p, np.float32)

            # Calculate angles
            vec_m = mid_p[1::2,:]-mid_p[0::2,:]
            dot_v = np.matmul(vec_m, np.transpose(vec_m))
            mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]
            mod_v = np.matmul(mod_v, np.transpose(mod_v))
            cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
            angles = np.arccos(cosine_angles)
            
            pos1 = np.argmax(angles, axis=1)
            maxt = np.amax(angles, axis=1)
            pos2 = np.argmax(maxt)
            
            # Determine which vertebrae are used for each angle
            vertebrae_info = {
                'angle1': {'vertebrae': [pos2, pos1[pos2]], 'angle': np.amax(maxt)/np.pi*180},
                'angle2': None,
                'angle3': None
            }
            
            # Check if S-shaped curve
            flag_s = self.is_S_curve(mid_p_v)
            
            if not flag_s:  # not S
                vertebrae_info['angle2'] = {'vertebrae': [0, pos2], 'angle': angles[0, pos2]/np.pi*180}
                vertebrae_info['angle3'] = {'vertebrae': [vnum, pos1[pos2]], 'angle': angles[vnum, pos1[pos2]]/np.pi*180}
            else:
                # S-shaped curve logic
                if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1]) < self.current_image.shape[0]:
                    angle2 = angles[pos2,:(pos2+1)]
                    pos1_1 = np.argmax(angle2)
                    angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
                    pos1_2 = np.argmax(angle3) + pos1[pos2]-1
                    
                    vertebrae_info['angle2'] = {'vertebrae': [pos1_1, pos2], 'angle': np.max(angle2)/np.pi*180}
                    vertebrae_info['angle3'] = {'vertebrae': [pos1_2, pos1[pos2]], 'angle': np.max(angle3)/np.pi*180}
                else:
                    angle2 = angles[pos2,:(pos2+1)]
                    pos1_1 = np.argmax(angle2)
                    angle3 = angles[pos1_1, :(pos1_1+1)]
                    pos1_2 = np.argmax(angle3)
                    
                    vertebrae_info['angle2'] = {'vertebrae': [pos1_1, pos2], 'angle': np.max(angle2)/np.pi*180}
                    vertebrae_info['angle3'] = {'vertebrae': [pos1_2, pos1_1], 'angle': np.max(angle3)/np.pi*180}
            
            return vertebrae_info
            
        except Exception as e:
            print(f"Error getting vertebrae info: {e}")
            return None
            
    def is_S_curve(self, mid_p_v):
        """Check if the spine has an S-shaped curve"""
        ll = []
        num = mid_p_v.shape[0]
        for i in range(num-2):
            term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
            term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
            ll.append(term1-term2)
        ll = np.asarray(ll, np.float32)[:, np.newaxis]
        ll_pair = np.matmul(ll, np.transpose(ll))
        a = sum(sum(ll_pair))
        b = sum(sum(abs(ll_pair)))
        return abs(a-b) >= 1e-4
        
    def _update_results(self, landmarks, cobb_angles, vertebrae_info):
        """Update results display"""
        self.landmarks = landmarks
        self.cobb_angles = cobb_angles
        self.vertebrae_info = vertebrae_info
        
        # Update results text
        self.results_text.delete("1.0", "end")
        
        if cobb_angles is not None:
            results_text = f"=== COBB ANGLE ANALYSIS RESULTS ===\n\n"
            results_text += f"Detected {len(landmarks)} landmarks\n\n"
            
            if vertebrae_info:
                results_text += f"Primary Cobb Angle: {cobb_angles[0]:.1f}°\n"
                results_text += f"Secondary Cobb Angle: {cobb_angles[1]:.1f}°\n"
                results_text += f"Tertiary Cobb Angle: {cobb_angles[2]:.1f}°\n\n"
                
                # Determine severity
                max_angle = max(cobb_angles)
                if max_angle < 10:
                    severity = "Normal"
                elif max_angle < 25:
                    severity = "Mild Scoliosis"
                elif max_angle < 40:
                    severity = "Moderate Scoliosis"
                else:
                    severity = "Severe Scoliosis"
                    
                results_text += f"Maximum Angle: {max_angle:.1f}°\n"
                results_text += f"Diagnosis: {severity}\n"
            else:
                results_text += f"Primary Cobb Angle: {cobb_angles[0]:.1f}°\n"
                results_text += f"Secondary Cobb Angle: {cobb_angles[1]:.1f}°\n"
                results_text += f"Tertiary Cobb Angle: {cobb_angles[2]:.1f}°\n"
        else:
            results_text = "Cobb angle calculation failed"
            
        self.results_text.insert("1.0", results_text)
        
        # Update display
        self.update_display()
        
    def update_display(self):
        """Update the image display with Cobb angle overlays"""
        if self.current_image is None:
            return
            
        # Create a copy of the image for drawing
        display_image = self.current_image.copy()
        
        if self.landmarks is not None and self.cobb_angles is not None:
            # Draw landmarks
            self.draw_landmarks(display_image)
            
            # Draw Cobb angles based on visibility settings
            if self.show_primary.get() and self.vertebrae_info and self.vertebrae_info['angle1']:
                self.draw_cobb_angle(display_image, self.vertebrae_info['angle1'], (255, 0, 0))  # Red
                
            if self.show_secondary.get() and self.vertebrae_info and self.vertebrae_info['angle2']:
                self.draw_cobb_angle(display_image, self.vertebrae_info['angle2'], (255, 0, 255))  # Magenta
                
            if self.show_tertiary.get() and self.vertebrae_info and self.vertebrae_info['angle3']:
                self.draw_cobb_angle(display_image, self.vertebrae_info['angle3'], (0, 255, 255))  # Cyan
        
        # Convert to PIL Image and update canvas
        pil_image = Image.fromarray(display_image)
        self.canvas.set_image(pil_image)
        
    def draw_landmarks(self, image):
        """Draw detected landmarks"""
        for i in range(0, len(self.landmarks), 4):
            pts = self.landmarks[i:i+4]
            for j, pt in enumerate(pts):
                if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                    
            # Draw rectangle connecting the 4 points
            if len(pts) == 4:
                pts_int = [(int(pt[0]), int(pt[1])) for pt in pts if not np.isnan(pt[0]) and not np.isnan(pt[1])]
                if len(pts_int) == 4:
                    cv2.line(image, pts_int[0], pts_int[1], (0, 255, 0), 2)  # Top
                    cv2.line(image, pts_int[1], pts_int[3], (0, 255, 0), 2)  # Right
                    cv2.line(image, pts_int[3], pts_int[2], (0, 255, 0), 2)  # Bottom
                    cv2.line(image, pts_int[2], pts_int[0], (0, 255, 0), 2)  # Left
                    
    def draw_cobb_angle(self, image, angle_info, color):
        """Draw a Cobb angle measurement"""
        vertebrae_pair = angle_info['vertebrae']
        angle_value = angle_info['angle']
        
        # Get vertebral midpoints
        pts = np.asarray(self.landmarks, np.float32)
        mid_p = []
        for i in range(0, len(pts), 4):
            pt1 = (pts[i,:]+pts[i+2,:])/2
            pt2 = (pts[i+1,:]+pts[i+3,:])/2
            mid_p.append(pt1)
            mid_p.append(pt2)
        mid_p = np.asarray(mid_p, np.float32)
        
        v1, v2 = vertebrae_pair
        
        if v1 >= len(mid_p)//2 or v2 >= len(mid_p)//2:
            return
            
        # Get the endplate points for the two vertebrae
        pt1_v1 = mid_p[v1 * 2]  # Top endplate of upper vertebra
        pt2_v1 = mid_p[v1 * 2 + 1]  # Bottom endplate of upper vertebra
        pt1_v2 = mid_p[v2 * 2]  # Top endplate of lower vertebra
        pt2_v2 = mid_p[v2 * 2 + 1]  # Bottom endplate of lower vertebra
        
        # Draw lines parallel to the endplates
        line1_start, line1_end = self.extend_line(pt1_v1, pt2_v1, 100)
        line2_start, line2_end = self.extend_line(pt1_v2, pt2_v2, 100)
        
        # Convert color to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        
        # Draw the parallel lines
        cv2.line(image, line1_start, line1_end, color_bgr, self.line_width.get())
        cv2.line(image, line2_start, line2_end, color_bgr, self.line_width.get())
        
        # Draw angle label
        mid_point = ((line1_start[0] + line2_start[0]) // 2, (line1_start[1] + line2_start[1]) // 2)
        cv2.putText(image, f"{angle_value:.1f}°", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, self.text_size.get() / 20, color_bgr, 2)
        
    def extend_line(self, pt1, pt2, length):
        """Extend a line by a given length"""
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0], pt2[1]
        
        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        mag = np.sqrt(dx*dx + dy*dy)
        
        if mag > 0:
            dx = dx / mag * length
            dy = dy / mag * length
            
            start_x = int(x1 - dx)
            start_y = int(y1 - dy)
            end_x = int(x2 + dx)
            end_y = int(y2 + dy)
            
            return (start_x, start_y), (end_x, end_y)
        return (int(x1), int(y1)), (int(x2), int(y2))
        
    def choose_line_color(self):
        """Choose line color"""
        color = colorchooser.askcolor(title="Choose Line Color", color=self.line_color)
        if color[1]:
            self.line_color = color[1]
            self.update_display()
            
    def choose_text_color(self):
        """Choose text color"""
        color = colorchooser.askcolor(title="Choose Text Color", color=self.text_color)
        if color[1]:
            self.text_color = color[1]
            self.update_display()
            
    def reset_view(self):
        """Reset the canvas view"""
        self.canvas.reset_view()
        
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

def main():
    app = CobbAngleGUI()
    app.run()

if __name__ == "__main__":
    main()
