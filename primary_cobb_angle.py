import torch
import numpy as np
import cv2
import os
import argparse
import signal
import sys
from models import spinal_net
import decoder
import cobb_evaluate
import draw_points

class PrimaryCobbAngleCalculator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        heads = {'hm': 1,  # num_classes = 1 for spinal landmarks
                 'reg': 2*1,
                 'wh': 2*4,}
        
        # Initialize model
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=False,
                                         down_ratio=4,
                                         final_kernel=1,
                                         head_conv=256)
        
        # Load pretrained weights
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize decoder
        self.decoder = decoder.DecDecoder(K=100, conf_thresh=0.2)
        
        # Image processing parameters
        self.input_h = 1024
        self.input_w = 512
        self.down_ratio = 4
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals"""
        print("\n\nReceived termination signal. Cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up OpenCV windows and resources"""
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Ensure windows are closed
            print("OpenCV windows closed.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        print(f'Loaded weights from {model_path}, epoch {checkpoint["epoch"]}')
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        
        self.model.load_state_dict(state_dict, strict=False)
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for inference"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Store original dimensions for scaling back
        self.original_h, self.original_w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.input_w, self.input_h))
        
        # Normalize
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        
        # Convert to tensor format
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, self.input_h, self.input_w)
        out_image = torch.from_numpy(out_image)
        
        return out_image, image
    
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
    
    def process_landmarks_for_cobb(self, pts2, original_image):
        """Process detected landmarks and prepare for Cobb angle calculation"""
        pts0 = pts2.copy()
        pts0[:,:10] *= self.down_ratio
        
        # Sort landmarks by y-coordinate (top to bottom)
        sort_ind = np.argsort(pts0[:,1])
        pts0 = pts0[sort_ind]
        
        # Scale landmarks back to original image dimensions
        x_index = range(0, 10, 2)
        y_index = range(1, 10, 2)
        pts0[:, x_index] = pts0[:, x_index] / self.input_w * self.original_w
        pts0[:, y_index] = pts0[:, y_index] / self.input_h * self.original_h
        
        # Convert to the format expected by Cobb angle calculation
        pr_landmarks = []
        for i, pt in enumerate(pts0):
            pr_landmarks.append(pt[2:4])  # Top-left
            pr_landmarks.append(pt[4:6])  # Top-right
            pr_landmarks.append(pt[6:8])  # Bottom-left
            pr_landmarks.append(pt[8:10]) # Bottom-right
        
        pr_landmarks = np.asarray(pr_landmarks, np.float32)  # [68, 2]
        
        return pr_landmarks, pts0
    
    def calculate_primary_cobb_angle(self, landmarks, image):
        """Calculate only the primary Cobb angle using the landmarks"""
        try:
            # Calculate all Cobb angles first
            cobb_angles = cobb_evaluate.cobb_angle_calc(landmarks, image.copy())
            # Return only the primary angle (first angle)
            return cobb_angles[0] if cobb_angles else None
        except Exception as e:
            print(f"Error calculating Cobb angle: {e}")
            return None
    
    def get_primary_cobb_vertebrae(self, landmarks, image):
        """Get the specific vertebrae used for primary Cobb angle calculation"""
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
        
        # Primary angle uses vertebrae at pos2 and pos1[pos2]
        primary_vertebrae = [pos2, pos1[pos2]]
        
        return primary_vertebrae, mid_p
    
    def draw_primary_cobb_angle(self, image, landmarks, primary_angle):
        """Draw only the primary Cobb angle measurement"""
        # Get the vertebrae used for primary Cobb angle
        primary_vertebrae, mid_p = self.get_primary_cobb_vertebrae(landmarks, image)
        
        # Draw all landmarks in blue first
        self.draw_landmarks_blue(image, landmarks)
        
        # Draw the primary Cobb angle measurement
        v1, v2 = primary_vertebrae
        
        if v1 >= len(mid_p)//2 or v2 >= len(mid_p)//2:
            return
        
        # Get the endplate points for the two vertebrae
        pt1_v1 = mid_p[v1 * 2]  # Top endplate of upper vertebra
        pt2_v1 = mid_p[v1 * 2 + 1]  # Bottom endplate of upper vertebra
        pt1_v2 = mid_p[v2 * 2]  # Top endplate of lower vertebra
        pt2_v2 = mid_p[v2 * 2 + 1]  # Bottom endplate of lower vertebra
        
        # Draw the vertebrae as highlighted rectangles
        self.draw_vertebra_highlight(image, pt1_v1, pt2_v1, (0, 0, 255))  # Red
        self.draw_vertebra_highlight(image, pt1_v2, pt2_v2, (0, 0, 255))  # Red
        
        # Draw lines parallel to the endplates (extended)
        line1_start, line1_end = self.extend_line(pt1_v1, pt2_v1, 150)
        line2_start, line2_end = self.extend_line(pt1_v2, pt2_v2, 150)
        
        # Draw the parallel lines
        cv2.line(image, line1_start, line1_end, (0, 0, 255), 3)  # Red
        cv2.line(image, line2_start, line2_end, (0, 0, 255), 3)  # Red
        
        # Calculate perpendicular lines that will intersect
        perp1_start, perp1_end, perp2_start, perp2_end, intersection = self.calculate_intersecting_perpendiculars(
            line1_start, line1_end, line2_start, line2_end)
        
        # Draw the extended perpendicular lines until intersection
        cv2.line(image, perp1_start, perp1_end, (0, 0, 255), 2)  # Red
        cv2.line(image, perp2_start, perp2_end, (0, 0, 255), 2)  # Red
        
        # Draw intersection point
        if intersection:
            cv2.circle(image, intersection, 5, (0, 0, 255), -1)  # Red
            cv2.circle(image, intersection, 5, (255, 255, 255), 2)  # White border
            
            # Draw angle arc at intersection
            self.draw_angle_arc_at_intersection(image, perp1_start, perp1_end, perp2_start, perp2_end, intersection, (0, 0, 255))
            
            # Add angle label
            self.draw_angle_label(image, intersection, primary_angle, (0, 0, 255))
    
    def draw_vertebra_highlight(self, canvas, pt1, pt2, color):
        """Draw a highlighted vertebra for Cobb angle measurement"""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # Calculate rectangle dimensions
        width = 25
        height = abs(y2 - y1) + 15
        
        # Draw filled rectangle with border
        cv2.rectangle(canvas, (x1-width//2, y1-height//2), (x1+width//2, y1+height//2), color, -1)
        cv2.rectangle(canvas, (x1-width//2, y1-height//2), (x1+width//2, y1+height//2), (255, 255, 255), 2)
        
        cv2.rectangle(canvas, (x2-width//2, y2-height//2), (x2+width//2, y2+height//2), color, -1)
        cv2.rectangle(canvas, (x2-width//2, y2-height//2), (x2+width//2, y2+height//2), (255, 255, 255), 2)
    
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
    
    def calculate_intersecting_perpendiculars(self, line1_start, line1_end, line2_start, line2_end):
        """Calculate perpendicular lines that will intersect"""
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end
        
        # Calculate midpoints of both lines
        mid1_x = (x1 + x2) // 2
        mid1_y = (y1 + y2) // 2
        mid2_x = (x3 + x4) // 2
        mid2_y = (y3 + y4) // 2
        
        # Calculate perpendicular directions for both lines
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        # Perpendicular vectors
        perp1_x = -dy1
        perp1_y = dx1
        perp2_x = -dy2
        perp2_y = dx2
        
        # Normalize perpendicular vectors
        mag1 = np.sqrt(perp1_x*perp1_x + perp1_y*perp1_y)
        mag2 = np.sqrt(perp2_x*perp2_x + perp2_y*perp2_y)
        
        if mag1 > 0 and mag2 > 0:
            perp1_x = perp1_x / mag1
            perp1_y = perp1_y / mag1
            perp2_x = perp2_x / mag2
            perp2_y = perp2_y / mag2
            
            # Calculate intersection point
            intersection = self.calculate_line_intersection(
                mid1_x, mid1_y, perp1_x, perp1_y,
                mid2_x, mid2_y, perp2_x, perp2_y
            )
            
            if intersection:
                # Extend perpendicular lines to intersection
                # Line 1 perpendicular
                perp1_start = (int(mid1_x - perp1_x * 100), int(mid1_y - perp1_y * 100))
                perp1_end = intersection
                
                # Line 2 perpendicular
                perp2_start = (int(mid2_x - perp2_x * 100), int(mid2_y - perp2_y * 100))
                perp2_end = intersection
                
                return perp1_start, perp1_end, perp2_start, perp2_end, intersection
        
        # Fallback: return simple perpendicular lines
        perp1_start = (int(mid1_x - perp1_x * 50), int(mid1_y - perp1_y * 50))
        perp1_end = (int(mid1_x + perp1_x * 50), int(mid1_y + perp1_y * 50))
        perp2_start = (int(mid2_x - perp2_x * 50), int(mid2_y - perp2_y * 50))
        perp2_end = (int(mid2_x + perp2_x * 50), int(mid2_y + perp2_y * 50))
        
        return perp1_start, perp1_end, perp2_start, perp2_end, None
    
    def calculate_line_intersection(self, x1, y1, dx1, dy1, x2, y2, dx2, dy2):
        """Calculate intersection of two lines given point and direction"""
        # Line 1: (x1, y1) + t1 * (dx1, dy1)
        # Line 2: (x2, y2) + t2 * (dx2, dy2)
        
        # Solve for t1 and t2 where lines intersect
        # x1 + t1*dx1 = x2 + t2*dx2
        # y1 + t1*dy1 = y2 + t2*dy2
        
        # Matrix equation: [dx1 -dx2; dy1 -dy2] * [t1; t2] = [x2-x1; y2-y1]
        det = dx1 * (-dy2) - dy1 * (-dx2)
        
        if abs(det) < 1e-6:  # Lines are parallel
            return None
        
        t1 = ((x2 - x1) * (-dy2) - (y2 - y1) * (-dx2)) / det
        
        # Calculate intersection point
        intersect_x = int(x1 + t1 * dx1)
        intersect_y = int(y1 + t1 * dy1)
        
        return (intersect_x, intersect_y)
    
    def draw_angle_arc_at_intersection(self, canvas, perp1_start, perp1_end, perp2_start, perp2_end, intersection, color):
        """Draw an arc showing the Cobb angle at the intersection point"""
        if intersection is None:
            return
            
        # Calculate angle between the two perpendicular lines
        vec1 = np.array([perp1_end[0] - perp1_start[0], perp1_end[1] - perp1_start[1]])
        vec2 = np.array([perp2_end[0] - perp2_start[0], perp2_end[1] - perp2_start[1]])
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Draw arc
        radius = 40
        start_angle = 0
        end_angle = int(angle * 180 / np.pi)
        
        # Draw arc using multiple line segments
        for i in range(start_angle, end_angle, 3):
            angle1 = i * np.pi / 180
            angle2 = (i + 3) * np.pi / 180
            
            x1 = intersection[0] + int(radius * np.cos(angle1))
            y1 = intersection[1] + int(radius * np.sin(angle1))
            x2 = intersection[0] + int(radius * np.cos(angle2))
            y2 = intersection[1] + int(radius * np.sin(angle2))
            
            cv2.line(canvas, (x1, y1), (x2, y2), color, 3)
    
    def draw_angle_label(self, canvas, center, angle_value, color):
        """Draw a clean angle value label"""
        if center is None:
            return
            
        text = f"{angle_value:.1f}°"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position text near the angle with better spacing
        text_x = center[0] + 15
        text_y = center[1] - 5
        
        # Draw a small, clean background circle
        cv2.circle(canvas, (text_x + text_width//2, text_y - text_height//2), 
                  max(text_width//2 + 8, text_height//2 + 5), (0, 0, 0), -1)
        cv2.circle(canvas, (text_x + text_width//2, text_y - text_height//2), 
                  max(text_width//2 + 8, text_height//2 + 5), color, 2)
        
        # Draw text
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color, thickness)
    
    def draw_results_panel(self, image, primary_angle, num_landmarks):
        """Draw a clean results panel with primary Cobb angle information"""
        # Create a semi-transparent background panel
        panel_bg = np.zeros((200, 400, 3), dtype=np.uint8)
        panel_bg[:] = (20, 20, 20)  # Dark gray background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        y_offset = 40
        
        # Title
        cv2.putText(panel_bg, "PRIMARY COBB ANGLE", (20, y_offset), font, 1.0, (255, 255, 255), thickness)
        y_offset += 50
        
        # Primary angle
        cv2.putText(panel_bg, f"Primary Angle: {primary_angle:.1f}°", (20, y_offset), font, 0.9, (0, 0, 255), thickness)
        y_offset += 40
        
        # Determine severity
        if primary_angle < 10:
            severity = "Normal"
            severity_color = (0, 255, 0)  # Green
        elif primary_angle < 25:
            severity = "Mild Scoliosis"
            severity_color = (0, 255, 255)  # Yellow
        elif primary_angle < 40:
            severity = "Moderate Scoliosis"
            severity_color = (0, 165, 255)  # Orange
        else:
            severity = "Severe Scoliosis"
            severity_color = (0, 0, 255)  # Red
        
        # Severity
        cv2.putText(panel_bg, f"Diagnosis: {severity}", (20, y_offset), font, 0.8, severity_color, thickness)
        y_offset += 40
        
        # Additional info
        cv2.putText(panel_bg, f"Landmarks Detected: {num_landmarks}", (20, y_offset), font, 0.7, (200, 200, 200), thickness)
        
        # Place panel on image (top-left corner)
        image[20:220, 20:420] = panel_bg
    
    def draw_landmarks_blue(self, image, landmarks):
        """Draw all detected landmarks in blue"""
        colors = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]  # All blue
        
        for i in range(0, len(landmarks), 4):
            pts = landmarks[i:i+4]
            for j, pt in enumerate(pts):
                if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 5, colors[j], -1)
                    cv2.putText(image, f'{j+1}', (int(pt[0])+8, int(pt[1])+8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw rectangle connecting the 4 points
            if len(pts) == 4:
                pts_int = [(int(pt[0]), int(pt[1])) for pt in pts if not np.isnan(pt[0]) and not np.isnan(pt[1])]
                if len(pts_int) == 4:
                    # Draw rectangle
                    cv2.line(image, pts_int[0], pts_int[1], (255, 0, 0), 2)  # Top
                    cv2.line(image, pts_int[1], pts_int[3], (255, 0, 0), 2)  # Right
                    cv2.line(image, pts_int[3], pts_int[2], (255, 0, 0), 2)  # Bottom
                    cv2.line(image, pts_int[2], pts_int[0], (255, 0, 0), 2)  # Left
    
    def display_results(self, original_image, landmarks, primary_angle, max_width=1000, max_height=800):
        """Display the results with primary Cobb angle information"""
        # Create a copy for visualization
        result_image = original_image.copy()
        
        # Draw the primary Cobb angle
        if primary_angle is not None:
            self.draw_primary_cobb_angle(result_image, landmarks, primary_angle)
            # Add results panel
            self.draw_results_panel(result_image, primary_angle, len(landmarks))
        else:
            # Simple error message
            self.draw_error_panel(result_image, len(landmarks))
        
        # Resize for display
        height, width = result_image.shape[:2]
        scale_x = max_width / width
        scale_y = max_height / height
        scale = min(scale_x, scale_y, 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(result_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            display_image = result_image
        
        # Display image
        cv2.imshow('Primary Cobb Angle Analysis', display_image)
        
        # Wait for key press with proper cleanup
        print("\nPress 'q' to quit, 's' to save image, or any other key to continue...")
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            self.cleanup()
            sys.exit(0)
        elif key == ord('s'):
            # Save the displayed image
            save_path = f"primary_cobb_analysis_saved_{cv2.getTickCount()}.jpg"
            cv2.imwrite(save_path, display_image)
            print(f"Image saved as: {save_path}")
        
        return result_image
    
    def draw_error_panel(self, image, num_landmarks):
        """Draw a simple error panel when Cobb angle calculation fails"""
        # Create a simple error panel
        panel_bg = np.zeros((120, 300, 3), dtype=np.uint8)
        panel_bg[:] = (20, 20, 20)  # Dark gray background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        cv2.putText(panel_bg, "PRIMARY COBB ANGLE", (20, 40), font, 0.9, (255, 255, 255), thickness)
        cv2.putText(panel_bg, "Calculation Failed", (20, 70), font, 0.8, (0, 0, 255), thickness)
        cv2.putText(panel_bg, f"Landmarks: {num_landmarks}", (20, 100), font, 0.7, (200, 200, 200), thickness)
        
        # Place panel on image (top-left corner)
        image[20:140, 20:320] = panel_bg
    
    def analyze_image(self, image_path, save_output=True, output_dir='primary_cobb_results'):
        """Complete analysis: detect landmarks, calculate primary Cobb angle, and display results"""
        print(f"Analyzing image: {image_path}")
        
        # Preprocess image
        image_tensor, resized_image = self.preprocess_image(image_path)
        
        # Detect landmarks
        pts2 = self.detect_landmarks(image_tensor)
        
        if len(pts2) == 0:
            print("No landmarks detected!")
            return None, None, None
        
        # Process landmarks for Cobb calculation
        landmarks, pts0 = self.process_landmarks_for_cobb(pts2, resized_image)
        
        # Load original image for final visualization
        original_image = cv2.imread(image_path)
        
        # Calculate primary Cobb angle
        primary_angle = self.calculate_primary_cobb_angle(landmarks, original_image)
        
        # Display results
        result_image = self.display_results(original_image, landmarks, primary_angle)
        
        # Print results
        print(f"\n=== PRIMARY COBB ANGLE ANALYSIS RESULTS ===")
        print(f"Detected {len(landmarks)} landmarks")
        if primary_angle is not None:
            print(f"Primary Cobb Angle: {primary_angle:.1f}°")
            
            # Determine severity
            if primary_angle < 10:
                severity = "Normal"
            elif primary_angle < 25:
                severity = "Mild Scoliosis"
            elif primary_angle < 40:
                severity = "Moderate Scoliosis"
            else:
                severity = "Severe Scoliosis"
            
            print(f"Diagnosis: {severity}")
        else:
            print("Primary Cobb angle calculation failed")
        
        # Save results if requested
        if save_output:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save result image
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_primary_cobb_analysis.jpg"), result_image)
            
            # Save landmarks coordinates
            np.save(os.path.join(output_dir, f"{base_name}_landmarks.npy"), landmarks)
            
            # Save primary Cobb angle
            if primary_angle is not None:
                np.save(os.path.join(output_dir, f"{base_name}_primary_cobb_angle.npy"), primary_angle)
                
                # Save text report
                with open(os.path.join(output_dir, f"{base_name}_primary_cobb_report.txt"), 'w') as f:
                    f.write("PRIMARY COBB ANGLE ANALYSIS REPORT\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Image: {image_path}\n")
                    try:
                        from datetime import datetime
                        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    except:
                        f.write(f"Analysis Date: {os.popen('date').read().strip() if os.name != 'nt' else 'Unknown'}\n")
                    f.write(f"Total Landmarks Detected: {len(landmarks)}\n\n")
                    f.write("PRIMARY COBB ANGLE MEASUREMENT:\n")
                    f.write("-" * 35 + "\n")
                    f.write(f"Primary Angle: {primary_angle:.1f}°\n\n")
                    
                    # Determine severity
                    if primary_angle < 10:
                        severity = "Normal"
                    elif primary_angle < 25:
                        severity = "Mild Scoliosis"
                    elif primary_angle < 40:
                        severity = "Moderate Scoliosis"
                    else:
                        severity = "Severe Scoliosis"
                    
                    f.write("CLINICAL ASSESSMENT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Primary Cobb Angle: {primary_angle:.1f}°\n")
                    f.write(f"Severity Classification: {severity}\n\n")
                    
                    f.write("SEVERITY CLASSIFICATION:\n")
                    f.write("-" * 25 + "\n")
                    f.write("• < 10°: Normal spine (no treatment needed)\n")
                    f.write("• 10-25°: Mild scoliosis (monitor for progression)\n")
                    f.write("• 25-40°: Moderate scoliosis (consider bracing)\n")
                    f.write("• > 40°: Severe scoliosis (surgical consultation)\n\n")
                    
                    f.write("IMPORTANT NOTES:\n")
                    f.write("-" * 15 + "\n")
                    f.write("• This analysis shows only the primary (largest) Cobb angle\n")
                    f.write("• The primary angle is the most clinically significant measurement\n")
                    f.write("• This analysis is for research and educational purposes\n")
                    f.write("• Clinical decisions should be made by qualified healthcare providers\n")
            
            print(f"\nResults saved to {output_dir}/")
        
        return landmarks, primary_angle, result_image

def main():
    parser = argparse.ArgumentParser(description='Calculate primary Cobb angle from spinal X-ray images')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input X-ray image')
    parser.add_argument('--model_path', type=str, default='pretrained_model/model_last.pth', 
                       help='Path to the pretrained model')
    parser.add_argument('--output_dir', type=str, default='primary_cobb_results', help='Output directory for results')
    parser.add_argument('--no_save', action='store_true', help='Do not save output files')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
    
    # Initialize calculator
    calculator = None
    try:
        calculator = PrimaryCobbAngleCalculator(args.model_path, args.device)
        
        # Analyze image
        landmarks, primary_angle, result_image = calculator.analyze_image(
            args.image_path, 
            save_output=not args.no_save,
            output_dir=args.output_dir
        )
        
        if landmarks is not None:
            print("Analysis completed successfully!")
        else:
            print("Analysis failed - no landmarks detected")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        if calculator:
            calculator.cleanup()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        if calculator:
            calculator.cleanup()
        print("Script terminated.")

if __name__ == "__main__":
    main()
