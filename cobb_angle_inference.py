import torch
import numpy as np
import cv2
import os
import argparse
from models import spinal_net
import decoder
import cobb_evaluate
import draw_points

class CobbAngleCalculator:
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
    
    def calculate_cobb_angles(self, landmarks, image):
        """Calculate Cobb angles using the landmarks"""
        try:
            # Calculate Cobb angles
            cobb_angles = cobb_evaluate.cobb_angle_calc(landmarks, image.copy())
            return cobb_angles
        except Exception as e:
            print(f"Error calculating Cobb angles: {e}")
            return None
    
    def display_results(self, original_image, landmarks, cobb_angles, max_width=1000, max_height=800):
        """Display the results with Cobb angle information"""
        # Create a copy for visualization
        result_image = original_image.copy()
        
        # Draw all landmarks in blue first
        self.draw_landmarks_blue(result_image, landmarks)
        
        # Calculate Cobb angles and get the specific vertebrae used
        if cobb_angles is not None:
            # Get the vertebrae used for Cobb angle calculation
            cobb_vertebrae = self.get_cobb_vertebrae(landmarks, result_image)
            # Draw Cobb angles with clear labeling
            self.draw_cobb_angles_with_labels(result_image, landmarks, cobb_angles, cobb_vertebrae)
        
        # Add text with Cobb angle information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)     # Black background
        
        # Create clean text overlay
        if cobb_angles is not None:
            # Determine severity based on maximum Cobb angle
            max_angle = max(cobb_angles)
            if max_angle < 10:
                severity = "Normal"
                severity_color = (0, 255, 0)  # Green
            elif max_angle < 25:
                severity = "Mild Scoliosis"
                severity_color = (0, 255, 255)  # Yellow
            elif max_angle < 40:
                severity = "Moderate Scoliosis"
                severity_color = (0, 165, 255)  # Orange
            else:
                severity = "Severe Scoliosis"
                severity_color = (0, 0, 255)  # Red
            
            # Get vertebral information for display
            vertebral_info = None
            try:
                vertebrae_info, mid_p = self.get_cobb_vertebrae(landmarks, result_image)
                vertebral_info = {
                    'primary': self.get_vertebral_names(vertebrae_info['angle1']['vertebrae']) if vertebrae_info['angle1'] else "N/A",
                    'secondary': self.get_vertebral_names(vertebrae_info['angle2']['vertebrae']) if vertebrae_info['angle2'] else "N/A",
                    'tertiary': self.get_vertebral_names(vertebrae_info['angle3']['vertebrae']) if vertebrae_info['angle3'] else "N/A"
                }
            except:
                vertebral_info = None
            
            # Create a clean results panel
            self.draw_results_panel(result_image, cobb_angles, max_angle, severity, severity_color, len(landmarks), vertebral_info)
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
        cv2.imshow('Cobb Angle Analysis', display_image)
        
        return result_image
    
    def get_cobb_vertebrae(self, landmarks, image):
        """Get the specific vertebrae used for Cobb angle calculation"""
        # Replicate the cobb_evaluate logic to get the exact vertebrae used
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
            if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1]) < image.shape[0]:
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
        
        return vertebrae_info, mid_p
    
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
    
    def draw_cobb_angles_with_labels(self, image, landmarks, cobb_angles, cobb_vertebrae):
        """Draw Cobb angles with proper measurement lines directly on the image"""
        # Get the vertebrae information
        vertebrae_info, mid_p = self.get_cobb_vertebrae(landmarks, image)
        
        # Calculate required canvas size for the measurement lines
        h, w = image.shape[:2]
        max_extension = 200  # Maximum extension needed for measurement lines
        
        # Check if we need to expand the canvas
        need_expansion = False
        for angle_name in ['angle1', 'angle2', 'angle3']:
            if vertebrae_info[angle_name] is not None:
                vertebrae_pair = vertebrae_info[angle_name]['vertebrae']
                if self.check_line_extension_needed(mid_p, vertebrae_pair, w, h, max_extension):
                    need_expansion = True
                    break
        
        if need_expansion:
            # Create expanded canvas with black background
            canvas_h = h + 2 * max_extension
            canvas_w = w + 2 * max_extension
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Place the original image in the center
            canvas[max_extension:max_extension+h, max_extension:max_extension+w] = image
            
            # Draw Cobb angle measurement lines on expanded canvas
            colors = [(0, 0, 255), (255, 0, 255), (0, 255, 255)]  # Red, Magenta, Cyan
            angle_names = ['angle1', 'angle2', 'angle3']
            
            for i, (angle_name, angle_value) in enumerate(zip(angle_names, cobb_angles)):
                if vertebrae_info[angle_name] is not None:
                    color = colors[i]
                    vertebrae_pair = vertebrae_info[angle_name]['vertebrae']
                    
                    # Draw the Cobb angle measurement lines
                    self.draw_cobb_measurement_lines(canvas, mid_p, vertebrae_pair, color, angle_value, 
                                                   offset_x=max_extension, offset_y=max_extension)
            
            # Add legend on expanded canvas
            self.draw_legend(canvas, colors, offset_x=max_extension, offset_y=50)
            
            # Copy the expanded result back to the original image
            image[:] = canvas[:h, :w]
        else:
            # Draw directly on the original image if no expansion needed
            colors = [(0, 0, 255), (255, 0, 255), (0, 255, 255)]  # Red, Magenta, Cyan
            angle_names = ['angle1', 'angle2', 'angle3']
            
            for i, (angle_name, angle_value) in enumerate(zip(angle_names, cobb_angles)):
                if vertebrae_info[angle_name] is not None:
                    color = colors[i]
                    vertebrae_pair = vertebrae_info[angle_name]['vertebrae']
                    
                    # Draw the Cobb angle measurement lines
                    self.draw_cobb_measurement_lines(image, mid_p, vertebrae_pair, color, angle_value, 
                                                offset_x=0, offset_y=0)
    
    def check_line_extension_needed(self, mid_p, vertebrae_pair, img_w, img_h, max_extension):
        """Check if measurement lines will extend beyond image boundaries"""
        v1, v2 = vertebrae_pair
        
        if v1 >= len(mid_p)//2 or v2 >= len(mid_p)//2:
            return False
        
        # Get endplate points
        pt1_v1 = mid_p[v1 * 2]
        pt2_v1 = mid_p[v1 * 2 + 1]
        pt1_v2 = mid_p[v2 * 2]
        pt2_v2 = mid_p[v2 * 2 + 1]
        
        # Calculate extended line endpoints
        line1_start, line1_end = self.extend_line(pt1_v1, pt2_v1, max_extension, 0, 0)
        line2_start, line2_end = self.extend_line(pt1_v2, pt2_v2, max_extension, 0, 0)
        
        # Check if any point extends beyond image boundaries
        for point in [line1_start, line1_end, line2_start, line2_end]:
            if point[0] < 0 or point[0] >= img_w or point[1] < 0 or point[1] >= img_h:
                return True
        
        return False
    
    def draw_cobb_measurement_lines(self, canvas, mid_p, vertebrae_pair, color, angle_value, offset_x=0, offset_y=0):
        """Draw the proper Cobb angle measurement lines with endplate parallels and perpendiculars"""
        v1, v2 = vertebrae_pair
        
        if v1 >= len(mid_p)//2 or v2 >= len(mid_p)//2:
            return
        
        # Get the endplate points for the two vertebrae
        pt1_v1 = mid_p[v1 * 2]  # Top endplate of upper vertebra
        pt2_v1 = mid_p[v1 * 2 + 1]  # Bottom endplate of upper vertebra
        pt1_v2 = mid_p[v2 * 2]  # Top endplate of lower vertebra
        pt2_v2 = mid_p[v2 * 2 + 1]  # Bottom endplate of lower vertebra
        
        # Draw the vertebrae as highlighted rectangles
        self.draw_vertebra_highlight(canvas, pt1_v1, pt2_v1, color, offset_x, offset_y)
        self.draw_vertebra_highlight(canvas, pt1_v2, pt2_v2, color, offset_x, offset_y)
        
        # Draw lines parallel to the endplates (extended)
        line1_start, line1_end = self.extend_line(pt1_v1, pt2_v1, 150, offset_x, offset_y)
        line2_start, line2_end = self.extend_line(pt1_v2, pt2_v2, 150, offset_x, offset_y)
        
        # Draw the parallel lines
        cv2.line(canvas, line1_start, line1_end, color, 3)
        cv2.line(canvas, line2_start, line2_end, color, 3)
        
        # Calculate perpendicular lines that will intersect
        perp1_start, perp1_end, perp2_start, perp2_end, intersection = self.calculate_intersecting_perpendiculars(
            line1_start, line1_end, line2_start, line2_end, offset_x, offset_y)
        
        # Draw the extended perpendicular lines until intersection
        cv2.line(canvas, perp1_start, perp1_end, color, 2)
        cv2.line(canvas, perp2_start, perp2_end, color, 2)
        
        # Draw intersection point
        if intersection:
            cv2.circle(canvas, intersection, 5, color, -1)
            cv2.circle(canvas, intersection, 5, (255, 255, 255), 2)
            
            # Draw angle arc at intersection
            self.draw_angle_arc_at_intersection(canvas, perp1_start, perp1_end, perp2_start, perp2_end, intersection, color)
            
            # Add angle label
            self.draw_angle_label(canvas, intersection, angle_value, color, offset_x, offset_y)
    
    def draw_vertebra_highlight(self, canvas, pt1, pt2, color, offset_x, offset_y):
        """Draw a highlighted vertebra for Cobb angle measurement"""
        x1, y1 = int(pt1[0]) + offset_x, int(pt1[1]) + offset_y
        x2, y2 = int(pt2[0]) + offset_x, int(pt2[1]) + offset_y
        
        # Calculate rectangle dimensions
        width = 25
        height = abs(y2 - y1) + 15
        
        # Draw filled rectangle with border
        cv2.rectangle(canvas, (x1-width//2, y1-height//2), (x1+width//2, y1+height//2), color, -1)
        cv2.rectangle(canvas, (x1-width//2, y1-height//2), (x1+width//2, y1+height//2), (255, 255, 255), 2)
        
        cv2.rectangle(canvas, (x2-width//2, y2-height//2), (x2+width//2, y2+height//2), color, -1)
        cv2.rectangle(canvas, (x2-width//2, y2-height//2), (x2+width//2, y2+height//2), (255, 255, 255), 2)
    
    def draw_cobb_angle_vertebrae(self, canvas, mid_p, vertebrae_pair, color, angle_value, offset_x=0, offset_y=0):
        """Draw the proper Cobb angle measurement like in medical diagrams"""
        v1, v2 = vertebrae_pair
        
        if v1 >= len(mid_p)//2 or v2 >= len(mid_p)//2:
            return
        
        # Get the endplate points for the two vertebrae
        # For vertebra v1 (upper end vertebra)
        pt1_v1 = mid_p[v1 * 2]  # Top endplate
        pt2_v1 = mid_p[v1 * 2 + 1]  # Bottom endplate
        
        # For vertebra v2 (lower end vertebra)  
        pt1_v2 = mid_p[v2 * 2]  # Top endplate
        pt2_v2 = mid_p[v2 * 2 + 1]  # Bottom endplate
        
        # Draw the vertebrae as rectangles
        self.draw_vertebra_rectangle(canvas, pt1_v1, pt2_v1, color, offset_x, offset_y)
        self.draw_vertebra_rectangle(canvas, pt1_v2, pt2_v2, color, offset_x, offset_y)
        
        # Draw lines parallel to the endplates
        # Line 1: parallel to upper end vertebra's superior endplate
        line1_start, line1_end = self.extend_line(pt1_v1, pt2_v1, 100, offset_x, offset_y)
        cv2.line(canvas, line1_start, line1_end, color, 3)
        
        # Line 2: parallel to lower end vertebra's inferior endplate  
        line2_start, line2_end = self.extend_line(pt1_v2, pt2_v2, 100, offset_x, offset_y)
        cv2.line(canvas, line2_start, line2_end, color, 3)
        
        # Draw perpendicular lines from the parallel lines
        perp1_start, perp1_end = self.draw_perpendicular_line(line1_start, line1_end, 50, offset_x, offset_y)
        perp2_start, perp2_end = self.draw_perpendicular_line(line2_start, line2_end, 50, offset_x, offset_y)
        
        cv2.line(canvas, perp1_start, perp1_end, color, 2)
        cv2.line(canvas, perp2_start, perp2_end, color, 2)
        
        # Calculate and draw the angle
        angle_center = self.calculate_angle_intersection(perp1_start, perp1_end, perp2_start, perp2_end)
        if angle_center:
            # Draw angle arc
            self.draw_angle_arc(canvas, perp1_start, perp1_end, perp2_start, perp2_end, angle_center, color, offset_x, offset_y)
            
            # Add angle label
            self.draw_angle_label(canvas, angle_center, angle_value, color, offset_x, offset_y)
    
    def draw_vertebra_rectangle(self, canvas, pt1, pt2, color, offset_x, offset_y):
        """Draw a vertebra as a rectangle"""
        x1, y1 = int(pt1[0]) + offset_x, int(pt1[1]) + offset_y
        x2, y2 = int(pt2[0]) + offset_x, int(pt2[1]) + offset_y
        
        # Calculate rectangle dimensions
        width = 20
        height = abs(y2 - y1) + 10
        
        # Draw rectangle
        cv2.rectangle(canvas, (x1-width//2, y1-height//2), (x1+width//2, y1+height//2), color, 2)
        cv2.rectangle(canvas, (x2-width//2, y2-height//2), (x2+width//2, y2+height//2), color, 2)
    
    def extend_line(self, pt1, pt2, length, offset_x, offset_y):
        """Extend a line by a given length"""
        x1, y1 = pt1[0] + offset_x, pt1[1] + offset_y
        x2, y2 = pt2[0] + offset_x, pt2[1] + offset_y
        
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
    
    def calculate_intersecting_perpendiculars(self, line1_start, line1_end, line2_start, line2_end, offset_x, offset_y):
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
    
    def draw_perpendicular_line(self, line_start, line_end, length, offset_x, offset_y):
        """Draw a perpendicular line from the midpoint of a line"""
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # Calculate perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        
        # Perpendicular vector
        perp_x = -dy
        perp_y = dx
        
        # Normalize
        mag = np.sqrt(perp_x*perp_x + perp_y*perp_y)
        if mag > 0:
            perp_x = perp_x / mag * length
            perp_y = perp_y / mag * length
            
            start_x = int(mid_x - perp_x)
            start_y = int(mid_y - perp_y)
            end_x = int(mid_x + perp_x)
            end_y = int(mid_y + perp_y)
            
            return (start_x, start_y), (end_x, end_y)
        return (mid_x, mid_y), (mid_x, mid_y)
    
    def calculate_angle_intersection(self, perp1_start, perp1_end, perp2_start, perp2_end):
        """Calculate the intersection point of two perpendicular lines"""
        # Simplified intersection calculation
        x1, y1 = perp1_start
        x2, y2 = perp1_end
        x3, y3 = perp2_start
        x4, y4 = perp2_end
        
        # Calculate intersection
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        intersect_x = int(x1 + t * (x2 - x1))
        intersect_y = int(y1 + t * (y2 - y1))
        
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
    
    def draw_angle_arc(self, canvas, perp1_start, perp1_end, perp2_start, perp2_end, center, color, offset_x, offset_y):
        """Draw an arc showing the Cobb angle"""
        if center is None:
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
        radius = 30
        start_angle = 0
        end_angle = int(angle * 180 / np.pi)
        
        # Draw arc using multiple line segments
        for i in range(start_angle, end_angle, 5):
            angle1 = i * np.pi / 180
            angle2 = (i + 5) * np.pi / 180
            
            x1 = center[0] + int(radius * np.cos(angle1))
            y1 = center[1] + int(radius * np.sin(angle1))
            x2 = center[0] + int(radius * np.cos(angle2))
            y2 = center[1] + int(radius * np.sin(angle2))
            
            cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
    
    def draw_angle_label(self, canvas, center, angle_value, color, offset_x, offset_y):
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
    
    def draw_legend(self, canvas, colors, offset_x=0, offset_y=0):
        """Draw a clean, organized legend"""
        # Draw a semi-transparent background for the legend
        legend_bg = np.zeros((200, 350, 3), dtype=np.uint8)
        legend_bg[:] = (20, 20, 20)  # Dark gray background
        
        # Create legend content
        legend_items = [
            ("COBB ANGLE ANALYSIS", (255, 255, 255), 1.2),
            ("", (255, 255, 255), 0.8),  # Spacing
            ("Primary Angle (Red)", colors[0], 0.8),
            ("Secondary Angle (Magenta)", colors[1], 0.8),
            ("Tertiary Angle (Cyan)", colors[2], 0.8),
            ("", (255, 255, 255), 0.8),  # Spacing
            ("Detected Vertebrae (Blue)", (255, 0, 0), 0.8)
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        y_offset = 30
        
        for text, color, scale in legend_items:
            if text:  # Skip empty spacing lines
                cv2.putText(legend_bg, text, (20, y_offset), font, scale, color, thickness)
            y_offset += 35
        
        # Place legend on canvas
        canvas[offset_y:offset_y+200, offset_x:offset_x+350] = legend_bg
    
    def draw_results_panel(self, image, cobb_angles, max_angle, severity, severity_color, num_landmarks, vertebral_info=None):
        """Draw a clean results panel with Cobb angle information"""
        # Create a semi-transparent background panel
        panel_bg = np.zeros((320, 450, 3), dtype=np.uint8)  # Increased size for vertebral info
        panel_bg[:] = (20, 20, 20)  # Dark gray background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        y_offset = 40
        
        # Title
        cv2.putText(panel_bg, "COBB ANGLE RESULTS", (20, y_offset), font, 1.0, (255, 255, 255), thickness)
        y_offset += 50
        
        # Cobb angles with vertebral information
        if vertebral_info:
            cv2.putText(panel_bg, f"Primary: {cobb_angles[0]:.1f}° ({vertebral_info['primary']})", (20, y_offset), font, 0.7, (0, 0, 255), thickness)
            y_offset += 35
            cv2.putText(panel_bg, f"Secondary: {cobb_angles[1]:.1f}° ({vertebral_info['secondary']})", (20, y_offset), font, 0.7, (255, 0, 255), thickness)
            y_offset += 35
            cv2.putText(panel_bg, f"Tertiary: {cobb_angles[2]:.1f}° ({vertebral_info['tertiary']})", (20, y_offset), font, 0.7, (0, 255, 255), thickness)
        else:
            cv2.putText(panel_bg, f"Primary: {cobb_angles[0]:.1f}°", (20, y_offset), font, 0.8, (0, 0, 255), thickness)
            y_offset += 35
            cv2.putText(panel_bg, f"Secondary: {cobb_angles[1]:.1f}°", (20, y_offset), font, 0.8, (255, 0, 255), thickness)
            y_offset += 35
            cv2.putText(panel_bg, f"Tertiary: {cobb_angles[2]:.1f}°", (20, y_offset), font, 0.8, (0, 255, 255), thickness)
        y_offset += 50
        
        # Maximum angle and severity
        cv2.putText(panel_bg, f"Maximum Angle: {max_angle:.1f}°", (20, y_offset), font, 0.9, (255, 255, 255), thickness)
        y_offset += 35
        cv2.putText(panel_bg, f"Diagnosis: {severity}", (20, y_offset), font, 0.9, severity_color, thickness)
        y_offset += 50
        
        # Additional info
        cv2.putText(panel_bg, f"Landmarks Detected: {num_landmarks}", (20, y_offset), font, 0.7, (200, 200, 200), thickness)
        
        # Place panel on image (top-left corner)
        image[20:340, 20:470] = panel_bg
    
    def draw_error_panel(self, image, num_landmarks):
        """Draw a simple error panel when Cobb angle calculation fails"""
        # Create a simple error panel
        panel_bg = np.zeros((120, 300, 3), dtype=np.uint8)
        panel_bg[:] = (20, 20, 20)  # Dark gray background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        cv2.putText(panel_bg, "COBB ANGLE ANALYSIS", (20, 40), font, 0.9, (255, 255, 255), thickness)
        cv2.putText(panel_bg, "Calculation Failed", (20, 70), font, 0.8, (0, 0, 255), thickness)
        cv2.putText(panel_bg, f"Landmarks: {num_landmarks}", (20, 100), font, 0.7, (200, 200, 200), thickness)
        
        # Place panel on image (top-left corner)
        image[20:140, 20:320] = panel_bg
    
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
    
    def get_vertebral_names(self, vertebrae_pair):
        """Convert vertebral indices to standard medical names"""
        v1, v2 = vertebrae_pair
        
        # Estimate vertebral levels based on position
        # This is an approximation - in practice, vertebral counting would be more precise
        total_vertebrae = 17  # Typical number of vertebrae detected (T1-L5)
        
        # Map indices to vertebral names
        vertebral_names = []
        for i in range(total_vertebrae):
            if i < 12:  # Thoracic vertebrae
                vertebral_names.append(f"T{i+1}")
            else:  # Lumbar vertebrae
                vertebral_names.append(f"L{i-11}")
        
        # Get the vertebral names for the pair
        if v1 < len(vertebral_names) and v2 < len(vertebral_names):
            return f"{vertebral_names[v1]}-{vertebral_names[v2]}"
        else:
            return f"V{v1+1}-V{v2+1}"  # Fallback to generic naming
    
    def analyze_image(self, image_path, save_output=True, output_dir='cobb_results'):
        """Complete analysis: detect landmarks, calculate Cobb angles, and display results"""
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
        
        # Calculate Cobb angles
        cobb_angles = self.calculate_cobb_angles(landmarks, original_image)
        
        # Get vertebral information for each angle
        vertebral_info = None
        if cobb_angles is not None:
            try:
                vertebrae_info, mid_p = self.get_cobb_vertebrae(landmarks, original_image)
                vertebral_info = {
                    'primary': self.get_vertebral_names(vertebrae_info['angle1']['vertebrae']) if vertebrae_info['angle1'] else "N/A",
                    'secondary': self.get_vertebral_names(vertebrae_info['angle2']['vertebrae']) if vertebrae_info['angle2'] else "N/A",
                    'tertiary': self.get_vertebral_names(vertebrae_info['angle3']['vertebrae']) if vertebrae_info['angle3'] else "N/A"
                }
            except Exception as e:
                print(f"Warning: Could not determine vertebral levels: {e}")
                vertebral_info = {
                    'primary': "N/A",
                    'secondary': "N/A", 
                    'tertiary': "N/A"
                }
        
        # Display results
        result_image = self.display_results(original_image, landmarks, cobb_angles)
        
        # Print results
        print(f"\n=== COBB ANGLE ANALYSIS RESULTS ===")
        print(f"Detected {len(landmarks)} landmarks")
        if cobb_angles is not None:
            if vertebral_info:
                print(f"Primary Cobb Angle: {cobb_angles[0]:.1f}° ({vertebral_info['primary']})")
                print(f"Secondary Cobb Angle: {cobb_angles[1]:.1f}° ({vertebral_info['secondary']})")
                print(f"Tertiary Cobb Angle: {cobb_angles[2]:.1f}° ({vertebral_info['tertiary']})")
            else:
                print(f"Primary Cobb Angle: {cobb_angles[0]:.1f}°")
                print(f"Secondary Cobb Angle: {cobb_angles[1]:.1f}°")
                print(f"Tertiary Cobb Angle: {cobb_angles[2]:.1f}°")
        else:
            print("Cobb angle calculation failed")
        
        # Save results if requested
        if save_output:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save result image
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_cobb_analysis.jpg"), result_image)
            
            # Save landmarks coordinates
            np.save(os.path.join(output_dir, f"{base_name}_landmarks.npy"), landmarks)
            
            # Save Cobb angles
            if cobb_angles is not None:
                np.save(os.path.join(output_dir, f"{base_name}_cobb_angles.npy"), cobb_angles)
                
                # Save text report with vertebral information
                with open(os.path.join(output_dir, f"{base_name}_report.txt"), 'w') as f:
                    f.write("COBB ANGLE ANALYSIS REPORT\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Image: {image_path}\n")
                    try:
                        from datetime import datetime
                        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    except:
                        f.write(f"Analysis Date: {os.popen('date').read().strip() if os.name != 'nt' else 'Unknown'}\n")
                    f.write(f"Total Landmarks Detected: {len(landmarks)}\n\n")
                    f.write("COBB ANGLE MEASUREMENTS:\n")
                    f.write("-" * 30 + "\n")
                    if vertebral_info:
                        f.write(f"Primary Angle: {cobb_angles[0]:.1f}° ({vertebral_info['primary']})\n")
                        f.write(f"Secondary Angle: {cobb_angles[1]:.1f}° ({vertebral_info['secondary']})\n")
                        f.write(f"Tertiary Angle: {cobb_angles[2]:.1f}° ({vertebral_info['tertiary']})\n\n")
                    else:
                        f.write(f"Primary Angle: {cobb_angles[0]:.1f}°\n")
                        f.write(f"Secondary Angle: {cobb_angles[1]:.1f}°\n")
                        f.write(f"Tertiary Angle: {cobb_angles[2]:.1f}°\n\n")
                    
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
                    
                    f.write("CLINICAL ASSESSMENT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Maximum Cobb Angle: {max_angle:.1f}°\n")
                    f.write(f"Severity Classification: {severity}\n")
                    if vertebral_info:
                        f.write(f"Primary Curve Location: {vertebral_info['primary']}\n\n")
                    else:
                        f.write(f"Primary Curve Location: Not determined\n\n")
                    
                    f.write("CLINICAL INTERPRETATION:\n")
                    f.write("-" * 25 + "\n")
                    f.write("• Primary angle represents the main curve and guides treatment decisions\n")
                    f.write("• Secondary angle is compensatory and helps maintain spinal balance\n")
                    f.write("• Tertiary angle is usually minor and may represent normal variation\n\n")
                    
                    f.write("SEVERITY CLASSIFICATION:\n")
                    f.write("-" * 25 + "\n")
                    f.write("• < 10°: Normal spine (no treatment needed)\n")
                    f.write("• 10-25°: Mild scoliosis (monitor for progression)\n")
                    f.write("• 25-40°: Moderate scoliosis (consider bracing)\n")
                    f.write("• > 40°: Severe scoliosis (surgical consultation)\n\n")
                    
                    f.write("IMPORTANT NOTES:\n")
                    f.write("-" * 15 + "\n")
                    f.write("• This analysis is for research and educational purposes\n")
                    f.write("• Clinical decisions should be made by qualified healthcare providers\n")
                    f.write("• Vertebral levels are estimated and should be verified clinically\n")
                    f.write("• Compare with previous measurements to assess progression\n")
            
            print(f"\nResults saved to {output_dir}/")
        
        print("\nPress any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return landmarks, cobb_angles, result_image

def main():
    parser = argparse.ArgumentParser(description='Calculate Cobb angles from spinal X-ray images')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input X-ray image')
    parser.add_argument('--model_path', type=str, default='pretrained_model/model_last.pth', 
                       help='Path to the pretrained model')
    parser.add_argument('--output_dir', type=str, default='cobb_results', help='Output directory for results')
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
    calculator = CobbAngleCalculator(args.model_path, args.device)
    
    # Analyze image
    try:
        landmarks, cobb_angles, result_image = calculator.analyze_image(
            args.image_path, 
            save_output=not args.no_save,
            output_dir=args.output_dir
        )
        
        if landmarks is not None:
            print("Analysis completed successfully!")
        else:
            print("Analysis failed - no landmarks detected")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 