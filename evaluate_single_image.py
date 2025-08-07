import torch
import numpy as np
import cv2
import os
import argparse
from models import spinal_net
import decoder
import pre_proc
import draw_points

class SingleImageEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        heads = {'hm': 1,  # num_classes = 1 for spinal landmarks
                 'reg': 2*1,
                 'wh': 2*4,}
        
        # Initialize model
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=False,  # Don't load pretrained weights since we'll load our own
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
    
    def process_landmarks(self, pts2, original_image):
        """Process detected landmarks and draw them on the image"""
        pts0 = pts2.copy()
        pts0[:,:10] *= self.down_ratio
        
        # Sort landmarks by y-coordinate (top to bottom)
        sort_ind = np.argsort(pts0[:,1])
        pts0 = pts0[sort_ind]
        
        # Draw landmarks on the image
        image_with_landmarks = original_image.copy()
        image_with_points = original_image.copy()
        
        # Draw landmarks using the existing draw_points module
        image_with_landmarks, image_with_points = draw_points.draw_landmarks_regress_test(
            pts0, image_with_landmarks, image_with_points
        )
        
        return image_with_landmarks, image_with_points, pts0
    
    def display_resized_image(self, window_name, image, max_width=800, max_height=600):
        """Display image in a resized window while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        # Calculate scaling factor to fit within max dimensions
        scale_x = max_width / width
        scale_y = max_height / height
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized_image = image
        
        cv2.imshow(window_name, resized_image)
    
    def evaluate_image(self, image_path, save_output=True, output_dir='output'):
        """Evaluate a single image and return results"""
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Detect landmarks
        pts2 = self.detect_landmarks(image_tensor)
        
        # Process and draw landmarks
        image_with_landmarks, image_with_points, landmarks = self.process_landmarks(pts2, original_image)
        
        # Save results if requested
        if save_output:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save images with landmarks
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_landmarks.jpg"), image_with_landmarks)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_points.jpg"), image_with_points)
            
            # Save landmarks coordinates
            np.save(os.path.join(output_dir, f"{base_name}_landmarks.npy"), landmarks)
            
            print(f"Results saved to {output_dir}/")
        
        # Display results with resized windows
        self.display_resized_image('Landmarks Detection', image_with_landmarks)
        self.display_resized_image('Points Detection', image_with_points)
        
        print(f"Detected {len(landmarks)} landmark groups")
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return landmarks, image_with_landmarks, image_with_points

def main():
    parser = argparse.ArgumentParser(description='Evaluate a single image using pretrained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='pretrained_model/model_last.pth', 
                       help='Path to the pretrained model')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
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
    
    # Initialize evaluator
    evaluator = SingleImageEvaluator(args.model_path, args.device)
    
    # Evaluate image
    try:
        landmarks, image_with_landmarks, image_with_points = evaluator.evaluate_image(
            args.image_path, 
            save_output=not args.no_save,
            output_dir=args.output_dir
        )
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main() 