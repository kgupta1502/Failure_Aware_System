import cv2
import torch
import numpy as np
from ultralytics import YOLO

class AdvancedFailureAwarePredictor:
    def __init__(self, model_path, safety_threshold=0.45):
        print(f"[*] Initializing Multi-Pillar Safety Framework: {model_path}")
        self.model = YOLO(model_path)
        self.safety_threshold = safety_threshold

    def calculate_global_visibility_hazard(self, img):
        """Pillar 1: Contextual Uncertainty via RMS Contrast."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        # Normalize contrast hazard (15 = extreme hazard, 60 = perfect clear visibility)
        contrast_hazard = 1.0 - (np.clip(contrast, 15, 60) - 15) / (60 - 15)
        return float(contrast_hazard)

    def extract_boxes(self, results):
        """Helper to extract confidences and xyxy arrays from YOLO results."""
        if results.boxes is None or len(results.boxes) == 0:
            return np.array([]), np.array([])
        conf = results.boxes.conf.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        return conf, boxes

    def calculate_epistemic_variance(self, conf1, boxes1, conf2, boxes2):
        """Pillar 2: Object-Level Epistemic Uncertainty (Prediction Fluctuation)."""
        if (len(conf1) == 0 and len(conf2) > 0) or (len(conf2) == 0 and len(conf1) > 0):
            return 0.60  
            
        if len(conf1) == 0 and len(conf2) == 0:
            return 0.0  
            
        count_difference = abs(len(conf1) - len(conf2)) / max(len(conf1), len(conf2))
        avg_conf_drop = abs(np.mean(conf1) - np.mean(conf2))
        
        variance_score = (0.5 * count_difference) + (0.5 * avg_conf_drop)
        return float(np.clip(variance_score, 0.0, 1.0))

    def process_frame(self, frame):
        """
        Processes a single frame array, runs the safety architecture,
        and returns the calculations along with an annotated HUD frame.
        """
        # 1. Compute Pillar 1: Global Visibility Hazard
        visibility_hazard = self.calculate_global_visibility_hazard(frame)
        
        # 2. Create Perturbed Stream (Fixed typo to convertScaleAbs)
        perturbed_img = cv2.convertScaleAbs(frame, alpha=0.8, beta=-10)
        
        # 3. Run Dual-Stream Inference
        results_orig = self.model(frame, verbose=False)[0]
        results_pert = self.model(perturbed_img, verbose=False)[0]
        
        conf_orig, boxes_orig = self.extract_boxes(results_orig)
        conf_pert, boxes_pert = self.extract_boxes(results_pert)
        
        # 4. Compute Pillar 2: Object Perturbation Variance
        object_variance = self.calculate_epistemic_variance(conf_orig, boxes_orig, conf_pert, boxes_pert)
        
        # 5. Calculate Unified System Reliability Index (SRI)
        mean_conf = np.mean(conf_orig) if len(conf_orig) > 0 else 0
        conf_deficit = 1.0 - mean_conf if len(conf_orig) > 0 else 0
        
        system_reliability_index = (0.4 * visibility_hazard) + (0.6 * max(object_variance, conf_deficit))
        system_reliability_index = float(np.clip(system_reliability_index, 0.0, 1.0))
        
        is_failed = system_reliability_index > self.safety_threshold
        
        # 6. Draw Real-Time Visual Output HUD
        # Generate base standard YOLO bounding boxes on the original frame
        annotated_frame = results_orig.plot()
        
        # Inject Custom Status Bar at the Top
        hud_color = (0, 0, 255) if is_failed else (0, 255, 0) # Red if compromised, Green if Safe
        status_text = "CRITICAL SAFETY RISK: HANDOVER CONTROL" if is_failed else "SYSTEM OPERATIONAL: SAFE"
        
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 45), hud_color, -1)
        cv2.putText(annotated_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Inject Telemetry Overlay Box at Bottom-Left
        cv2.rectangle(annotated_frame, (10, annotated_frame.shape[0] - 130), (380, annotated_frame.shape[0] - 10), (0, 0, 0), -1)
        metrics = [
            f"Objects Tracked : {len(boxes_orig)}",
            f"Visibility Haz  : {visibility_hazard:.4f}",
            f"Model Flicker   : {object_variance:.4f}",
            f"System SRI Index: {system_reliability_index:.4f} (Max: {self.safety_threshold})"
        ]
        for i, text in enumerate(metrics):
            y_pos = annotated_frame.shape[0] - 105 + (i * 25)
            cv2.putText(annotated_frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        return annotated_frame, system_reliability_index

    def predict_video(self, video_path):
        """Opens a video stream file and streams real-time telemetry processing."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"[-] Cannot open or read video stream source: {video_path}")
            
        print(f"[*] Real-Time Processing Started. Press 'q' inside the video window to quit.")
        
        # Configure window to be resizable
        cv2.namedWindow("Failure-Aware Driving Interface", cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[*] Video stream processing completed successfully or feed disconnected.")
                break
                
            # Process single frame through our architecture pipeline
            annotated_frame, sri_score = self.process_frame(frame)
            
            # Render visual output
            cv2.imshow("Failure-Aware Driving Interface", annotated_frame)
            
            # Frame break handler: checks if 'q' key is pressed to close down safely
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[-] Processing forcefully aborted by user.")
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    WEIGHTS = "runs/detect/runs/detect/rtdetr_focal_loss_run-4/weights/best.pt"
    
    # Initialize predictor
    safety_net = AdvancedFailureAwarePredictor(model_path=WEIGHTS, safety_threshold=0.45)
    
    # Target video stream path
    test_video = "data/processed/master_dataset/val/images/Pedestrian near miss DashCam video, Chester, UK - Olivia Barnes (1080p, h264).mp4"
    
    try:
        safety_net.predict_video(test_video)
    except Exception as e:
        print(f"[-] Execution Fatal Error: {e}")