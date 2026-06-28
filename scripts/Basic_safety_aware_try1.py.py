import cv2
import numpy as np
from ultralytics import YOLO

class SafetySupervisorySystem:
    def __init__(self, model_path="yolo11n.pt", threshold=0.35):
        """
        Initializes the Safety Supervisor with a YOLO model and a safety threshold.
        """
        self.model = YOLO(model_path)
        self.threshold = threshold
        print(f"[*] Safety System initialized with {model_path}. Threshold: {threshold}")

    def _get_uncertainty(self, img):
        """
        Calculates epistemic uncertainty by measuring prediction variance 
        between a clean frame and a slightly perturbed (gamma-shifted) frame.
        """
        # Create a perturbed version (Gamma shift to test model stability)
        perturbed = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
        
        # Run dual-stream inference
        r1 = self.model(img, verbose=False)[0]
        r2 = self.model(perturbed, verbose=False)[0]
        
        # Extract confidence scores
        conf1 = r1.boxes.conf.cpu().numpy() if len(r1.boxes) > 0 else []
        conf2 = r2.boxes.conf.cpu().numpy() if len(r2.boxes) > 0 else []
        
        # Calculate variation: How much did the count or confidence change?
        # A significant change under minor perturbation indicates low model stability.
        count_delta = abs(len(conf1) - len(conf2))
        conf_delta = abs(np.mean(conf1) - np.mean(conf2)) if (len(conf1) > 0 and len(conf2) > 0) else 0.5
        
        uncertainty = (0.5 * min(count_delta / 3.0, 1.0)) + (0.5 * conf_delta)
        return float(np.clip(uncertainty, 0.0, 1.0))

    def process(self, source):
        """
        Processes video or image input and displays the live Safety Dashboard.
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"[-] Error: Could not open source {source}")
            return

        print("[*] Dashboard starting... Press 'q' to exit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Compute Uncertainty (The Safety Check)
            uncertainty = self._get_uncertainty(frame)
            sri = 1.0 - uncertainty # SRI = System Reliability Index
            
            # 2. Run Main Detection
            results = self.model(frame, verbose=False)[0]
            annotated = results.plot()
            
            # 3. Presentable HUD (Heads-Up Display)
            color = (0, 255, 0) if sri > (1 - self.threshold) else (0, 0, 255)
            status = "SYSTEM OPERATIONAL" if sri > (1 - self.threshold) else "CRITICAL RISK: HANDOVER"
            
            # Overlay SRI info
            cv2.rectangle(annotated, (10, 10), (450, 100), (0, 0, 0), -1)
            cv2.putText(annotated, f"SRI: {sri:.2f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(annotated, status, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show output
            cv2.imshow("Safety Supervisory Dashboard", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to your best-performing model weights
    WEIGHTS = "runs/detect/rtdetr_altered_weights_run-3/weights/best.pt"
    
    # Path to the video or image file
    INPUT_SRC = "data/processed/master_dataset/val/images/Pedestrian near miss DashCam video, Chester, UK - Olivia Barnes (1080p, h264).mp4"
    
    system = SafetySupervisorySystem(WEIGHTS)
    system.process(INPUT_SRC)