import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
    
    def process_frame(self, frame):
        """
        Process a single frame and return landmarks.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0] # Return first face
        return None

    def get_roi_coordinates(self, landmarks, frame_shape):
        """
        Extract coordinates for Forehead and Cheeks based on landmarks.
        Returns a dictionary of ROIs.
        """
        h, w, _ = frame_shape
        
        # Landmark indices (approximate for MediaPipe FaceMesh)
        # Forehead: Central upper region
        forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        # Simplified Rectangle approximation might be better for stability, but using convex hull of these points:
        
        # Better simplified regions for rPPG stability:
        # Forehead
        forehead_pts = [109, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67] # Broad forehead area
        # Actually, let's stick to a simpler defined regions for ROI averaging to reduce noise
        
        # Forehead (Center)
        fh_indices = [68, 104, 69, 108, 151, 337, 299, 333, 298]
        
        # Left Cheek
        lc_indices = [205, 207, 214, 212, 165, 92, 186, 57, 43, 106, 182]
        
        # Right Cheek
        rc_indices = [373, 374, 280, 425, 427, 411, 287, 273, 422, 391, 326]

        rois = {
            'forehead': self._get_mask(fh_indices, landmarks, h, w),
            'left_cheek': self._get_mask(lc_indices, landmarks, h, w),
            'right_cheek': self._get_mask(rc_indices, landmarks, h, w)
        }
        
        return rois

    def _get_mask(self, indices, landmarks, h, w):
        points = []
        for idx in indices:
            pt = landmarks.landmark[idx]
            points.append([int(pt.x * w), int(pt.y * h)])
        
        points = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        return mask
