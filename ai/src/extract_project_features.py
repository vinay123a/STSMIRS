import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.skeleton_extractor import SkeletonExtractor

def extract_features():
    # Prefer config.json from the ai/ parent folder when running from src/
    cfg_candidate = Path(os.path.join(os.path.dirname(__file__), '..', 'config.json')).resolve()
    config_path = str(cfg_candidate) if cfg_candidate.exists() else 'config.json'
    extractor = SkeletonExtractor(config_path)
    # Prefer the project's ai/datasets folder if present (user-provided videos)
    default_videos = Path(os.path.join(os.path.dirname(__file__), '..', 'datasets')).resolve()
    if default_videos.exists():
        video_root = default_videos
    else:
        video_root = Path(r"c:\ai folder\fight-detection-ai\project_videos")
    output_root = Path("data/features")
    
    # Mapping folder names to our model classes
    class_map = {
        "fall": "Fall",
        "fighting": "Fighting",
        "loitering": "Loitering",
        "lying_still": "Lying_Still",
        "running": "Running",
        "walking": "Walking"
    }

    for folder, class_name in class_map.items():
        folder_path = video_root / folder
        if not folder_path.exists():
            continue
            
        out_dir = output_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {class_name} videos...")
        
        for video_file in folder_path.glob("*.mp4"):
            print(f"  Extracting skeletons from {video_file.name}")
            video_skeletons = []

            # If the extractor has a model, use its video streaming API which is more robust
            if getattr(extractor, 'model', None) is not None:
                try:
                    # stream inference over video file (ultralytics handles video internally)
                    results_stream = extractor.model(str(video_file), stream=True, conf=extractor.confidence_threshold)
                    for res in results_stream:
                        # res may be a Results object; try to read keypoints
                        try:
                            kpts_arr = None
                            if hasattr(res, 'keypoints') and res.keypoints is not None:
                                kpts_arr = res.keypoints.data.cpu().numpy()

                            if isinstance(kpts_arr, np.ndarray) and kpts_arr.size > 0:
                                kpts = kpts_arr[0]
                                normalized = extractor.normalize_keypoints(kpts, res.orig_shape[0], res.orig_shape[1])
                                video_skeletons.append(normalized.flatten())
                            else:
                                video_skeletons.append(np.zeros(51))
                        except Exception as e_frame:
                            print(f"[Extractor] Frame result handling error: {e_frame}")
                            video_skeletons.append(np.zeros(51))
                except Exception as e_stream:
                    print(f"[Extractor] Video stream inference failed: {e_stream}")
                    # fallback to cv2 per-frame processing
            else:
                print("[Extractor] No model available; skipping video")
                continue
                    
            
            # Save the sequence as a .npy file
            if video_skeletons:
                npy_path = out_dir / f"{video_file.stem}.npy"
                np.save(npy_path, np.array(video_skeletons))

if __name__ == "__main__":
    extract_features()
