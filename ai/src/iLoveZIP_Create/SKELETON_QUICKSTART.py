#!/usr/bin/env python3
"""
STSMIRS — Quick Start: Skeleton-Based Face Recognition & Action Detection
Follow these steps to set up and run the skeleton system
"""

import os
import sys
from pathlib import Path

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_requirements():
    """Check if all requirements are installed."""
    print_section("1. CHECKING REQUIREMENTS")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('ultralytics', 'YOLO'),
        ('numpy', 'NumPy'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            missing.append(package)
    
    if missing:
        print("\n  Install missing packages:")
        print(f"    pip install -r requirements.txt")
        return False
    
    print("\n  ✓ All requirements met!")
    return True

def setup_directories():
    """Create necessary directories."""
    print_section("2. SETTING UP DIRECTORIES")
    
    dirs = [
        'models',
        'data/skeleton_features',
        'data/features',
        'runs/skeleton_eval',
        'outputs',
        'faces'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created/verified: {dir_path}")

def download_models():
    """Download required models."""
    print_section("3. DOWNLOADING MODELS")
    
    print("  Models will auto-download on first use:")
    print("    - yolov8n-pose.pt (skeleton extraction)")
    print("    - yolov8n.pt (person detection)")
    print("\n  Models are downloaded to: ~/.cache/ultralytics/")
    print("\n  First run will take 2-3 minutes for downloads.")

def test_skeleton_extraction():
    """Test skeleton extraction on a sample frame."""
    print_section("4. TESTING SKELETON EXTRACTION")
    
    print("  Testing skeleton extraction...")
    
    try:
        import cv2
        import numpy as np
        from src.skeleton_extractor import SkeletonExtractor
        
        extractor = SkeletonExtractor('config.json')
        
        if extractor.model is None:
            print("  ⚠ Skeleton model not loaded. Will download on first use.")
            return True
        
        # Create dummy frame
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        result = extractor.extract_skeleton(frame)
        
        print(f"  ✓ Skeleton extraction working")
        print(f"    - Detected {len(result['keypoints'])} persons (expected 0 for blank frame)")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ Error during test: {e}")
        return False

def test_action_detector():
    """Test skeleton action detector."""
    print_section("5. TESTING ACTION DETECTOR")
    
    print("  Testing skeleton action detector...")
    
    try:
        from src.skeleton_action_detector import SkeletonActionDetector
        
        detector = SkeletonActionDetector('config.json')
        
        if detector.model is None:
            print("  ⚠ Action model not loaded. Will use untrained model.")
        else:
            print("  ✓ Skeleton action detector loaded")
        
        print(f"    - Supported actions: {', '.join(detector.classes)}")
        print(f"    - Sequence length: {detector.seq_len} frames")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def show_pipeline_diagram():
    """Show pipeline architecture."""
    print_section("6. SYSTEM ARCHITECTURE")
    
    diagram = """
    Camera/Video Input
        ↓
    [Person Detection] (YOLOv8)
        ↓
    [Skeleton Extraction] (YOLOv8 Pose)
        ├─→ Normalized Keypoints
        ├─→ Joint Angles
        └─→ Limb Lengths
        ↓
    ┌───────────────────┬─────────────────────┐
    ↓                   ↓                     ↓
[Enhanced Face]   [Action Detection]    [Event Trigger]
[Recognition]     [LSTM + Skeleton]         ↓
    ↓                   ↓                 [Alert]
[Person ID]       [Action Label]       [Logging]
    """
    print(diagram)

def show_quick_commands():
    """Show quick start commands."""
    print_section("7. QUICK START COMMANDS")
    
    commands = {
        "Run Live Pipeline": "python src/main.py --config config.json --source 0",
        "Train Skeleton Model": "python src/train_skeleton_lstm.py --config config.json",
        "Extract Skeleton Features": (
            "python tools/extract_skeleton_features.py "
            "--mode videos --source dataset --output data/skeleton_features"
        ),
        "Evaluate Model": (
            "python tools/skeleton_eval.py --mode evaluate "
            "--dataset dataset --output results.json"
        ),
        "Visualize Skeleton": (
            "python tools/skeleton_eval.py --mode visualize "
            "--video test.mp4 --output annotated.mp4"
        ),
        "Download Models Only": (
            "python -c \"from ultralytics import YOLO; "
            "YOLO('yolov8n-pose.pt'); YOLO('yolov8n.pt')\""
        ),
    }
    
    for desc, cmd in commands.items():
        print(f"\n  {desc}:")
        print(f"    $ {cmd}")

def show_configuration_tips():
    """Show configuration tips."""
    print_section("8. CONFIGURATION TIPS")
    
    tips = [
        ("Skeleton Confidence", 
         "Adjust 'keypoint_confidence_threshold' (default 0.3) if skeleton not detected"),
        
        ("Action Detection", 
         "Tune 'skeleton_action.postprocess.confidence_threshold' (default 0.5)"),
        
        ("Performance",
         "Use 'frame_skip: 2' in skeleton config to process every 2nd frame"),
        
        ("GPU/CPU",
         "Set 'skeleton.device' to 'cuda', 'cpu', or 'auto' (default)"),
        
        ("Model Size",
         "Use 'yolov8s-pose.pt' or 'yolov8l-pose.pt' for different speed/accuracy tradeoffs"),
    ]
    
    for setting, tip in tips:
        print(f"\n  {setting}:")
        print(f"    → {tip}")

def show_next_steps():
    """Show next steps."""
    print_section("9. NEXT STEPS")
    
    steps = [
        ("1. Prepare Dataset", 
         "Organize videos by action class in 'dataset/' folder"),
        
        ("2. Extract Features",
         "Run: python tools/extract_skeleton_features.py --mode videos"),
        
        ("3. Train Model",
         "Run: python src/train_skeleton_lstm.py"),
        
        ("4. Evaluate",
         "Run: python tools/skeleton_eval.py --mode evaluate"),
        
        ("5. Deploy",
         "Run: python src/main.py to start live pipeline"),
    ]
    
    for num, (title, cmd) in enumerate(steps, 1):
        print(f"\n  Step {num}: {title}")
        print(f"    {cmd}")

def show_documentation():
    """Show documentation links."""
    print_section("10. DOCUMENTATION")
    
    docs = {
        "Full Documentation": "SKELETON_README.md",
        "Module Reference": "src/skeleton_*.py (docstrings)",
        "Config Reference": "config.json (skeleton* sections)",
        "Examples": "tools/skeleton_eval.py",
    }
    
    for title, path in docs.items():
        print(f"  {title}:")
        print(f"    → {path}")

def main():
    """Run quick start wizard."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*10 + "STSMIRS — Skeleton System Quick Start" + " "*22 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Run checks
    if not check_requirements():
        print("\n❌ Please install requirements first:")
        print("   pip install -r requirements.txt")
        return
    
    setup_directories()
    download_models()
    
    # Test components
    test_skeleton_extraction()
    test_action_detector()
    
    show_pipeline_diagram()
    show_quick_commands()
    show_configuration_tips()
    show_next_steps()
    show_documentation()
    
    # Final message
    print_section("READY TO START!")
    print("""
  You're all set! Here are the recommended next steps:
  
  1. For live demo:
     $ python src/main.py --config config.json --source 0
  
  2. For training on your data:
     $ python tools/extract_skeleton_features.py --mode videos \\
         --source dataset --output data/skeleton_features
     $ python src/train_skeleton_lstm.py --config config.json
  
  3. For evaluation:
     $ python tools/skeleton_eval.py --mode evaluate \\
         --dataset dataset --output results.json
  
  Questions or issues? Check SKELETON_README.md for detailed documentation.
    """)
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
