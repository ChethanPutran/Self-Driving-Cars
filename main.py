from pipeline.autonomous_perception import AutonomousPerceptionPipeline    
# Demo execution
def main():
    """Main demonstration"""
    print("=== Autonomous Driving Perception System ===\n")
    
    # Initialize pipeline
    pipeline = AutonomousPerceptionPipeline()
    
    print("Pipeline initialized successfully!")
    print("\nComponents loaded:")
    print("> Classical Computer Vision (SIFT, Optical Flow, Stereo)")
    print("> Deep Learning (YOLOv5, U-Net, Monodepth2)")
    print("> Sensor Fusion (Camera + LiDAR)")
    print("> Tracking (Kalman Filter, DeepSORT)")
    print("> BEV Representation & Path Planning")
    
    # Example usage with sample data
    pipeline.run_on_video('sample_driving.mp4', 'output.mp4', max_frames=50)
    
    print("\nTo run on actual data:")
    print("1. Provide video path to run_on_video() method")
    print("2. Or call process_frame() for individual frames")
    

if __name__ == "__main__":
    pipeline = main()
