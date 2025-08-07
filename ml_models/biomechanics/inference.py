"""
Real-time Inference for Biomechanics Model
Process video streams and generate live feedback
"""

import cv2
import numpy as np
import torch
import time
from typing import Optional, Tuple
import logging
from gnn_lstm import BiomechanicsModel, BiomechanicsResult
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

logger = logging.getLogger(__name__)

class RealTimeBiomechanicsAnalyzer:
    """Real-time biomechanics analyzer for video streams"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = BiomechanicsModel()
        if model_path and torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Analysis state
        self.frame_count = 0
        self.analysis_history = []
        self.current_exercise = "squat"
        
        # Real-time visualization
        self.heatmap_queue = queue.Queue(maxsize=10)
        self.analysis_queue = queue.Queue(maxsize=10)
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0.0
    
    def analyze_video_stream(self, video_source: int = 0, exercise_type: str = "squat"):
        """Analyze real-time video stream from camera"""
        self.current_exercise = exercise_type
        
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error("Could not open video source")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting real-time biomechanics analysis...")
        logger.info(f"Exercise type: {exercise_type}")
        logger.info("Press 'q' to quit, 's' to save analysis")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Analyze frame
                result = self.analyze_frame(frame, exercise_type)
                
                # Update performance metrics
                self._update_fps()
                
                # Visualize results
                annotated_frame = self._visualize_results(frame, result)
                
                # Display frame
                cv2.imshow('FitBalance Biomechanics Analysis', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_analysis()
                elif key == ord('h'):
                    self._show_heatmap_window()
                
        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Analysis completed")
    
    def analyze_video_file(self, video_path: str, exercise_type: str = "squat", output_path: Optional[str] = None):
        """Analyze video file and optionally save results"""
        self.current_exercise = exercise_type
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Analyzing video: {video_path}")
        logger.info(f"FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        frame_count = 0
        analysis_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze frame
                result = self.analyze_frame(frame, exercise_type)
                result.frame_number = frame_count
                result.timestamp = frame_count / fps
                
                analysis_results.append(result)
                
                # Visualize results
                annotated_frame = self._visualize_results(frame, result)
                
                # Write frame if output path provided
                if writer:
                    writer.write(annotated_frame)
                
                # Display progress
                frame_count += 1
                if frame_count % 30 == 0:  # Every 30 frames
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Display frame (optional)
                cv2.imshow('FitBalance Analysis', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Save analysis results
        self._save_analysis_results(analysis_results, video_path)
        
        logger.info(f"Analysis completed. Processed {frame_count} frames.")
        return analysis_results
    
    def analyze_frame(self, frame: np.ndarray, exercise_type: str = "squat") -> BiomechanicsResult:
        """Analyze a single frame"""
        start_time = time.time()
        
        # Run analysis
        with torch.no_grad():
            result = self.model.analyze_frame(frame, exercise_type)
        
        # Update frame count and history
        self.frame_count += 1
        result.frame_number = self.frame_count
        result.timestamp = time.time()
        
        # Store in history (keep last 100 frames)
        self.analysis_history.append(result)
        if len(self.analysis_history) > 100:
            self.analysis_history.pop(0)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing time to result
        result.processing_time = processing_time
        
        return result
    
    def _visualize_results(self, frame: np.ndarray, result: BiomechanicsResult) -> np.ndarray:
        """Visualize analysis results on frame"""
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw form score
        score_text = f"Form Score: {result.form_score:.1f}/100"
        cv2.putText(annotated_frame, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw FPS
        fps_text = f"FPS: {self.avg_fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw risk factors
        if result.risk_factors:
            risk_text = f"Risks: {', '.join(result.risk_factors[:3])}"
            cv2.putText(annotated_frame, risk_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw joint angles
        y_offset = 120
        for joint_angle in result.joint_angles[:5]:  # Show first 5 joints
            color = (0, 255, 0) if not joint_angle.is_abnormal else (0, 0, 255)
            angle_text = f"{joint_angle.joint_name}: {joint_angle.angle:.1f}°"
            cv2.putText(annotated_frame, angle_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        # Draw torques
        y_offset = 250
        for torque in result.torques[:3]:  # Show first 3 torques
            color_map = {"low": (0, 255, 0), "medium": (0, 255, 255), 
                        "high": (0, 165, 255), "critical": (0, 0, 255)}
            color = color_map.get(torque.risk_level, (255, 255, 255))
            torque_text = f"{torque.joint_name}: {torque.torque_magnitude:.1f} N⋅m ({torque.risk_level})"
            cv2.putText(annotated_frame, torque_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        # Draw recommendations
        if result.recommendations:
            y_offset = 350
            cv2.putText(annotated_frame, "Recommendations:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 25
            for rec in result.recommendations[:2]:  # Show first 2 recommendations
                cv2.putText(annotated_frame, f"• {rec}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        return annotated_frame
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.avg_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _show_heatmap_window(self):
        """Show heatmap in separate window"""
        if not self.analysis_history:
            return
        
        # Get latest heatmap
        latest_result = self.analysis_history[-1]
        heatmap = latest_result.heatmap_data
        
        # Normalize heatmap for display
        heatmap_display = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
        
        # Resize for display
        heatmap_resized = cv2.resize(heatmap_colored, (400, 300))
        
        cv2.imshow('Torque Heatmap', heatmap_resized)
    
    def _save_analysis(self):
        """Save current analysis results"""
        if not self.analysis_history:
            logger.warning("No analysis data to save")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"biomechanics_analysis_{timestamp}.json"
        
        # Convert results to serializable format
        analysis_data = []
        for result in self.analysis_history:
            analysis_data.append({
                'frame_number': result.frame_number,
                'timestamp': result.timestamp,
                'form_score': result.form_score,
                'risk_factors': result.risk_factors,
                'recommendations': result.recommendations,
                'joint_angles': [
                    {
                        'joint_name': ja.joint_name,
                        'angle': ja.angle,
                        'is_abnormal': ja.is_abnormal
                    } for ja in result.joint_angles
                ],
                'torques': [
                    {
                        'joint_name': t.joint_name,
                        'torque_magnitude': t.torque_magnitude,
                        'risk_level': t.risk_level
                    } for t in result.torques
                ]
            })
        
        # Save to file
        import json
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Analysis saved to {filename}")
    
    def _save_analysis_results(self, results: list, video_path: str):
        """Save analysis results for video file"""
        import json
        
        # Create output filename
        base_name = video_path.rsplit('.', 1)[0]
        output_file = f"{base_name}_analysis.json"
        
        # Convert results to serializable format
        analysis_data = []
        for result in results:
            analysis_data.append({
                'frame_number': result.frame_number,
                'timestamp': result.timestamp,
                'form_score': result.form_score,
                'risk_factors': result.risk_factors,
                'recommendations': result.recommendations,
                'joint_angles': [
                    {
                        'joint_name': ja.joint_name,
                        'angle': ja.angle,
                        'is_abnormal': ja.is_abnormal
                    } for ja in result.joint_angles
                ],
                'torques': [
                    {
                        'joint_name': t.joint_name,
                        'torque_magnitude': t.torque_magnitude,
                        'risk_level': t.risk_level
                    } for t in result.torques
                ]
            })
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_file}")
    
    def get_performance_summary(self) -> dict:
        """Get performance summary of the analysis"""
        if not self.analysis_history:
            return {}
        
        form_scores = [r.form_score for r in self.analysis_history]
        risk_factors = []
        for r in self.analysis_history:
            risk_factors.extend(r.risk_factors)
        
        return {
            'total_frames': len(self.analysis_history),
            'average_form_score': np.mean(form_scores),
            'min_form_score': np.min(form_scores),
            'max_form_score': np.max(form_scores),
            'common_risk_factors': list(set(risk_factors)),
            'average_fps': self.avg_fps
        }

def main():
    """Main function for real-time analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Biomechanics Analysis')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for camera, or path to video file)')
    parser.add_argument('--exercise', type=str, default='squat',
                       choices=['squat', 'deadlift', 'pushup'],
                       help='Exercise type to analyze')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (for file analysis)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = RealTimeBiomechanicsAnalyzer(model_path=args.model)
    
    # Determine if source is camera or file
    if args.source.isdigit():
        # Camera
        analyzer.analyze_video_stream(int(args.source), args.exercise)
    else:
        # Video file
        analyzer.analyze_video_file(args.source, args.exercise, args.output)
    
    # Print performance summary
    summary = analyzer.get_performance_summary()
    if summary:
        print("\n=== Performance Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 