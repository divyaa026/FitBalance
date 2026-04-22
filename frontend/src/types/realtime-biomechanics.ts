/**
 * Real-time Biomechanics Types
 */

export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}

export interface FormIssue {
  type: string;
  severity: number;
  message: string;
  is_critical?: boolean;
}

export type PoseQualityType = 'NONE' | 'PARTIAL' | 'FULL';
export type MovementStateType = 'STATIC' | 'MOVING';

export interface PoseQuality {
  type: PoseQualityType;
  coverage: number;  // 0-1, percentage of keypoints detected
  missing_joints: string[];
  message: string;
  avg_confidence: number;
}

export interface LiveFramePayload {
  frame: string;  // base64 encoded JPEG
  exercise_type: string;
}

export interface FormAnalysisResponse {
  form_score: number;
  feedback: string;
  keypoints: Keypoint[];
  issues: FormIssue[];
  joint_angles: Record<string, number>;
  processing_time_ms: number;
  error?: string;
  // New fields
  pose_quality: PoseQuality;
  movement_state: MovementStateType;
  reps: number;
  correct_reps: number;
  incorrect_reps: number;
  should_speak: boolean;
  processed_frame?: string;  // Base64 encoded annotated frame from MediaPipe
}

export interface RealtimeStatus {
  available: boolean;
  device: string;
  active_connections: number;
}

export type ExerciseType = 
  | 'squat' 
  | 'deadlift' 
  | 'bench_press' 
  | 'pushup' 
  | 'lunge' 
  | 'overhead_press'
  | 'row';

export const EXERCISE_OPTIONS: { value: ExerciseType; label: string }[] = [
  { value: 'squat', label: 'Squat' },
  { value: 'deadlift', label: 'Deadlift' },
  { value: 'bench_press', label: 'Bench Press' },
  { value: 'pushup', label: 'Push-up' },
  { value: 'lunge', label: 'Lunge' },
  { value: 'overhead_press', label: 'Overhead Press' },
  { value: 'row', label: 'Row' },
];

// COCO skeleton connections for drawing
export const SKELETON_CONNECTIONS: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4],  // Head
  [5, 6],  // Shoulders
  [5, 7], [7, 9],  // Left arm
  [6, 8], [8, 10],  // Right arm
  [5, 11], [6, 12],  // Torso
  [11, 12],  // Hips
  [11, 13], [13, 15],  // Left leg
  [12, 14], [14, 16],  // Right leg
];

// Keypoint colors by body part - used for PARTIAL pose
export const KEYPOINT_COLORS: Record<number, string> = {
  0: '#FF6B6B',  // nose
  1: '#FF6B6B', 2: '#FF6B6B',  // eyes
  3: '#FF6B6B', 4: '#FF6B6B',  // ears
  5: '#4ECDC4', 6: '#4ECDC4',  // shoulders
  7: '#45B7D1', 8: '#45B7D1',  // elbows
  9: '#96CEB4', 10: '#96CEB4',  // wrists
  11: '#FFEAA7', 12: '#FFEAA7',  // hips
  13: '#DDA0DD', 14: '#DDA0DD',  // knees
  15: '#98D8C8', 16: '#98D8C8',  // ankles
};

// Confidence thresholds for keypoint coloring
export const CONFIDENCE_THRESHOLDS = {
  HIGH: 0.7,    // Green
  MEDIUM: 0.5,  // Yellow
  LOW: 0.3,     // Red
};

// Get keypoint color based on confidence
export function getKeypointColor(confidence: number): string {
  if (confidence >= CONFIDENCE_THRESHOLDS.HIGH) return '#22C55E';  // Green
  if (confidence >= CONFIDENCE_THRESHOLDS.MEDIUM) return '#EAB308'; // Yellow
  if (confidence >= CONFIDENCE_THRESHOLDS.LOW) return '#EF4444';    // Red
  return '#6B7280';  // Gray (very low confidence)
}

// Get skeleton line color based on minimum confidence of endpoints
export function getSkeletonColor(conf1: number, conf2: number): string {
  const minConf = Math.min(conf1, conf2);
  if (minConf >= CONFIDENCE_THRESHOLDS.HIGH) return '#22C55E';
  if (minConf >= CONFIDENCE_THRESHOLDS.MEDIUM) return '#EAB308';
  return '#EF4444';
}
