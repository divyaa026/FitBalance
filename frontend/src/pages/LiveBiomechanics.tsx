import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Activity, Volume2, VolumeX, Settings, Play, Square, AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import {
  FormAnalysisResponse,
  Keypoint,
  ExerciseType,
  PoseQuality,
  PoseQualityType,
  MovementStateType,
  EXERCISE_OPTIONS,
  SKELETON_CONNECTIONS,
  getKeypointColor,
  getSkeletonColor,
} from '@/types/realtime-biomechanics';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

// Voice feedback throttling - lowered for more responsive feedback
const VOICE_COOLDOWN_MS = 2000;

export default function LiveBiomechanics() {
  // State
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [exerciseType, setExerciseType] = useState<ExerciseType>('squat');
  const [formScore, setFormScore] = useState<number>(0);
  const [feedback, setFeedback] = useState<string>('');
  const [issues, setIssues] = useState<FormAnalysisResponse['issues']>([]);
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  
  // NEW: Enhanced state for pose quality tracking
  const [poseQuality, setPoseQuality] = useState<PoseQuality | null>(null);
  const [movementState, setMovementState] = useState<MovementStateType>('STATIC');
  const [repCount, setRepCount] = useState<number>(0);
  const [correctReps, setCorrectReps] = useState<number>(0);
  const [incorrectReps, setIncorrectReps] = useState<number>(0);
  const [processedFrame, setProcessedFrame] = useState<string | null>(null);
  const [lastFeedbackTime, setLastFeedbackTime] = useState<number>(0);
  const [lastSpokenFeedback, setLastSpokenFeedback] = useState<string>('');
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const processedFrameRef = useRef<HTMLImageElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const { toast } = useToast();
  const userId = useRef('user_' + Date.now()).current;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  // Track last speech time with ref to avoid stale closure issues
  const lastSpeechTimeRef = useRef<number>(0);
  const lastSpokenTextRef = useRef<string>('');
  
  // Speak feedback with throttling and deduplication
  const speakFeedback = useCallback((text: string, forceSpeak: boolean = false) => {
    if (!voiceEnabled || !text) return;
    
    // Ensure speech synthesis is available
    if (!('speechSynthesis' in window)) {
      console.warn('Speech synthesis not supported');
      return;
    }
    
    const now = Date.now();
    
    // Don't repeat same feedback (using ref to avoid stale state)
    if (text === lastSpokenTextRef.current && !forceSpeak) return;
    
    // Respect cooldown (using ref to avoid stale state)
    if (now - lastSpeechTimeRef.current < VOICE_COOLDOWN_MS && !forceSpeak) return;
    
    // Cancel any ongoing speech first
    window.speechSynthesis.cancel();
    
    // Create new utterance each time (more reliable)
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1;
    utterance.volume = 1.0;
    
    // Try to use a good voice
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.startsWith('en'));
    if (englishVoice) utterance.voice = englishVoice;
    
    window.speechSynthesis.speak(utterance);
    
    lastSpeechTimeRef.current = now;
    lastSpokenTextRef.current = text;
    setLastFeedbackTime(now);
    setLastSpokenFeedback(text);
  }, [voiceEnabled]);

  // Draw skeleton overlay on canvas with confidence-based coloring
  const drawSkeleton = useCallback((keypoints: Keypoint[], poseQuality: PoseQuality | null) => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || keypoints.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get the displayed size of the video element (CSS size)
    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;
    
    // Get native video dimensions
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    
    // Match canvas to displayed size for proper overlay
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    // Calculate scale factors and offset for object-contain behavior
    const videoAspect = videoWidth / videoHeight;
    const displayAspect = displayWidth / displayHeight;
    
    let scaleX: number, scaleY: number, offsetX = 0, offsetY = 0;
    
    if (videoAspect > displayAspect) {
      // Video is wider - letterbox top/bottom
      scaleX = displayWidth / videoWidth;
      scaleY = scaleX;
      offsetY = (displayHeight - videoHeight * scaleY) / 2;
    } else {
      // Video is taller - pillarbox left/right
      scaleY = displayHeight / videoHeight;
      scaleX = scaleY;
      offsetX = (displayWidth - videoWidth * scaleX) / 2;
    }

    // Clear previous frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Helper to transform keypoint coordinates
    const transformX = (x: number) => x * scaleX + offsetX;
    const transformY = (y: number) => y * scaleY + offsetY;

    // Draw skeleton connections with confidence-based colors
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    
    SKELETON_CONNECTIONS.forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];
      
      if (kp1 && kp2 && kp1.confidence > 0.3 && kp2.confidence > 0.3) {
        ctx.beginPath();
        // Color based on minimum confidence of the two endpoints
        ctx.strokeStyle = getSkeletonColor(kp1.confidence, kp2.confidence);
        ctx.moveTo(transformX(kp1.x), transformY(kp1.y));
        ctx.lineTo(transformX(kp2.x), transformY(kp2.y));
        ctx.stroke();
      }
    });

    // Draw keypoints with confidence-based colors
    keypoints.forEach((kp) => {
      if (kp.confidence > 0.2) {
        const color = getKeypointColor(kp.confidence);
        const radius = kp.confidence > 0.5 ? 6 : 4;
        
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(transformX(kp.x), transformY(kp.y), radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    // Draw pose quality indicator in corner
    if (poseQuality) {
      const indicatorColor = poseQuality.type === 'FULL' ? '#22C55E' : 
                             poseQuality.type === 'PARTIAL' ? '#EAB308' : '#EF4444';
      ctx.fillStyle = indicatorColor;
      ctx.beginPath();
      ctx.arc(30, 30, 12, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, []);

  // Capture and send frame
  const captureAndSendFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Resize to 480px width for faster processing (maintain aspect ratio)
    const targetWidth = 480;
    const scale = targetWidth / video.videoWidth;
    canvas.width = targetWidth;
    canvas.height = video.videoHeight * scale;

    // Draw video frame to canvas (scaled)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 JPEG (lower quality for speed)
    const base64Frame = canvas.toDataURL('image/jpeg', 0.6);

    // Send to WebSocket
    ws.send(JSON.stringify({
      frame: base64Frame,
      exercise_type: exerciseType,
    }));
  }, [exerciseType]);

  // Handle WebSocket message
  const handleWSMessage = useCallback((event: MessageEvent) => {
    try {
      const data: FormAnalysisResponse = JSON.parse(event.data);
      
      if (data.error) {
        console.warn('Analysis warning:', data.error);
        return;
      }

      // Update all state from response
      setFormScore(data.form_score);
      setIssues(data.issues || []);
      setKeypoints(data.keypoints || []);
      setProcessingTime(data.processing_time_ms);
      
      // NEW: Update pose quality and movement state
      if (data.pose_quality) {
        setPoseQuality(data.pose_quality);
      }
      if (data.movement_state) {
        setMovementState(data.movement_state);
      }
      if (typeof data.reps === 'number') {
        setRepCount(data.reps);
      }
      if (typeof data.correct_reps === 'number') {
        setCorrectReps(data.correct_reps);
      }
      if (typeof data.incorrect_reps === 'number') {
        setIncorrectReps(data.incorrect_reps);
      }
      
      // Display processed frame from MediaPipe (with skeleton already drawn)
      if (data.processed_frame) {
        setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
      }

      // Update feedback display and speak
      if (data.feedback && data.feedback.length > 0) {
        setFeedback(data.feedback);
        
        // Always try to speak feedback - frontend handles debouncing
        // Backend's should_speak is just a hint, but we have our own logic
        speakFeedback(data.feedback);
      }
    } catch (err) {
      console.error('Failed to parse WS message:', err);
    }
  }, [speakFeedback]);

  // Start camera and WebSocket
  const startStreaming = async () => {
    try {
      setCameraError(null);
      
      // Initialize speech synthesis on user interaction (browsers require this)
      if ('speechSynthesis' in window) {
        // Wake up the speech engine
        window.speechSynthesis.cancel();
        // Force load voices - some browsers need this
        const loadVoices = () => {
          const voices = window.speechSynthesis.getVoices();
          console.log('Voices loaded:', voices.length);
        };
        loadVoices();
        // Chrome needs this event
        window.speechSynthesis.onvoiceschanged = loadVoices;
        
        // Test speech immediately to "warm up" the engine
        const testUtterance = new SpeechSynthesisUtterance('');
        testUtterance.volume = 0;
        window.speechSynthesis.speak(testUtterance);
      }

      // Get camera stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      
      // Set initial pose quality
      setPoseQuality({
        type: 'NONE',
        coverage: 0,
        missing_joints: [],
        message: 'Initializing...',
        avg_confidence: 0
      });

      // Connect WebSocket
      const ws = new WebSocket(`${WS_URL}/biomechanics/realtime/${userId}`);
      
      ws.onopen = () => {
        setIsConnected(true);
        toast({
          title: 'Connected',
          description: 'Real-time analysis started',
        });
        
        // Start sending frames (10 FPS for real-time feedback)
        frameIntervalRef.current = setInterval(captureAndSendFrame, 150);
      };

      ws.onmessage = handleWSMessage;

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        toast({
          title: 'Connection Error',
          description: 'Failed to connect to analysis server',
          variant: 'destructive',
        });
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (frameIntervalRef.current) {
          clearInterval(frameIntervalRef.current);
        }
      };

      wsRef.current = ws;
      setIsStreaming(true);

    } catch (err: any) {
      console.error('Camera error:', err);
      setCameraError(err.message || 'Failed to access camera');
      toast({
        title: 'Camera Error',
        description: 'Please allow camera access and try again',
        variant: 'destructive',
      });
    }
  };

  // Stop streaming
  const stopStreaming = () => {
    // Stop frame capture
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Clear overlay
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }

    // Cancel speech
    window.speechSynthesis.cancel();

    // Reset all state
    setIsStreaming(false);
    setIsConnected(false);
    setKeypoints([]);
    setFormScore(0);
    setFeedback('');
    setPoseQuality(null);
    setMovementState('STATIC');
    setRepCount(0);
    setCorrectReps(0);
    setIncorrectReps(0);
    setProcessedFrame(null);
    setLastSpokenFeedback('');
  };

  // Reset rep counter
  const resetRepCount = () => {
    setRepCount(0);
    setCorrectReps(0);
    setIncorrectReps(0);
    toast({
      title: 'Reps Reset',
      description: 'Rep counter has been reset to 0',
    });
  };

  // Get score color
  const getScoreColor = (score: number) => {
    if (score >= 85) return 'text-green-500';
    if (score >= 70) return 'text-yellow-500';
    if (score >= 50) return 'text-orange-500';
    return 'text-red-500';
  };

  const getScoreBgColor = (score: number) => {
    if (score >= 85) return 'bg-green-500';
    if (score >= 70) return 'bg-yellow-500';
    if (score >= 50) return 'bg-orange-500';
    return 'bg-red-500';
  };

  // Get pose quality badge color and text
  const getPoseQualityBadge = () => {
    if (!poseQuality) return { variant: 'secondary' as const, text: 'Waiting...' };
    switch (poseQuality.type) {
      case 'FULL':
        return { variant: 'default' as const, text: movementState === 'MOVING' ? '● Active' : '● Ready' };
      case 'PARTIAL':
        return { variant: 'secondary' as const, text: '◐ Partial' };
      case 'NONE':
        return { variant: 'destructive' as const, text: '○ No Pose' };
      default:
        return { variant: 'secondary' as const, text: 'Detecting...' };
    }
  };

  // Get score description based on pose quality and movement
  const getScoreDescription = () => {
    if (!poseQuality || poseQuality.type === 'NONE') {
      return 'Step into frame to begin';
    }
    if (poseQuality.type === 'PARTIAL') {
      return 'Adjust position for full body view';
    }
    if (movementState === 'STATIC') {
      return 'Ready position - start your workout';
    }
    // Moving with full pose
    if (formScore >= 85) return 'Excellent form!';
    if (formScore >= 70) return 'Good form, minor adjustments needed';
    if (formScore >= 50) return 'Focus on technique';
    if (formScore > 0) return 'Needs improvement';
    return 'Analyzing...';
  };

  return (
    <div className="container mx-auto p-4 max-w-7xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Activity className="h-8 w-8 text-primary" />
          Live Form Analysis
        </h1>
        <p className="text-muted-foreground mt-2">
          Real-time AI-powered exercise form feedback with voice coaching
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Feed - Main Area */}
        <div className="lg:col-span-2">
          <Card className="overflow-hidden">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Camera Feed
                </CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setVoiceEnabled(!voiceEnabled)}
                    title={voiceEnabled ? 'Mute voice feedback' : 'Enable voice feedback'}
                  >
                    {voiceEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="relative aspect-video bg-black">
                {/* Hidden video element for camera capture */}
                <video
                  ref={videoRef}
                  className={`w-full h-full object-contain ${processedFrame ? 'hidden' : ''}`}
                  playsInline
                  muted
                />
                
                {/* Processed frame from MediaPipe (with skeleton drawn) */}
                {processedFrame && (
                  <img
                    src={processedFrame}
                    alt="Processed frame"
                    className="w-full h-full object-contain"
                  />
                )}

                {/* Hidden canvas for frame capture */}
                <canvas ref={canvasRef} className="hidden" />

                {/* Form Score Badge */}
                {isStreaming && formScore > 0 && (
                  <div className="absolute top-4 right-4">
                    <div className={`w-20 h-20 rounded-full ${getScoreBgColor(formScore)} flex items-center justify-center shadow-lg`}>
                      <span className="text-white text-2xl font-bold">{Math.round(formScore)}</span>
                    </div>
                  </div>
                )}

                {/* Connection and Pose Quality Status */}
                {isStreaming && (
                  <div className="absolute top-4 left-4 flex flex-col gap-2">
                    <Badge variant={isConnected ? 'default' : 'destructive'}>
                      {isConnected ? '● Live' : '○ Connecting...'}
                    </Badge>
                    <Badge variant={getPoseQualityBadge().variant}>
                      {getPoseQualityBadge().text}
                    </Badge>
                  </div>
                )}

                {/* Rep Counter - show when moving */}
                {isStreaming && repCount > 0 && (
                  <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
                    <div className="bg-primary text-primary-foreground px-4 py-2 rounded-full shadow-lg flex items-center gap-2">
                      <span className="text-2xl font-bold">{repCount}</span>
                      <span className="text-sm">reps</span>
                    </div>
                  </div>
                )}

                {/* Processing time and movement state */}
                {isStreaming && processingTime > 0 && (
                  <div className="absolute bottom-4 left-4 flex gap-2">
                    <Badge variant="secondary">
                      {processingTime.toFixed(0)}ms
                    </Badge>
                    {movementState === 'MOVING' && (
                      <Badge variant="default" className="bg-green-500">
                        Moving
                      </Badge>
                    )}
                  </div>
                )}

                {/* Camera error */}
                {cameraError && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                    <Alert variant="destructive" className="max-w-md">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{cameraError}</AlertDescription>
                    </Alert>
                  </div>
                )}

                {/* Start prompt */}
                {!isStreaming && !cameraError && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-white">
                      <Camera className="h-16 w-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg">Click "Start Analysis" to begin</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="p-4 flex flex-wrap gap-4 items-center justify-between border-t">
                <div className="flex gap-4 items-center">
                  <Select
                    value={exerciseType}
                    onValueChange={(v) => setExerciseType(v as ExerciseType)}
                    disabled={isStreaming}
                  >
                    <SelectTrigger className="w-40">
                      <SelectValue placeholder="Exercise" />
                    </SelectTrigger>
                    <SelectContent>
                      {EXERCISE_OPTIONS.map(opt => (
                        <SelectItem key={opt.value} value={opt.value}>
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex gap-2">
                  {!isStreaming ? (
                    <Button onClick={startStreaming} className="gap-2">
                      <Play className="h-4 w-4" />
                      Start Analysis
                    </Button>
                  ) : (
                    <Button onClick={stopStreaming} variant="destructive" className="gap-2">
                      <Square className="h-4 w-4" />
                      Stop
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar - Feedback & Stats */}
        <div className="space-y-6">
          {/* Form Score Card */}
          <Card>
            <CardHeader>
              <CardTitle>Form Score</CardTitle>
              <CardDescription>Real-time quality assessment</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center mb-4">
                <span className={`text-6xl font-bold ${getScoreColor(formScore)}`}>
                  {Math.round(formScore)}
                </span>
                <span className="text-2xl text-muted-foreground">/100</span>
              </div>
              <Progress value={formScore} className="h-3" />
              <p className="text-sm text-muted-foreground mt-2 text-center">
                {getScoreDescription()}
              </p>
            </CardContent>
          </Card>

          {/* Rep Counter Card */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <CardTitle>Rep Counter</CardTitle>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={resetRepCount}
                  disabled={!isStreaming}
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Reset
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                  <span className="text-4xl font-bold text-green-600 dark:text-green-400">{correctReps}</span>
                  <p className="text-sm text-green-700 dark:text-green-300 mt-1">Correct</p>
                </div>
                <div className="text-center p-3 bg-red-100 dark:bg-red-900/30 rounded-lg">
                  <span className="text-4xl font-bold text-red-600 dark:text-red-400">{incorrectReps}</span>
                  <p className="text-sm text-red-700 dark:text-red-300 mt-1">Incorrect</p>
                </div>
              </div>
              <p className="text-muted-foreground text-center mt-3 text-sm">
                {movementState === 'MOVING' ? 'Keep going!' : 'Face sideways to begin squats'}
              </p>
            </CardContent>
          </Card>

          {/* Current Feedback */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Volume2 className="h-5 w-5" />
                Live Feedback
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="min-h-[80px] flex items-center justify-center">
                {feedback ? (
                  <p className={`text-xl font-medium text-center ${
                    poseQuality?.type === 'NONE' ? 'text-destructive' :
                    poseQuality?.type === 'PARTIAL' ? 'text-yellow-600' :
                    formScore < 60 ? 'text-orange-500' : ''
                  }`}>{feedback}</p>
                ) : (
                  <p className="text-muted-foreground text-center">
                    {isStreaming ? 'Position yourself for analysis...' : 'Feedback will appear here during analysis'}
                  </p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Detected Issues */}
          <Card>
            <CardHeader>
              <CardTitle>Form Issues</CardTitle>
              <CardDescription>Areas needing attention</CardDescription>
            </CardHeader>
            <CardContent>
              {issues.length > 0 ? (
                <div className="space-y-2">
                  {issues.map((issue, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center justify-between p-2 rounded-lg ${
                        issue.is_critical ? 'bg-red-100 dark:bg-red-900/20 border border-red-300' : 'bg-muted/50'
                      }`}
                    >
                      <span className={`text-sm ${issue.is_critical ? 'font-semibold text-red-700 dark:text-red-400' : ''}`}>
                        {issue.is_critical && '⚠️ '}{issue.message}
                      </span>
                      <Badge
                        variant={issue.is_critical ? 'destructive' : issue.severity > 0.6 ? 'secondary' : 'outline'}
                      >
                        {issue.is_critical ? 'Critical' : issue.severity > 0.7 ? 'High' : issue.severity > 0.4 ? 'Med' : 'Low'}
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-4">
                  {isStreaming 
                    ? (movementState === 'MOVING' 
                        ? 'No issues detected - great form!' 
                        : 'Start exercising to detect issues')
                    : 'Start analysis to detect issues'}
                </p>
              )}
            </CardContent>
          </Card>

          {/* Exercise Tips */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                {exerciseType.replace('_', ' ').toUpperCase()} Tips
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="text-sm space-y-2 text-muted-foreground">
                {exerciseType === 'squat' && (
                  <>
                    <li className="font-semibold text-primary">📷 Stand SIDEWAYS to camera</li>
                    <li>• Show full body from head to feet</li>
                    <li>• Keep feet shoulder-width apart</li>
                    <li>• Push knees out over toes</li>
                    <li>• Maintain neutral spine</li>
                    <li>• Aim for parallel or below</li>
                  </>
                )}
                {exerciseType === 'deadlift' && (
                  <>
                    <li>• Keep bar close to body</li>
                    <li>• Hinge at the hips</li>
                    <li>• Neutral spine throughout</li>
                    <li>• Drive through heels</li>
                  </>
                )}
                {exerciseType === 'pushup' && (
                  <>
                    <li>• Hands shoulder-width apart</li>
                    <li>• Keep body in straight line</li>
                    <li>• Elbows at 45° angle</li>
                    <li>• Full range of motion</li>
                  </>
                )}
                {exerciseType === 'bench_press' && (
                  <>
                    <li>• Grip slightly wider than shoulders</li>
                    <li>• Retract shoulder blades</li>
                    <li>• Touch chest, don't bounce</li>
                    <li>• Drive feet into floor</li>
                  </>
                )}
                {exerciseType === 'lunge' && (
                  <>
                    <li>• Front knee over ankle</li>
                    <li>• Back knee toward floor</li>
                    <li>• Torso upright</li>
                    <li>• Control the descent</li>
                  </>
                )}
                {(exerciseType === 'overhead_press' || exerciseType === 'row') && (
                  <>
                    <li>• Maintain core stability</li>
                    <li>• Control the movement</li>
                    <li>• Full range of motion</li>
                    <li>• Breathe properly</li>
                  </>
                )}
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
