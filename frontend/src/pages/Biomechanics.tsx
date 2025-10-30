import { useState, useRef, useCallback, useEffect } from "react";
import { Video, Upload, TrendingUp, AlertCircle, Camera, Square, Play, StopCircle, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useBiomechanics } from "@/hooks/use-fitbalance";
import { useToast } from "@/hooks/use-toast";
import SimpleCameraTest from "@/components/SimpleCameraTest";
import TorqueHeatmap from '@/components/TorqueHeatmap';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

const exercises = [
  "Squat",
  "Deadlift",
  "Bench Press",
  "Overhead Press",
  "Row",
  "Lunge",
];

export default function Biomechanics() {
  const [selectedExercise, setSelectedExercise] = useState<string>("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState<string | null>(null);
  const [cameraPermission, setCameraPermission] = useState<'granted' | 'denied' | 'prompt'>('prompt');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunks = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);

  const { isLoading: isAnalyzing, error, analysis, analyzeMovement } = useBiomechanics();
  const { toast } = useToast();

  // Initialize camera when component mounts
  useEffect(() => {
    checkCameraPermission();
    return () => {
      // Cleanup camera stream when component unmounts
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]); // Add stream dependency to cleanup properly

  // Separate effect to handle video element updates
  useEffect(() => {
    if (stream && videoRef.current) {
      console.log('Effect: Setting up video element with stream');
      const video = videoRef.current;

      // Reset video element completely
      video.srcObject = null;
      video.load();

      // Set new source
      video.srcObject = stream;

      const handleLoadedMetadata = () => {
        console.log('Effect: Video metadata loaded', {
          width: video.videoWidth,
          height: video.videoHeight,
          readyState: video.readyState
        });

        // Ensure video plays
        video.play().catch(err => {
          console.error('Effect: Play failed:', err);
        });
      };

      const handleCanPlay = () => {
        console.log('Effect: Video can play');
        if (video.paused) {
          video.play().catch(console.error);
        }
      };

      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('canplay', handleCanPlay);

      // Cleanup function
      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('canplay', handleCanPlay);
      };
    }
  }, [stream]);

  const checkCameraPermission = async () => {
    try {
      const result = await navigator.permissions.query({ name: 'camera' as PermissionName });
      setCameraPermission(result.state);
    } catch (err) {
      console.warn('Permission API not supported, will try direct access');
    }
  };

  const startCamera = async () => {
    try {
      console.log('Requesting camera access...');
      setVideoError(null);

      // Start with the simplest possible constraints
      let mediaStream;

      try {
        // Try simple constraints first
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
        console.log('Simple constraints worked!');
      } catch (simpleError) {
        console.log('Simple constraints failed, trying with specific settings...');
        // If simple fails, try with more specific constraints
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { min: 320, ideal: 640, max: 1280 },
            height: { min: 240, ideal: 480, max: 720 },
            frameRate: { ideal: 30, max: 60 }
          },
          audio: false
        });
      }

      console.log('Camera stream obtained!');
      console.log('Stream details:', {
        id: mediaStream.id,
        active: mediaStream.active,
        videoTracks: mediaStream.getVideoTracks().length
      });

      if (mediaStream.getVideoTracks().length === 0) {
        throw new Error('No video tracks available');
      }

      const videoTrack = mediaStream.getVideoTracks()[0];
      console.log('Video track settings:', videoTrack.getSettings());
      console.log('Video track state:', videoTrack.readyState);

      setStream(mediaStream);
      setVideoError(null);
      setCameraPermission('granted');

      // Don't set srcObject here - let the useEffect handle it

      toast({
        title: "Camera Access Granted",
        description: "Camera is ready for recording.",
      });
    } catch (err) {
      console.error('Error accessing camera:', err);
      setCameraPermission('denied');
      setVideoError(`Camera error: ${err.message}`);
      toast({
        title: "Camera Access Denied",
        description: `Please allow camera access. Error: ${err.message}`,
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startRecording = useCallback(() => {
    if (!stream || !selectedExercise) return;

    recordedChunks.current = [];

    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=vp8'
    });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks.current, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      setRecordedVideoUrl(url);

      // Auto-analyze the recorded video
      analyzeRecordedVideo(blob);
    };

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setIsRecording(true);

    toast({
      title: "Recording Started",
      description: "Recording your exercise. Click stop when finished.",
    });
  }, [stream, selectedExercise]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      toast({
        title: "Recording Stopped",
        description: "Processing your video for analysis...",
      });
    }
  }, [isRecording]);

  const analyzeRecordedVideo = async (videoBlob: Blob) => {
    try {
      const file = new File([videoBlob], `exercise-${selectedExercise}-${Date.now()}.webm`, {
        type: 'video/webm'
      });

      await analyzeMovement(file, selectedExercise, 'user123');

      toast({
        title: "Analysis Complete",
        description: "Your exercise has been analyzed successfully!",
      });
    } catch (err) {
      console.error('Analysis failed:', err);
      toast({
        title: "Analysis Failed",
        description: "Unable to analyze the video. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('File upload triggered!', event);

    const file = event.target.files?.[0];
    console.log('Selected file:', file);

    if (!file) {
      console.log('No file selected');
      return;
    }

    if (!selectedExercise) {
      console.log('No exercise selected');
      toast({
        title: "Exercise Not Selected",
        description: "Please select an exercise type before uploading.",
        variant: "destructive",
      });
      return;
    }

    // Validate file type for video
    if (!file.type.startsWith('video/')) {
      console.log('Invalid file type:', file.type);
      toast({
        title: "Invalid File Type",
        description: "Please select a valid video file (MP4, WebM, MOV, etc.).",
        variant: "destructive",
      });
      return;
    }

    console.log('Starting upload process for file:', file.name, file.type, file.size);

    try {
      toast({
        title: "Uploading Video",
        description: "Your video is being analyzed...",
      });

      console.log('Calling analyzeMovement...');
      const result = await analyzeMovement(file, selectedExercise, 'user123');
      console.log('Analysis result:', result);

      toast({
        title: "Upload Successful",
        description: "Your video has been analyzed successfully!",
      });
    } catch (err) {
      console.error('Upload error:', err);
      toast({
        title: "Upload Failed",
        description: `Unable to analyze the uploaded video: ${err.message || 'Unknown error'}`,
        variant: "destructive",
      });
    }

    // Reset file input
    if (event.target) {
      event.target.value = '';
    }
  };

  const handlePhotoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !selectedExercise) {
      if (!selectedExercise) {
        toast({
          title: "Exercise Not Selected",
          description: "Please select an exercise type before uploading.",
          variant: "destructive",
        });
      }
      return;
    }

    // Validate file type for image
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid File Type",
        description: "Please select a valid image file (JPG, PNG, GIF, etc.).",
        variant: "destructive",
      });
      return;
    }

    try {
      toast({
        title: "Uploading Photo",
        description: "Your photo is being analyzed...",
      });

      // For now, we'll treat photos similar to videos for analysis
      // In a real implementation, you might have a separate endpoint for photos
      await analyzeMovement(file, selectedExercise, 'user123');

      toast({
        title: "Upload Successful",
        description: "Your photo has been analyzed successfully!",
      });
    } catch (err) {
      console.error('Photo upload error:', err);
      toast({
        title: "Upload Failed",
        description: "Unable to analyze the uploaded photo. Please try again.",
        variant: "destructive",
      });
    }

    // Reset file input
    if (event.target) {
      event.target.value = '';
    }
  };

  const getRiskLevelColor = (riskLevel?: string) => {
    switch (riskLevel) {
      case 'low': return 'text-green-500';
      case 'medium': return 'text-yellow-500';
      case 'high': return 'text-red-500';
      default: return 'text-muted-foreground';
    }
  };

  const getFormScoreColor = (score?: number) => {
    if (!score) return 'text-muted-foreground';
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="min-h-screen px-4 py-8 relative z-10">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary mb-4">
            <Video className="h-4 w-4" />
            <span className="text-sm font-medium">Biomechanics Analysis</span>
          </div>
          <h1 className="text-4xl font-bold mb-2">Movement Coaching</h1>
          <p className="text-muted-foreground">
            Record your exercises for real-time form analysis and personalized coaching
          </p>
        </div>

        {/* Video Capture Section */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardHeader>
            <CardTitle>Record Exercise</CardTitle>
            <CardDescription>
              Select your exercise type and record a video for analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select value={selectedExercise} onValueChange={setSelectedExercise}>
              <SelectTrigger>
                <SelectValue placeholder="Select exercise type" />
              </SelectTrigger>
              <SelectContent>
                {exercises.map((exercise) => (
                  <SelectItem key={exercise} value={exercise.toLowerCase()}>
                    {exercise}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Camera Permission Alert */}
            {cameraPermission === 'denied' && (
              <Alert className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Camera access is required to record videos. Please enable camera permissions in your browser settings.
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {videoError && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{videoError}</AlertDescription>
              </Alert>
            )}

            <div className="aspect-video bg-gray-200 rounded-lg overflow-hidden border-2 border-dashed border-border relative">
              {stream ? (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    controls={false}
                    className="w-full h-full"
                    style={{
                      objectFit: 'cover',
                      backgroundColor: '#000000',
                      display: 'block',
                      transform: 'scaleX(-1)' // Mirror for better UX
                    }}
                    onLoadedMetadata={(e) => {
                      const video = e.target as HTMLVideoElement;
                      console.log('Video loaded metadata:', {
                        videoWidth: video.videoWidth,
                        videoHeight: video.videoHeight,
                        duration: video.duration,
                        readyState: video.readyState
                      });
                    }}
                    onCanPlay={(e) => {
                      const video = e.target as HTMLVideoElement;
                      console.log('Video can play');
                      if (video.paused) {
                        video.play().catch(e => console.error('Play failed:', e));
                      }
                    }}
                    onPlay={() => console.log('Video is playing - should be visible!')}
                    onError={(e) => {
                      console.error('Video error:', e);
                      setVideoError('Video playback error');
                    }}
                    onLoadedData={() => console.log('Video data loaded')}
                    onTimeUpdate={() => {
                      // This fires when video is actually playing
                      // console.log('Video time update - video is definitely playing');
                    }}
                  />
                  {/* Debug info overlay - more detailed */}
                  <div className="absolute bottom-2 left-2 bg-black/80 text-white text-xs px-2 py-1 rounded max-w-xs">
                    <div>Stream: {stream ? 'Active' : 'None'}</div>
                    <div>Tracks: {stream?.getVideoTracks().length || 0}</div>
                    <div>State: {stream?.getVideoTracks()[0]?.readyState || 'N/A'}</div>
                    <div>Video: {videoRef.current?.videoWidth || 0}x{videoRef.current?.videoHeight || 0}</div>
                    <div>Playing: {videoRef.current?.paused === false ? 'Yes' : 'No'}</div>
                  </div>
                </>
              ) : recordedVideoUrl ? (
                <video
                  src={recordedVideoUrl}
                  controls
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <Camera className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground mb-4">
                      {cameraPermission === 'granted'
                        ? 'Camera ready - Select an exercise to start recording'
                        : 'Click "Start Camera" to begin'
                      }
                    </p>
                  </div>
                </div>
              )}

              {/* Recording indicator */}
              {isRecording && (
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  Recording
                </div>
              )}
            </div>

            <div className="flex gap-4 flex-wrap">
              {!stream ? (
                <Button
                  onClick={startCamera}
                  className="flex-1 gradient-biomechanics text-white font-medium"
                  disabled={isAnalyzing}
                >
                  <Camera className="mr-2 h-4 w-4" />
                  Start Camera
                </Button>
              ) : !isRecording ? (
                <>
                  <Button
                    onClick={startRecording}
                    className="flex-1 gradient-biomechanics text-white font-medium"
                    disabled={!selectedExercise || isAnalyzing}
                  >
                    <Video className="mr-2 h-4 w-4" />
                    Start Recording
                  </Button>
                  <Button
                    onClick={stopCamera}
                    variant="outline"
                    className="flex-1 border-red-500 text-red-500 hover:bg-red-500 hover:text-white font-medium"
                  >
                    <Square className="mr-2 h-4 w-4" />
                    Stop Camera
                  </Button>
                </>
              ) : (
                <Button
                  onClick={stopRecording}
                  variant="destructive"
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white font-medium"
                >
                  <StopCircle className="mr-2 h-4 w-4" />
                  Stop Recording
                </Button>
              )}

              {/* Simplified Upload Video Button */}
              <div className="flex-1">
                <input
                  id="video-upload"
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
                <Button
                  variant="outline"
                  className="w-full border-blue-500 text-blue-600 hover:bg-blue-500 hover:text-white font-medium"
                  disabled={!selectedExercise || isAnalyzing}
                  onClick={() => {
                    const input = document.getElementById('video-upload') as HTMLInputElement;
                    if (input) input.click();
                  }}
                >
                  <Upload className="mr-2 h-4 w-4" />
                  Upload Video
                </Button>
              </div>

              {/* Simplified Upload Photo Button */}
              <div className="flex-1">
                <input
                  id="photo-upload"
                  type="file"
                  accept="image/*"
                  onChange={handlePhotoUpload}
                  style={{ display: 'none' }}
                />
                <Button
                  variant="outline"
                  className="w-full border-green-500 text-green-600 hover:bg-green-500 hover:text-white font-medium"
                  disabled={!selectedExercise || isAnalyzing}
                  onClick={() => {
                    const input = document.getElementById('photo-upload') as HTMLInputElement;
                    if (input) input.click();
                  }}
                >
                  <Camera className="mr-2 h-4 w-4" />
                  Upload Photo
                </Button>
              </div>
            </div>

            {isAnalyzing && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Analyzing movement...</span>
                  <span className="font-medium">Processing</span>
                </div>
                <Progress value={undefined} className="h-2" />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Results Placeholder */}
        <div className="grid md:grid-cols-2 gap-6 animate-slide-up">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Form Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className={`text-5xl font-bold mb-2 ${getFormScoreColor(analysis?.form_score)}`}>
                  {analysis?.form_score ? Math.round(analysis.form_score) : '--'}
                </div>
                <p className="text-sm text-muted-foreground">
                  {analysis?.form_score
                    ? `${analysis.form_score >= 80 ? 'Excellent' : analysis.form_score >= 60 ? 'Good' : 'Needs Improvement'} form`
                    : 'Record an exercise to see your form score'
                  }
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-orange-500" />
                Risk Level
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className={`text-3xl font-bold mb-2 ${getRiskLevelColor(analysis?.torques?.[0]?.risk_level)}`}>
                  {analysis?.torques?.[0]?.risk_level?.toUpperCase() || '--'}
                </div>
                <p className="text-sm text-muted-foreground">
                  {analysis?.torques?.[0]?.risk_level
                    ? 'Based on joint torque analysis'
                    : 'Risk assessment will appear here'
                  }
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Recommendations Placeholder */}
        <Card className="glass-card mt-6 animate-slide-up">
          <CardHeader>
            <CardTitle>AI Coaching Recommendations</CardTitle>
            <CardDescription>
              Personalized tips to improve your form and prevent injuries
            </CardDescription>
          </CardHeader>
          <CardContent>
            {analysis?.recommendations ? (
              <div className="space-y-3">
                {analysis.recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                    <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center text-white text-sm font-medium flex-shrink-0 mt-0.5">
                      {index + 1}
                    </div>
                    <p className="text-sm">{recommendation}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                Complete an analysis to receive personalized coaching recommendations
              </div>
            )}
          </CardContent>
        </Card>

        {/* Simple Camera Test for debugging */}
        <div className="mt-6">
          <SimpleCameraTest />
        </div>
      </div>
    </div>
  );
}
