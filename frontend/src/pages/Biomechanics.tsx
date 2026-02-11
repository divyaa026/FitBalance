import { useState, useRef } from "react";
import { Video, Upload, TrendingUp, AlertCircle, Camera, Lightbulb, Activity, CheckCircle, AlertTriangle, XCircle, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useBiomechanics } from "@/hooks/use-fitbalance";
import { useToast } from "@/hooks/use-toast";
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { TorqueHeatmap } from '@/components/TorqueHeatmap';
import { BodyStressMap } from '@/components/BodyStressMap';
import { Badge } from "@/components/ui/badge";

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
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [exerciseDetectionError, setExerciseDetectionError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);

  const { isLoading: isAnalyzing, error, analysis, analyzeMovement } = useBiomechanics();
  const { toast } = useToast();


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

    // Create URL for video preview
    const videoUrl = URL.createObjectURL(file);
    setUploadedVideoUrl(videoUrl);
    setUploadedImageUrl(null);  // Clear any uploaded image
    setExerciseDetectionError(null);  // Clear previous errors

    try {
      toast({
        title: "Uploading Video",
        description: "Your video is being analyzed...",
      });

      console.log('Calling analyzeMovement...');
      const result = await analyzeMovement(file, selectedExercise, 'user123');
      console.log('Analysis result:', result);

      // Check if exercise was detected
      if (result && !result.is_valid_exercise) {
        setExerciseDetectionError(result.error_message || "No exercise detected in the video.");
        toast({
          title: "No Exercise Detected",
          description: result.error_message || "Please upload a video showing a person performing an exercise.",
          variant: "destructive",
        });
      } else {
        setExerciseDetectionError(null);
        toast({
          title: "Upload Successful",
          description: "Your video has been analyzed successfully!",
        });
      }
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

    console.log('Starting photo upload for:', file.name, file.type, file.size);

    // Create URL for image preview
    const imageUrl = URL.createObjectURL(file);
    setUploadedImageUrl(imageUrl);
    setUploadedVideoUrl(null);  // Clear any uploaded video
    setExerciseDetectionError(null);  // Clear previous errors

    try {
      toast({
        title: "Uploading Photo",
        description: "Your photo is being analyzed...",
      });

      const result = await analyzeMovement(file, selectedExercise, 'user123');
      console.log('Photo analysis result:', result);

      // Check if exercise was detected
      if (result && !result.is_valid_exercise) {
        setExerciseDetectionError(result.error_message || "No exercise form detected in the image.");
        toast({
          title: "No Exercise Form Detected",
          description: result.error_message || "Please upload an image showing your exercise form.",
          variant: "destructive",
        });
      } else {
        setExerciseDetectionError(null);
        toast({
          title: "Upload Successful",
          description: "Your photo has been analyzed successfully!",
        });
      }
    } catch (err) {
      console.error('Photo upload error:', err);
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      toast({
        title: "Upload Failed",
        description: `Unable to analyze the uploaded photo: ${errorMsg}`,
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
            Upload your exercise videos for AI-powered form analysis and personalized coaching
          </p>
        </div>

        {/* Video Capture Section */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardHeader>
            <CardTitle>Analyze Exercise</CardTitle>
            <CardDescription>
              Select your exercise type and upload a video or photo for analysis
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

            {/* Exercise Detection Error Alert */}
            {exerciseDetectionError && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  <strong>No Exercise Detected:</strong> {exerciseDetectionError}
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="aspect-video bg-gray-200 rounded-lg overflow-hidden border-2 border-dashed border-border relative">
              {uploadedImageUrl ? (
                <img
                  src={uploadedImageUrl}
                  alt="Uploaded exercise form"
                  className="w-full h-full object-contain bg-black"
                />
              ) : uploadedVideoUrl ? (
                <video
                  src={uploadedVideoUrl}
                  controls
                  autoPlay
                  loop
                  className="w-full h-full object-contain bg-black"
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <Video className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground mb-4">
                      Upload a video or photo to analyze your exercise form
                    </p>
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-4 flex-wrap">
              {/* Upload Video Button */}
              <div className="flex-1">
                <input
                  id="video-upload"
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  style={{ position: 'absolute', left: '-9999px', width: '1px', height: '1px', overflow: 'hidden' }}
                />
                {(!selectedExercise || isAnalyzing) ? (
                  <Button
                    variant="outline"
                    className="w-full border-blue-500 text-blue-600 hover:bg-blue-500 hover:text-white font-medium"
                    disabled
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    Upload Video
                  </Button>
                ) : (
                  <label htmlFor="video-upload" className="w-full block">
                    <div className="inline-block w-full text-center border border-blue-500 text-blue-600 hover:bg-blue-500 hover:text-white font-medium px-4 py-2 rounded-md cursor-pointer">
                      <Upload className="mr-2 h-4 w-4 inline-block align-middle" />
                      <span className="align-middle">Upload Video</span>
                    </div>
                  </label>
                )}
              </div>

              {/* Upload Photo Button */}
              <div className="flex-1">
                <input
                  id="photo-upload"
                  ref={photoInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handlePhotoUpload}
                  style={{ position: 'absolute', left: '-9999px', width: '1px', height: '1px', overflow: 'hidden' }}
                />
                {(!selectedExercise || isAnalyzing) ? (
                  <Button
                    variant="outline"
                    className="w-full border-green-500 text-green-600 hover:bg-green-500 hover:text-white font-medium"
                    disabled
                  >
                    <Camera className="mr-2 h-4 w-4" />
                    Upload Photo
                  </Button>
                ) : (
                  <label htmlFor="photo-upload" className="w-full block">
                    <div className="inline-block w-full text-center border border-green-500 text-green-600 hover:bg-green-500 hover:text-white font-medium px-4 py-2 rounded-md cursor-pointer">
                      <Camera className="mr-2 h-4 w-4 inline-block align-middle" />
                      <span className="align-middle">Upload Photo</span>
                    </div>
                  </label>
                )}
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

        {analysis && (
  <>
    <Card className="glass-card mt-8 animate-slide-up">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          Form Analysis Results
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center mb-6">
          <div className="relative inline-flex items-center justify-center w-32 h-32">
            <svg className="w-full h-full transform -rotate-90">
              <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="8" fill="none" className="text-muted" />
              <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="8" fill="none"
                strokeDasharray={`${2 * Math.PI * 56}`}
                strokeDashoffset={`${2 * Math.PI * 56 * (1 - (analysis.form_score ?? 0) / 100)}`}
                className={`
                  ${analysis.form_score >= 80 ? 'text-green-500' : ''}
                  ${analysis.form_score >= 60 && analysis.form_score < 80 ? 'text-yellow-500' : ''}
                  ${analysis.form_score < 60 ? 'text-red-500' : ''}
                `}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-4xl font-bold">{Math.round(analysis.form_score ?? 0)}</span>
              <span className="text-xs text-muted-foreground">Form Score</span>
            </div>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            {analysis.form_score >= 80 && 'Excellent form! Keep it up.'}
            {analysis.form_score >= 60 && analysis.form_score < 80 && 'Good form with room for improvement.'}
            {analysis.form_score < 60 && 'Form needs work. Review recommendations below.'}
          </p>
        </div>
        {/* Joint Angles */}
        {analysis?.joint_angles?.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-semibold text-sm">Joint Angles</h4>
            {analysis.joint_angles.map((ja) => {
              const joint = (ja as any).joint_name ?? 'joint';
              const angleVal = (ja as any).angle ?? 0;
              const angleNum = typeof angleVal === 'number' ? angleVal : Number(angleVal);
              const displayAngle = Number.isFinite(angleNum) ? Math.round(angleNum) : '--';
              const progressValue = Number.isFinite(angleNum) ? (angleNum / 180) * 100 : 0;
              return (
                <div key={joint} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="capitalize">{String(joint).replace('_', ' ')}</span>
                    <span className="font-medium">{displayAngle}°</span>
                  </div>
                  <Progress value={progressValue} className="h-2" />
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
    {/* Risk Factors (derived from torque analysis) */}
    {analysis?.torques?.some(t => t.risk_level && t.risk_level !== 'low') && (
      <Card className="glass-card mt-8 animate-slide-up border-orange-500/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-orange-600">
            <AlertCircle className="h-5 w-5" />
            Risk Factors Detected
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {analysis?.torques?.filter(t => t.risk_level && t.risk_level !== 'low').map((t, index) => (
              <Alert key={index} variant="destructive" className="bg-orange-50 border-orange-200">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {`${t.joint_name.replace(/_/g, ' ')} - ${String(t.risk_level).toUpperCase()}`}
                </AlertDescription>
              </Alert>
            ))}
          </div>
        </CardContent>
      </Card>
    )}
    
    {/* Form Errors - Specific incorrect positions */}
    {analysis?.form_errors && analysis.form_errors.length > 0 && (
      <Card className="glass-card mt-8 animate-slide-up border-red-500/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <XCircle className="h-5 w-5" />
            Form Corrections Needed
          </CardTitle>
          <CardDescription>
            Specific areas where your form needs improvement
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {analysis.form_errors.map((error, index) => (
              <div 
                key={index} 
                className={`p-4 rounded-lg border-l-4 ${
                  error.severity === 'severe' ? 'bg-red-100 border-red-600' :
                  error.severity === 'moderate' ? 'bg-orange-100 border-orange-600' :
                  'bg-yellow-100 border-yellow-600'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {error.severity === 'severe' ? (
                      <XCircle className="h-5 w-5 text-red-600" />
                    ) : error.severity === 'moderate' ? (
                      <AlertTriangle className="h-5 w-5 text-orange-600" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-yellow-600" />
                    )}
                    <span className={`font-semibold ${
                      error.severity === 'severe' ? 'text-red-900' :
                      error.severity === 'moderate' ? 'text-orange-900' :
                      'text-yellow-900'
                    }`}>{error.body_part}</span>
                  </div>
                  <Badge 
                    variant={error.severity === 'severe' ? 'destructive' : 'secondary'}
                    className={
                      error.severity === 'severe' ? 'bg-red-600 text-white' :
                      error.severity === 'moderate' ? 'bg-orange-600 text-white' :
                      'bg-yellow-600 text-black'
                    }
                  >
                    {error.severity.toUpperCase()}
                  </Badge>
                </div>
                <p className={`text-sm font-medium mb-1 ${
                  error.severity === 'severe' ? 'text-red-800' :
                  error.severity === 'moderate' ? 'text-orange-800' :
                  'text-yellow-800'
                }`}>{error.issue}</p>
                <p className={`text-xs mb-2 ${
                  error.severity === 'severe' ? 'text-red-700' :
                  error.severity === 'moderate' ? 'text-orange-700' :
                  'text-yellow-700'
                }`}>
                  Current: {error.current_value.toFixed(1)}° | Expected: {error.expected_range[0]}° - {error.expected_range[1]}°
                </p>
                <div className={`p-2 rounded text-sm ${
                  error.severity === 'severe' ? 'bg-red-200 text-red-900' :
                  error.severity === 'moderate' ? 'bg-orange-200 text-orange-900' :
                  'bg-yellow-200 text-yellow-900'
                }`}>
                  <span className="font-medium">💡 Fix: </span>{error.correction_tip}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )}
    
    {/* Recommendations */}
    {analysis.recommendations && analysis.recommendations.length > 0 && (
      <Card className="glass-card mt-8 animate-slide-up">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            Personalized Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {analysis.recommendations.map((rec, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-muted/30 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm">{rec}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )}
  {/* Body Stress Map: Visual representation of joint stress */}
  {(analysis as any)?.heatmap_data && (
        <Card className="glass-card mt-8 animate-slide-up">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            Joint Stress Analysis
          </CardTitle>
          <CardDescription>
            Visual representation of stress on your joints during the exercise. Click on a joint to see details.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <BodyStressMap data={(analysis as any).heatmap_data} formScore={analysis?.form_score} />
        </CardContent>
      </Card>
    )}
  </>
)}

        {/* Analysis Results Placeholder */}
        <div className="grid md:grid-cols-2 gap-6 mt-10 animate-slide-up">
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
        <Card className="glass-card mt-8 animate-slide-up">
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
      </div>
    </div>
  );
}
