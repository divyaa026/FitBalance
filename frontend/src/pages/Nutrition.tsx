import { useState, useRef, useEffect } from "react";
import { Camera, Upload, TrendingUp, Target, AlertCircle, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useNutrition } from "@/hooks/use-fitbalance";
import { useToast } from "@/hooks/use-toast";
import { useNutritionHistory, useNutritionStats } from '@/hooks/use-fitbalance';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import {
  LineChart as ReLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

export default function Nutrition() {
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const { isLoading: isAnalyzing, error, analysis, analyzeMeal } = useNutrition();
  const { toast } = useToast();

  const [historyDays, setHistoryDays] = useState(7);
  const currentUserId = 123; // In production, get from auth context

  const { data: history, isLoading: isLoadingHistory } = useNutritionHistory(currentUserId, historyDays);
  const { data: stats, isLoading: isLoadingStats } = useNutritionStats(currentUserId, 7);

  // Helper for weekly chart
  const processWeeklyData = (meals: any[]) => {
    const dailyData: { [key: string]: { protein: number; calories: number; count: number } } = {};
    meals.forEach(meal => {
      const date = new Date(meal.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      if (!dailyData[date]) dailyData[date] = { protein: 0, calories: 0, count: 0 };
      dailyData[date].protein += meal.total_protein || 0;
      dailyData[date].calories += meal.total_calories || 0;
      dailyData[date].count += 1;
    });
    return Object.entries(dailyData)
      .map(([date, data]) => ({ date, protein: Math.round(data.protein), calories: Math.round(data.calories) }))
      .slice(-7);
  };

  // Effect to handle video element when stream changes
  useEffect(() => {
    if (cameraStream && videoRef.current) {
      const video = videoRef.current;
      
      console.log('Setting up video element with stream');
      
      // Reset video element
      video.srcObject = null;
      video.load();
      
      // Set the new stream
      video.srcObject = cameraStream;
      
      const handleLoadedMetadata = () => {
        console.log('Video metadata loaded:', {
          width: video.videoWidth,
          height: video.videoHeight,
          readyState: video.readyState
        });
        
        // Force play
        video.play().then(() => {
          console.log('Video playing successfully');
        }).catch(err => {
          console.error('Video play failed:', err);
          setCameraError('Video playback failed');
        });
      };
      
      const handleCanPlay = () => {
        console.log('Video can play');
        if (video.paused) {
          video.play().catch(console.error);
        }
      };
      
      const handleError = (e) => {
        console.error('Video error:', e);
        setCameraError('Video display error');
      };
      
      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('canplay', handleCanPlay);
      video.addEventListener('error', handleError);
      
      // Cleanup
      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('canplay', handleCanPlay);
        video.removeEventListener('error', handleError);
      };
    }
  }, [cameraStream]);

  const startCamera = async () => {
    try {
      setCameraError(null);
      
      // Try different camera constraints for laptops
      let stream;
      
      try {
        // First try: Simple constraints
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
        console.log('Simple constraints worked');
      } catch (err) {
        console.log('Simple constraints failed, trying specific settings');
        // Second try: Specific constraints
        stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          },
          audio: false
        });
      }
      
      console.log('Camera stream obtained:', {
        id: stream.id,
        active: stream.active,
        tracks: stream.getVideoTracks().length
      });
      
      setCameraStream(stream);
      
      toast({
        title: "Camera Started",
        description: "Point your camera at your meal and click capture.",
      });
    } catch (err) {
      console.error('Camera error:', err);
      setCameraError(`Camera access failed: ${err.message}`);
      toast({
        title: "Camera Access Failed",
        description: "Please allow camera access and try again.",
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const capturePhoto = () => {
    if (!cameraStream || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob and analyze
    canvas.toBlob(async (blob) => {
      if (blob) {
        const imageUrl = URL.createObjectURL(blob);
        setCapturedImage(imageUrl);
        
        // Create file from blob for analysis
        const file = new File([blob], `meal-${Date.now()}.jpg`, { type: 'image/jpeg' });
        
        try {
          toast({
            title: "Analyzing Meal",
            description: "AI is analyzing your meal photo...",
          });
          
          await analyzeMeal(file, 'user123', []);
          
          toast({
            title: "Analysis Complete",
            description: "Your meal has been analyzed successfully!",
          });
        } catch (err) {
          console.error('Analysis error:', err);
          toast({
            title: "Analysis Failed",
            description: "Unable to analyze the meal. Please try again.",
            variant: "destructive",
          });
        }
      }
    }, 'image/jpeg', 0.8);

    // Stop camera after capture
    stopCamera();
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid File Type",
        description: "Please select a valid image file (JPG, PNG, etc.).",
        variant: "destructive",
      });
      return;
    }

    try {
      toast({
        title: "Uploading Photo",
        description: "Your meal photo is being analyzed...",
      });
      
      await analyzeMeal(file, 'user123', []);
      
      // Show preview of uploaded image
      const imageUrl = URL.createObjectURL(file);
      setCapturedImage(imageUrl);
      
      toast({
        title: "Upload Successful",
        description: "Your meal has been analyzed successfully!",
      });
    } catch (err) {
      console.error('Upload error:', err);
      toast({
        title: "Upload Failed",
        description: "Unable to analyze the uploaded image. Please try again.",
        variant: "destructive",
      });
    }
    
    // Reset file input
    if (event.target) {
      event.target.value = '';
    }
  };
  return (
    <div className="min-h-screen px-4 py-8 relative z-10">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/10 text-secondary mb-4">
            <Camera className="h-4 w-4" />
            <span className="text-sm font-medium">Intelligent Nutrition</span>
          </div>
          <h1 className="text-4xl font-bold mb-2">Meal Analysis</h1>
          <p className="text-muted-foreground">
            Capture your meals for instant AI-powered nutritional insights
          </p>
        </div>

        {/* Daily Progress Dashboard */}
        {(isLoadingStats ? <LoadingSpinner message="Loading stats..." /> :
          <Card className="glass-card mb-6 animate-slide-up">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Daily Nutrition Progress
              </CardTitle>
              <CardDescription>Your nutrition goals for today</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Protein Progress */}
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Protein</span>
                    <span className="text-sm text-muted-foreground">{stats?.total_protein || 0}g / 150g</span>
                  </div>
                  <Progress value={((stats?.total_protein || 0) / 150) * 100} className="h-3" />
                </div>
                {/* Calories Progress */}
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Calories</span>
                    <span className="text-sm text-muted-foreground">{stats?.total_calories || 0} / 2000</span>
                  </div>
                  <Progress value={((stats?.total_calories || 0) / 2000) * 100} className="h-3" />
                </div>
                {/* Meal Count */}
                <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Camera className="h-4 w-4 text-secondary" />
                    <span className="text-sm font-medium">Meals Logged Today</span>
                  </div>
                  <span className="text-2xl font-bold text-primary">{stats?.total_meals || 0}</span>
                </div>
              </div>
            </CardContent>
          </Card>)
        }

        {/* Error Alert */}
        {(error || cameraError) && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error || cameraError}</AlertDescription>
          </Alert>
        )}

        {/* Food Capture */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardHeader>
            <CardTitle>Analyze Meal</CardTitle>
            <CardDescription>
              Take a photo of your meal for instant nutritional analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center border-2 border-dashed border-border overflow-hidden">
              {cameraStream ? (
                <div className="relative w-full h-full">
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                    style={{ 
                      backgroundColor: '#000000',
                      transform: 'scaleX(-1)' // Mirror for better UX with front camera
                    }}
                    onLoadedMetadata={(e) => {
                      const video = e.target as HTMLVideoElement;
                      console.log('Video loaded:', {
                        width: video.videoWidth,
                        height: video.videoHeight,
                        readyState: video.readyState
                      });
                    }}
                    onPlay={() => console.log('Video is playing')}
                    onError={(e) => {
                      console.error('Video error:', e);
                      setCameraError('Video display error');
                    }}
                  />
                  {/* Debug overlay for laptops */}
                  <div className="absolute bottom-2 left-2 bg-black/80 text-white text-xs px-2 py-1 rounded">
                    <div>Status: {cameraStream?.active ? 'Active' : 'Inactive'}</div>
                    <div>Tracks: {cameraStream?.getVideoTracks().length || 0}</div>
                    <div>Ready: {videoRef.current?.readyState || 0}</div>
                  </div>
                </div>
              ) : capturedImage ? (
                <img
                  src={capturedImage}
                  alt="Captured meal"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="text-center">
                  <Camera className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground mb-4">
                    {cameraError ? 'Camera error - try refreshing' : 'Click "Capture Meal" to start camera'}
                  </p>
                </div>
              )}
            </div>

            <div className="flex gap-4 flex-wrap">
              {!cameraStream ? (
                <>
                  <Button 
                    className="flex-1 gradient-nutrition text-white font-medium"
                    onClick={startCamera}
                    disabled={isAnalyzing}
                  >
                    <Camera className="mr-2 h-4 w-4" />
                    Capture Meal
                  </Button>
                  
                  {/* Upload Photo Button */}
                  <div className="flex-1">
                    <input
                      id="meal-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      style={{ display: 'none' }}
                    />
                    <Button 
                      variant="outline" 
                      className="w-full border-green-500 text-green-600 hover:bg-green-500 hover:text-white font-medium"
                      onClick={() => {
                        const input = document.getElementById('meal-upload') as HTMLInputElement;
                        if (input) input.click();
                      }}
                      disabled={isAnalyzing}
                    >
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Photo
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <Button 
                    className="flex-1 gradient-nutrition text-white font-medium"
                    onClick={capturePhoto}
                    disabled={isAnalyzing}
                  >
                    <Camera className="mr-2 h-4 w-4" />
                    Take Photo
                  </Button>
                  <Button 
                    variant="outline"
                    className="flex-1 border-red-500 text-red-500 hover:bg-red-500 hover:text-white font-medium"
                    onClick={stopCamera}
                  >
                    Cancel
                  </Button>
                  
                  {/* Enhanced debug button for laptop camera issues */}
                  <Button 
                    variant="outline"
                    size="sm"
                    className="border-blue-500 text-blue-600 hover:bg-blue-500 hover:text-white font-medium"
                    onClick={async () => {
                      console.log('=== CAMERA DEBUG ===');
                      console.log('Stream:', cameraStream);
                      console.log('Stream active:', cameraStream?.active);
                      console.log('Video tracks:', cameraStream?.getVideoTracks());
                      console.log('Video element:', videoRef.current);
                      console.log('Video srcObject:', videoRef.current?.srcObject);
                      console.log('Video readyState:', videoRef.current?.readyState);
                      console.log('Video paused:', videoRef.current?.paused);
                      console.log('Video dimensions:', videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight);
                      
                      if (videoRef.current && cameraStream) {
                        const video = videoRef.current;
                        
                        // Force refresh the video element
                        video.srcObject = null;
                        video.load();
                        
                        setTimeout(() => {
                          video.srcObject = cameraStream;
                          
                          video.play().then(() => {
                            console.log('Manual refresh worked!');
                            toast({
                              title: "Debug Refresh",
                              description: "Video element refreshed successfully",
                            });
                          }).catch(err => {
                            console.error('Manual refresh failed:', err);
                            toast({
                              title: "Debug Failed",
                              description: `Video refresh failed: ${err.message}`,
                              variant: "destructive",
                            });
                          });
                        }, 100);
                      }
                    }}
                  >
                    Fix Video
                  </Button>
                </>
              )}
            </div>
            
            {/* Hidden canvas for photo capture */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </CardContent>
        </Card>

        {/* Analysis Results */}
        <div className="grid md:grid-cols-2 gap-6 animate-slide-up">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-secondary" />
                Meal Quality Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="text-5xl font-bold text-secondary mb-4">
                  {analysis?.meal_quality_score ? Math.round(analysis.meal_quality_score) : '--'}
                </div>
                <Progress 
                  value={analysis?.meal_quality_score || 0} 
                  className="h-2 mb-2" 
                />
                <p className="text-sm text-muted-foreground">
                  {analysis?.meal_quality_score 
                    ? `${analysis.meal_quality_score >= 80 ? 'Excellent' : analysis.meal_quality_score >= 60 ? 'Good' : 'Fair'} nutritional quality`
                    : 'Analyze a meal to see quality score'
                  }
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Protein Target
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="text-5xl font-bold text-primary mb-4">
                  {analysis?.total_protein ? Math.round((analysis.total_protein / 150) * 100) : 0}%
                </div>
                <Progress 
                  value={analysis?.total_protein ? (analysis.total_protein / 150) * 100 : 0} 
                  className="h-2 mb-2" 
                />
                <p className="text-sm text-muted-foreground">
                  {analysis?.total_protein 
                    ? `${Math.round(analysis.total_protein)}g of 150g daily goal`
                    : 'Daily protein goal progress'
                  }
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Detected Foods */}
        {analysis?.detected_foods && analysis.detected_foods.length > 0 && (
          <Card className="glass-card mt-6 animate-slide-up">
            <CardHeader>
              <CardTitle>Detected Foods</CardTitle>
              <CardDescription>
                AI identified these foods in your meal
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {analysis.detected_foods.map((food, index) => (
                  <div key={index} className="flex justify-between items-center p-4 bg-muted/50 rounded-lg">
                    <div>
                      <h4 className="font-semibold">{food.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        Confidence: {Math.round(food.confidence * 100)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{Math.round(food.calories)} cal</p>
                      <p className="text-sm text-muted-foreground">{Math.round(food.protein_content)}g protein</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recommendations */}
        <Card className="glass-card mt-6 animate-slide-up">
          <CardHeader>
            <CardTitle>Nutritional Insights</CardTitle>
            <CardDescription>
              AI-powered recommendations based on your goals
            </CardDescription>
          </CardHeader>
          <CardContent>
            {analysis?.recommendations && analysis.recommendations.length > 0 ? (
              <div className="space-y-3">
                {analysis.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <div className="w-2 h-2 bg-secondary rounded-full mt-2 flex-shrink-0" />
                    <p className="text-sm">{rec}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                {isAnalyzing 
                  ? "Analyzing your meal..." 
                  : "Analyze your first meal to receive personalized nutritional insights"
                }
              </div>
            )}
          </CardContent>
        </Card>

        {/* Meal History */}
        <Card className="glass-card mt-6 animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-secondary" /> Recent Meals
              </div>
              <Button variant="outline" size="sm" onClick={() => setHistoryDays(historyDays === 7 ? 30 : 7)}>
                {historyDays === 7 ? 'Last 7 Days' : 'Last 30 Days'}
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoadingHistory ? <LoadingSpinner message="Loading meal history..." /> :
              (history?.meals && history.meals.length > 0 ? (
                <div className="space-y-3">
                  {history.meals.map((meal: any, index: number) => (
                    <div key={index} className="p-4 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <p className="font-medium">{new Date(meal.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                          <p className="text-xs text-muted-foreground">{new Date(meal.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-semibold text-primary">{Math.round(meal.total_protein)}g protein</p>
                          <p className="text-xs text-muted-foreground">{Math.round(meal.total_calories)} cal</p>
                        </div>
                      </div>
                      {/* Detected Foods */}
                      {meal.detected_foods && Object.keys(meal.detected_foods).length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {Object.keys(meal.detected_foods).map((foodName, i) => (
                            <span key={i} className="text-xs px-2 py-1 bg-secondary/20 text-secondary rounded-full">{foodName.replace(/_/g, ' ')}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Camera className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No meals logged yet</p>
                  <p className="text-sm">Start by capturing your first meal!</p>
                </div>
              ))}
          </CardContent>
        </Card>

        {/* Weekly Analytics Chart */}
        <Card className="glass-card mt-6 animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" /> Weekly Nutrition Trends
            </CardTitle>
            <CardDescription>Your protein and calorie intake over time</CardDescription>
          </CardHeader>
          <CardContent>
            {history?.meals && history.meals.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <ReLineChart data={processWeeklyData(history.meals)}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" stroke="currentColor" style={{ fontSize: '12px' }} />
                  <YAxis stroke="currentColor" style={{ fontSize: '12px' }} />
                  <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }} />
                  <Legend />
                  <Line type="monotone" dataKey="protein" stroke="hsl(var(--primary))" strokeWidth={2} name="Protein (g)" />
                  <Line type="monotone" dataKey="calories" stroke="hsl(var(--secondary))" strokeWidth={2} name="Calories" />
                </ReLineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <p>Not enough data to show trends</p>
                <p className="text-sm">Log more meals to see your progress!</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
