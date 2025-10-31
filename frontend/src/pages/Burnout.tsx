import { useState } from "react";
import { Activity, TrendingDown, AlertTriangle, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
<<<<<<< HEAD
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useBurnout } from "@/hooks/use-fitbalance";
import { useToast } from "@/hooks/use-toast";
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
=======
import { BurnoutAssessment } from "@/components/BurnoutAssessment";
import { PerformancePrediction } from "@/components/PerformancePrediction";
>>>>>>> 633c84e602780eab5038f97c9beaa390e270d288

export default function Burnout() {
  const [workoutFrequency, setWorkoutFrequency] = useState(4);
  const [sleepHours, setSleepHours] = useState(7.5);
  const [stressLevel, setStressLevel] = useState(5);
  const [recoveryDays, setRecoveryDays] = useState(2);
  const [performanceTrend, setPerformanceTrend] = useState("stable");

  const { isLoading, error, burnoutData, analyzeBurnout } = useBurnout();
  const { toast } = useToast();

  const handleAnalyze = async () => {
    try {
      await analyzeBurnout({
        user_id: "123",
        workout_frequency: workoutFrequency,
        sleep_hours: sleepHours,
        stress_level: stressLevel,
        recovery_time: recoveryDays,
        performance_trend: performanceTrend,
      });

      toast({
        title: "Analysis Complete",
        description: "Your burnout risk has been calculated.",
      });
    } catch (err) {
      toast({
        title: "Analysis Failed",
        description: "Unable to analyze burnout risk. Please try again.",
        variant: "destructive",
      });
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case "low":
        return "text-green-500";
      case "medium":
        return "text-yellow-500";
      case "high":
        return "text-orange-500";
      case "critical":
        return "text-red-500";
      default:
        return "text-gray-500";
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case "low":
        return <Shield className="h-8 w-8 text-green-500" />;
      case "medium":
        return <AlertTriangle className="h-8 w-8 text-yellow-500" />;
      case "high":
        return <AlertTriangle className="h-8 w-8 text-orange-500" />;
      case "critical":
        return <AlertTriangle className="h-8 w-8 text-red-500" />;
      default:
        return <Activity className="h-8 w-8 text-gray-500" />;
    }
  };

  return (
    <div className="min-h-screen px-4 py-8 relative z-10">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-destructive/10 text-destructive mb-4">
            <TrendingDown className="h-4 w-4" />
            <span className="text-sm font-medium">Burnout Prevention</span>
          </div>
          <h1 className="text-4xl font-bold mb-2">Athletic Burnout Risk</h1>
          <p className="text-muted-foreground">
            Predictive analytics to prevent overtraining and optimize recovery
          </p>
        </div>

        {/* Input Form */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardHeader>
            <CardTitle>Enter Your Metrics</CardTitle>
            <CardDescription>
              Provide current training and recovery data for analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Workout Frequency */}
            <div className="space-y-2">
              <Label>Workouts Per Week: {workoutFrequency}</Label>
              <Slider
                value={[workoutFrequency]}
                onValueChange={(value) => setWorkoutFrequency(value[0])}
                min={1}
                max={7}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                How many times do you train per week?
              </p>
            </div>
<<<<<<< HEAD

            {/* Sleep Hours */}
            <div className="space-y-2">
              <Label>Average Sleep Hours: {sleepHours.toFixed(1)}</Label>
              <Slider
                value={[sleepHours]}
                onValueChange={(value) => setSleepHours(value[0])}
                min={4}
                max={10}
                step={0.5}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Average hours of sleep per night
              </p>
            </div>

            {/* Stress Level */}
            <div className="space-y-2">
              <Label>Stress Level: {stressLevel}/10</Label>
              <Slider
                value={[stressLevel]}
                onValueChange={(value) => setStressLevel(value[0])}
                min={1}
                max={10}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Overall stress level (1=very low, 10=very high)
              </p>
            </div>

            {/* Recovery Days */}
            <div className="space-y-2">
              <Label>Recovery Days Between Intense Sessions: {recoveryDays}</Label>
              <Slider
                value={[recoveryDays]}
                onValueChange={(value) => setRecoveryDays(value[0])}
                min={0}
                max={7}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Days of rest between high-intensity workouts
              </p>
            </div>

            {/* Performance Trend */}
            <div className="space-y-2">
              <Label>Performance Trend</Label>
              <Select value={performanceTrend} onValueChange={setPerformanceTrend}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="improving">Improving</SelectItem>
                  <SelectItem value="stable">Stable</SelectItem>
                  <SelectItem value="declining">Declining</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                How has your performance been trending lately?
              </p>
            </div>

            {/* Analyze Button */}
            <Button
              className="w-full gradient-burnout text-white font-medium"
              onClick={handleAnalyze}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="mr-2 h-4 w-4" />
                  Analyze Burnout Risk
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Results */}
        {burnoutData && (
          <>
            {/* Risk Score Card */}
            <div className="grid md:grid-cols-3 gap-6 mb-6 animate-slide-up">
              <Card className="glass-card col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {getRiskIcon(burnoutData.risk_level)}
                    Burnout Risk Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Risk Level */}
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Risk Level</span>
                        <span className={`text-2xl font-bold ${getRiskColor(burnoutData.risk_level)}`}>
                          {burnoutData.risk_level.toUpperCase()}
                        </span>
                      </div>
                      <Progress value={burnoutData.risk_score} className="h-3" />
                      <p className="text-xs text-muted-foreground mt-1">
                        Risk Score: {Math.round(burnoutData.risk_score)}/100
                      </p>
                    </div>

                    {/* Time to Burnout */}
                    {burnoutData.time_to_burnout && (
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Estimated Time to Burnout</p>
                        <p className="text-3xl font-bold">
                          {Math.round(burnoutData.time_to_burnout / 30)} months
                        </p>
                        <p className="text-xs text-muted-foreground">
                          ({Math.round(burnoutData.time_to_burnout)} days)
                        </p>
                      </div>
                    )}

                    {/* Survival Probability */}
                    <div className="p-4 bg-muted/30 rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">1-Year Survival Probability</p>
                      <p className="text-3xl font-bold text-green-500">
                        {Math.round(burnoutData.survival_probability * 100)}%
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Chance of avoiding burnout in the next year
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Risk Factors */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-lg">Risk Factors</CardTitle>
                </CardHeader>
                <CardContent>
                  {burnoutData.risk_factors && burnoutData.risk_factors.length > 0 ? (
                    <div className="space-y-2">
                      {burnoutData.risk_factors.map((factor: string, index: number) => (
                        <div key={index} className="flex items-start gap-2 text-sm">
                          <AlertTriangle className="h-4 w-4 text-orange-500 flex-shrink-0 mt-0.5" />
                          <span className="text-xs">
                            {factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-green-600">
                      No significant risk factors detected
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Survival Curve */}
            {burnoutData.survival_curve_data && (
              <Card className="glass-card mb-6 animate-slide-up">
                <CardHeader>
                  <CardTitle>Survival Curve</CardTitle>
                  <CardDescription>
                    Probability of avoiding burnout over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <ReLineChart
                      data={burnoutData.survival_curve_data.times.map(
                        (time: number, index: number) => ({
                          days: time,
                          months: Math.round(time / 30),
                          probability:
                            burnoutData.survival_curve_data.survival_probabilities[index] * 100,
                        })
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis
                        dataKey="months"
                        label={{ value: 'Months', position: 'insideBottom', offset: -5 }}
                        stroke="currentColor"
                        style={{ fontSize: '12px' }}
                      />
                      <YAxis
                        label={{ value: 'Survival Probability (%)', angle: -90, position: 'insideLeft' }}
                        domain={[0, 100]}
                        stroke="currentColor"
                        style={{ fontSize: '12px' }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'hsl(var(--background))',
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px',
                        }}
                        formatter={(value: number) => [
                          `${(value as number).toFixed(1)}%`,
                          'Survival Probability',
                        ]}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="probability"
                        stroke="hsl(var(--destructive))"
                        strokeWidth={3}
                        name="No Burnout Probability"
                        dot={false}
                      />
                    </ReLineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            {burnoutData.recommendations && burnoutData.recommendations.length > 0 && (
              <Card className="glass-card animate-slide-up">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="h-5 w-5 text-green-500" />
                    Personalized Recommendations
                  </CardTitle>
                  <CardDescription>
                    Actions to reduce burnout risk and optimize recovery
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {burnoutData.recommendations.map((rec: string, index: number) => (
                      <div
                        key={index}
                        className="flex items-start gap-3 p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors"
                      >
                        <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-green-600 font-bold text-sm">{index + 1}</span>
                        </div>
                        <p className="text-sm">{rec}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}
=======

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Risk Score</span>
                  <span className="font-medium">25/100</span>
                </div>
                <Progress value={25} className="h-2" />
              </div>

              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Recovery Status</span>
                  <span className="font-medium text-green-500">Optimal</span>
                </div>
                <Progress value={85} className="h-2" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Risk Factors */}
        <div className="grid md:grid-cols-3 gap-6 mb-6 animate-slide-up">
          {[
            { icon: Activity, label: "Training Load", value: "Moderate", color: "text-green-500" },
            { icon: TrendingDown, label: "Sleep Quality", value: "Good", color: "text-blue-500" },
            { icon: AlertTriangle, label: "Stress Level", value: "Low", color: "text-yellow-500" },
          ].map((factor, index) => (
            <Card key={index} className="glass-card">
              <CardContent className="pt-6">
                <div className="text-center">
                  <factor.icon className={`h-8 w-8 mx-auto mb-3 ${factor.color}`} />
                  <div className="text-sm text-muted-foreground mb-1">
                    {factor.label}
                  </div>
                  <div className="text-lg font-bold">{factor.value}</div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Assessment and Prediction */}
        <div className="grid md:grid-cols-2 gap-6 mb-6 animate-slide-up">
          <BurnoutAssessment
            data={{
              weeklyLoad: 12,
              recoverySleep: 7.5,
              stressScore: 45,
              prediction: 'Low',
              confidence: 85
            }}
          />
          <PerformancePrediction
            predictions={[
              { date: 'Mon', predicted: 85, actual: 82 },
              { date: 'Tue', predicted: 88, actual: 86 },
              { date: 'Wed', predicted: 92, actual: 90 },
              { date: 'Thu', predicted: 95, actual: 94 },
              { date: 'Fri', predicted: 96 },
              { date: 'Sat', predicted: 98 },
              { date: 'Sun', predicted: 97 },
            ]}
          />

          {/* Recommendations */}
          <Card className="glass-card animate-slide-up">
            <CardHeader>
              <CardTitle>Recovery Recommendations</CardTitle>
              <CardDescription>
                Personalized strategies to maintain peak performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  "Maintain current training intensity for optimal adaptation",
                  "Focus on quality sleep to support recovery (7-9 hours)",
                  "Incorporate active recovery sessions 2-3 times per week",
                  "Monitor nutrition to support training demands",
                ].map((rec, index) => (
                  <div key={index} className="flex items-start gap-3 p-4 rounded-lg bg-muted/50">
                    <div className="h-6 w-6 rounded-full bg-green-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-green-500 text-sm font-bold">{index + 1}</span>
                    </div>
                    <p className="text-sm">{rec}</p>
                  </div>
                ))}
              </div>

              <Button className="w-full mt-6 gradient-burnout text-white font-medium">
                Update Assessment
              </Button>
            </CardContent>
          </Card>
        </div>
>>>>>>> 633c84e602780eab5038f97c9beaa390e270d288
      </div>
    </div>
  );
}
