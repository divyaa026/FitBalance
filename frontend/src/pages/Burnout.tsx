import { Activity, AlertTriangle, TrendingDown, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { BurnoutAssessment } from "@/components/BurnoutAssessment";
import { PerformancePrediction } from "@/components/PerformancePrediction";

export default function Burnout() {
  return (
    <div className="min-h-screen px-4 py-8 relative z-10">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-orange-500/10 text-orange-500 mb-4">
            <Activity className="h-4 w-4" />
            <span className="text-sm font-medium">Burnout Prevention</span>
          </div>
          <h1 className="text-4xl font-bold mb-2">Recovery Analysis</h1>
          <p className="text-muted-foreground">
            Predictive analytics to optimize performance and prevent overtraining
          </p>
        </div>

        {/* Current Status */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-6 w-6 text-green-500" />
              Current Risk Level
            </CardTitle>
            <CardDescription>
              Based on your recent activity and recovery patterns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center mb-6">
              <div className="text-6xl font-bold text-green-500 mb-2">Low</div>
              <p className="text-muted-foreground">
                You're maintaining a healthy training balance
              </p>
            </div>

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
      </div>
    </div>
  );
}
