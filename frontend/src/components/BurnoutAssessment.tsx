import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Line } from "recharts";
import { ResponsiveContainer, LineChart } from "recharts";
import { Brain, LineChart as LineChartIcon, TrendingDown } from "lucide-react";

interface BurnoutAssessmentProps {
    data: {
        weeklyLoad: number;
        recoverySleep: number;
        stressScore: number;
        prediction: 'Low' | 'Moderate' | 'High';
        confidence: number;
    };
}

export function BurnoutAssessment({ data }: BurnoutAssessmentProps) {
    // Example data for the trend chart
    const trendData = [
        { week: 'W1', score: 35 },
        { week: 'W2', score: 42 },
        { week: 'W3', score: 38 },
        { week: 'W4', score: data.stressScore },
    ];

    return (
        <Card className="glass-card">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    Burnout Assessment
                </CardTitle>
                <CardDescription>
                    Current burnout risk evaluation based on your activity patterns
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid gap-4">
                    <div className="grid grid-cols-3 gap-4">
                        <div className="space-y-2">
                            <p className="text-sm font-medium">Weekly Load</p>
                            <p className="text-2xl font-bold">{data.weeklyLoad}h</p>
                        </div>
                        <div className="space-y-2">
                            <p className="text-sm font-medium">Recovery Sleep</p>
                            <p className="text-2xl font-bold">{data.recoverySleep}h</p>
                        </div>
                        <div className="space-y-2">
                            <p className="text-sm font-medium">Stress Score</p>
                            <p className="text-2xl font-bold">{data.stressScore}</p>
                        </div>
                    </div>

                    <div className="pt-4">
                        <div className="flex items-center justify-between">
                            <h4 className="text-sm font-medium">Stress Score Trend</h4>
                            <LineChartIcon className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <div className="h-[100px] mt-2">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={trendData}>
                                    <Line
                                        type="monotone"
                                        dataKey="score"
                                        stroke="currentColor"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="rounded-lg bg-muted p-4">
                        <div className="flex items-center gap-2">
                            <TrendingDown className={
                                data.prediction === 'Low' ? 'text-green-500' :
                                    data.prediction === 'Moderate' ? 'text-yellow-500' :
                                        'text-red-500'
                            } />
                            <div>
                                <p className="font-medium">{data.prediction} Risk</p>
                                <p className="text-sm text-muted-foreground">
                                    {data.confidence}% confidence in prediction
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}