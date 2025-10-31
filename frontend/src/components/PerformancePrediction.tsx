import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, ResponsiveContainer, Line, XAxis, YAxis, Tooltip } from "recharts";
import { Activity, TrendingUp, Trophy } from "lucide-react";

interface PerformancePredictionProps {
    predictions: Array<{
        date: string;
        predicted: number;
        actual?: number;
    }>;
}

export function PerformancePrediction({ predictions }: PerformancePredictionProps) {
    const bestPerformance = Math.max(...predictions.map(p => p.predicted));
    const bestDay = predictions.find(p => p.predicted === bestPerformance);

    return (
        <Card className="glass-card">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-primary" />
                    Performance Prediction
                </CardTitle>
                <CardDescription>
                    Forecasted performance based on your training patterns and recovery
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="space-y-6">
                    <div className="h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={predictions}>
                                <XAxis
                                    dataKey="date"
                                    stroke="#888888"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                />
                                <YAxis
                                    stroke="#888888"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    tickFormatter={(value) => `${value}%`}
                                />
                                <Tooltip />
                                <Line
                                    type="monotone"
                                    dataKey="predicted"
                                    strokeWidth={2}
                                    stroke="currentColor"
                                    dot={false}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="actual"
                                    strokeWidth={2}
                                    stroke="#22c55e"
                                    dot={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <Card>
                            <CardContent className="pt-6">
                                <div className="flex items-center gap-2">
                                    <Trophy className="h-4 w-4 text-yellow-500" />
                                    <div>
                                        <p className="text-sm font-medium">Peak Performance</p>
                                        <p className="text-2xl font-bold">{bestPerformance}%</p>
                                        <p className="text-xs text-muted-foreground">
                                            Expected on {bestDay?.date}
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardContent className="pt-6">
                                <div className="flex items-center gap-2">
                                    <TrendingUp className="h-4 w-4 text-green-500" />
                                    <div>
                                        <p className="text-sm font-medium">Performance Trend</p>
                                        <p className="text-2xl font-bold">
                                            {predictions[predictions.length - 1].predicted}%
                                        </p>
                                        <p className="text-xs text-muted-foreground">
                                            Current trajectory
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}