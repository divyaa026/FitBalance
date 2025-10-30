import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { Lightbulb, Brain, Trophy } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface NutritionInsight {
    id: string;
    type: 'tip' | 'warning' | 'achievement';
    title: string;
    description: string;
    impact: number;
}

interface TrendData {
    date: string;
    value: number;
    target: number;
}

interface NutritionInsightsProps {
    insights: NutritionInsight[];
    trends: {
        protein: TrendData[];
        calories: TrendData[];
    };
}

export function NutritionInsights({ insights, trends }: NutritionInsightsProps) {
    const getInsightIcon = (type: NutritionInsight['type']) => {
        switch (type) {
            case 'tip':
                return <Lightbulb className="h-5 w-5" />;
            case 'warning':
                return <Brain className="h-5 w-5" />;
            case 'achievement':
                return <Trophy className="h-5 w-5" />;
        }
    };

    const getInsightStyle = (type: NutritionInsight['type']) => {
        switch (type) {
            case 'tip':
                return 'bg-blue-500/10 text-blue-500';
            case 'warning':
                return 'bg-yellow-500/10 text-yellow-500';
            case 'achievement':
                return 'bg-green-500/10 text-green-500';
        }
    };

    return (
        <Card className="glass-card">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5 text-primary" />
                    Nutritional Insights
                </CardTitle>
                <CardDescription>
                    Smart analysis of your nutritional patterns
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid gap-6">
                    <div className="h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={trends.calories}>
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
                                    tickFormatter={(value) => `${value}k`}
                                />
                                <Tooltip />
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    strokeWidth={2}
                                    stroke="currentColor"
                                    dot={false}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="target"
                                    strokeWidth={2}
                                    stroke="#22c55e"
                                    strokeDasharray="5 5"
                                    dot={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="space-y-4">
                        {insights.map((insight) => (
                            <Alert key={insight.id}>
                                <div className={`p-2 rounded-full ${getInsightStyle(insight.type)}`}>
                                    {getInsightIcon(insight.type)}
                                </div>
                                <AlertTitle className="ml-2">{insight.title}</AlertTitle>
                                <AlertDescription className="ml-2">
                                    {insight.description}
                                </AlertDescription>
                            </Alert>
                        ))}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}