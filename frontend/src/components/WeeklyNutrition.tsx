import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { Utensils, Carrot, Fish, Apple } from "lucide-react";

interface WeeklyNutritionProps {
    data: Array<{
        day: string;
        calories: number;
        protein: number;
        carbs: number;
        fat: number;
        target: number;
    }>;
}

export function WeeklyNutrition({ data }: WeeklyNutritionProps) {
    const averageCalories = Math.round(
        data.reduce((acc, day) => acc + day.calories, 0) / data.length
    );

    const averageProtein = Math.round(
        data.reduce((acc, day) => acc + day.protein, 0) / data.length
    );

    return (
        <Card className="glass-card">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Utensils className="h-5 w-5 text-primary" />
                    Weekly Nutritional Analysis
                </CardTitle>
                <CardDescription>
                    Your nutrition patterns over the past week
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="space-y-6">
                    <div className="h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={data}>
                                <XAxis
                                    dataKey="day"
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
                                <Bar
                                    dataKey="calories"
                                    fill="currentColor"
                                    radius={[4, 4, 0, 0]}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                        <div className="space-y-2">
                            <div className="flex items-center gap-2">
                                <Utensils className="h-4 w-4" />
                                <p className="text-sm font-medium">Avg. Calories</p>
                            </div>
                            <p className="text-2xl font-bold">{averageCalories}</p>
                            <p className="text-xs text-muted-foreground">kcal per day</p>
                        </div>
                        <div className="space-y-2">
                            <div className="flex items-center gap-2">
                                <Fish className="h-4 w-4" />
                                <p className="text-sm font-medium">Avg. Protein</p>
                            </div>
                            <p className="text-2xl font-bold">{averageProtein}g</p>
                            <p className="text-xs text-muted-foreground">protein per day</p>
                        </div>
                        <div className="space-y-2">
                            <div className="flex items-center gap-2">
                                <Apple className="h-4 w-4" />
                                <p className="text-sm font-medium">Goal Progress</p>
                            </div>
                            <p className="text-2xl font-bold">
                                {Math.round((averageCalories / data[0].target) * 100)}%
                            </p>
                            <p className="text-xs text-muted-foreground">of daily target</p>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}