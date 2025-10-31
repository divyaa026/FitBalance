import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Utensils, Clock } from "lucide-react";

interface RecentMeal {
    id: string;
    name: string;
    timestamp: string;
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    ingredients: string[];
    imageUrl?: string;
}

interface RecentMealsProps {
    meals: RecentMeal[];
}

export function RecentMeals({ meals }: RecentMealsProps) {
    return (
        <Card className="glass-card">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Utensils className="h-5 w-5 text-primary" />
                    Recent Meals
                </CardTitle>
                <CardDescription>
                    Your latest meal entries and their nutritional content
                </CardDescription>
            </CardHeader>
            <CardContent>
                <ScrollArea className="h-[400px] pr-4">
                    <div className="space-y-4">
                        {meals.map((meal) => (
                            <Card key={meal.id}>
                                <CardContent className="p-4">
                                    <div className="flex gap-4">
                                        {meal.imageUrl && (
                                            <img
                                                src={meal.imageUrl}
                                                alt={meal.name}
                                                className="h-20 w-20 rounded-md object-cover"
                                            />
                                        )}
                                        <div className="flex-1">
                                            <div className="flex items-center justify-between mb-2">
                                                <h4 className="font-medium">{meal.name}</h4>
                                                <div className="flex items-center text-sm text-muted-foreground">
                                                    <Clock className="h-4 w-4 mr-1" />
                                                    {new Date(meal.timestamp).toLocaleTimeString([], {
                                                        hour: '2-digit',
                                                        minute: '2-digit'
                                                    })}
                                                </div>
                                            </div>

                                            <div className="flex gap-2 mb-3">
                                                <Badge variant="secondary">{meal.calories} kcal</Badge>
                                                <Badge variant="outline">P: {meal.protein}g</Badge>
                                                <Badge variant="outline">C: {meal.carbs}g</Badge>
                                                <Badge variant="outline">F: {meal.fat}g</Badge>
                                            </div>

                                            <div className="flex flex-wrap gap-1">
                                                {meal.ingredients.map((ingredient, index) => (
                                                    <Badge
                                                        key={index}
                                                        variant="secondary"
                                                        className="bg-accent/30"
                                                    >
                                                        {ingredient}
                                                    </Badge>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}