import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, BookOpenCheck, MoveRight } from "lucide-react";
import { FeaturedArticles } from "@/components/FeaturedArticles";

export function FitnessArticles() {
    const articles = [
        {
            id: "1",
            title: "Understanding Progressive Overload: The Key to Strength Gains",
            description: "Learn how to properly implement progressive overload in your training program for maximum muscle growth and strength development.",
            category: "Training",
            readTime: 5,
            imageUrl: "https://images.unsplash.com/photo-1517963879433-6ad2b056d712",
            url: "#",
        },
        {
            id: "2",
            title: "The Science Behind Post-Workout Nutrition",
            description: "Discover the optimal nutrition strategies to maximize recovery and muscle growth after your workouts.",
            category: "Nutrition",
            readTime: 7,
            imageUrl: "https://images.unsplash.com/photo-1490645935967-10de6ba17061",
            url: "#",
        },
        {
            id: "3",
            title: "Mental Health Benefits of Regular Exercise",
            description: "Explore how consistent physical activity can improve your mental well-being and reduce stress levels.",
            category: "Wellness",
            readTime: 6,
            imageUrl: "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b",
            url: "#",
        }
    ];

    return (
        <div className="min-h-screen px-4 py-8 relative z-10">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="mb-8 animate-fade-in">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 text-accent mb-4">
                        <BookOpenCheck className="h-4 w-4" />
                        <span className="text-sm font-medium">Latest Articles</span>
                    </div>
                    <h1 className="text-4xl font-bold mb-2">Stay Informed</h1>
                    <p className="text-muted-foreground">
                        Expert insights to help you reach your fitness goals
                    </p>
                </div>

                {/* Featured Articles */}
                <div className="grid gap-6">
                    <FeaturedArticles articles={articles} />

                    {/* Categories */}
                    <div className="grid md:grid-cols-3 gap-6 animate-slide-up">
                        {[
                            {
                                title: "Training Tips",
                                description: "Expert advice to improve your workouts",
                                icon: Activity,
                                count: 12,
                            },
                            {
                                title: "Nutrition Guides",
                                description: "Healthy eating and meal planning",
                                icon: BookOpenCheck,
                                count: 8,
                            },
                            {
                                title: "Recovery Insights",
                                description: "Optimize your rest and recovery",
                                icon: MoveRight,
                                count: 15,
                            },
                        ].map((category, index) => (
                            <Card key={index} className="glass-card">
                                <CardHeader>
                                    <div className="flex items-center justify-between">
                                        <category.icon className="h-5 w-5 text-primary" />
                                        <span className="text-sm text-muted-foreground">
                                            {category.count} articles
                                        </span>
                                    </div>
                                    <CardTitle className="mt-2">{category.title}</CardTitle>
                                    <CardDescription>{category.description}</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <Button className="w-full" variant="secondary">
                                        View All
                                    </Button>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}