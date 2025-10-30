import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Book } from "lucide-react";

interface Article {
    id: string;
    title: string;
    description: string;
    category: string;
    readTime: number;
    imageUrl: string;
    url: string;
}

interface FeaturedArticlesProps {
    articles: Article[];
}

export function FeaturedArticles({ articles }: FeaturedArticlesProps) {
    return (
        <Card className="glass-card">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Book className="h-5 w-5 text-primary" />
                    Featured Articles
                </CardTitle>
                <CardDescription>
                    Stay informed with the latest fitness and nutrition insights
                </CardDescription>
            </CardHeader>
            <CardContent>
                <ScrollArea className="h-[500px] pr-4">
                    <div className="grid gap-6">
                        {articles.map((article) => (
                            <Card key={article.id} className="overflow-hidden">
                                <div className="aspect-video w-full">
                                    <img
                                        src={article.imageUrl}
                                        alt={article.title}
                                        className="object-cover w-full h-full"
                                    />
                                </div>
                                <CardHeader>
                                    <div className="flex items-center justify-between mb-2">
                                        <Badge className="bg-primary/10 text-primary hover:bg-primary/20">
                                            {article.category}
                                        </Badge>
                                        <span className="text-sm text-muted-foreground">
                                            {article.readTime} min read
                                        </span>
                                    </div>
                                    <CardTitle className="line-clamp-2">{article.title}</CardTitle>
                                    <CardDescription className="line-clamp-3">
                                        {article.description}
                                    </CardDescription>
                                </CardHeader>
                                <CardFooter>
                                    <Button
                                        className="w-full"
                                        variant="secondary"
                                        onClick={() => window.open(article.url, '_blank')}
                                    >
                                        Read More
                                    </Button>
                                </CardFooter>
                            </Card>
                        ))}
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}