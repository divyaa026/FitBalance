import { ArrowLeft, Clock, Tag } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function ProgressiveOverloadArticle() {
  return (
    <div className="min-h-screen px-4 py-8 relative z-10">
      <div className="max-w-4xl mx-auto">
        {/* Back Button */}
        <div className="mb-8">
          <Link to="/">
            <Button variant="ghost" className="text-purple-300 hover:text-purple-200">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Button>
          </Link>
        </div>

        {/* Article Header */}
        <div className="mb-8 animate-fade-in">
          <div className="flex items-center gap-2 mb-4">
            <div className="text-sm font-medium text-primary bg-primary/10 px-2.5 py-0.5 rounded-full">
              Training
            </div>
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              5 min read
            </div>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent">
            The Science of Progressive Overload
          </h1>

          <p className="text-lg text-muted-foreground leading-relaxed">
            Learn how to properly implement progressive overload in your training program for maximum muscle growth and strength development.
          </p>
        </div>

        {/* Featured Image */}
        <div className="mb-8 animate-slide-up">
          <div className="aspect-[16/9] rounded-xl overflow-hidden">
            <img
              src="https://images.unsplash.com/photo-1517963879433-6ad2b056d712"
              alt="Progressive Overload Training"
              className="w-full h-full object-cover"
            />
          </div>
        </div>

        {/* Article Content */}
        <div className="prose prose-lg prose-invert max-w-none animate-slide-up">
          <h2>What is Progressive Overload?</h2>
          <p>
            Progressive overload is the systematic increase of training demands over time to continually challenge your muscles and promote growth.
            It's the fundamental principle behind all successful strength training programs and is essential for long-term progress.
          </p>

          <h2>Why Progressive Overload Works</h2>
          <p>
            Your body adapts to the demands placed upon it. When you consistently lift the same weights for the same number of repetitions,
            your muscles become efficient at handling that specific workload. To continue building strength and muscle, you must gradually
            increase the demands placed on your body.
          </p>

          <h2>Methods of Progressive Overload</h2>

          <h3>1. Increase Weight</h3>
          <p>
            The most straightforward method is to gradually increase the amount of weight you're lifting. This could mean adding 2.5-5kg
            to your lifts every 1-2 weeks, depending on your experience level and the exercise.
          </p>

          <h3>2. Increase Repetitions</h3>
          <p>
            If you can't increase the weight, try performing more repetitions with the same weight. For example, if you were doing 3 sets
            of 8-10 reps, try to work up to 3 sets of 10-12 reps before increasing the weight.
          </p>

          <h3>3. Increase Sets</h3>
          <p>
            Adding an extra set to your exercises can provide additional volume and stimulus for growth. This is particularly effective
            for intermediate lifters who have already built a solid foundation.
          </p>

          <h3>4. Decrease Rest Time</h3>
          <p>
            Reducing rest periods between sets increases the metabolic stress on your muscles, which can contribute to hypertrophy.
            This method is commonly used in programs like German Volume Training.
          </p>

          <h3>5. Increase Frequency</h3>
          <p>
            Training a muscle group more frequently throughout the week can provide more opportunities for growth. This works well
            for advanced lifters who have higher recovery capacity.
          </p>

          <h2>Implementing Progressive Overload Safely</h2>

          <h3>Track Your Progress</h3>
          <p>
            Keep a detailed training log that records your weights, sets, reps, and how the exercises felt. This data is crucial for
            making informed decisions about when and how to progress.
          </p>

          <h3>Progress Gradually</h3>
          <p>
            Small, consistent increases are more sustainable than large jumps. Aim for 5-10% increases in weight or volume every 1-2 weeks,
            depending on your experience level.
          </p>

          <h3>Listen to Your Body</h3>
          <p>
            Progressive overload should be balanced with adequate recovery. If you're consistently feeling excessively fatigued or
            experiencing joint pain, it may be time to deload or focus on technique.
          </p>

          <h3>Use Multiple Methods</h3>
          <p>
            Don't rely on just one method of progression. Rotate between increasing weight, reps, sets, and frequency to keep your
            training varied and prevent plateaus.
          </p>

          <h2>Common Mistakes to Avoid</h2>

          <h3>Progressing Too Quickly</h3>
          <p>
            Jumping weights too aggressively can lead to injury and burnout. Remember that consistency over time beats intensity in bursts.
          </p>

          <h3>Ignoring Recovery</h3>
          <p>
            Progressive overload without adequate recovery leads to overtraining. Make sure you're getting enough sleep, nutrition,
            and active recovery.
          </p>

          <h3>Neglecting Technique</h3>
          <p>
            As you increase weight, your form can suffer. Always prioritize proper technique over lifting heavier weights.
          </p>

          <h2>Sample Progressive Overload Program</h2>

          <p>Here's a simple 8-week progression for the bench press:</p>

          <div className="bg-muted/50 p-6 rounded-lg my-6">
            <h4 className="font-semibold mb-4">Bench Press Progression</h4>
            <div className="space-y-2 text-sm">
              <div><strong>Week 1-2:</strong> 3 sets × 8-10 reps @ 70kg</div>
              <div><strong>Week 3-4:</strong> 3 sets × 8-10 reps @ 72.5kg</div>
              <div><strong>Week 5-6:</strong> 4 sets × 8-10 reps @ 72.5kg</div>
              <div><strong>Week 7-8:</strong> 3 sets × 10-12 reps @ 72.5kg</div>
            </div>
          </div>

          <h2>Conclusion</h2>
          <p>
            Progressive overload is the cornerstone of effective strength training. By systematically increasing training demands over time,
            you ensure continued progress and prevent plateaus. Remember to progress gradually, track your workouts, and prioritize recovery
            for long-term success in your fitness journey.
          </p>
        </div>

        {/* Related Articles */}
        <div className="mt-12 animate-slide-up">
          <h3 className="text-2xl font-bold mb-6">Related Articles</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <Link to="/articles/post-workout-nutrition">
              <Card className="glass-card hover:scale-105 transition-all duration-300 cursor-pointer">
                <CardContent className="p-6">
                  <div className="flex items-center gap-2 mb-3">
                    <Tag className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium text-primary">Nutrition</span>
                  </div>
                  <h4 className="font-semibold mb-2">Optimizing Post-Workout Nutrition</h4>
                  <p className="text-sm text-muted-foreground">Discover the optimal nutrition strategies to maximize recovery...</p>
                </CardContent>
              </Card>
            </Link>

            <Link to="/articles/mental-benefits">
              <Card className="glass-card hover:scale-105 transition-all duration-300 cursor-pointer">
                <CardContent className="p-6">
                  <div className="flex items-center gap-2 mb-3">
                    <Tag className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium text-primary">Wellness</span>
                  </div>
                  <h4 className="font-semibold mb-2">Mental Benefits of Exercise</h4>
                  <p className="text-sm text-muted-foreground">Explore how consistent physical activity can improve your mental well-being...</p>
                </CardContent>
              </Card>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}