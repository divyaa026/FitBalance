import { ArrowLeft, Clock, Tag } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function PostWorkoutNutritionArticle() {
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
              Nutrition
            </div>
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              7 min read
            </div>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent">
            Optimizing Post-Workout Nutrition
          </h1>

          <p className="text-lg text-muted-foreground leading-relaxed">
            Discover the optimal nutrition strategies to maximize recovery and muscle growth after your workouts.
          </p>
        </div>

        {/* Featured Image */}
        <div className="mb-8 animate-slide-up">
          <div className="aspect-[16/9] rounded-xl overflow-hidden">
            <img
              src="https://images.unsplash.com/photo-1490645935967-10de6ba17061"
              alt="Post-Workout Nutrition"
              className="w-full h-full object-cover"
            />
          </div>
        </div>

        {/* Article Content */}
        <div className="prose prose-lg prose-invert max-w-none animate-slide-up">
          <h2>The Importance of Post-Workout Nutrition</h2>
          <p>
            The period immediately following your workout is often called the "anabolic window" - a critical time when your body is primed
            for nutrient uptake and muscle repair. Proper post-workout nutrition can significantly enhance recovery, reduce muscle soreness,
            and maximize the benefits of your training.
          </p>

          <h2>The Three Pillars of Post-Workout Nutrition</h2>

          <h3>1. Protein for Muscle Repair</h3>
          <p>
            Protein is essential for muscle repair and growth. During exercise, your muscles undergo microscopic damage that needs to be
            repaired. Consuming high-quality protein post-workout provides the amino acids necessary for this repair process.
          </p>

          <div className="bg-muted/50 p-6 rounded-lg my-6">
            <h4 className="font-semibold mb-4">Recommended Protein Intake</h4>
            <ul className="space-y-2">
              <li><strong>Strength Training:</strong> 20-40g of protein within 30-60 minutes post-workout</li>
              <li><strong>Endurance Training:</strong> 10-20g of protein within 30-60 minutes post-workout</li>
              <li><strong>Daily Total:</strong> 1.6-2.2g per kg of body weight for active individuals</li>
            </ul>
          </div>

          <h3>2. Carbohydrates for Glycogen Replenishment</h3>
          <p>
            Carbohydrates are crucial for replenishing glycogen stores depleted during exercise. This is particularly important for
            athletes and those engaging in intense or long-duration training sessions.
          </p>

          <h3>3. Micronutrients for Recovery</h3>
          <p>
            Vitamins and minerals play supporting roles in recovery. Antioxidants help reduce exercise-induced oxidative stress,
            while electrolytes support hydration and muscle function.
          </p>

          <h2>Timing Your Post-Workout Nutrition</h2>

          <h3>The First 30 Minutes</h3>
          <p>
            While the "anabolic window" concept has been somewhat overstated, there's still value in consuming nutrients relatively
            soon after training. Aim to have your post-workout meal or shake within 30-60 minutes of finishing your workout.
          </p>

          <h3>Extended Recovery Window</h3>
          <p>
            The most critical period for nutrient timing is actually the 2-4 hours following exercise. During this time, your body
            continues to repair and adapt to the training stimulus.
          </p>

          <h2>Optimal Post-Workout Meals</h2>

          <h3>Protein Shakes</h3>
          <p>
            Whey protein shakes are convenient and rapidly absorbed. Mix with water or milk and consume immediately after training.
            Add fruit for carbohydrates and healthy fats.
          </p>

          <h3>Whole Food Meals</h3>
          <p>
            For a more complete recovery meal, combine protein-rich foods with complex carbohydrates and vegetables:
          </p>

          <div className="bg-muted/50 p-6 rounded-lg my-6">
            <h4 className="font-semibold mb-4">Sample Post-Workout Meals</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-medium mb-2">Option 1: Chicken & Rice</h5>
                <ul className="text-sm space-y-1">
                  <li>150g grilled chicken breast</li>
                  <li>200g brown rice</li>
                  <li>Mixed vegetables</li>
                  <li>~45g protein, 80g carbs</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium mb-2">Option 2: Greek Yogurt Bowl</h5>
                <ul className="text-sm space-y-1">
                  <li>300g Greek yogurt</li>
                  <li>100g berries</li>
                  <li>50g granola</li>
                  <li>30g almonds</li>
                  <li>~35g protein, 60g carbs</li>
                </ul>
              </div>
            </div>
          </div>

          <h2>Special Considerations</h2>

          <h3>Fasted Training</h3>
          <p>
            If you train in a fasted state, prioritize protein immediately after your workout. The carbohydrates can come in your
            next regular meal.
          </p>

          <h3>Evening Workouts</h3>
          <p>
            Post-workout nutrition timing is still important in the evening. Have your recovery meal before bed to support overnight
            recovery processes.
          </p>

          <h3>Plant-Based Athletes</h3>
          <p>
            Plant-based proteins like pea, hemp, and rice protein can be just as effective as animal-based proteins when consumed
            in adequate amounts. Consider combining different plant proteins to ensure complete amino acid profiles.
          </p>

          <h2>Common Mistakes to Avoid</h2>

          <h3>Insufficient Protein</h3>
          <p>
            Many people don't consume enough protein post-workout. Aim for at least 20-30g of high-quality protein in your recovery meal.
          </p>

          <h3>Too Much Sugar</h3>
          <p>
            While carbohydrates are important, avoid loading up on sugary foods or drinks. Opt for complex carbohydrates that provide
            sustained energy.
          </p>

          <h3>Ignoring Hydration</h3>
          <p>
            Rehydration is just as important as nutrition. Drink water or an electrolyte beverage to replace fluids lost during exercise.
          </p>

          <h2>Supplements for Recovery</h2>

          <h3>Creatine</h3>
          <p>
            Creatine supplementation can enhance recovery and improve high-intensity exercise performance. Take 3-5g daily,
            regardless of training status.
          </p>

          <h3>Beta-Alanine</h3>
          <p>
            Beta-alanine can help buffer lactic acid in muscles, potentially reducing fatigue during intense training.
          </p>

          <h3>Branched-Chain Amino Acids (BCAAs)</h3>
          <p>
            BCAAs may help reduce muscle soreness and support recovery, though whole protein sources are generally preferable.
          </p>

          <h2>Individualizing Your Approach</h2>
          <p>
            The optimal post-workout nutrition strategy depends on several factors:
          </p>

          <ul>
            <li><strong>Training Type:</strong> Strength vs. endurance training have different requirements</li>
            <li><strong>Training Intensity:</strong> Higher intensity workouts require more recovery nutrients</li>
            <li><strong>Body Composition Goals:</strong> Building muscle vs. losing fat vs. maintaining</li>
            <li><strong>Individual Tolerance:</strong> Some people digest certain foods better than others</li>
            <li><strong>Dietary Restrictions:</strong> Vegan, vegetarian, allergies, etc.</li>
          </ul>

          <h2>Tracking and Adjusting</h2>
          <p>
            Monitor how your body responds to different post-workout nutrition strategies. Keep track of:
          </p>

          <ul>
            <li>Recovery rate (how quickly soreness subsides)</li>
            <li>Energy levels throughout the day</li>
            <li>Performance in subsequent workouts</li>
            <li>Body composition changes over time</li>
          </ul>

          <p>
            Adjust your approach based on these metrics. What works for one person may not work for another.
          </p>

          <h2>Conclusion</h2>
          <p>
            Post-workout nutrition is a crucial component of any training program. By prioritizing protein for muscle repair,
            carbohydrates for glycogen replenishment, and proper timing, you can optimize your recovery and maximize the benefits
            of your training. Remember that consistency and individualization are key - experiment with different approaches
            and find what works best for your body and goals.
          </p>
        </div>

        {/* Related Articles */}
        <div className="mt-12 animate-slide-up">
          <h3 className="text-2xl font-bold mb-6">Related Articles</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <Link to="/articles/progressive-overload">
              <Card className="glass-card hover:scale-105 transition-all duration-300 cursor-pointer">
                <CardContent className="p-6">
                  <div className="flex items-center gap-2 mb-3">
                    <Tag className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium text-primary">Training</span>
                  </div>
                  <h4 className="font-semibold mb-2">The Science of Progressive Overload</h4>
                  <p className="text-sm text-muted-foreground">Learn how to properly implement progressive overload in your training program...</p>
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