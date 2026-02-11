import { ArrowLeft, Clock, Tag } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function MentalBenefitsArticle() {
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
              Wellness
            </div>
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              6 min read
            </div>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent">
            Mental Benefits of Exercise
          </h1>

          <p className="text-lg text-muted-foreground leading-relaxed">
            Explore how consistent physical activity can improve your mental well-being and reduce stress levels.
          </p>
        </div>

        {/* Featured Image */}
        <div className="mb-8 animate-slide-up">
          <div className="aspect-[16/9] rounded-xl overflow-hidden">
            <img
              src="https://images.unsplash.com/photo-1544367567-0f2fcb009e0b"
              alt="Mental Benefits of Exercise"
              className="w-full h-full object-cover"
            />
          </div>
        </div>

        {/* Article Content */}
        <div className="prose prose-lg prose-invert max-w-none animate-slide-up">
          <h2>The Mind-Body Connection</h2>
          <p>
            The relationship between physical activity and mental health is profound and well-established. Regular exercise doesn't
            just strengthen your muscles and cardiovascular system—it also acts as a powerful antidepressant, anxiolytic, and cognitive
            enhancer. Understanding these connections can motivate you to make exercise a consistent part of your lifestyle.
          </p>

          <h2>Exercise and Depression</h2>
          <p>
            Numerous studies have shown that regular physical activity can be as effective as antidepressant medication for treating
            mild to moderate depression. Exercise promotes the release of endorphins, serotonin, and other neurotransmitters that
            improve mood and create feelings of well-being.
          </p>

          <div className="bg-muted/50 p-6 rounded-lg my-6">
            <h4 className="font-semibold mb-4">Key Mechanisms</h4>
            <ul className="space-y-2">
              <li><strong>Endorphin Release:</strong> Natural painkillers that create euphoria</li>
              <li><strong>Serotonin Production:</strong> Mood-regulating neurotransmitter</li>
              <li><strong>BDNF Increase:</strong> Brain-derived neurotrophic factor supports neuron growth</li>
              <li><strong>Neuroplasticity:</strong> Enhanced brain adaptability and resilience</li>
            </ul>
          </div>

          <h2>Anxiety Reduction</h2>
          <p>
            Exercise serves as a natural anxiety reliever by reducing levels of stress hormones like cortisol and adrenaline.
            Physical activity also provides a healthy outlet for pent-up energy and tension, helping to break the cycle of anxious thoughts.
          </p>

          <h3>Types of Exercise for Anxiety</h3>
          <ul>
            <li><strong>Aerobic Exercise:</strong> Running, cycling, swimming - sustained moderate activity</li>
            <li><strong>Yoga:</strong> Combines movement with mindfulness and breathing</li>
            <li><strong>Martial Arts:</strong> Provides structure and focus while building confidence</li>
            <li><strong>Team Sports:</strong> Social interaction combined with physical activity</li>
          </ul>

          <h2>Stress Management</h2>
          <p>
            Chronic stress takes a toll on both mental and physical health. Exercise provides a healthy coping mechanism by:
          </p>

          <ul>
            <li>Reducing cortisol levels</li>
            <li>Improving sleep quality</li>
            <li>Building resilience to stress</li>
            <li>Providing a sense of accomplishment</li>
            <li>Creating "me time" for mental processing</li>
          </ul>

          <h2>Cognitive Benefits</h2>

          <h3>Enhanced Brain Function</h3>
          <p>
            Regular exercise has been shown to improve various aspects of cognitive function, including:
          </p>

          <ul>
            <li><strong>Memory:</strong> Better short-term and long-term memory retention</li>
            <li><strong>Attention:</strong> Improved focus and concentration</li>
            <li><strong>Processing Speed:</strong> Faster information processing</li>
            <li><strong>Executive Function:</strong> Better planning and decision-making</li>
          </ul>

          <h3>Neuroprotection</h3>
          <p>
            Exercise may help protect against age-related cognitive decline and neurodegenerative diseases. Studies suggest
            that physically active individuals have a lower risk of developing Alzheimer's disease and other forms of dementia.
          </p>

          <h2>Sleep Quality Improvement</h2>
          <p>
            Regular physical activity can significantly improve sleep quality and duration. Exercise helps regulate your circadian
            rhythm and reduces the time it takes to fall asleep. However, timing matters—avoid intense exercise close to bedtime
            as it may be stimulating.
          </p>

          <div className="bg-muted/50 p-6 rounded-lg my-6">
            <h4 className="font-semibold mb-4">Sleep Optimization Tips</h4>
            <ul className="space-y-2">
              <li>Exercise 3-4 hours before bedtime</li>
              <li>Maintain consistent exercise timing</li>
              <li>Include relaxation exercises like yoga</li>
              <li>Avoid screens 1 hour before bed</li>
              <li>Create a cool, dark sleep environment</li>
            </ul>
          </div>

          <h2>Body Image and Self-Esteem</h2>
          <p>
            Regular exercise can improve body image and self-esteem through multiple mechanisms:
          </p>

          <ul>
            <li><strong>Physical Changes:</strong> Improved fitness and appearance</li>
            <li><strong>Achievement:</strong> Setting and reaching goals</li>
            <li><strong>Mastery:</strong> Developing new skills and competencies</li>
            <li><strong>Social Benefits:</strong> Improved social interactions and relationships</li>
          </ul>

          <h2>Social Connection</h2>
          <p>
            Many forms of exercise provide opportunities for social interaction, which is crucial for mental health. Group fitness
            classes, team sports, and exercise partners can help combat loneliness and build supportive relationships.
          </p>

          <h2>Exercise as Meditation</h2>
          <p>
            Certain forms of exercise, particularly yoga, tai chi, and mindful walking, can serve as moving meditation. These activities
            combine physical movement with present-moment awareness, providing similar mental benefits to seated meditation.
          </p>

          <h2>Getting Started: Making Exercise a Habit</h2>

          <h3>Start Small</h3>
          <p>
            Begin with activities you enjoy and can sustain. Even 10-15 minutes of daily movement can provide mental health benefits.
            Gradually increase duration and intensity as you build confidence and fitness.
          </p>

          <h3>Find Enjoyable Activities</h3>
          <p>
            Choose exercises that bring you joy rather than viewing them as punishment. This could include dancing, hiking, swimming,
            cycling, or any physical activity that makes you feel good.
          </p>

          <h3>Set Realistic Goals</h3>
          <p>
            Focus on consistency rather than intensity. Set achievable goals like "exercise 3 times per week" rather than
            "lose 20 pounds in 2 months."
          </p>

          <h3>Track Your Progress</h3>
          <p>
            Keep a journal of how exercise makes you feel mentally and physically. This can help maintain motivation and
            identify patterns in your mental health.
          </p>

          <h2>Special Considerations</h2>

          <h3>Mental Health Conditions</h3>
          <p>
            While exercise can be beneficial for mental health, it's not a substitute for professional treatment. If you're dealing
            with severe depression, anxiety, or other mental health conditions, consult with a healthcare professional before
            starting a new exercise program.
          </p>

          <h3>Exercise and Medication</h3>
          <p>
            Exercise can complement medication and therapy for mental health conditions. Always consult your healthcare provider
            about how exercise fits into your treatment plan.
          </p>

          <h3>Overtraining and Mental Health</h3>
          <p>
            While moderate exercise is beneficial, excessive exercise can lead to overtraining syndrome, which can negatively
            impact mental health. Listen to your body and include adequate rest and recovery.
          </p>

          <h2>The Long-Term Impact</h2>
          <p>
            The mental health benefits of exercise compound over time. Regular physical activity creates a positive feedback loop:
            improved mood leads to better adherence to exercise, which further improves mental health. This creates a foundation
            for long-term mental wellness and resilience.
          </p>

          <h2>Conclusion</h2>
          <p>
            The mental benefits of exercise extend far beyond physical fitness. Regular physical activity can improve mood, reduce
            anxiety, enhance cognitive function, and provide a powerful tool for managing stress and mental health challenges.
            Whether you're looking to prevent mental health issues or support existing treatment, exercise should be considered
            a fundamental component of mental wellness strategy.
          </p>

          <p>
            Remember that any movement is better than no movement. Find activities you enjoy, start small, and build gradually.
            Your mind and body will thank you for it.
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
          </div>
        </div>
      </div>
    </div>
  );
}