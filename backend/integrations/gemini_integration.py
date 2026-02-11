"""
Google Gemini API Integration for Enhanced Recommendations
Provides personalized insights, form feedback, and burnout guidance
"""

import os
from typing import Dict, List, Optional
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI package not installed. Install with: pip install google-generativeai")

class GeminiIntegration:
    """Gemini API integration for FitBalance"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or provided")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use Gemini Pro model
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_exercise_feedback(self, exercise_type: str, form_score: float,
                                   joint_angles: Dict[str, float],
                                   risk_factors: List[str]) -> str:
        """Generate personalized exercise form feedback"""
        
        prompt = f"""
You are a certified personal trainer and biomechanics expert analyzing exercise form data from computer vision.

Exercise: {exercise_type}
Form Score: {form_score}/100
Key Joint Angles: {json.dumps(joint_angles, indent=2)}
Risk Factors Detected: {', '.join(risk_factors) if risk_factors else 'None'}

Provide:
1. Brief assessment of current form (2-3 sentences)
2. 3-4 specific corrections to improve form
3. One safety tip to prevent injury
4. Encouragement based on their score

Keep response concise, actionable, and motivating. Use second person ("your", "you").
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating exercise feedback: {e}")
            return self._fallback_exercise_feedback(exercise_type, form_score, risk_factors)
    
    def generate_nutrition_plan(self, user_profile: Dict, goals: List[str],
                               dietary_restrictions: List[str],
                               current_intake: Dict) -> str:
        """Generate personalized nutrition recommendations"""
        
        prompt = f"""
You are a registered dietitian creating a personalized nutrition plan.

User Profile:
- Age: {user_profile.get('age', 'N/A')}
- Gender: {user_profile.get('gender', 'N/A')}
- Activity Level: {user_profile.get('activity_level', 'moderate')}
- Current Weight: {user_profile.get('weight', 'N/A')} kg

Goals: {', '.join(goals)}
Dietary Restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}

Current Average Daily Intake:
- Protein: {current_intake.get('protein', 0)}g
- Carbs: {current_intake.get('carbs', 0)}g
- Fats: {current_intake.get('fats', 0)}g
- Calories: {current_intake.get('calories', 0)} kcal

Provide:
1. Assessment of current intake vs goals
2. Recommended daily macros (protein/carbs/fats/calories)
3. 5 specific food suggestions aligned with restrictions
4. One meal timing tip for their goals

Be concise, practical, and evidence-based.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating nutrition plan: {e}")
            return self._fallback_nutrition_plan(user_profile, goals, current_intake)
    
    def generate_burnout_guidance(self, risk_level: str, risk_factors: List[str],
                                 user_metrics: Dict) -> str:
        """Generate personalized burnout prevention guidance"""
        
        prompt = f"""
You are a wellness coach specializing in burnout prevention and recovery.

Risk Level: {risk_level}
Primary Risk Factors: {', '.join(risk_factors)}

Recent Metrics (30-day averages):
- Sleep Quality: {user_metrics.get('sleep_quality', 0)}/100
- Stress Level: {user_metrics.get('stress_level', 0)}/100
- Workload: {user_metrics.get('workload', 0)}/100
- Social Support: {user_metrics.get('social_support', 0)}/100
- Exercise Frequency: {user_metrics.get('exercise_frequency', 0)} days/week
- Work-Life Balance: {user_metrics.get('work_life_balance', 0)}/100

Provide:
1. Brief empathetic acknowledgment of their situation
2. 3 immediate actionable steps to reduce risk
3. One long-term strategy for sustainable wellbeing
4. When to seek professional help (if risk is high)

Be supportive, non-judgmental, and practical. Prioritize evidence-based interventions.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating burnout guidance: {e}")
            return self._fallback_burnout_guidance(risk_level, risk_factors)
    
    def generate_workout_plan(self, fitness_level: str, goals: List[str],
                            available_equipment: List[str],
                            time_per_session: int) -> str:
        """Generate personalized workout plan"""
        
        prompt = f"""
You are a certified strength and conditioning specialist creating a workout plan.

Fitness Level: {fitness_level}
Goals: {', '.join(goals)}
Available Equipment: {', '.join(available_equipment)}
Time Available: {time_per_session} minutes per session

Create a 3-day per week workout plan with:
1. Brief workout structure overview
2. Specific exercises for each day (with sets/reps)
3. Progression strategy for next 4 weeks
4. Recovery and rest day recommendations

Be specific, practical, and aligned with their goals and constraints.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating workout plan: {e}")
            return self._fallback_workout_plan(fitness_level, goals, time_per_session)
    
    def analyze_progress(self, metric_history: List[Dict], metric_name: str,
                        goal: str) -> str:
        """Analyze progress towards a specific goal"""
        
        # Get recent trend
        recent_values = [m['value'] for m in metric_history[-30:]]
        avg_recent = sum(recent_values) / len(recent_values) if recent_values else 0
        
        older_values = [m['value'] for m in metric_history[-60:-30]] if len(metric_history) >= 60 else recent_values
        avg_older = sum(older_values) / len(older_values) if older_values else avg_recent
        
        trend = "improving" if avg_recent > avg_older else "declining" if avg_recent < avg_older else "stable"
        
        prompt = f"""
You are a fitness and wellness coach analyzing user progress.

Metric: {metric_name}
Goal: {goal}
Current Average (last 30 days): {avg_recent:.1f}
Previous Average (30-60 days ago): {avg_older:.1f}
Trend: {trend}

Provide:
1. Brief progress assessment (1-2 sentences)
2. What's working well
3. One specific adjustment to accelerate progress
4. Motivational message

Be encouraging, data-driven, and actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error analyzing progress: {e}")
            return self._fallback_progress_analysis(metric_name, trend)
    
    # Fallback methods for when API fails
    
    def _fallback_exercise_feedback(self, exercise_type: str, form_score: float, 
                                   risk_factors: List[str]) -> str:
        if form_score >= 80:
            return f"Great form on {exercise_type}! Keep maintaining proper technique and gradual progression."
        elif form_score >= 60:
            return f"Good effort on {exercise_type}. Focus on: {', '.join(risk_factors[:2])} to improve form and safety."
        else:
            return f"Your {exercise_type} form needs improvement. Key areas: {', '.join(risk_factors)}. Consider working with a trainer."
    
    def _fallback_nutrition_plan(self, user_profile: Dict, goals: List[str], 
                                current_intake: Dict) -> str:
        return f"Based on your {', '.join(goals)}, focus on balanced macros: adequate protein (1.6-2.2g/kg), complex carbs, healthy fats. Consult a dietitian for personalized guidance."
    
    def _fallback_burnout_guidance(self, risk_level: str, risk_factors: List[str]) -> str:
        if risk_level == "high":
            return f"Your burnout risk is high. Immediate steps: improve sleep, reduce workload, seek support. Consider professional counseling."
        return f"Manage burnout risk by addressing: {', '.join(risk_factors[:3])}. Prioritize rest, boundaries, and self-care."
    
    def _fallback_workout_plan(self, fitness_level: str, goals: List[str], 
                              time_per_session: int) -> str:
        return f"For {fitness_level} level and {', '.join(goals)}, focus on compound movements 3x/week in {time_per_session}-min sessions. Progressive overload is key."
    
    def _fallback_progress_analysis(self, metric_name: str, trend: str) -> str:
        if trend == "improving":
            return f"Your {metric_name} is {trend}! Keep up the great work. Stay consistent with current strategies."
        return f"Your {metric_name} is {trend}. Review your approach and consider adjusting intensity or frequency."

# Example usage
def test_gemini_integration():
    """Test Gemini integration with sample data"""
    
    # Check if API key exists
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set. Set it with: $env:GEMINI_API_KEY='your-key-here'")
        return
    
    try:
        gemini = GeminiIntegration()
        
        # Test exercise feedback
        print("\n🏋️ Exercise Feedback Test:")
        print("-" * 60)
        feedback = gemini.generate_exercise_feedback(
            exercise_type="squat",
            form_score=72.5,
            joint_angles={
                "left_knee": 95,
                "right_knee": 92,
                "hip": 88
            },
            risk_factors=["Knee valgus", "Forward lean"]
        )
        print(feedback)
        
        print("\n✅ Gemini integration test complete!")
        
    except Exception as e:
        print(f"❌ Error testing Gemini: {e}")

if __name__ == "__main__":
    test_gemini_integration()
