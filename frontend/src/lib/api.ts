// FitBalance API Service
// Connects the React frontend with the FastAPI backend

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Types for API responses
export interface BiomechanicsAnalysis {
  form_score: number;
  joint_angles: Array<{
    joint_name: string;
    angle: number;
    is_abnormal: boolean;
  }>;
  torques: Array<{
    joint_name: string;
    torque_magnitude: number;
    risk_level: string;
  }>;
  recommendations: string[];
  exercise_type: string;
  user_id: string;
}

export interface FoodItem {
  name: string;
  protein_content: number;
  calories: number;
  confidence: number;
  tags?: string[];
  nutrients?: Record<string, number>;
}

export interface NutritionAnalysis {
  detected_foods: FoodItem[];
  total_protein: number;
  total_calories: number;
  total_carbs?: number;
  total_fat?: number;
  total_fiber?: number;
  meal_quality_score: number;
  recommendations: string[];
  shap_explanation?: string;
}

export interface BurnoutAnalysis {
  risk_score: number;
  risk_level: 'low' | 'moderate' | 'high' | 'critical';
  time_to_burnout: number;
  survival_probability: number;
  risk_factors: string[];
  recommendations: string[];
  user_id: string;
}

export interface SurvivalCurve {
  time_points: number[];
  survival_probabilities: number[];
  user_id: string;
}

// API Service Class
class FitBalanceAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // Helper method for making HTTP requests
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Helper method for file uploads
  private async uploadFile<T>(
    endpoint: string,
    file: File,
    additionalFields: Record<string, string> = {},
    fileFieldName: string = 'file'
  ): Promise<T> {
    const formData = new FormData();
    formData.append(fileFieldName, file);
    
    Object.entries(additionalFields).forEach(([key, value]) => {
      formData.append(key, value);
    });

    const response = await fetch(`${this.baseURL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `Upload Error: ${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch {
        // If we can't parse JSON, use the default error message
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }

  // Health check
  async healthCheck(): Promise<{ status: string; modules: Record<string, string> }> {
    return this.request('/health');
  }

  // Biomechanics Endpoints
  async analyzeBiomechanics(
    videoFile: File,
    exerciseType: string = 'squat',
    userId: string = 'default'
  ): Promise<BiomechanicsAnalysis> {
    return this.uploadFile('/biomechanics/analyze', videoFile, {
      exercise_type: exerciseType,
      user_id: userId,
    }, 'video_file');
  }

  async getTorqueHeatmap(
    userId: string,
    exerciseType: string = 'squat'
  ): Promise<{ heatmap: any; user_id: string; exercise: string }> {
    return this.request(`/biomechanics/heatmap/${userId}?exercise_type=${exerciseType}`);
  }

  // Nutrition Endpoints
  async analyzeMeal(
    mealPhoto: File,
    userId: string = 'default',
    dietaryRestrictions: string[] = []
  ): Promise<NutritionAnalysis> {
    return this.uploadFile('/nutrition/analyze-meal', mealPhoto, {
      user_id: userId,
      dietary_restrictions: JSON.stringify(dietaryRestrictions),
    }, 'meal_photo');
  }

  async getNutritionRecommendations(
    userId: string,
    targetProtein?: number,
    activityLevel: string = 'moderate'
  ): Promise<{ recommendations: string[]; target_protein: number; activity_level: string }> {
    const params = new URLSearchParams({
      activity_level: activityLevel,
    });
    
    if (targetProtein) {
      params.append('target_protein', targetProtein.toString());
    }

    return this.request(`/nutrition/recommendations/${userId}?${params}`);
  }

  // Burnout Endpoints
  async analyzeBurnoutRisk(
    userId: string,
    workoutFrequency: number,
    sleepHours: number,
    stressLevel: number,
    recoveryTime: number,
    performanceTrend: string = 'stable'
  ): Promise<BurnoutAnalysis> {
    return this.request('/burnout/analyze', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        workout_frequency: workoutFrequency,
        sleep_hours: sleepHours,
        stress_level: stressLevel,
        recovery_time: recoveryTime,
        performance_trend: performanceTrend,
      }),
    });
  }

  async getSurvivalCurve(userId: string): Promise<SurvivalCurve> {
    return this.request(`/burnout/survival-curve/${userId}`);
  }

  async getBurnoutRecommendations(
    userId: string
  ): Promise<{ recommendations: string[] }> {
    return this.request(`/burnout/recommendations/${userId}`);
  }
}

// Export singleton instance
export const api = new FitBalanceAPI();

// Export default for easy importing
export default api;