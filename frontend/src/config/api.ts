/**
 * API Configuration
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
<<<<<<< HEAD
  // Nutrition endpoints
  NUTRITION_ANALYZE: `${API_BASE_URL}/nutrition/analyze`,
  NUTRITION_HISTORY: (userId: number, days: number = 7) => 
    `${API_BASE_URL}/nutrition/history/${userId}?days=${days}`,
  NUTRITION_STATS: (userId: number, days: number = 7) => 
    `${API_BASE_URL}/nutrition/stats/${userId}?days=${days}`,
  NUTRITION_FOODS: `${API_BASE_URL}/nutrition/foods`,
  NUTRITION_RECOMMENDATIONS: (userId: string) => 
    `${API_BASE_URL}/nutrition/recommendations/${userId}`,

  // Biomechanics endpoints
  BIOMECHANICS_ANALYZE: `${API_BASE_URL}/biomechanics/analyze`,
  BIOMECHANICS_HEATMAP: (userId: string, exercise: string) => 
    `${API_BASE_URL}/biomechanics/heatmap/${userId}/${exercise}`,

  // Burnout endpoints
  BURNOUT_ANALYZE: `${API_BASE_URL}/burnout/analyze`,
  BURNOUT_SURVIVAL: (userId: string) => 
    `${API_BASE_URL}/burnout/survival-curve/${userId}`,
  BURNOUT_RECOMMENDATIONS: (userId: string) => 
    `${API_BASE_URL}/burnout/recommendations/${userId}`,

  // Health sync endpoints
  HEALTH_SYNC_GOOGLE: `${API_BASE_URL}/health/sync/google-fit`,
  HEALTH_SYNC_FITBIT: `${API_BASE_URL}/health/sync/fitbit`,
=======
    // Nutrition endpoints
    NUTRITION_ANALYZE: `${API_BASE_URL}/nutrition/analyze`,
    NUTRITION_HISTORY: (userId: number | string, days: number = 7) =>
        `${API_BASE_URL}/nutrition/history/${userId}?days=${days}`,
    NUTRITION_STATS: (userId: number | string, days: number = 7) =>
        `${API_BASE_URL}/nutrition/stats/${userId}?days=${days}`,
    NUTRITION_FOODS: `${API_BASE_URL}/nutrition/foods`,
    NUTRITION_RECOMMENDATIONS: (userId: string | number) =>
        `${API_BASE_URL}/nutrition/recommendations/${userId}`,

    // Biomechanics endpoints
    BIOMECHANICS_ANALYZE: `${API_BASE_URL}/biomechanics/analyze`,
    BIOMECHANICS_HEATMAP: (userId: string | number, exercise: string) =>
        `${API_BASE_URL}/biomechanics/heatmap/${userId}/${exercise}`,

    // Burnout endpoints
    BURNOUT_ANALYZE: `${API_BASE_URL}/burnout/analyze`,
    BURNOUT_SURVIVAL: (userId: string | number) =>
        `${API_BASE_URL}/burnout/survival-curve/${userId}`,
    BURNOUT_RECOMMENDATIONS: (userId: string | number) =>
        `${API_BASE_URL}/burnout/recommendations/${userId}`,

    // Health sync endpoints
    HEALTH_SYNC_GOOGLE: `${API_BASE_URL}/health/sync/google-fit`,
    HEALTH_SYNC_FITBIT: `${API_BASE_URL}/health/sync/fitbit`,
>>>>>>> 633c84e602780eab5038f97c9beaa390e270d288
};

export default API_BASE_URL;
