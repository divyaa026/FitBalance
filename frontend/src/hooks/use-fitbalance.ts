import { useState, useCallback } from 'react';
import { api, BiomechanicsAnalysis, NutritionAnalysis, BurnoutAnalysis, SurvivalCurve } from '@/lib/api';
import { API_ENDPOINTS } from '@/config/api';
import { useQuery } from '@tanstack/react-query';

// Custom hook for biomechanics analysis
export function useBiomechanics() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<BiomechanicsAnalysis | null>(null);

  const analyzeMovement = useCallback(async (
    videoFile: File,
    exerciseType: string,
    userId: string = 'default'
  ) => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await api.analyzeBiomechanics(videoFile, exerciseType, userId);
      setAnalysis(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getHeatmap = useCallback(async (userId: string, exerciseType: string) => {
    try {
      setError(null);
      return await api.getTorqueHeatmap(userId, exerciseType);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get heatmap';
      setError(errorMessage);
      throw err;
    }
  }, []);

  return {
    isLoading,
    error,
    analysis,
    analyzeMovement,
    getHeatmap,
  };
}

// Custom hook for nutrition analysis
export function useNutrition() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<NutritionAnalysis | null>(null);

  const analyzeMeal = useCallback(async (
    mealPhoto: File,
    userId: string = 'default',
    dietaryRestrictions: string[] = []
  ) => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await api.analyzeMeal(mealPhoto, userId, dietaryRestrictions);
      setAnalysis(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Meal analysis failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getRecommendations = useCallback(async (
    userId: string,
    targetProtein?: number,
    activityLevel: string = 'moderate'
  ) => {
    try {
      setError(null);
      return await api.getNutritionRecommendations(userId, targetProtein, activityLevel);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get recommendations';
      setError(errorMessage);
      throw err;
    }
  }, []);

  return {
    isLoading,
    error,
    analysis,
    analyzeMeal,
    getRecommendations,
  };
}

// Additional nutrition hooks
import API_BASE_URL from '@/config/api';
import { API_ENDPOINTS } from '@/config/api';
import { useQuery } from '@tanstack/react-query';

export function useNutritionHistory(userId: number, days: number = 7) {
  return useQuery({
    queryKey: ['nutrition-history', userId, days],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.NUTRITION_HISTORY(userId, days));
      if (!response.ok) throw new Error('Failed to fetch nutrition history');
      return response.json();
    },
  });
}

export function useNutritionStats(userId: number, days: number = 7) {
  return useQuery({
    queryKey: ['nutrition-stats', userId, days],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.NUTRITION_STATS(userId, days));
      if (!response.ok) throw new Error('Failed to fetch nutrition stats');
      return response.json();
    },
  });
}

export function useFoodDatabase() {
  return useQuery({
    queryKey: ['food-database'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.NUTRITION_FOODS);
      if (!response.ok) throw new Error('Failed to fetch food database');
      return response.json();
    },
  });
}

// Custom hook for burnout analysis
export function useBurnout() {
  const [burnoutData, setBurnoutData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
<<<<<<< HEAD

  const analyzeBurnout = async (data: {
    user_id: string;
    workout_frequency: number;
    sleep_hours: number;
    stress_level: number;
    recovery_time: number;
    performance_trend: string;
  }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(API_ENDPOINTS.BURNOUT_ANALYZE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
=======
  const [analysis, setAnalysis] = useState<BurnoutAnalysis | null>(null);
  const [burnoutData, setBurnoutData] = useState<any>(null);
  const [survivalCurve, setSurvivalCurve] = useState<SurvivalCurve | null>(null);

  const analyzeBurnout = useCallback(async (
    data: {
      user_id: string;
      workout_frequency: number;
      sleep_hours: number;
      stress_level: number;
      recovery_time: number;
      performance_trend?: string;
    }
  ) => {
    try {
      setIsLoading(true);
      setError(null);
      const payload = {
        user_id: data.user_id,
        workout_frequency: data.workout_frequency,
        sleep_hours: data.sleep_hours,
        stress_level: data.stress_level,
        recovery_time: data.recovery_time,
        performance_trend: data.performance_trend || 'stable',
      };

      const response = await fetch(API_ENDPOINTS.BURNOUT_ANALYZE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
>>>>>>> 633c84e602780eab5038f97c9beaa390e270d288
      });

      if (!response.ok) throw new Error('Burnout analysis failed');

      const result = await response.json();
<<<<<<< HEAD
      setBurnoutData(result);
    } catch (err: any) {
      setError(err.message);
=======
      setAnalysis(result);
      setBurnoutData(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Risk analysis failed';
      setError(errorMessage);
>>>>>>> 633c84e602780eab5038f97c9beaa390e270d288
      throw err;
    } finally {
      setIsLoading(false);
    }
<<<<<<< HEAD
=======
  }, []);

  const getSurvivalCurve = useCallback(async (userId: string) => {
    try {
      setError(null);
      const result = await api.getSurvivalCurve(userId);
      setSurvivalCurve(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get survival curve';
      setError(errorMessage);
      throw err;
    }
  }, []);

  const getRecommendations = useCallback(async (userId: string) => {
    try {
      setError(null);
      return await api.getBurnoutRecommendations(userId);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get recommendations';
      setError(errorMessage);
      throw err;
    }
  }, []);

  return {
    isLoading,
    error,
    analysis,
    burnoutData,
    survivalCurve,
    analyzeBurnout,
    getSurvivalCurve,
    getRecommendations,
>>>>>>> 633c84e602780eab5038f97c9beaa390e270d288
  };

  return { isLoading, error, burnoutData, analyzeBurnout };
}

// Health check hook
export function useHealthCheck() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);

  const checkHealth = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await api.healthCheck();
      setStatus(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Health check failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    isLoading,
    error,
    status,
    checkHealth,
  };
}

// Add nutrition history hook
export function useNutritionHistory(userId: number, days: number = 7) {
  return useQuery({
    queryKey: ['nutrition-history', userId, days],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.NUTRITION_HISTORY(userId, days));
      if (!response.ok) throw new Error('Failed to fetch nutrition history');
      return response.json();
    },
  });
}

// Add nutrition stats hook
export function useNutritionStats(userId: number, days: number = 7) {
  return useQuery({
    queryKey: ['nutrition-stats', userId, days],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.NUTRITION_STATS(userId, days));
      if (!response.ok) throw new Error('Failed to fetch nutrition stats');
      return response.json();
    },
  });
}

// Add food database hook
export function useFoodDatabase() {
  return useQuery({
    queryKey: ['food-database'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.NUTRITION_FOODS);
      if (!response.ok) throw new Error('Failed to fetch food database');
      return response.json();
    },
  });
}