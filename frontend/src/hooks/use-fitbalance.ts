import { useState, useCallback } from 'react';
import { api, BiomechanicsAnalysis, NutritionAnalysis, BurnoutAnalysis, SurvivalCurve } from '@/lib/api';

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

// Custom hook for burnout analysis
export function useBurnout() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<BurnoutAnalysis | null>(null);
  const [survivalCurve, setSurvivalCurve] = useState<SurvivalCurve | null>(null);

  const analyzeRisk = useCallback(async (
    userId: string,
    workoutFrequency: number,
    sleepHours: number,
    stressLevel: number,
    recoveryTime: number,
    performanceTrend: string = 'stable'
  ) => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await api.analyzeBurnoutRisk(
        userId,
        workoutFrequency,
        sleepHours,
        stressLevel,
        recoveryTime,
        performanceTrend
      );
      setAnalysis(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Risk analysis failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
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
    survivalCurve,
    analyzeRisk,
    getSurvivalCurve,
    getRecommendations,
  };
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