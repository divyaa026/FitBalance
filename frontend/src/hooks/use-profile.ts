import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UserProfile, FitnessGoal, Achievement, Stats } from '@/types/profile';

// Simulated API functions - replace with actual API calls
const fetchProfile = async (): Promise<UserProfile> => {
    // Simulated API response
    return {
        id: '1',
        name: 'FitBalance User',
        email: 'user@example.com',
        level: 'Intermediate',
        joinedDate: new Date().toISOString(),
    };
};

const fetchGoals = async (): Promise<FitnessGoal[]> => {
    // Simulated API response
    return [
        {
            id: '1',
            type: 'nutrition',
            name: 'Daily Protein Intake',
            target: 150,
            unit: 'g',
            current: 100,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        },
        {
            id: '2',
            type: 'workout',
            name: 'Weekly Workouts',
            target: 5,
            unit: 'sessions',
            current: 3,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        },
        {
            id: '3',
            type: 'form',
            name: 'Form Score Average',
            target: 85,
            unit: 'points',
            current: 75,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        },
    ];
};

const fetchAchievements = async (): Promise<Achievement[]> => {
    // Simulated API response
    return [
        {
            id: '1',
            title: 'First Workout',
            description: 'Completed your first workout session',
            icon: 'dumbbell',
            earnedDate: new Date().toISOString(),
            type: 'workout',
            rarity: 'common',
        },
        {
            id: '2',
            title: 'Perfect Form',
            description: 'Achieved 100% form score in a workout',
            icon: 'target',
            earnedDate: new Date().toISOString(),
            type: 'form',
            rarity: 'rare',
        },
    ];
};

const fetchStats = async (): Promise<Stats> => {
    // Simulated API response
    return {
        totalWorkouts: 15,
        goalsCompleted: 5,
        achievementsEarned: 2,
        workoutStreak: 3,
        perfectFormStreak: 1,
        nutritionStreak: 2,
    };
};

export const useProfile = () => {
    const queryClient = useQueryClient();
    const [isEditing, setIsEditing] = useState(false);

    const profile = useQuery({
        queryKey: ['profile'],
        queryFn: fetchProfile,
    });

    const goals = useQuery({
        queryKey: ['goals'],
        queryFn: fetchGoals,
    });

    const achievements = useQuery({
        queryKey: ['achievements'],
        queryFn: fetchAchievements,
    });

    const stats = useQuery({
        queryKey: ['stats'],
        queryFn: fetchStats,
    });

    const updateProfile = useMutation({
        mutationFn: async (updatedProfile: Partial<UserProfile>) => {
            // Simulated API call
            await new Promise(resolve => setTimeout(resolve, 1000));
            return updatedProfile;
        },
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['profile'] });
        },
    });

    const updateGoal = useMutation({
        mutationFn: async (updatedGoal: Partial<FitnessGoal>) => {
            // Simulated API call
            await new Promise(resolve => setTimeout(resolve, 1000));
            return updatedGoal;
        },
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['goals'] });
        },
    });

    return {
        profile: profile.data,
        goals: goals.data,
        achievements: achievements.data,
        stats: stats.data,
        isLoading: profile.isLoading || goals.isLoading || achievements.isLoading || stats.isLoading,
        isError: profile.isError || goals.isError || achievements.isError || stats.isError,
        updateProfile: updateProfile.mutate,
        updateGoal: updateGoal.mutate,
        isEditing,
        setIsEditing,
    };
};