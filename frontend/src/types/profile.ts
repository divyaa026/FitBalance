export interface UserProfile {
    id: string;
    name: string;
    email: string;
    level: 'Beginner' | 'Intermediate' | 'Advanced';
    avatar?: string;
    bio?: string;
    joinedDate: string;
}

export interface FitnessGoal {
    id: string;
    type: 'nutrition' | 'workout' | 'form' | 'custom';
    name: string;
    target: number;
    unit: string;
    current: number;
    deadline?: string;
    createdAt: string;
    updatedAt: string;
}

export interface Achievement {
    id: string;
    title: string;
    description: string;
    icon: string;
    earnedDate: string;
    type: 'nutrition' | 'workout' | 'form' | 'special';
    rarity: 'common' | 'rare' | 'epic' | 'legendary';
}

export interface Stats {
    totalWorkouts: number;
    goalsCompleted: number;
    achievementsEarned: number;
    workoutStreak: number;
    perfectFormStreak: number;
    nutritionStreak: number;
}