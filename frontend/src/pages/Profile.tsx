import { User, Target, Award, Settings, TrendingUp, Activity, Plus, Medal, Dumbbell, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { EditProfileDialog } from "@/components/EditProfileDialog";
import { EditGoalDialog } from "@/components/EditGoalDialog";
import { useProfile } from "@/hooks/use-profile";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useState } from "react";

export default function Profile() {
  const {
    profile,
    goals,
    achievements,
    stats,
    isLoading,
    updateProfile,
    updateGoal,
  } = useProfile();

  const [editProfileOpen, setEditProfileOpen] = useState(false);
  const [editGoalOpen, setEditGoalOpen] = useState(false);
  const [selectedGoal, setSelectedGoal] = useState<string | null>(null);

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="min-h-screen px-4 py-8 relative z-10">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 text-accent mb-4">
            <User className="h-4 w-4" />
            <span className="text-sm font-medium">Your Profile</span>
          </div>
          <h1 className="text-4xl font-bold mb-2">Profile & Settings</h1>
          <p className="text-muted-foreground">
            Track your progress and manage your fitness journey
          </p>
        </div>

        {/* User Info Card */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardContent className="pt-6">
            <div className="flex items-center gap-6">
              <Avatar className="h-24 w-24">
                {profile?.avatar ? (
                  <AvatarImage src={profile.avatar} alt={profile?.name} />
                ) : (
                  <AvatarFallback className="gradient-profile text-white text-3xl">
                    {profile?.name?.charAt(0) || 'FB'}
                  </AvatarFallback>
                )}
              </Avatar>

              <div className="flex-1">
                <h2 className="text-2xl font-bold mb-1">{profile?.name}</h2>
                <p className="text-muted-foreground mb-4">{profile?.level} Level</p>
                <p className="text-sm text-muted-foreground mb-4">{profile?.bio}</p>

                <div className="flex gap-4">
                  <Button variant="secondary" size="sm" onClick={() => setEditProfileOpen(true)}>
                    <Settings className="mr-2 h-4 w-4" />
                    Edit Profile
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => {
                    setSelectedGoal(null);
                    setEditGoalOpen(true);
                  }}>
                    <Plus className="mr-2 h-4 w-4" />
                    Add Goal
                  </Button>
                </div>
              </div>

              <EditProfileDialog
                profile={profile || {}}
                onUpdate={updateProfile}
                open={editProfileOpen}
                onOpenChange={setEditProfileOpen}
              />
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-6 animate-slide-up">
          {[
            { icon: Activity, label: "Workouts", value: stats?.totalWorkouts || 0, change: `+${stats?.workoutStreak || 0} streak`, gradient: "gradient-biomechanics" },
            { icon: Target, label: "Goals Met", value: stats?.goalsCompleted || 0, change: `${((goals?.length || 1) / (stats?.goalsCompleted || 1) * 100).toFixed(0)}%`, gradient: "gradient-nutrition" },
            { icon: Award, label: "Achievements", value: stats?.achievementsEarned || 0, change: `${achievements?.length || 0} total`, gradient: "gradient-burnout" },
          ].map((stat, index) => (
            <Card key={index} className="glass-card">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between mb-4">
                  <div className={`h-12 w-12 rounded-xl ${stat.gradient} flex items-center justify-center`}>
                    <stat.icon className="h-6 w-6 text-white" />
                  </div>
                  <div className="flex items-center gap-1 text-green-500 text-sm">
                    <TrendingUp className="h-4 w-4" />
                    {stat.change}
                  </div>
                </div>
                <div className="text-3xl font-bold mb-1">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Goals Section */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-primary" />
                  Fitness Goals
                </CardTitle>
                <CardDescription>
                  Set and track your personal fitness objectives
                </CardDescription>
              </div>
              <Button variant="secondary" size="sm" onClick={() => {
                setSelectedGoal(null);
                setEditGoalOpen(true);
              }}>
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {goals?.map((goal) => {
                const progress = Math.min((goal.current / goal.target) * 100, 100);
                return (
                  <div key={goal.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{goal.name}</span>
                          <Badge variant="outline">{goal.type}</Badge>
                        </div>
                        <span className="text-sm text-muted-foreground">
                          {goal.current} / {goal.target} {goal.unit}
                        </span>
                      </div>
                      <Button variant="ghost" size="sm" onClick={() => {
                        setSelectedGoal(goal.id);
                        setEditGoalOpen(true);
                      }}>
                        <Settings className="h-4 w-4" />
                      </Button>
                    </div>
                    <Progress value={progress} />
                  </div>
                );
              })}
            </div>

            {goals?.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No goals set yet. Click the + button to add your first goal!</p>
              </div>
            )}
          </CardContent>
        </Card>

        <EditGoalDialog
          goal={goals?.find(g => g.id === selectedGoal)}
          onUpdate={updateGoal}
          open={editGoalOpen}
          onOpenChange={setEditGoalOpen}
          isNewGoal={!selectedGoal}
        />

        {/* Achievements */}
        <Card className="glass-card animate-slide-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award className="h-5 w-5 text-yellow-500" />
              Achievements
            </CardTitle>
            <CardDescription>
              Your fitness milestones and accomplishments
            </CardDescription>
          </CardHeader>
          <CardContent>
            {achievements && achievements.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {achievements.map((achievement) => (
                  <Card key={achievement.id} className="border border-accent/20">
                    <CardContent className="pt-6">
                      <div className="flex items-start gap-4">
                        <div className={`h-12 w-12 rounded-xl flex items-center justify-center
                          ${achievement.rarity === 'legendary' ? 'gradient-profile' :
                            achievement.rarity === 'epic' ? 'gradient-biomechanics' :
                              achievement.rarity === 'rare' ? 'gradient-nutrition' :
                                'gradient-burnout'}`}>
                          {achievement.icon === 'dumbbell' ? <Dumbbell className="h-6 w-6 text-white" /> :
                            achievement.icon === 'target' ? <Target className="h-6 w-6 text-white" /> :
                              achievement.icon === 'medal' ? <Medal className="h-6 w-6 text-white" /> :
                                <Zap className="h-6 w-6 text-white" />}
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <h4 className="font-semibold">{achievement.title}</h4>
                            <Badge variant="outline" className={
                              achievement.rarity === 'legendary' ? 'border-yellow-500 text-yellow-500' :
                                achievement.rarity === 'epic' ? 'border-purple-500 text-purple-500' :
                                  achievement.rarity === 'rare' ? 'border-blue-500 text-blue-500' :
                                    'border-gray-500 text-gray-500'
                            }>
                              {achievement.rarity}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">{achievement.description}</p>
                          <p className="text-xs text-muted-foreground mt-2">
                            Earned on {new Date(achievement.earnedDate).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Award className="h-16 w-16 mx-auto mb-4 opacity-50" />
                <p>Complete your first workout to start earning achievements!</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
