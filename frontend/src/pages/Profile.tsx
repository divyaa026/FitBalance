import { User, Target, Award, Settings, TrendingUp, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

export default function Profile() {
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
                <AvatarFallback className="gradient-profile text-white text-3xl">
                  FB
                </AvatarFallback>
              </Avatar>
              
              <div className="flex-1">
                <h2 className="text-2xl font-bold mb-1">FitBalance User</h2>
                <p className="text-muted-foreground mb-4">Intermediate Level</p>
                
                <div className="flex gap-4">
                  <Button variant="outline" size="sm">
                    <Settings className="mr-2 h-4 w-4" />
                    Edit Profile
                  </Button>
                  <Button variant="outline" size="sm">
                    <Target className="mr-2 h-4 w-4" />
                    Set Goals
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-6 animate-slide-up">
          {[
            { icon: Activity, label: "Workouts", value: "0", change: "+0%", gradient: "gradient-biomechanics" },
            { icon: Target, label: "Goals Met", value: "0", change: "+0%", gradient: "gradient-nutrition" },
            { icon: Award, label: "Achievements", value: "0", change: "+0", gradient: "gradient-burnout" },
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
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-primary" />
              Fitness Goals
            </CardTitle>
            <CardDescription>
              Set and track your personal fitness objectives
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { goal: "Daily Protein Intake", current: "0g", target: "150g", progress: 0 },
                { goal: "Weekly Workouts", current: "0", target: "5", progress: 0 },
                { goal: "Form Score Average", current: "0", target: "85+", progress: 0 },
              ].map((goal, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{goal.goal}</span>
                    <span className="text-muted-foreground">
                      {goal.current} / {goal.target}
                    </span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full gradient-profile transition-all duration-500"
                      style={{ width: `${goal.progress}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            
            <Button className="w-full mt-6 gradient-profile text-white">
              Update Goals
            </Button>
          </CardContent>
        </Card>

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
            <div className="text-center py-12 text-muted-foreground">
              <Award className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p>Complete your first workout to start earning achievements!</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
