# Person 3: Frontend Developer - Task Guide

**Project:** FitBalance  
**Duration:** 2-3 weeks (60-80 hours)  
**Contribution:** 20-25% of total project  
**Role:** EXCLUSIVE FRONTEND WORK - All React/TypeScript UI development

---

## 📋 Overview

You are responsible for **ALL frontend work**:
1. Complete **Nutrition Page** (camera, history, charts, stats)
2. Complete **Biomechanics Page** (form analysis, heatmaps, visualizations)
3. Complete **Burnout Page** (risk assessment, survival curves)
4. **Mobile Responsiveness** across all pages
5. **API Integration** with backend endpoints
6. **Testing** and **documentation**

**No backend work, no ML work, no DevOps** - focus 100% on frontend user experience.

---

## 🎯 Your Deliverables

- ✅ Complete Nutrition page with history, stats, and charts
- ✅ Complete Biomechanics page with form analysis and heatmaps
- ✅ Complete Burnout page with risk assessment and survival curves
- ✅ Mobile-responsive design for all pages
- ✅ Loading states and error handling
- ✅ Toast notifications system
- ✅ API integration with all backend endpoints
- ✅ Frontend testing checklist completed
- ✅ Frontend documentation

---

## 📂 File Structure You'll Work With

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/                     # shadcn/ui components (pre-built)
│   │   ├── Layout.tsx              # Existing
│   │   ├── MobileNav.tsx           # ← YOU: Create
│   │   ├── LoadingSpinner.tsx      # ← YOU: Create
│   │   └── TorqueHeatmap.tsx       # ← YOU: Create
│   ├── pages/
│   │   ├── Home.tsx                # ✅ Already complete
│   │   ├── Nutrition.tsx           # ← YOU: Enhance heavily
│   │   ├── Biomechanics.tsx        # ← YOU: Enhance heavily
│   │   ├── Burnout.tsx             # ← YOU: Create from scratch
│   │   └── Profile.tsx             # ✅ Already exists
│   ├── hooks/
│   │   ├── use-fitbalance.ts       # ← YOU: Add new hooks
│   │   └── use-toast.ts            # ✅ Already exists
│   ├── config/
│   │   └── api.ts                  # ← YOU: Create
│   ├── test/
│   │   ├── setup.ts                # ← YOU: Create
│   │   ├── Nutrition.test.tsx      # ← YOU: Create
│   │   └── Burnout.test.tsx        # ← YOU: Create
│   ├── App.tsx                     # ← YOU: Minor updates
│   └── main.tsx                    # ✅ Already exists
├── package.json                    # ← YOU: Add dependencies
├── vitest.config.ts                # ← YOU: Create
├── .env.local                      # ← YOU: Create
└── README.md                       # ← YOU: Update

```

---

## 🚀 TASK 1: Set Up Development Environment
**Time:** 2 hours

### Step 1.1: Install Dependencies

```powershell
cd c:\Users\divya\Desktop\projects\FitBalance\frontend

# Install existing dependencies
npm install

# Install additional charting libraries
npm install recharts @nivo/heatmap

# Install testing dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event vitest @vitest/ui jsdom

# Verify installation
npm list --depth=0
```

**Expected output:** All packages installed without errors.

---

### Step 1.2: Configure API Base URL

**Create:** `frontend/src/config/api.ts`

```typescript
/**
 * API Configuration
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
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
};

export default API_BASE_URL;
```

**Create:** `frontend/.env.local`

```env
VITE_API_URL=http://localhost:8000
```

**Verify:**
```powershell
# Start dev server to test
npm run dev

# Open browser: http://localhost:8081
# Check console for errors
```

---

## 📊 TASK 2: Complete Nutrition Page
**Time:** 20 hours

The existing Nutrition page has basic camera capture. You'll add:
- Daily progress dashboard
- Meal history with timeline
- Weekly analytics chart
- Enhanced UI/UX

### Step 2.1: Update API Hooks (4 hours)

**Update:** `frontend/src/hooks/use-fitbalance.ts`

Add these new hooks at the end of the file:

```typescript
import { API_ENDPOINTS } from '@/config/api';

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
```

---

### Step 2.2: Add Daily Progress Dashboard (4 hours)

**Update:** `frontend/src/pages/Nutrition.tsx`

**First, add new state and hooks at the top of the component (after existing state):**

```typescript
import { BarChart3 } from "lucide-react"; // Add to existing imports
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'; // New import

// Add to existing state (around line 15)
const [historyDays, setHistoryDays] = useState(7);
const currentUserId = 123; // In production, get from auth context

// Add new hooks (after existing hooks)
const { data: history, isLoading: isLoadingHistory } = useNutritionHistory(currentUserId, historyDays);
const { data: stats } = useNutritionStats(currentUserId, 7);
```

**Then, add this component AFTER the "Today's Stats" section (around line 290):**

```typescript
{/* Daily Progress Dashboard */}
<Card className="glass-card mb-6 animate-slide-up">
  <CardHeader>
    <CardTitle className="flex items-center gap-2">
      <TrendingUp className="h-5 w-5 text-primary" />
      Daily Nutrition Progress
    </CardTitle>
    <CardDescription>Your nutrition goals for today</CardDescription>
  </CardHeader>
  <CardContent>
    <div className="space-y-4">
      {/* Protein Progress */}
      <div>
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium">Protein</span>
          <span className="text-sm text-muted-foreground">
            {stats?.total_protein || 0}g / 150g
          </span>
        </div>
        <Progress 
          value={((stats?.total_protein || 0) / 150) * 100} 
          className="h-3"
        />
      </div>

      {/* Calories Progress */}
      <div>
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium">Calories</span>
          <span className="text-sm text-muted-foreground">
            {stats?.total_calories || 0} / 2000
          </span>
        </div>
        <Progress 
          value={((stats?.total_calories || 0) / 2000) * 100} 
          className="h-3"
        />
      </div>

      {/* Meal Count */}
      <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
        <div className="flex items-center gap-2">
          <Camera className="h-4 w-4 text-secondary" />
          <span className="text-sm font-medium">Meals Logged Today</span>
        </div>
        <span className="text-2xl font-bold text-primary">
          {stats?.total_meals || 0}
        </span>
      </div>
    </div>
  </CardContent>
</Card>
```

---

### Step 2.3: Add Meal History Section (6 hours)

**Add this AFTER the recommendations section (around line 570):**

```typescript
{/* Meal History */}
<Card className="glass-card mt-6 animate-slide-up">
  <CardHeader>
    <CardTitle className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-secondary" />
        Recent Meals
      </div>
      <Button 
        variant="outline" 
        size="sm" 
        onClick={() => setHistoryDays(historyDays === 7 ? 30 : 7)}
      >
        {historyDays === 7 ? 'Last 7 Days' : 'Last 30 Days'}
      </Button>
    </CardTitle>
  </CardHeader>
  <CardContent>
    {isLoadingHistory ? (
      <div className="text-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
        <p className="text-sm text-muted-foreground mt-2">Loading history...</p>
      </div>
    ) : history?.meals && history.meals.length > 0 ? (
      <div className="space-y-3">
        {history.meals.map((meal: any, index: number) => (
          <div key={index} className="p-4 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
            <div className="flex justify-between items-start mb-2">
              <div>
                <p className="font-medium">
                  {new Date(meal.timestamp).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric'
                  })}
                </p>
                <p className="text-xs text-muted-foreground">
                  {new Date(meal.timestamp).toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold text-primary">
                  {Math.round(meal.total_protein)}g protein
                </p>
                <p className="text-xs text-muted-foreground">
                  {Math.round(meal.total_calories)} cal
                </p>
              </div>
            </div>
            
            {/* Detected Foods */}
            {meal.detected_foods && Object.keys(meal.detected_foods).length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {Object.keys(meal.detected_foods).map((foodName, i) => (
                  <span 
                    key={i}
                    className="text-xs px-2 py-1 bg-secondary/20 text-secondary rounded-full"
                  >
                    {foodName.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    ) : (
      <div className="text-center py-8 text-muted-foreground">
        <Camera className="h-12 w-12 mx-auto mb-3 opacity-50" />
        <p>No meals logged yet</p>
        <p className="text-sm">Start by capturing your first meal!</p>
      </div>
    )}
  </CardContent>
</Card>
```

---

### Step 2.4: Add Weekly Analytics Chart (6 hours)

**Add this helper function at the TOP of the component (before the return statement):**

```typescript
const processWeeklyData = (meals: any[]) => {
  // Group meals by date
  const dailyData: { [key: string]: { protein: number; calories: number; count: number } } = {};
  
  meals.forEach(meal => {
    const date = new Date(meal.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    
    if (!dailyData[date]) {
      dailyData[date] = { protein: 0, calories: 0, count: 0 };
    }
    
    dailyData[date].protein += meal.total_protein || 0;
    dailyData[date].calories += meal.total_calories || 0;
    dailyData[date].count += 1;
  });
  
  // Convert to array and sort by date
  return Object.entries(dailyData)
    .map(([date, data]) => ({
      date,
      protein: Math.round(data.protein),
      calories: Math.round(data.calories),
    }))
    .slice(-7); // Last 7 days
};
```

**Add the chart component AFTER meal history:**

```typescript
{/* Weekly Analytics Chart */}
<Card className="glass-card mt-6 animate-slide-up">
  <CardHeader>
    <CardTitle className="flex items-center gap-2">
      <BarChart3 className="h-5 w-5 text-primary" />
      Weekly Nutrition Trends
    </CardTitle>
    <CardDescription>Your protein and calorie intake over time</CardDescription>
  </CardHeader>
  <CardContent>
    {history?.meals && history.meals.length > 0 ? (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={processWeeklyData(history.meals)}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis 
            dataKey="date" 
            stroke="currentColor"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="currentColor"
            style={{ fontSize: '12px' }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'hsl(var(--background))', 
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px'
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="protein" 
            stroke="hsl(var(--primary))" 
            strokeWidth={2}
            name="Protein (g)"
          />
          <Line 
            type="monotone" 
            dataKey="calories" 
            stroke="hsl(var(--secondary))" 
            strokeWidth={2}
            name="Calories"
          />
        </LineChart>
      </ResponsiveContainer>
    ) : (
      <div className="text-center py-12 text-muted-foreground">
        <p>Not enough data to show trends</p>
        <p className="text-sm">Log more meals to see your progress!</p>
      </div>
    )}
  </CardContent>
</Card>
```

**Test the Nutrition page:**
```powershell
npm run dev

# Navigate to http://localhost:8081/nutrition
# Verify:
# - Daily progress shows
# - Meal history loads
# - Chart displays (if data available)
```

---

## 🏋️ TASK 3: Complete Biomechanics Page
**Time:** 16 hours

### Step 3.1: Add Exercise Results Display (8 hours)

**Update:** `frontend/src/pages/Biomechanics.tsx`

**Add new imports at the top:**

```typescript
import { CheckCircle, Lightbulb } from "lucide-react"; // Add to existing imports
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
```

**Find where the analysis results would display (after the analysis is complete) and add:**

```typescript
{/* Form Score Card - After analysis completes */}
{biomechanicsData && (
  <>
    <Card className="glass-card mt-6 animate-slide-up">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          Form Analysis Results
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Overall Form Score */}
        <div className="text-center mb-6">
          <div className="relative inline-flex items-center justify-center w-32 h-32">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                className="text-muted"
              />
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                strokeDasharray={`${2 * Math.PI * 56}`}
                strokeDashoffset={`${2 * Math.PI * 56 * (1 - biomechanicsData.form_score / 100)}`}
                className={`
                  ${biomechanicsData.form_score >= 80 ? 'text-green-500' : ''}
                  ${biomechanicsData.form_score >= 60 && biomechanicsData.form_score < 80 ? 'text-yellow-500' : ''}
                  ${biomechanicsData.form_score < 60 ? 'text-red-500' : ''}
                `}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-4xl font-bold">
                {Math.round(biomechanicsData.form_score)}
              </span>
              <span className="text-xs text-muted-foreground">Form Score</span>
            </div>
          </div>
          
          <p className="mt-4 text-sm text-muted-foreground">
            {biomechanicsData.form_score >= 80 && 'Excellent form! Keep it up.'}
            {biomechanicsData.form_score >= 60 && biomechanicsData.form_score < 80 && 'Good form with room for improvement.'}
            {biomechanicsData.form_score < 60 && 'Form needs work. Review recommendations below.'}
          </p>
        </div>

        {/* Joint Angles */}
        <div className="space-y-3">
          <h4 className="font-semibold text-sm">Joint Angles</h4>
          {Object.entries(biomechanicsData.joint_angles).map(([joint, angle]) => (
            <div key={joint} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="capitalize">{joint.replace('_', ' ')}</span>
                <span className="font-medium">{Math.round(angle as number)}°</span>
              </div>
              <Progress 
                value={(angle as number / 180) * 100} 
                className="h-2"
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>

    {/* Risk Factors */}
    {biomechanicsData.risk_factors && biomechanicsData.risk_factors.length > 0 && (
      <Card className="glass-card mt-6 animate-slide-up border-orange-500/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-orange-600">
            <AlertCircle className="h-5 w-5" />
            Risk Factors Detected
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {biomechanicsData.risk_factors.map((risk: string, index: number) => (
              <Alert key={index} variant="destructive" className="bg-orange-50 border-orange-200">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {risk.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </AlertDescription>
              </Alert>
            ))}
          </div>
        </CardContent>
      </Card>
    )}

    {/* Recommendations */}
    {biomechanicsData.recommendations && biomechanicsData.recommendations.length > 0 && (
      <Card className="glass-card mt-6 animate-slide-up">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            Personalized Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {biomechanicsData.recommendations.map((rec: string, index: number) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-muted/30 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm">{rec}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )}
  </>
)}
```

---

### Step 3.2: Add Torque Heatmap Visualization (8 hours)

**Create new component:** `frontend/src/components/TorqueHeatmap.tsx`

```typescript
import { ResponsiveHeatMap } from '@nivo/heatmap';

interface TorqueHeatmapProps {
  data: { [key: string]: number[][] };
  jointName: string;
}

export function TorqueHeatmap({ data, jointName }: TorqueHeatmapProps) {
  // Transform data for nivo heatmap
  const heatmapKey = `${jointName}_torque`;
  const rawData = data[heatmapKey];
  
  if (!rawData || !Array.isArray(rawData)) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No heatmap data available for {jointName}
      </div>
    );
  }

  const transformedData = rawData.map((row, rowIndex) => ({
    id: `Row ${rowIndex}`,
    data: Array.isArray(row) 
      ? row.map((value, colIndex) => ({
          x: `Col ${colIndex}`,
          y: value
        }))
      : []
  }));

  return (
    <div className="h-64">
      <ResponsiveHeatMap
        data={transformedData}
        margin={{ top: 10, right: 10, bottom: 40, left: 40 }}
        valueFormat=">-.2f"
        axisTop={null}
        axisRight={null}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: 'Time',
          legendPosition: 'middle',
          legendOffset: 32
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: 'Torque',
          legendPosition: 'middle',
          legendOffset: -30
        }}
        colors={{
          type: 'diverging',
          scheme: 'red_yellow_blue',
          divergeAt: 0.5
        }}
        emptyColor="#555555"
        legends={[
          {
            anchor: 'bottom',
            translateX: 0,
            translateY: 30,
            length: 400,
            thickness: 8,
            direction: 'row',
            tickPosition: 'after',
            tickSize: 3,
            tickSpacing: 4,
            tickOverlap: false,
            tickFormat: '>-.2s',
            title: 'Torque (Nm) →',
            titleAlign: 'start',
            titleOffset: 4
          }
        ]}
      />
    </div>
  );
}
```

**Add to Biomechanics page (after recommendations):**

```typescript
import { TorqueHeatmap } from '@/components/TorqueHeatmap';

{/* Torque Heatmaps */}
{biomechanicsData?.heatmap_data && (
  <Card className="glass-card mt-6 animate-slide-up">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Activity className="h-5 w-5 text-primary" />
        Torque Heatmap Analysis
      </CardTitle>
      <CardDescription>
        Visual representation of joint stress during movement
      </CardDescription>
    </CardHeader>
    <CardContent>
      <Tabs defaultValue="knee" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="knee">Knee</TabsTrigger>
          <TabsTrigger value="hip">Hip</TabsTrigger>
          <TabsTrigger value="ankle">Ankle</TabsTrigger>
          <TabsTrigger value="back">Back</TabsTrigger>
        </TabsList>
        
        <TabsContent value="knee" className="mt-4">
          <TorqueHeatmap data={biomechanicsData.heatmap_data} jointName="knee" />
        </TabsContent>
        <TabsContent value="hip" className="mt-4">
          <TorqueHeatmap data={biomechanicsData.heatmap_data} jointName="hip" />
        </TabsContent>
        <TabsContent value="ankle" className="mt-4">
          <TorqueHeatmap data={biomechanicsData.heatmap_data} jointName="ankle" />
        </TabsContent>
        <TabsContent value="back" className="mt-4">
          <TorqueHeatmap data={biomechanicsData.heatmap_data} jointName="back" />
        </TabsContent>
      </Tabs>
    </CardContent>
  </Card>
)}
```

---

## 🔥 TASK 4: Complete Burnout Page (CREATE FROM SCRATCH)
**Time:** 12 hours

**Create:** `frontend/src/pages/Burnout.tsx`

```typescript
import { useState } from "react";
import { Activity, TrendingDown, AlertTriangle, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useBurnout } from "@/hooks/use-fitbalance";
import { useToast } from "@/hooks/use-toast";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Burnout() {
  const [workoutFrequency, setWorkoutFrequency] = useState(4);
  const [sleepHours, setSleepHours] = useState(7.5);
  const [stressLevel, setStressLevel] = useState(5);
  const [recoveryDays, setRecoveryDays] = useState(2);
  const [performanceTrend, setPerformanceTrend] = useState("stable");
  
  const { isLoading, error, burnoutData, analyzeBurnout } = useBurnout();
  const { toast } = useToast();

  const handleAnalyze = async () => {
    try {
      await analyzeBurnout({
        user_id: "123",
        workout_frequency: workoutFrequency,
        sleep_hours: sleepHours,
        stress_level: stressLevel,
        recovery_time: recoveryDays,
        performance_trend: performanceTrend
      });

      toast({
        title: "Analysis Complete",
        description: "Your burnout risk has been calculated.",
      });
    } catch (err) {
      toast({
        title: "Analysis Failed",
        description: "Unable to analyze burnout risk. Please try again.",
        variant: "destructive",
      });
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-500';
      case 'medium': return 'text-yellow-500';
      case 'high': return 'text-orange-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'low': return <Shield className="h-8 w-8 text-green-500" />;
      case 'medium': return <AlertTriangle className="h-8 w-8 text-yellow-500" />;
      case 'high': return <AlertTriangle className="h-8 w-8 text-orange-500" />;
      case 'critical': return <AlertTriangle className="h-8 w-8 text-red-500" />;
      default: return <Activity className="h-8 w-8 text-gray-500" />;
    }
  };

  return (
    <div className="min-h-screen px-4 py-8 relative z-10">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-destructive/10 text-destructive mb-4">
            <TrendingDown className="h-4 w-4" />
            <span className="text-sm font-medium">Burnout Prevention</span>
          </div>
          <h1 className="text-4xl font-bold mb-2">Athletic Burnout Risk</h1>
          <p className="text-muted-foreground">
            Predictive analytics to prevent overtraining and optimize recovery
          </p>
        </div>

        {/* Input Form */}
        <Card className="glass-card mb-6 animate-slide-up">
          <CardHeader>
            <CardTitle>Enter Your Metrics</CardTitle>
            <CardDescription>
              Provide current training and recovery data for analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Workout Frequency */}
            <div className="space-y-2">
              <Label>Workouts Per Week: {workoutFrequency}</Label>
              <Slider
                value={[workoutFrequency]}
                onValueChange={(value) => setWorkoutFrequency(value[0])}
                min={1}
                max={7}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                How many times do you train per week?
              </p>
            </div>

            {/* Sleep Hours */}
            <div className="space-y-2">
              <Label>Average Sleep Hours: {sleepHours.toFixed(1)}</Label>
              <Slider
                value={[sleepHours]}
                onValueChange={(value) => setSleepHours(value[0])}
                min={4}
                max={10}
                step={0.5}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Average hours of sleep per night
              </p>
            </div>

            {/* Stress Level */}
            <div className="space-y-2">
              <Label>Stress Level: {stressLevel}/10</Label>
              <Slider
                value={[stressLevel]}
                onValueChange={(value) => setStressLevel(value[0])}
                min={1}
                max={10}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Overall stress level (1=very low, 10=very high)
              </p>
            </div>

            {/* Recovery Days */}
            <div className="space-y-2">
              <Label>Recovery Days Between Intense Sessions: {recoveryDays}</Label>
              <Slider
                value={[recoveryDays]}
                onValueChange={(value) => setRecoveryDays(value[0])}
                min={0}
                max={7}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Days of rest between high-intensity workouts
              </p>
            </div>

            {/* Performance Trend */}
            <div className="space-y-2">
              <Label>Performance Trend</Label>
              <Select value={performanceTrend} onValueChange={setPerformanceTrend}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="improving">Improving</SelectItem>
                  <SelectItem value="stable">Stable</SelectItem>
                  <SelectItem value="declining">Declining</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                How has your performance been trending lately?
              </p>
            </div>

            {/* Analyze Button */}
            <Button 
              className="w-full gradient-burnout text-white font-medium"
              onClick={handleAnalyze}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="mr-2 h-4 w-4" />
                  Analyze Burnout Risk
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Results */}
        {burnoutData && (
          <>
            {/* Risk Score Card */}
            <div className="grid md:grid-cols-3 gap-6 mb-6 animate-slide-up">
              <Card className="glass-card col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {getRiskIcon(burnoutData.risk_level)}
                    Burnout Risk Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Risk Level */}
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Risk Level</span>
                        <span className={`text-2xl font-bold ${getRiskColor(burnoutData.risk_level)}`}>
                          {burnoutData.risk_level.toUpperCase()}
                        </span>
                      </div>
                      <Progress 
                        value={burnoutData.risk_score} 
                        className="h-3"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Risk Score: {Math.round(burnoutData.risk_score)}/100
                      </p>
                    </div>

                    {/* Time to Burnout */}
                    {burnoutData.time_to_burnout && (
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Estimated Time to Burnout</p>
                        <p className="text-3xl font-bold">
                          {Math.round(burnoutData.time_to_burnout / 30)} months
                        </p>
                        <p className="text-xs text-muted-foreground">
                          ({Math.round(burnoutData.time_to_burnout)} days)
                        </p>
                      </div>
                    )}

                    {/* Survival Probability */}
                    <div className="p-4 bg-muted/30 rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">1-Year Survival Probability</p>
                      <p className="text-3xl font-bold text-green-500">
                        {Math.round(burnoutData.survival_probability * 100)}%
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Chance of avoiding burnout in the next year
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Risk Factors */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-lg">Risk Factors</CardTitle>
                </CardHeader>
                <CardContent>
                  {burnoutData.risk_factors && burnoutData.risk_factors.length > 0 ? (
                    <div className="space-y-2">
                      {burnoutData.risk_factors.map((factor: string, index: number) => (
                        <div key={index} className="flex items-start gap-2 text-sm">
                          <AlertTriangle className="h-4 w-4 text-orange-500 flex-shrink-0 mt-0.5" />
                          <span className="text-xs">
                            {factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-green-600">
                      No significant risk factors detected
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Survival Curve */}
            {burnoutData.survival_curve_data && (
              <Card className="glass-card mb-6 animate-slide-up">
                <CardHeader>
                  <CardTitle>Survival Curve</CardTitle>
                  <CardDescription>
                    Probability of avoiding burnout over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={burnoutData.survival_curve_data.times.map((time: number, index: number) => ({
                      days: time,
                      months: Math.round(time / 30),
                      probability: burnoutData.survival_curve_data.survival_probabilities[index] * 100
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis 
                        dataKey="months"
                        label={{ value: 'Months', position: 'insideBottom', offset: -5 }}
                        stroke="currentColor"
                        style={{ fontSize: '12px' }}
                      />
                      <YAxis 
                        label={{ value: 'Survival Probability (%)', angle: -90, position: 'insideLeft' }}
                        domain={[0, 100]}
                        stroke="currentColor"
                        style={{ fontSize: '12px' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'hsl(var(--background))', 
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px'
                        }}
                        formatter={(value: number) => [`${value.toFixed(1)}%`, 'Survival Probability']}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="probability" 
                        stroke="hsl(var(--destructive))" 
                        strokeWidth={3}
                        name="No Burnout Probability"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            {burnoutData.recommendations && burnoutData.recommendations.length > 0 && (
              <Card className="glass-card animate-slide-up">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="h-5 w-5 text-green-500" />
                    Personalized Recommendations
                  </CardTitle>
                  <CardDescription>
                    Actions to reduce burnout risk and optimize recovery
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {burnoutData.recommendations.map((rec: string, index: number) => (
                      <div key={index} className="flex items-start gap-3 p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
                        <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-green-600 font-bold text-sm">{index + 1}</span>
                        </div>
                        <p className="text-sm">{rec}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  );
}
```

**Add the burnout hook to:** `frontend/src/hooks/use-fitbalance.ts`

```typescript
export function useBurnout() {
  const [burnoutData, setBurnoutData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      });

      if (!response.ok) throw new Error('Burnout analysis failed');

      const result = await response.json();
      setBurnoutData(result);
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return { isLoading, error, burnoutData, analyzeBurnout };
}
```

**Update routing in:** `frontend/src/App.tsx`

```typescript
import Burnout from "./pages/Burnout"; // Add import

// Add route
<Route path="/burnout" element={<Burnout />} />
```

---

## 📱 TASK 5: Mobile Responsiveness & UX Polish
**Time:** 8 hours

### Step 5.1: Add Mobile Navigation (4 hours)

**Create:** `frontend/src/components/MobileNav.tsx`

```typescript
import { useState } from 'react';
import { Menu, X, Home, Activity, Apple, TrendingDown, User } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';

export function MobileNav() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { path: '/', icon: Home, label: 'Home' },
    { path: '/biomechanics', icon: Activity, label: 'Biomechanics' },
    { path: '/nutrition', icon: Apple, label: 'Nutrition' },
    { path: '/burnout', icon: TrendingDown, label: 'Burnout' },
    { path: '/profile', icon: User, label: 'Profile' },
  ];

  return (
    <>
      {/* Mobile Menu Button */}
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden fixed top-4 right-4 z-50"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X /> : <Menu />}
      </Button>

      {/* Mobile Menu Overlay */}
      {isOpen && (
        <div className="fixed inset-0 bg-background/95 backdrop-blur-md z-40 md:hidden">
          <nav className="flex flex-col items-center justify-center h-full space-y-8">
            {navItems.map(({ path, icon: Icon, label }) => (
              <Link
                key={path}
                to={path}
                onClick={() => setIsOpen(false)}
                className={`flex items-center gap-3 text-2xl font-medium ${
                  location.pathname === path
                    ? 'text-primary'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <Icon className="h-6 w-6" />
                {label}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </>
  );
}
```

**Add to:** `frontend/src/App.tsx`

```typescript
import { MobileNav } from './components/MobileNav'; // Add import

// Inside BrowserRouter, add:
<MobileNav />
```

---

### Step 5.2: Create Loading Spinner Component (2 hours)

**Create:** `frontend/src/components/LoadingSpinner.tsx`

```typescript
import { Activity } from "lucide-react";

export function LoadingSpinner({ message = "Loading..." }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="relative">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-primary"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <Activity className="h-6 w-6 text-primary animate-pulse" />
        </div>
      </div>
      <p className="mt-4 text-sm text-muted-foreground animate-pulse">{message}</p>
    </div>
  );
}
```

**Use it in pages:** Replace `<div className="animate-spin...">` with `<LoadingSpinner />`

---

### Step 5.3: Standardize Toast Notifications (2 hours)

Update all toast patterns across pages:

```typescript
// Success pattern
toast({
  title: "✅ Success",
  description: "Action completed successfully",
  duration: 3000,
});

// Error pattern
toast({
  title: "❌ Error",
  description: "Something went wrong. Please try again.",
  variant: "destructive",
  duration: 5000,
});

// Info pattern
toast({
  title: "ℹ️ Info",
  description: "Helpful information for the user",
  duration: 4000,
});
```

---

## 🧪 TASK 6: Testing Setup
**Time:** 4 hours

### Step 6.1: Configure Vitest

**Create:** `frontend/vitest.config.ts`

```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

**Create:** `frontend/src/test/setup.ts`

```typescript
import '@testing-library/jest-dom';
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';

expect.extend(matchers);

afterEach(() => {
  cleanup();
});
```

**Update:** `frontend/package.json`

Add scripts:
```json
{
  "scripts": {
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage"
  }
}
```

---

### Step 6.2: Create Component Tests

**Create:** `frontend/src/test/Nutrition.test.tsx`

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import Nutrition from '../pages/Nutrition';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    <BrowserRouter>
      {children}
    </BrowserRouter>
  </QueryClientProvider>
);

describe('Nutrition Page', () => {
  it('renders nutrition page title', () => {
    render(<Nutrition />, { wrapper });
    expect(screen.getByText('Meal Analysis')).toBeInTheDocument();
  });

  it('displays camera capture button', () => {
    render(<Nutrition />, { wrapper });
    expect(screen.getByText(/Capture Meal/i)).toBeInTheDocument();
  });

  it('displays upload photo button', () => {
    render(<Nutrition />, { wrapper });
    expect(screen.getByText(/Upload Photo/i)).toBeInTheDocument();
  });
});
```

**Create:** `frontend/src/test/Burnout.test.tsx`

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import Burnout from '../pages/Burnout';

const queryClient = new QueryClient();

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    <BrowserRouter>
      {children}
    </BrowserRouter>
  </QueryClientProvider>
);

describe('Burnout Page', () => {
  it('renders burnout page title', () => {
    render(<Burnout />, { wrapper });
    expect(screen.getByText('Athletic Burnout Risk')).toBeInTheDocument();
  });

  it('displays workout frequency slider', () => {
    render(<Burnout />, { wrapper });
    expect(screen.getByText(/Workouts Per Week/i)).toBeInTheDocument();
  });

  it('displays analyze button', () => {
    render(<Burnout />, { wrapper });
    expect(screen.getByText(/Analyze Burnout Risk/i)).toBeInTheDocument();
  });
});
```

**Run tests:**
```powershell
npm test
```

---

## 📚 TASK 7: Documentation
**Time:** 4 hours

**Create:** `frontend/FRONTEND_GUIDE.md`

```markdown
# FitBalance Frontend Guide

## Tech Stack
- **React 18** with TypeScript
- **Vite** for build tooling
- **TailwindCSS** for styling
- **shadcn/ui** for components
- **React Router** for navigation
- **React Query** for API state management
- **Recharts** for data visualization
- **Nivo** for heatmaps

## Project Structure
```
frontend/src/
├── components/     # Reusable UI components
│   ├── ui/        # shadcn/ui components
│   ├── Layout.tsx
│   ├── MobileNav.tsx
│   ├── LoadingSpinner.tsx
│   └── TorqueHeatmap.tsx
├── pages/         # Page components
│   ├── Home.tsx
│   ├── Nutrition.tsx
│   ├── Biomechanics.tsx
│   ├── Burnout.tsx
│   └── Profile.tsx
├── hooks/         # Custom React hooks
│   ├── use-fitbalance.ts
│   └── use-toast.ts
├── lib/          # Utilities
│   └── utils.ts
├── config/       # Configuration
│   └── api.ts
├── test/         # Tests
├── App.tsx       # Root component
└── main.tsx      # Entry point
```

## Development

### Start Dev Server
```bash
npm run dev
```
Open: http://localhost:8081

### Build for Production
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Run Tests
```bash
npm test           # Run tests
npm run test:ui    # Run with UI
npm run test:coverage  # Coverage report
```

## API Integration

All API calls go through `src/hooks/use-fitbalance.ts` using React Query.

### Example:
```typescript
const { analysis, isLoading, analyzeMeal } = useNutrition();

await analyzeMeal(file, userId, restrictions);
console.log(analysis.total_protein);
```

## Environment Variables

Create `.env.local`:
```
VITE_API_URL=http://localhost:8000
```

## Component Guidelines

1. Use TypeScript for all components
2. Follow shadcn/ui patterns
3. Use Tailwind utility classes
4. Implement loading and error states
5. Ensure mobile responsiveness
6. Add accessibility attributes

## Mobile Responsiveness

- Breakpoints: `sm` (640px), `md` (768px), `lg` (1024px), `xl` (1280px)
- Test on: iPhone SE, iPhone 12, iPad, Desktop
- Use `md:` prefix for desktop-specific styles

## Pages Overview

### Nutrition (`/nutrition`)
- Camera capture for meal photos
- Daily progress dashboard
- Meal history timeline
- Weekly analytics chart
- Detected foods display
- Recommendations

### Biomechanics (`/biomechanics`)
- Video upload for form analysis
- Exercise selection
- Form score visualization
- Joint angles display
- Torque heatmaps
- Risk factors and recommendations

### Burnout (`/burnout`)
- Metric input sliders
- Risk assessment
- Survival curve chart
- Risk factors list
- Personalized recommendations

## Troubleshooting

### API calls failing
- Check `.env.local` has correct `VITE_API_URL`
- Verify backend is running on port 8000
- Check browser console for CORS errors

### Build errors
- Clear node_modules: `rm -rf node_modules && npm install`
- Check TypeScript errors: `npx tsc --noEmit`

### Styling broken
- Restart dev server
- Check Tailwind config
- Verify PostCSS setup

## Deployment

Can be deployed to:
- **Vercel** (recommended) - `vercel deploy`
- **Netlify** - Connect GitHub repo
- **AWS S3 + CloudFront**
- **Azure Static Web Apps**

## Performance Tips

1. Use lazy loading for routes
2. Optimize images (WebP format)
3. Enable code splitting
4. Use React.memo for expensive components
5. Implement virtual scrolling for long lists
```

---

## ✅ Final Checklist

### Nutrition Page
- [ ] Camera opens successfully
- [ ] Photo upload works
- [ ] Analysis returns results
- [ ] Daily progress displays
- [ ] Meal history loads
- [ ] Weekly chart shows
- [ ] Stats update correctly
- [ ] Mobile responsive

### Biomechanics Page
- [ ] Video upload works
- [ ] Exercise selection functional
- [ ] Form score displays
- [ ] Joint angles shown
- [ ] Heatmap renders
- [ ] Recommendations appear
- [ ] Mobile responsive

### Burnout Page
- [ ] All sliders work
- [ ] Analysis submits
- [ ] Results display correctly
- [ ] Survival curve renders
- [ ] Recommendations show
- [ ] Mobile responsive

### General
- [ ] Navigation works
- [ ] All pages load
- [ ] API errors handled
- [ ] Loading states show
- [ ] Toast notifications work
- [ ] Mobile menu functions
- [ ] Tests pass

---

## 📊 Success Metrics

Your work is complete when:
1. ✅ All 3 pages fully functional
2. ✅ Mobile responsive (< 768px tested)
3. ✅ All API integrations working
4. ✅ Tests pass 100%
5. ✅ Documentation complete

---

## 🆘 Getting Help

If stuck:
1. **Check console:** Browser DevTools
2. **Test API:** Use Postman/curl to verify backend
3. **Review docs:** Read error messages
4. **Ask team:** Coordinate with Person 2 (Backend)

---

## 📅 Timeline

**Week 1:**
- Days 1-2: Setup + Nutrition page
- Days 3-5: Biomechanics page

**Week 2:**
- Days 1-3: Burnout page
- Days 4-5: Mobile responsiveness

**Week 3 (Buffer):**
- Testing and polish
- Documentation
- Bug fixes

---

## 🎉 Completion

Once done:
1. Commit code to git
2. Update README
3. Share with team
4. Demo to Person 4 (DevOps)

**Your contribution: 20-25% of total project** 🏆

Good luck! 🚀
