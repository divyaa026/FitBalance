import { Link } from "react-router-dom";
import { Activity, Apple, BarChart3, ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const features = [
  {
    icon: Activity,
    title: "Biomechanics Coaching",
    description: "Real-time movement analysis with joint angle detection and personalized form correction",
    gradient: "gradient-biomechanics",
    link: "/biomechanics",
  },
  {
    icon: Apple,
    title: "Intelligent Nutrition",
    description: "AI-powered food recognition with protein optimization and meal quality scoring",
    gradient: "gradient-nutrition",
    link: "/nutrition",
  },
  {
    icon: BarChart3,
    title: "Burnout Prevention",
    description: "Predictive analytics for athletic performance and personalized recovery management",
    gradient: "gradient-burnout",
    link: "/burnout",
  },
];

export default function Home() {
  return (
    <div className="min-h-screen relative z-10 pt-16">
      {/* Hero Section */}
            {/* Hero Section */}
      <section className="relative px-4 py-8 md:py-12 h-screen flex items-center justify-center">
        <div className="max-w-7xl mx-auto relative z-10 w-full -mt-16">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Side - Text Content */}
            <div className="text-left animate-fade-in space-y-6 flex flex-col justify-center">
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold leading-tight">
                <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-600 bg-clip-text text-transparent">
                  Transform Your
                </span>
                <br />
                <span className="bg-gradient-to-r from-white via-purple-200 to-white bg-clip-text text-transparent">
                  Fitness Journey
                </span>
              </h1>
              
              <p className="text-lg md:text-xl text-purple-200 leading-relaxed max-w-xl">
                Unlock your potential with AI-powered biomechanics analysis, 
                personalized nutrition tracking, and burnout prevention. 
                Your complete wellness companion.
              </p>
              
              <div className="flex flex-wrap items-center gap-4 pt-2">
                <Link to="/biomechanics">
                  <Button size="lg" className="gradient-hero text-white hover:opacity-90 transition-all duration-300 px-8 py-4 text-lg font-semibold rounded-full shadow-2xl hover:shadow-purple-500/25">
                    Start Your Journey
                    <ArrowRight className="ml-2 h-6 w-6" />
                  </Button>
                </Link>
                <Link to="/profile">
                  <Button size="lg" variant="outline" className="border-2 border-purple-400 text-purple-200 hover:bg-purple-400 hover:text-white px-8 py-4 text-lg font-semibold rounded-full transition-all duration-300">
                    View Profile
                  </Button>
                </Link>
              </div>
            </div>

            {/* Right Side - Enhanced Fitness Visualization */}
            <div className="relative lg:flex justify-center items-center animate-slide-up">
              <div className="relative w-full max-w-md mx-auto">
                {/* Main phone mockup */}
                <div className="relative bg-gradient-to-br from-purple-600/30 via-pink-500/20 to-purple-700/30 rounded-[2.5rem] p-2 shadow-2xl backdrop-blur-sm border border-purple-400/20">
                  <div className="bg-gradient-to-br from-purple-900/60 to-pink-900/40 rounded-[2rem] p-6 h-96 backdrop-blur-md border border-purple-300/10">
                    
                    {/* Phone header */}
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-gradient-to-r from-purple-400 to-pink-400 rounded-lg flex items-center justify-center">
                          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M13.5.67s.74 2.65.74 4.8c0 2.06-1.35 3.73-3.41 3.73-2.07 0-3.63-1.67-3.63-3.73l.03-.36C5.21 7.51 4 10.62 4 14c0 4.42 3.58 8 8 8s8-3.58 8-8C20 8.61 17.41 3.8 13.5.67zM11.71 19c-1.78 0-3.22-1.4-3.22-3.14 0-1.62 1.05-2.76 2.81-3.12 1.77-.36 3.6-1.21 4.62-2.58.39 1.29.59 2.65.59 4.04 0 2.65-2.15 4.8-4.8 4.8z"/>
                          </svg>
                        </div>
                        <span className="text-white font-semibold text-lg">FitBalance</span>
                      </div>
                      <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-lg shadow-green-400/50"></div>
                    </div>
                    
                    {/* Stats cards */}
                    <div className="space-y-4">
                      <div className="bg-gradient-to-r from-purple-500/20 to-purple-600/20 rounded-xl p-4 border border-purple-400/20 backdrop-blur-sm">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="text-2xl font-bold text-white">98%</div>
                            <div className="text-purple-200 text-sm">Form Accuracy</div>
                          </div>
                          <div className="w-12 h-12 bg-purple-500/30 rounded-lg flex items-center justify-center">
                            <svg className="w-6 h-6 text-purple-200" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                      
                      <div className="bg-gradient-to-r from-pink-500/20 to-pink-600/20 rounded-xl p-4 border border-pink-400/20 backdrop-blur-sm">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="text-2xl font-bold text-white">2,450</div>
                            <div className="text-pink-200 text-sm">Calories Burned</div>
                          </div>
                          <div className="w-12 h-12 bg-pink-500/30 rounded-lg flex items-center justify-center">
                            <svg className="w-6 h-6 text-pink-200" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm3.5 6L12 10.5 8.5 8 12 6.5 15.5 8zM12 13.5l3.5-2.5v3L12 16.5 8.5 14v-3L12 13.5z"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                      
                      <div className="bg-gradient-to-r from-purple-400/20 to-pink-400/20 rounded-xl p-4 border border-purple-300/20 backdrop-blur-sm">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="text-2xl font-bold text-white">7.2</div>
                            <div className="text-purple-200 text-sm">Wellness Score</div>
                          </div>
                          <div className="w-12 h-12 bg-gradient-to-r from-purple-400/30 to-pink-400/30 rounded-lg flex items-center justify-center">
                            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Progress indicator */}
                    <div className="mt-6">
                      <div className="flex justify-between text-sm text-purple-200 mb-2">
                        <span>Today's Progress</span>
                        <span>87%</span>
                      </div>
                      <div className="bg-purple-800/50 rounded-full h-2 overflow-hidden">
                        <div className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-500 h-full rounded-full animate-pulse" style={{width: '87%'}}></div>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Floating badges - cleaner design */}
                <div className="absolute -top-3 -right-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 rounded-full text-sm font-semibold shadow-xl animate-bounce">
                  AI Powered
                </div>
                
                <div className="absolute -bottom-3 -left-3 bg-gradient-to-r from-pink-500 to-purple-500 text-white px-4 py-2 rounded-full text-sm font-semibold shadow-xl animate-bounce delay-300">
                  Real-time
                </div>
                
                {/* Side floating icons */}
                <div className="absolute top-1/4 -left-6 w-10 h-10 bg-purple-500/80 rounded-full flex items-center justify-center animate-pulse shadow-lg">
                  <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                  </svg>
                </div>
                
                <div className="absolute bottom-1/4 -right-6 w-10 h-10 bg-pink-500/80 rounded-full flex items-center justify-center animate-pulse delay-500 shadow-lg">
                  <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 11H7v9h2v-9zm4 0h-2v9h2v-9zm4 0h-2v9h2v-9zm2.5-8.5V5c0 1.45-1.18 2.5-2.61 2.5H16l.69 1H18v9H6v-9h1.31l.69-1H6.39C4.96 7.5 3.78 6.45 3.78 5v-2.5z"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Background floating elements - reduced */}
        <div className="absolute top-1/3 left-1/4 w-24 h-24 bg-purple-500/10 rounded-full blur-xl animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/4 w-20 h-20 bg-pink-500/10 rounded-full blur-xl animate-pulse delay-300"></div>
      </section>

      {/* Features Grid */}
      <section className="px-4 py-24 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 animate-slide-up">
            <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent">
              Three Pillars of Wellness
            </h2>
            <p className="text-purple-200 text-lg max-w-2xl mx-auto leading-relaxed">
              Comprehensive health tracking powered by machine learning and computer vision technology
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 animate-slide-up">
            {features.map((feature, index) => (
              <Link key={index} to={feature.link}>
                <Card className="glass-card hover:scale-105 transition-all duration-500 cursor-pointer h-full border-purple-500/20 bg-purple-900/20 backdrop-blur-xl">
                  <CardHeader>
                    <div className={`w-16 h-16 rounded-3xl ${feature.gradient} flex items-center justify-center mb-6 shadow-lg`}>
                      <feature.icon className="h-8 w-8 text-white" />
                    </div>
                    <CardTitle className="text-2xl text-purple-100 mb-4">{feature.title}</CardTitle>
                    <CardDescription className="text-lg text-purple-200 leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center text-purple-300 font-semibold text-lg hover:text-purple-200 transition-colors">
                      Explore
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="px-4 py-20 bg-purple-900/30 backdrop-blur-sm border-y border-purple-500/20">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { value: "Real-time", label: "AI Analysis" },
              { value: "100K+", label: "Food Database" },
              { value: "AI-Powered", label: "Insights" },
              { value: "24/7", label: "Tracking" },
            ].map((stat, index) => (
              <div key={index} className="text-center group hover:scale-105 transition-transform duration-300">
                <div className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-3 group-hover:from-purple-300 group-hover:to-pink-300 transition-all duration-300">
                  {stat.value}
                </div>
                <div className="text-lg text-purple-200 font-medium">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
