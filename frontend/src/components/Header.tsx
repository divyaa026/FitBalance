import { Link } from "react-router-dom";
import { Home, Dumbbell } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 bg-card/90 backdrop-blur-xl border-b border-white/20 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-lg hover:opacity-80 transition-opacity">
          <div className="h-8 w-8 rounded-lg gradient-hero flex items-center justify-center">
            <Dumbbell className="h-5 w-5 text-white" />
          </div>
          <span className="bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
            FitBalance
          </span>
        </Link>
        
        <Link to="/">
          <Button variant="ghost" size="sm" className="gap-2">
            <Home className="h-4 w-4" />
            Home
          </Button>
        </Link>
      </div>
    </header>
  );
};
