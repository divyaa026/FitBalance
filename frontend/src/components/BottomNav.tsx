import { NavLink } from "react-router-dom";
import { Activity, Apple, BarChart3, User } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  {
    to: "/biomechanics",
    icon: Activity,
    label: "Biomechanics",
    gradient: "gradient-biomechanics",
  },
  {
    to: "/nutrition",
    icon: Apple,
    label: "Nutrition",
    gradient: "gradient-nutrition",
  },
  {
    to: "/burnout",
    icon: BarChart3,
    label: "Burnout",
    gradient: "gradient-burnout",
  },
  {
    to: "/profile",
    icon: User,
    label: "Profile",
    gradient: "gradient-profile",
  },
];

export const BottomNav = () => {
  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-card/90 backdrop-blur-xl border-t border-white/20 z-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-around h-16">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "flex flex-col items-center justify-center gap-1 px-4 py-2 rounded-lg transition-all",
                  isActive ? "text-primary" : "text-muted-foreground hover:text-foreground"
                )
              }
            >
              {({ isActive }) => (
                <>
                  <div
                    className={cn(
                      "p-2 rounded-lg transition-all",
                      isActive ? item.gradient : "bg-transparent"
                    )}
                  >
                    <item.icon
                      className={cn(
                        "h-5 w-5 transition-colors",
                        isActive ? "text-white" : "text-current"
                      )}
                    />
                  </div>
                  <span className="text-xs font-medium">{item.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
};
