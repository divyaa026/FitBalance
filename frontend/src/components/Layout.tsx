import { Outlet } from "react-router-dom";
import { BottomNav } from "./BottomNav";
import { Header } from "./Header";

export const Layout = () => {
  return (
    <div className="min-h-screen pb-20 relative">
      <Header />
      <div className="relative z-10 pt-16">
        <Outlet />
      </div>
      <BottomNav />
    </div>
  );
};
