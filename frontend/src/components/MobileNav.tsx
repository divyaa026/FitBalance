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
            <Button
                variant="ghost"
                size="icon"
                className="md:hidden fixed top-4 right-4 z-50"
                onClick={() => setIsOpen(!isOpen)}
            >
                {isOpen ? <X /> : <Menu />}
            </Button>

            {isOpen && (
                <div className="fixed inset-0 bg-background/95 backdrop-blur-md z-40 md:hidden">
                    <nav className="flex flex-col items-center justify-center h-full space-y-8">
                        {navItems.map(({ path, icon: Icon, label }) => (
                            <Link
                                key={path}
                                to={path}
                                onClick={() => setIsOpen(false)}
                                className={`flex items-center gap-3 text-2xl font-medium ${location.pathname === path
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

export default MobileNav;
