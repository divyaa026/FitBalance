import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
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
    beforeEach(() => {
        render(<Nutrition />, { wrapper });
    });

    it('renders nutrition page title', () => {
        expect(screen.getByText('Meal Analysis')).toBeInTheDocument();
    });

    it('displays camera capture button', () => {
        expect(screen.getByText(/Capture Meal/i)).toBeInTheDocument();
    });

    it('displays upload photo button', () => {
        expect(screen.getByText(/Upload Photo/i)).toBeInTheDocument();
    });

    it('shows daily progress dashboard', () => {
        expect(screen.getByText('Daily Nutrition Progress')).toBeInTheDocument();
    });

    it('shows meal history section', () => {
        expect(screen.getByText('Recent Meals')).toBeInTheDocument();
    });

    it('shows weekly trends chart section', () => {
        expect(screen.getByText('Weekly Nutrition Trends')).toBeInTheDocument();
    });
});