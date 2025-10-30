import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
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
    beforeEach(() => {
        render(<Burnout />, { wrapper });
    });

    it('renders burnout page title', () => {
        expect(screen.getByText('Athletic Burnout Risk')).toBeInTheDocument();
    });

    it('displays workout frequency slider', () => {
        expect(screen.getByText(/Workouts Per Week/i)).toBeInTheDocument();
    });

    it('displays analyze button', () => {
        expect(screen.getByText(/Analyze Burnout Risk/i)).toBeInTheDocument();
    });

    it('shows survival curve section', () => {
        expect(screen.getByText('Survival Curve')).toBeInTheDocument();
    });

    it('shows recommendations section', () => {
        expect(screen.getByText('Recovery Recommendations')).toBeInTheDocument();
    });
});