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
