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
