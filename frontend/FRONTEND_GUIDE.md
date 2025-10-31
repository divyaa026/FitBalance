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

## Final Checklist

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

## Success Metrics

Your work is complete when:
1. ✅ All 3 pages fully functional
2. ✅ Mobile responsive (< 768px tested)
3. ✅ All API integrations working
4. ✅ Tests pass 100%
5. ✅ Documentation complete

## Getting Help
If stuck:
1. **Check console:** Browser DevTools
2. **Test API:** Use Postman/curl to verify backend
3. **Review docs:** Read error messages
4. **Ask team:** Coordinate with Person 2 (Backend)

## Timeline

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

## 🎉 Completion
Once done:
1. Commit code to git
2. Update README
3. Share with team
4. Demo to Person 4 (DevOps)

**Your contribution: 20-25% of total project** 🏆

Good luck! 🚀
