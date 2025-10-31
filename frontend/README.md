# FitBalance Frontend

[![React 18](https://img.shields.io/badge/React-18-blue)](https://react.dev/) [![Vite](https://img.shields.io/badge/Vite-frontend-brightgreen)](https://vitejs.dev/) [![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue)](https://www.typescriptlang.org/)

**Modern nutrition, biomechanics, and burnout dashboard with camera, video, analytics, and full API connectivity.**

---

## 🚀 Quick Start

1. **Install dependencies (from `frontend/` directory):**
   ```bash
   npm install
   ```

2. **Configure the Backend API URL:**
   - Copy `.env.local.example` to `.env.local` (or create manually).
   - Ensure `VITE_API_URL` matches your backend (default: `http://localhost:8000`).

3. **Run the Dev Server:**
   ```bash
   npm run dev
   # Open http://localhost:8081
   ```

4. **Run Tests:**
   ```bash
   npm test
   # or with UI coverage
   npm run test:ui
   npm run test:coverage
   ```

5. **Build for Production:**
   ```bash
   npm run build
   npm run preview
   # Open the preview URL in your browser
   ```

---

## 🖥️ Pages & Functionality

- **/nutrition**  
  Camera-based meal capture, daily stats dashboard, history chart, and AI-driven nutritional analysis.
- **/biomechanics**  
  Video upload/capture, form analysis, torque heatmaps, risk detection, and smart coaching recommendations.
- **/burnout**  
  Burnout risk calculator, survival curves, recommendations, and actionable metrics.
- **Mobile Responsive**  
  Unified design with MobileNav and adaptive UI for iPhone, Android, and desktop.

---

## 📚 Documentation
- See [`FRONTEND_GUIDE.md`](./FRONTEND_GUIDE.md) for: 
  - Full tech stack outline
  - Directory/project structure
  - Component design standards
  - Pages feature walkthrough and checklist
  - Troubleshooting, testing, and deployment

---

## 🧪 Testing
- Unit and integration tests: [Vitest](https://vitest.dev/)
- UI coverage and debugging: `npm run test:ui`
- Snapshot, hooks, and async query tests built in `/src/test/`

---

## 🤝 Contributing
- PRs welcome. See [contributing guidelines](../docs/CONTRIBUTING.md)

---

## 🛠️ Tech Stack Quick Reference
- React 18 + TypeScript (SPA)
- Vite build tool
- TailwindCSS utility styling
- shadcn/ui + Lucide icons
- React Router
- React Query (TanStack)
- Recharts + @nivo/heatmap (analytics/visualization)
- Vitest + Testing Library (testing)

---

## 🟢 Status
**All Person 3 (Frontend) deliverables are complete & tested.**

API backend connectivity required for full analytics and chart data. If you encounter issues, verify backend API is live and `.env.local` is configured correctly.

---

## © 2025 FitBalance | Built for project demo and production launch.
