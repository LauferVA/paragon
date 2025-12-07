# Web Application Domain Template

This template extends the generic templates with web application-specific questions.

## Frontend Architecture
**Questions to ask:**
- What frontend framework will you use (React, Vue, Svelte, vanilla JS)?
- Will this be a single-page application (SPA) or multi-page application (MPA)?
- What state management approach will you use (Redux, Zustand, Context, none)?
- Will you use server-side rendering (SSR) or static site generation (SSG)?
- What build tool will you use (Vite, Webpack, Parcel)?
- What CSS approach will you use (Tailwind, CSS Modules, Styled Components, plain CSS)?
- Do you need TypeScript or is JavaScript sufficient?

**Example answers:**
> - React 18 with TypeScript
> - Single-page application (SPA)
> - Zustand for global state, local state for components
> - Client-side rendering (CSR) only
> - Vite for fast dev builds
> - Tailwind CSS for styling
> - TypeScript for type safety

---

## UI/UX Design
**Questions to ask:**
- Do you have existing design mockups or wireframes?
- What design system or component library will you use (Material-UI, Ant Design, custom)?
- What are the key user interfaces/screens?
- What are the responsive breakpoints (mobile, tablet, desktop)?
- Are there accessibility requirements (WCAG 2.1 AA, screen reader support)?
- What browsers must be supported (Chrome, Firefox, Safari, IE11)?
- Are there specific UX patterns to follow (e.g., always confirm deletions)?

**Example answers:**
> - Figma designs provided by design team
> - Custom component library based on Radix UI primitives
> - Key screens: Dashboard, Graph Viewer, Settings
> - Responsive: Mobile (320px+), Tablet (768px+), Desktop (1024px+)
> - WCAG 2.1 AA compliance required
> - Modern browsers only (no IE11)
> - Destructive actions require confirmation modal

---

## Data Fetching & APIs
**Questions to ask:**
- How will the frontend communicate with the backend (REST, GraphQL, gRPC-Web)?
- What API endpoints will be consumed?
- Will you use a data fetching library (React Query, SWR, Apollo)?
- How will you handle loading states?
- How will you handle errors (retry, fallback, error boundaries)?
- Do you need real-time updates (WebSockets, Server-Sent Events, polling)?
- What caching strategy will you use?

**Example answers:**
> - REST API with JSON responses
> - Endpoints: /api/services, /api/dependencies, /api/graph
> - React Query for caching and automatic refetching
> - Loading: Skeleton screens
> - Errors: Toast notifications + error boundary for crashes
> - WebSocket for real-time dependency updates
> - Cache: 5-minute stale-while-revalidate

---

## Routing & Navigation
**Questions to ask:**
- What routing library will you use (React Router, Next.js router, TanStack Router)?
- What are the main routes/pages?
- Do you need nested routing?
- How will authentication affect routing (protected routes)?
- Do you need deep linking support?
- What is the navigation structure (sidebar, top nav, breadcrumbs)?

**Example answers:**
> - React Router v6
> - Routes: /, /services/:id, /graph, /settings
> - Nested routes for service tabs (/services/:id/dependencies, /services/:id/conflicts)
> - Protected routes require authentication (redirect to /login)
> - Deep linking supported (shareable URLs)
> - Navigation: Sidebar with top breadcrumbs

---

## Authentication & Authorization
**Questions to ask:**
- How will users authenticate (login form, OAuth, SSO)?
- Where are auth tokens stored (localStorage, sessionStorage, cookies)?
- How will you handle token refresh?
- How do you protect routes (HOC, route guards)?
- How do you handle authorization (role-based, permission-based)?
- What happens when auth expires (redirect, refresh, modal)?

**Example answers:**
> - OAuth 2.0 with GitHub/GitLab
> - Tokens stored in httpOnly cookies (XSS protection)
> - Automatic token refresh via refresh token
> - Protected routes use ProtectedRoute wrapper component
> - Role-based: admin, developer, viewer
> - Auth expiry: Automatic refresh if possible, else redirect to login

---

## State Management
**Questions to ask:**
- What global state needs to be managed (user, theme, cache)?
- What local component state is needed?
- How will you handle form state (React Hook Form, Formik, plain state)?
- Do you need persistent state (localStorage, IndexedDB)?
- How will you handle optimistic updates?
- How will you synchronize state across tabs?

**Example answers:**
> - Global: User profile, selected service, theme
> - Local: Form inputs, modal open/closed, filters
> - Form state: React Hook Form with Zod validation
> - Persistent: Theme preference in localStorage
> - Optimistic: Update UI immediately, rollback on error
> - Tab sync: Use BroadcastChannel API for theme changes

---

## Performance Optimization
**Questions to ask:**
- What are your performance budgets (bundle size, FCP, LCP)?
- Will you use code splitting (route-based, component-based)?
- What assets need to be optimized (images, fonts, icons)?
- Will you use lazy loading for images/components?
- Do you need to optimize for slow networks (offline support, service worker)?
- What analytics/monitoring will you use (bundle analyzer, Lighthouse, Web Vitals)?

**Example answers:**
> - Bundle size: < 200KB gzipped for initial load
> - Core Web Vitals: LCP < 2.5s, FID < 100ms, CLS < 0.1
> - Route-based code splitting for each major page
> - Images: WebP format with fallback, lazy loaded
> - PWA with offline support for core features
> - Monitoring: Lighthouse CI, Sentry for errors, Vercel Analytics

---

## Forms & Validation
**Questions to ask:**
- What forms need to be implemented?
- What validation rules apply (client-side, server-side)?
- How will validation errors be displayed?
- Do you need multi-step forms or wizards?
- How will you handle unsaved changes (navigation warnings)?
- Do you need file uploads?

**Example answers:**
> - Forms: Login, service metadata edit, settings
> - Validation: Zod schemas (client-side), server validates again
> - Errors: Inline below fields, summary at top
> - Multi-step: Service creation wizard (3 steps)
> - Unsaved changes: Browser beforeunload warning
> - File upload: Service config file upload (YAML/JSON)

---

## Testing Strategy
**Questions to ask:**
- What testing frameworks will you use (Vitest, Jest, Cypress, Playwright)?
- What will you unit test vs integration test vs e2e test?
- What is your test coverage target?
- How will you test user interactions?
- How will you mock API calls?
- Will you use visual regression testing?

**Example answers:**
> - Vitest for unit/integration, Playwright for e2e
> - Unit: Utilities, hooks
> - Integration: Component behavior with mocked APIs
> - E2E: Critical user flows (login, create service, view graph)
> - Coverage target: 80%
> - API mocking: MSW (Mock Service Worker)
> - Visual regression: Chromatic for Storybook

---

## Deployment & Hosting
**Questions to ask:**
- Where will the frontend be hosted (Vercel, Netlify, AWS S3/CloudFront)?
- What is the build process?
- How are environment variables managed (build-time, runtime)?
- What is the deployment strategy (preview deploys, canary, blue/green)?
- Do you need a CDN?
- What domain/subdomain will be used?

**Example answers:**
> - Hosted on Vercel
> - Build: `vite build` produces static assets
> - Env vars: Build-time for public vars, runtime config endpoint for secrets
> - Deployment: Auto-deploy on main branch, preview deploys for PRs
> - CDN: Vercel's built-in edge network
> - Domain: app.paragon.dev

---

## SEO & Meta Tags
**Questions to ask:**
- Do you need SEO optimization (if public-facing)?
- What meta tags are needed (og:image, twitter:card)?
- Do you need dynamic meta tags per page?
- Will you have a sitemap?
- Do you need structured data (JSON-LD)?

**Example answers:**
> - No SEO needed (internal tool, auth-required)
> - Basic meta tags: title, description, favicon
> - Static meta tags (not dynamic)
> - No sitemap needed
> - No structured data needed

---

## Browser Support & Polyfills
**Questions to ask:**
- What browsers and versions must be supported?
- Do you need polyfills for older browsers?
- What features require progressive enhancement?
- How will you handle unsupported browsers?

**Example answers:**
> - Modern browsers: Last 2 versions of Chrome, Firefox, Safari, Edge
> - No IE11 support
> - No polyfills needed (target ES2020+)
> - Unsupported browser: Show upgrade banner

---

## Internationalization (i18n)
**Questions to ask:**
- Do you need to support multiple languages?
- What localization library will you use (react-i18next, FormatJS)?
- What languages are required?
- How will language selection work (auto-detect, user preference)?
- Do you need to localize dates, numbers, currencies?

**Example answers:**
> - English only for v1
> - Future: Spanish, French (use react-i18next)
> - Language selection: User preference in settings
> - Date/number formatting: Use Intl API

---

## Third-Party Integrations
**Questions to ask:**
- What third-party services will be integrated (analytics, error tracking, etc.)?
- What SDKs/scripts need to be loaded?
- How will you handle GDPR/privacy for third-party services?
- Do you need consent management for cookies/tracking?

**Example answers:**
> - Sentry for error tracking
> - Posthog for product analytics
> - Load scripts asynchronously to avoid blocking
> - GDPR: Cookie banner with opt-in for analytics (opt-out for errors)
