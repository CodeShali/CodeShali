# DataEcho — AI Data Transparency SaaS

## Overview
DataEcho lets users search any company and see: (1) what LLMs know from training data, (2) known data breaches, and (3) recently shared/leaked company information.

**Live target:** dataecho.ai  
**Tagline:** "See what AI knows about you."

## Tech Stack
- **Frontend:** Next.js 14 (App Router) + Tailwind CSS + shadcn/ui
- **Backend:** Python FastAPI (`backend/`)
- **Database:** PostgreSQL + Prisma ORM
- **Auth:** NextAuth.js (Google + GitHub OAuth)
- **Payments:** Stripe subscriptions
- **LLM:** Anthropic Claude API with web_search tool
- **Logos:** Clearbit Logo API

## Project Structure
```
dataecho/
├── app/                     # Next.js App Router
│   ├── (auth)/login/        # Login page
│   ├── (dashboard)/         # Protected dashboard routes
│   │   ├── layout.tsx       # Dashboard shell
│   │   ├── page.tsx         # Home: search + recent feed
│   │   └── audit/[company]/ # Company audit page
│   ├── api/                 # Next.js API routes
│   │   ├── auth/            # NextAuth handlers
│   │   ├── audit/           # Calls FastAPI, stores in DB
│   │   ├── chat/            # Chat with audit context
│   │   ├── usage/           # Daily usage check
│   │   └── stripe/          # Checkout + portal + webhooks
│   └── globals.css
├── backend/                 # FastAPI Python service
│   ├── main.py
│   ├── routers/audit.py
│   └── services/claude_service.py  # Claude + web_search
├── components/
│   ├── auth/LoginForm.tsx
│   ├── audit/               # CompanyHeader, RiskBanner, DataCard, LayerSection, ChatWidget
│   ├── dashboard/           # SearchBar, UsageCounter, RecentFeed
│   └── shared/              # Navbar, Sidebar, PaywallModal, UpgradeButton
├── lib/                     # auth.ts, stripe.ts, prisma.ts, utils.ts
├── prisma/schema.prisma
└── middleware.ts             # Protects /dashboard routes
```

## Running Locally

### Frontend
```bash
cd dataecho
cp .env.example .env.local
# fill in env vars
npm install
npx prisma db push
npm run dev
```

### Backend
```bash
cd dataecho/backend
cp .env.example .env
pip install -r requirements.txt
python main.py
```

## Environment Variables
See `.env.example` for all required variables. Key ones:
- `ANTHROPIC_API_KEY` — Claude API access
- `DATABASE_URL` — PostgreSQL connection string
- `NEXTAUTH_SECRET` — Random secret for session signing
- `GOOGLE_CLIENT_ID/SECRET` — Google OAuth
- `GITHUB_CLIENT_ID/SECRET` — GitHub OAuth
- `STRIPE_SECRET_KEY` — Stripe secret
- `STRIPE_WEBHOOK_SECRET` — From Stripe dashboard

## Plans & Limits
| Plan       | Audits/day | Card expansions | Chat messages |
|------------|-----------|-----------------|---------------|
| FREE       | 3         | 1 per audit     | 3 per audit   |
| PRO ($12/mo)| 20       | Unlimited       | Unlimited     |
| ENTERPRISE | Unlimited | Unlimited       | Unlimited     |

## Design System
- Background: `#060a12` (primary), `#0a0e1a` (secondary), `#0f1829` (cards)
- Fonts: Syne (display/headings), Inter (body), JetBrains Mono (labels)
- Accents: amber `#f59e0b`, blue `#38bdf8`, red `#f87171`, purple `#a78bfa`, green `#4ade80`
- Risk colors: Low=green, Medium=amber, High=orange, Critical=red

## Key Architectural Decisions
- NextAuth JWT strategy stores plan in token to avoid DB round-trip per request
- Usage counting is done in the Next.js API layer (not FastAPI) for simpler auth context
- FastAPI backend is stateless — it only calls Claude and returns JSON
- Company audit results stored as JSON blob in `Audit.result` for flexibility
- Clearbit logo with first-letter fallback — never breaks layout
