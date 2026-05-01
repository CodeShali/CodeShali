'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import {
  Brain, ShieldOff, Server, Globe, ChevronRight, CheckCircle2,
  Search, Zap, Lock, BarChart3, Share2, ArrowRight, Menu, X,
  Building2, AlertTriangle, Layers, Database, Wifi,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

// ─── Landing Navbar ────────────────────────────────────────────────
function LandingNav() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', fn)
    return () => window.removeEventListener('scroll', fn)
  }, [])

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-[#060a12]/90 backdrop-blur-md border-b border-[#111827]' : ''
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-lg bg-accent-amber/10 border border-accent-amber/20 flex items-center justify-center">
            <span className="font-display text-sm font-bold text-accent-amber">D</span>
          </div>
          <span className="font-display text-lg font-bold text-text-primary">DataEcho</span>
        </Link>

        <nav className="hidden md:flex items-center gap-8 text-sm text-text-secondary">
          <a href="#features" className="hover:text-text-primary transition-colors">Features</a>
          <a href="#how-it-works" className="hover:text-text-primary transition-colors">How it works</a>
          <a href="#pricing" className="hover:text-text-primary transition-colors">Pricing</a>
        </nav>

        <div className="hidden md:flex items-center gap-3">
          <Link href="/login">
            <Button variant="ghost" size="sm" className="text-text-secondary">Sign in</Button>
          </Link>
          <Link href="/login">
            <Button size="sm" variant="pro" className="gap-1.5">
              Get started free <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          </Link>
        </div>

        <button
          className="md:hidden p-2 text-text-muted hover:text-text-secondary"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden border-t border-[#111827] bg-[#060a12] px-6 py-4 space-y-3">
          {['#features', '#how-it-works', '#pricing'].map((href) => (
            <a
              key={href}
              href={href}
              onClick={() => setMobileOpen(false)}
              className="block text-sm text-text-secondary hover:text-text-primary py-2 capitalize"
            >
              {href.replace('#', '').replace('-', ' ')}
            </a>
          ))}
          <Link href="/login" onClick={() => setMobileOpen(false)}>
            <Button variant="pro" className="w-full mt-2">Get started free</Button>
          </Link>
        </div>
      )}
    </header>
  )
}

// ─── Animated Product Demo ─────────────────────────────────────────
const DEMO_COMPANY = 'Adobe'
const DEMO_TYPING_MS = 80

function ProductDemo() {
  const [phase, setPhase] = useState<'idle' | 'typing' | 'loading' | 'results'>('idle')
  const [typed, setTyped] = useState('')
  const timerRef = useRef<ReturnType<typeof setTimeout>>()

  const startCycle = () => {
    setPhase('idle')
    setTyped('')

    timerRef.current = setTimeout(() => {
      setPhase('typing')
      let i = 0
      const typeNext = () => {
        if (i < DEMO_COMPANY.length) {
          setTyped(DEMO_COMPANY.slice(0, i + 1))
          i++
          timerRef.current = setTimeout(typeNext, DEMO_TYPING_MS)
        } else {
          timerRef.current = setTimeout(() => {
            setPhase('loading')
            timerRef.current = setTimeout(() => {
              setPhase('results')
              timerRef.current = setTimeout(startCycle, 6000)
            }, 2200)
          }, 400)
        }
      }
      typeNext()
    }, 800)
  }

  useEffect(() => {
    startCycle()
    return () => clearTimeout(timerRef.current)
  }, [])

  return (
    <div className="relative w-full max-w-md mx-auto">
      {/* Ambient glow */}
      <div className="absolute -inset-8 bg-amber-500/5 rounded-3xl blur-2xl animate-glow-breathe pointer-events-none" />
      <div className="absolute -inset-4 bg-sky-500/5 rounded-3xl blur-xl animate-glow-breathe pointer-events-none" style={{ animationDelay: '1.5s' }} />

      <div className="relative rounded-2xl border border-[#1e293b] bg-[#0a0e1a] shadow-2xl overflow-hidden">
        {/* Window chrome */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-[#111827] bg-[#060a12]">
          <div className="h-3 w-3 rounded-full bg-red-500/60" />
          <div className="h-3 w-3 rounded-full bg-amber-500/60" />
          <div className="h-3 w-3 rounded-full bg-green-500/60" />
          <div className="flex-1 mx-4 h-5 rounded bg-[#0f1829] border border-[#111827] flex items-center px-2">
            <span className="text-[10px] font-mono text-text-muted">dataecho.ai/audit</span>
          </div>
        </div>

        <div className="p-5 space-y-4 min-h-[320px]">
          {/* Search bar */}
          <div className="relative flex items-center gap-2 bg-[#060a12] border border-[#1e293b] rounded-xl px-4 py-3">
            <Search className="h-4 w-4 text-text-muted flex-shrink-0" />
            <span className="text-sm text-text-primary font-mono">
              {typed || (phase === 'idle' ? '' : '')}
              {(phase === 'typing' || (phase === 'idle' && typed === '')) && (
                <span className="inline-block w-0.5 h-3.5 bg-accent-amber ml-0.5 animate-blink" />
              )}
              {!typed && phase === 'idle' && (
                <span className="text-text-muted">Search any company...</span>
              )}
            </span>
            <div className="ml-auto">
              <div className={`h-7 w-7 rounded-lg flex items-center justify-center transition-all ${
                typed ? 'bg-accent-amber/20 border border-accent-amber/30' : 'bg-[#0f1829] border border-[#111827]'
              }`}>
                <ArrowRight className={`h-3.5 w-3.5 ${typed ? 'text-accent-amber' : 'text-text-muted'}`} />
              </div>
            </div>
          </div>

          {/* Loading state */}
          {phase === 'loading' && (
            <div className="flex flex-col items-center justify-center py-8 space-y-4 animate-fade-in">
              <div className="relative h-12 w-12">
                <div className="absolute inset-0 rounded-full border-2 border-accent-amber/20" />
                <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-accent-amber animate-spin" />
              </div>
              <div className="space-y-1 text-center">
                <p className="text-sm font-medium text-text-secondary">Running 3-layer audit...</p>
                <div className="flex items-center justify-center gap-4 text-xs font-mono text-text-muted">
                  <span className="flex items-center gap-1"><Brain className="h-3 w-3" /> AI scan</span>
                  <span className="flex items-center gap-1"><ShieldOff className="h-3 w-3" /> HIBP</span>
                  <span className="flex items-center gap-1"><Server className="h-3 w-3" /> Shodan</span>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {phase === 'results' && (
            <div className="space-y-3 animate-fade-in">
              {/* Company header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="h-9 w-9 rounded-xl bg-white flex items-center justify-center text-xs font-bold text-gray-800">A</div>
                  <div>
                    <p className="text-sm font-display font-bold text-text-primary">Adobe Inc.</p>
                    <p className="text-xs text-text-muted font-mono">adobe.com</p>
                  </div>
                </div>
                <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-full border border-orange-400/30 bg-orange-400/10 text-orange-400">
                  HIGH RISK
                </span>
              </div>

              {/* Risk bar */}
              <div className="rounded-xl border border-orange-500/20 bg-orange-500/5 px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-text-muted">RISK SCORE</span>
                  <span className="text-sm font-bold font-mono text-orange-400">78 / 100</span>
                </div>
                <div className="h-1.5 bg-[#111827] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-amber-400 to-orange-400 transition-all duration-1000"
                    style={{ width: '78%' }}
                  />
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-1.5">
                {[
                  { label: 'LLM', color: '#38bdf8', icon: Brain },
                  { label: 'Breaches', color: '#f87171', icon: ShieldOff, badge: '3' },
                  { label: 'Infra', color: '#fb923c', icon: Server },
                ].map(({ label, color, icon: Icon, badge }) => (
                  <div
                    key={label}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border"
                    style={{ backgroundColor: `${color}10`, borderColor: `${color}30`, color }}
                  >
                    <Icon className="h-3 w-3" />
                    {label}
                    {badge && (
                      <span className="text-[10px] px-1 rounded-full" style={{ backgroundColor: `${color}20`, color }}>
                        {badge}
                      </span>
                    )}
                  </div>
                ))}
              </div>

              {/* Sample breach card */}
              <div className="rounded-xl border border-green-400/20 bg-green-400/5 px-3 py-2.5">
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle2 className="h-3 w-3 text-green-400 flex-shrink-0" />
                  <span className="text-[10px] font-mono text-green-400">VERIFIED · HIBP</span>
                </div>
                <p className="text-xs font-semibold text-text-primary">Adobe 2013 Breach</p>
                <p className="text-[11px] text-text-muted mt-0.5">152M records · Passwords, Emails, Hints</p>
              </div>
            </div>
          )}
        </div>

        {/* Scan line overlay during loading */}
        {phase === 'loading' && (
          <div
            className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-accent-amber/60 to-transparent pointer-events-none animate-scan-line"
          />
        )}
      </div>
    </div>
  )
}

// ─── Stats marquee ─────────────────────────────────────────────────
const MARQUEE_ITEMS = [
  { icon: Building2, text: '50,000+ companies audited' },
  { icon: ShieldOff, text: 'HIBP breach database' },
  { icon: Server, text: 'Shodan infrastructure scans' },
  { icon: Database, text: 'GitHub code leak detection' },
  { icon: Wifi, text: 'Certificate transparency logs' },
  { icon: Brain, text: 'Claude AI-powered analysis' },
  { icon: CheckCircle2, text: 'Verified real-time data' },
  { icon: Lock, text: 'Zero data retention policy' },
]

function StatsMarquee() {
  const items = [...MARQUEE_ITEMS, ...MARQUEE_ITEMS]
  return (
    <div className="overflow-hidden border-y border-[#111827] bg-[#060a12]/60 py-4">
      <div className="flex animate-marquee gap-12 w-max">
        {items.map((item, i) => {
          const Icon = item.icon
          return (
            <div key={i} className="flex items-center gap-2.5 whitespace-nowrap">
              <Icon className="h-4 w-4 text-accent-amber flex-shrink-0" />
              <span className="text-sm font-mono text-text-muted">{item.text}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── How it works steps ────────────────────────────────────────────
const STEPS = [
  {
    number: '01',
    icon: Search,
    title: 'Enter a company name',
    description: 'Type any company name or domain. DataEcho resolves the domain and queues a multi-source intelligence scan.',
    color: '#f59e0b',
  },
  {
    number: '02',
    icon: Layers,
    title: 'Real data fetched in parallel',
    description: 'We simultaneously query Have I Been Pwned, Shodan, GitHub code search, certificate transparency logs, and Claude AI — all in under 10 seconds.',
    color: '#38bdf8',
  },
  {
    number: '03',
    icon: BarChart3,
    title: 'AI-enriched report delivered',
    description: 'Claude analyzes verified data and generates a structured 3-layer report: LLM knowledge, confirmed breaches, and infrastructure exposure.',
    color: '#4ade80',
  },
]

// ─── Feature cards ─────────────────────────────────────────────────
const FEATURES = [
  {
    icon: Brain,
    title: 'LLM Knowledge Layer',
    description: 'See exactly what AI models like Claude, GPT, and Gemini know about a company from training data — leadership, financials, controversies, legal issues.',
    color: '#38bdf8',
    items: ['Executive profiles', 'Product history', 'Public controversies', 'Partnership data'],
  },
  {
    icon: ShieldOff,
    title: 'Breach Intelligence',
    description: 'Real verified breach records from Have I Been Pwned\'s authoritative database, enriched with AI context about impact and remediation.',
    color: '#f87171',
    items: ['HIBP verified records', 'Records affected count', 'Data types exposed', 'Historical timeline'],
    badge: 'VERIFIED',
  },
  {
    icon: Server,
    title: 'Infrastructure Exposure',
    description: 'Certificate transparency logs reveal every subdomain. Shodan maps open ports and known CVEs on live infrastructure.',
    color: '#fb923c',
    items: ['Subdomain mapping', 'Open port detection', 'CVE vulnerability scan', 'Service fingerprinting'],
    badge: 'VERIFIED',
  },
  {
    icon: Globe,
    title: 'Code & Data Leaks',
    description: 'GitHub code search finds public repositories containing company domains, internal URLs, and potentially leaked configuration files.',
    color: '#a78bfa',
    items: ['Public repo scanning', 'Internal URL detection', 'Config file leaks', 'Employee disclosures'],
    badge: 'VERIFIED',
  },
]

// ─── Pricing ───────────────────────────────────────────────────────
const PLANS = [
  {
    name: 'Free',
    price: '$0',
    period: 'forever',
    badge: null,
    description: 'For individuals exploring data transparency.',
    features: [
      '3 audits per day',
      'All 4 intelligence layers',
      'Shareable audit links',
      '1 card expansion per audit',
      'Standard AI engine',
    ],
    cta: 'Start for free',
    ctaVariant: 'outline' as const,
    highlight: false,
  },
  {
    name: 'Pro',
    price: '$12',
    period: '/month',
    badge: 'Most Popular',
    description: 'For professionals and security teams.',
    features: [
      '20 audits per day',
      'Unlimited card expansions',
      'PDF export',
      'AI chat with audit data',
      'Advanced AI engine',
      'Priority support',
    ],
    cta: 'Start Pro trial',
    ctaVariant: 'pro' as const,
    highlight: true,
  },
  {
    name: 'Enterprise',
    price: '$49',
    period: '/month',
    badge: null,
    description: 'For organizations with compliance needs.',
    features: [
      'Unlimited audits',
      'Unlimited everything',
      'Premium AI engine',
      'Team access',
      'API access',
      'SLA + dedicated support',
    ],
    cta: 'Contact sales',
    ctaVariant: 'outline' as const,
    highlight: false,
  },
]

// ─── Main page ─────────────────────────────────────────────────────
export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#060a12] text-text-primary overflow-x-hidden">
      <LandingNav />

      {/* ── Hero ── */}
      <section className="relative pt-32 pb-20 md:pt-40 md:pb-28 overflow-hidden">
        {/* Grid background */}
        <div
          className="absolute inset-0 opacity-30 pointer-events-none"
          style={{
            backgroundImage:
              'linear-gradient(rgba(248,250,252,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(248,250,252,0.03) 1px, transparent 1px)',
            backgroundSize: '64px 64px',
          }}
        />
        {/* Radial glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-amber-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute top-1/4 right-0 w-[400px] h-[400px] bg-sky-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative max-w-6xl mx-auto px-6">
          <div className="flex flex-col lg:flex-row items-center gap-16">
            {/* Left — copy */}
            <div className="flex-1 text-center lg:text-left">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-accent-amber/20 bg-accent-amber/5 text-xs font-mono text-accent-amber mb-6">
                <span className="h-1.5 w-1.5 rounded-full bg-accent-amber animate-pulse-amber" />
                Real breach data · Live infrastructure · AI analysis
              </div>

              <h1 className="font-display text-5xl md:text-6xl lg:text-7xl font-bold leading-[1.05] tracking-tight mb-6">
                <span className="text-text-primary">See what AI</span>
                <br />
                <span
                  className="bg-clip-text text-transparent"
                  style={{
                    backgroundImage: 'linear-gradient(135deg, #f59e0b 0%, #fb923c 50%, #f87171 100%)',
                  }}
                >
                  knows about
                </span>
                <br />
                <span className="text-text-primary">any company.</span>
              </h1>

              <p className="text-lg text-text-secondary leading-relaxed max-w-xl mb-8 mx-auto lg:mx-0">
                DataEcho runs a 3-layer intelligence audit — combining verified breach databases,
                live infrastructure scanning, and Claude AI — to show you exactly what's publicly exposed about any company.
              </p>

              <div className="flex flex-col sm:flex-row gap-3 justify-center lg:justify-start">
                <Link href="/login">
                  <Button size="lg" variant="pro" className="gap-2 text-base px-6">
                    <Zap className="h-4 w-4" />
                    Run your first audit free
                  </Button>
                </Link>
                <a href="#how-it-works">
                  <Button size="lg" variant="outline" className="gap-2 text-base px-6">
                    See how it works
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </a>
              </div>

              <div className="flex items-center gap-5 mt-8 justify-center lg:justify-start">
                {[
                  { label: 'Free to start' },
                  { label: 'No credit card' },
                  { label: 'Instant results' },
                ].map(({ label }) => (
                  <div key={label} className="flex items-center gap-1.5 text-xs text-text-muted">
                    <CheckCircle2 className="h-3.5 w-3.5 text-green-400 flex-shrink-0" />
                    {label}
                  </div>
                ))}
              </div>
            </div>

            {/* Right — animated demo */}
            <div className="flex-1 w-full max-w-lg">
              <ProductDemo />
            </div>
          </div>
        </div>
      </section>

      {/* ── Marquee ── */}
      <StatsMarquee />

      {/* ── How it works ── */}
      <section id="how-it-works" className="py-24 relative">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <p className="text-xs font-mono text-accent-amber uppercase tracking-widest mb-3">How it works</p>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-text-primary mb-4">
              From search to insight<br />in seconds
            </h2>
            <p className="text-text-secondary max-w-lg mx-auto">
              One search triggers a parallel scan across 5 real data sources, synthesized by Claude AI into a single structured report.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6 relative">
            {/* Connector line */}
            <div className="hidden md:block absolute top-12 left-[calc(16.7%+1rem)] right-[calc(16.7%+1rem)] h-px bg-gradient-to-r from-amber-500/20 via-sky-500/20 to-green-500/20 pointer-events-none" />

            {STEPS.map((step, i) => {
              const Icon = step.icon
              return (
                <div
                  key={i}
                  className="relative rounded-2xl border border-[#111827] bg-[#0a0e1a] p-7 hover:border-[#1e293b] transition-all group"
                >
                  <div
                    className="h-12 w-12 rounded-xl flex items-center justify-center mb-5 transition-all group-hover:scale-110"
                    style={{ backgroundColor: `${step.color}15`, border: `1px solid ${step.color}25` }}
                  >
                    <Icon className="h-6 w-6" style={{ color: step.color }} />
                  </div>
                  <div
                    className="absolute top-6 right-6 font-display text-5xl font-bold opacity-5"
                    style={{ color: step.color }}
                  >
                    {step.number}
                  </div>
                  <h3 className="font-display text-lg font-bold text-text-primary mb-2">{step.title}</h3>
                  <p className="text-sm text-text-secondary leading-relaxed">{step.description}</p>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className="py-24 bg-[#0a0e1a] border-y border-[#111827]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <p className="text-xs font-mono text-accent-amber uppercase tracking-widest mb-3">Intelligence layers</p>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-text-primary mb-4">
              Four layers of<br />verified intelligence
            </h2>
            <p className="text-text-secondary max-w-lg mx-auto">
              Unlike tools that only guess, DataEcho fetches from real authoritative databases — then AI reasons over the verified data.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-5">
            {FEATURES.map((feature, i) => {
              const Icon = feature.icon
              return (
                <div
                  key={i}
                  className="rounded-2xl border border-[#111827] bg-[#060a12] p-7 hover:border-[#1e293b] transition-all group"
                >
                  <div className="flex items-start gap-4 mb-5">
                    <div
                      className="h-11 w-11 rounded-xl flex items-center justify-center flex-shrink-0 transition-all group-hover:scale-110"
                      style={{ backgroundColor: `${feature.color}12`, border: `1px solid ${feature.color}20` }}
                    >
                      <Icon className="h-5 w-5" style={{ color: feature.color }} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-display text-lg font-bold text-text-primary">{feature.title}</h3>
                        {feature.badge && (
                          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-green-400/10 border border-green-400/20 text-green-400">
                            {feature.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-text-secondary leading-relaxed">{feature.description}</p>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {feature.items.map((item) => (
                      <span
                        key={item}
                        className="text-xs font-mono px-2.5 py-1 rounded-lg border"
                        style={{
                          backgroundColor: `${feature.color}08`,
                          borderColor: `${feature.color}20`,
                          color: feature.color,
                        }}
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>

          {/* AI reasoning callout */}
          <div className="mt-8 rounded-2xl border border-accent-amber/20 bg-accent-amber/5 p-6 flex flex-col md:flex-row items-center gap-5">
            <div className="h-12 w-12 rounded-xl bg-accent-amber/10 border border-accent-amber/20 flex items-center justify-center flex-shrink-0">
              <Brain className="h-6 w-6 text-accent-amber" />
            </div>
            <div>
              <h4 className="font-display font-bold text-text-primary mb-1">Claude AI synthesizes everything</h4>
              <p className="text-sm text-text-secondary">
                Real data from HIBP, Shodan, GitHub, and crt.sh is fed as verified context to Claude.
                The AI reasons over facts — it doesn't guess. HIBP breach records are never contradicted or fabricated.
              </p>
            </div>
            <Link href="/login" className="shrink-0">
              <Button variant="default" size="sm" className="gap-1.5 whitespace-nowrap">
                Try it now <ArrowRight className="h-3.5 w-3.5" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Use cases ── */}
      <section className="py-24">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <p className="text-xs font-mono text-accent-amber uppercase tracking-widest mb-3">Who uses DataEcho</p>
            <h2 className="font-display text-4xl font-bold text-text-primary mb-4">Built for security-conscious teams</h2>
          </div>

          <div className="grid md:grid-cols-3 gap-5">
            {[
              {
                icon: Lock,
                title: 'Security Teams',
                description: 'Monitor your own company\'s public exposure. Get alerted to new breach records, subdomain sprawl, and CVEs on your infrastructure.',
                color: '#f87171',
              },
              {
                icon: Search,
                title: 'Due Diligence',
                description: 'Research vendors, partners, and acquisition targets. Understand their breach history and infrastructure posture before signing.',
                color: '#38bdf8',
              },
              {
                icon: AlertTriangle,
                title: 'Competitive Intel',
                description: 'Understand what AI models know about any competitor. See their public controversies, leadership profiles, and data incidents.',
                color: '#a78bfa',
              },
            ].map(({ icon: Icon, title, description, color }) => (
              <div key={title} className="rounded-2xl border border-[#111827] bg-[#0a0e1a] p-6 hover:border-[#1e293b] transition-all group">
                <div
                  className="h-10 w-10 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-all"
                  style={{ backgroundColor: `${color}12`, border: `1px solid ${color}20` }}
                >
                  <Icon className="h-5 w-5" style={{ color }} />
                </div>
                <h3 className="font-display font-bold text-text-primary mb-2">{title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pricing ── */}
      <section id="pricing" className="py-24 bg-[#0a0e1a] border-y border-[#111827]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <p className="text-xs font-mono text-accent-amber uppercase tracking-widest mb-3">Pricing</p>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-text-primary mb-4">
              Start free, scale when ready
            </h2>
            <p className="text-text-secondary max-w-lg mx-auto">
              All plans include real breach data, infrastructure scanning, and AI analysis. No credit card required to start.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-5 max-w-4xl mx-auto">
            {PLANS.map((plan) => (
              <div
                key={plan.name}
                className={`relative rounded-2xl p-7 flex flex-col transition-all ${
                  plan.highlight
                    ? 'border border-accent-amber/30 bg-gradient-to-b from-amber-500/5 to-transparent shadow-2xl shadow-amber-500/5'
                    : 'border border-[#111827] bg-[#060a12] hover:border-[#1e293b]'
                }`}
              >
                {plan.badge && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <Badge variant="pro" className="text-xs px-3 py-0.5 shadow-lg">
                      {plan.badge}
                    </Badge>
                  </div>
                )}

                <div className="mb-6">
                  <p className="font-display text-sm font-semibold text-text-secondary mb-1">{plan.name}</p>
                  <div className="flex items-baseline gap-1 mb-2">
                    <span className="font-display text-4xl font-bold text-text-primary">{plan.price}</span>
                    <span className="text-text-muted text-sm">{plan.period}</span>
                  </div>
                  <p className="text-xs text-text-muted">{plan.description}</p>
                </div>

                <ul className="space-y-2.5 mb-7 flex-1">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-center gap-2.5 text-sm text-text-secondary">
                      <CheckCircle2 className="h-4 w-4 text-green-400 flex-shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>

                <Link href="/login" className="block">
                  <Button
                    variant={plan.ctaVariant}
                    className={`w-full ${plan.highlight ? '' : ''}`}
                  >
                    {plan.cta}
                  </Button>
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Final CTA ── */}
      <section className="py-28 relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-amber-500/8 rounded-full blur-3xl animate-glow-breathe" />
        </div>
        <div className="relative max-w-3xl mx-auto px-6 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-accent-amber/20 bg-accent-amber/5 text-xs font-mono text-accent-amber mb-6">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-amber animate-pulse-amber" />
            Free forever · No credit card
          </div>
          <h2 className="font-display text-5xl md:text-6xl font-bold text-text-primary mb-5 leading-tight">
            What does AI know<br />
            <span
              className="bg-clip-text text-transparent"
              style={{ backgroundImage: 'linear-gradient(135deg, #f59e0b, #fb923c)' }}
            >
              about your company?
            </span>
          </h2>
          <p className="text-lg text-text-secondary mb-8 max-w-xl mx-auto">
            Run your first audit in 30 seconds. See verified breach records, exposed infrastructure, and AI-synthesized intelligence — all in one report.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link href="/login">
              <Button size="xl" variant="pro" className="gap-2 text-base px-8">
                <Zap className="h-5 w-5" />
                Run a free audit now
              </Button>
            </Link>
            <Link href="/login">
              <Button size="xl" variant="outline" className="gap-2 text-base px-8">
                <Share2 className="h-5 w-5" />
                View a sample report
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-[#111827] bg-[#060a12]">
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="grid md:grid-cols-4 gap-10 mb-10">
            <div>
              <div className="flex items-center gap-2.5 mb-4">
                <div className="h-8 w-8 rounded-lg bg-accent-amber/10 border border-accent-amber/20 flex items-center justify-center">
                  <span className="font-display text-sm font-bold text-accent-amber">D</span>
                </div>
                <span className="font-display text-lg font-bold text-text-primary">DataEcho</span>
              </div>
              <p className="text-sm text-text-muted leading-relaxed">
                AI-powered data transparency audits. See what's publicly exposed about any company.
              </p>
            </div>

            {[
              {
                heading: 'Product',
                links: [
                  { label: 'Features', href: '#features' },
                  { label: 'How it works', href: '#how-it-works' },
                  { label: 'Pricing', href: '#pricing' },
                ],
              },
              {
                heading: 'Intelligence',
                links: [
                  { label: 'HIBP Breaches', href: '#features' },
                  { label: 'Shodan Scan', href: '#features' },
                  { label: 'GitHub Leaks', href: '#features' },
                  { label: 'crt.sh Subdomains', href: '#features' },
                ],
              },
              {
                heading: 'Company',
                links: [
                  { label: 'Sign in', href: '/login' },
                  { label: 'Get started', href: '/login' },
                ],
              },
            ].map(({ heading, links }) => (
              <div key={heading}>
                <p className="text-xs font-mono font-semibold text-text-muted uppercase tracking-wider mb-4">{heading}</p>
                <ul className="space-y-2.5">
                  {links.map(({ label, href }) => (
                    <li key={label}>
                      <a href={href} className="text-sm text-text-secondary hover:text-text-primary transition-colors">
                        {label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <div className="flex flex-col md:flex-row items-center justify-between pt-8 border-t border-[#111827] gap-4">
            <p className="text-xs text-text-muted font-mono">
              © {new Date().getFullYear()} DataEcho. All rights reserved.
            </p>
            <p className="text-xs text-text-muted font-mono">
              Powered by Claude AI · HIBP · Shodan · GitHub · crt.sh
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
