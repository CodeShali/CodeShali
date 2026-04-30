'use client'

import { useEffect, useState } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import {
  TrendingUp, TrendingDown, DollarSign, Zap, Users, Activity,
  BarChart3, RefreshCw, AlertTriangle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Metrics {
  period: { days: number; since: string }
  users: { total: number; byPlan: Record<string, number> }
  revenue: { mrr: number; inPeriod: number }
  costs: {
    totalUsd: number
    byPlan: Record<string, { audits: number; cost: number; inputTokens: number; outputTokens: number }>
    byModel: Record<string, { audits: number; cost: number }>
    daily: Record<string, number>
  }
  profit: { inPeriod: number; marginPct: number }
  auditCount: number
}

const PLAN_MODEL: Record<string, string> = {
  FREE: 'Haiku 4.5',
  PRO: 'Sonnet 4.6',
  ENTERPRISE: 'Opus 4.7',
}

const PLAN_PRICE: Record<string, string> = {
  FREE: '$0/mo',
  PRO: '$12/mo',
  ENTERPRISE: '$49/mo',
}

function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  color = 'text-text-primary',
  positive,
}: {
  label: string
  value: string
  sub?: string
  icon: React.ElementType
  color?: string
  positive?: boolean
}) {
  return (
    <div className="rounded-xl border border-[#111827] bg-[#0a0e1a] p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-mono text-text-muted uppercase tracking-wider mb-2">{label}</p>
          <p className={`text-2xl font-bold font-display ${color}`}>{value}</p>
          {sub && <p className="text-xs text-text-muted mt-1">{sub}</p>}
        </div>
        <div className={`p-2 rounded-lg bg-[#0f1829] ${color}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
    </div>
  )
}

function PlanRow({
  plan,
  userCount,
  data,
}: {
  plan: string
  userCount: number
  data?: { audits: number; cost: number; inputTokens: number; outputTokens: number }
}) {
  const costPerAudit = data && data.audits > 0 ? data.cost / data.audits : 0
  const colors: Record<string, string> = {
    FREE: 'text-text-secondary',
    PRO: 'text-accent-amber',
    ENTERPRISE: 'text-purple-400',
  }

  return (
    <div className="flex items-center gap-4 py-3 border-b border-[#111827] last:border-0">
      <div className="w-28">
        <span className={`text-sm font-semibold ${colors[plan] || 'text-text-secondary'}`}>{plan}</span>
        <p className="text-xs text-text-muted">{PLAN_PRICE[plan]}</p>
      </div>
      <div className="flex-1 text-xs font-mono text-text-muted">{PLAN_MODEL[plan] || '—'}</div>
      <div className="w-16 text-right text-sm text-text-secondary">{userCount}</div>
      <div className="w-16 text-right text-sm text-text-secondary">{data?.audits ?? 0}</div>
      <div className="w-24 text-right text-sm font-mono text-red-400">
        ${(data?.cost ?? 0).toFixed(4)}
      </div>
      <div className="w-24 text-right text-sm font-mono text-text-muted">
        ${costPerAudit.toFixed(5)}/audit
      </div>
    </div>
  )
}

export default function AdminPage() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`/api/admin/metrics?days=${days}`)
      if (res.status === 403) {
        setError('Access denied. Admin only.')
        return
      }
      const data = await res.json()
      setMetrics(data)
    } catch {
      setError('Failed to load metrics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (status === 'unauthenticated') router.replace('/login')
    if (status === 'authenticated') fetchMetrics()
  }, [status, days])

  if (status === 'loading') return null

  return (
    <div className="min-h-screen bg-[#060a12] text-text-primary">
      <div className="border-b border-[#111827] px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BarChart3 className="h-5 w-5 text-accent-amber" />
          <span className="font-display text-lg font-bold">Admin — Cost & Revenue</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex rounded-lg border border-[#111827] overflow-hidden text-xs">
            {[7, 30, 90].map((d) => (
              <button
                key={d}
                onClick={() => setDays(d)}
                className={`px-3 py-1.5 font-mono transition-colors ${
                  days === d ? 'bg-accent-amber/10 text-accent-amber' : 'text-text-muted hover:text-text-secondary'
                }`}
              >
                {d}d
              </button>
            ))}
          </div>
          <Button size="sm" variant="outline" onClick={fetchMetrics} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {error && (
          <div className="flex items-center gap-2 rounded-xl border border-red-500/20 bg-red-500/5 px-4 py-3 text-sm text-red-400">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}

        {metrics && (
          <>
            {/* Key metrics */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                label={`Revenue (${days}d)`}
                value={`$${metrics.revenue.inPeriod.toFixed(2)}`}
                sub={`MRR: $${metrics.revenue.mrr}/mo`}
                icon={DollarSign}
                color="text-green-400"
              />
              <StatCard
                label={`API Cost (${days}d)`}
                value={`$${metrics.costs.totalUsd.toFixed(4)}`}
                sub={`${metrics.auditCount} audits`}
                icon={Zap}
                color="text-red-400"
              />
              <StatCard
                label={`Profit (${days}d)`}
                value={`$${metrics.profit.inPeriod.toFixed(2)}`}
                sub={`${metrics.profit.marginPct}% margin`}
                icon={metrics.profit.inPeriod >= 0 ? TrendingUp : TrendingDown}
                color={metrics.profit.inPeriod >= 0 ? 'text-green-400' : 'text-red-400'}
              />
              <StatCard
                label="Total Users"
                value={String(metrics.users.total)}
                sub={`FREE: ${metrics.users.byPlan['FREE'] || 0} / PRO: ${metrics.users.byPlan['PRO'] || 0}`}
                icon={Users}
                color="text-sky-400"
              />
            </div>

            {/* Cost by plan */}
            <div className="rounded-xl border border-[#111827] bg-[#0a0e1a] overflow-hidden">
              <div className="px-5 py-4 border-b border-[#111827] flex items-center gap-2">
                <Activity className="h-4 w-4 text-text-muted" />
                <h2 className="text-sm font-medium text-text-secondary">Cost Breakdown by Plan</h2>
                <span className="text-xs text-text-muted ml-auto">Last {days} days</span>
              </div>
              <div className="px-5">
                {/* Header */}
                <div className="flex items-center gap-4 py-2 text-xs font-mono text-text-muted">
                  <div className="w-28">Plan</div>
                  <div className="flex-1">Model</div>
                  <div className="w-16 text-right">Users</div>
                  <div className="w-16 text-right">Audits</div>
                  <div className="w-24 text-right">Total Cost</div>
                  <div className="w-24 text-right">Avg/Audit</div>
                </div>
                {['FREE', 'PRO', 'ENTERPRISE'].map((plan) => (
                  <PlanRow
                    key={plan}
                    plan={plan}
                    userCount={metrics.users.byPlan[plan] || 0}
                    data={metrics.costs.byPlan[plan]}
                  />
                ))}
              </div>
            </div>

            {/* Model breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-xl border border-[#111827] bg-[#0a0e1a] overflow-hidden">
                <div className="px-5 py-4 border-b border-[#111827]">
                  <h2 className="text-sm font-medium text-text-secondary">Model Usage</h2>
                </div>
                <div className="px-5 py-3 space-y-3">
                  {Object.entries(metrics.costs.byModel).map(([model, data]) => (
                    <div key={model} className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-mono text-text-primary">{model}</p>
                        <p className="text-xs text-text-muted">{data.audits} audits</p>
                      </div>
                      <span className="text-sm font-mono text-red-400">${data.cost.toFixed(4)}</span>
                    </div>
                  ))}
                  {Object.keys(metrics.costs.byModel).length === 0 && (
                    <p className="text-sm text-text-muted py-2">No model data yet</p>
                  )}
                </div>
              </div>

              {/* Plan config reference */}
              <div className="rounded-xl border border-[#111827] bg-[#0a0e1a] overflow-hidden">
                <div className="px-5 py-4 border-b border-[#111827]">
                  <h2 className="text-sm font-medium text-text-secondary">Plan Config (Live)</h2>
                  <p className="text-xs text-text-muted mt-0.5">Edit backend/services/claude_service.py to change</p>
                </div>
                <div className="px-5 py-3 space-y-3">
                  {[
                    { plan: 'FREE', model: 'claude-haiku-4-5', maxTokens: 1500, searches: 2, color: 'text-text-secondary' },
                    { plan: 'PRO', model: 'claude-sonnet-4-6', maxTokens: 3000, searches: 5, color: 'text-accent-amber' },
                    { plan: 'ENTERPRISE', model: 'claude-opus-4-7', maxTokens: 6000, searches: 10, color: 'text-purple-400' },
                  ].map((cfg) => (
                    <div key={cfg.plan} className="flex items-start justify-between gap-4">
                      <div>
                        <p className={`text-sm font-semibold ${cfg.color}`}>{cfg.plan}</p>
                        <p className="text-xs font-mono text-text-muted">{cfg.model}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-text-secondary">{cfg.maxTokens} max tokens</p>
                        <p className="text-xs text-text-muted">{cfg.searches} web searches</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Daily cost chart (table form) */}
            {Object.keys(metrics.costs.daily).length > 0 && (
              <div className="rounded-xl border border-[#111827] bg-[#0a0e1a] overflow-hidden">
                <div className="px-5 py-4 border-b border-[#111827]">
                  <h2 className="text-sm font-medium text-text-secondary">Daily API Costs</h2>
                </div>
                <div className="px-5 py-3 overflow-x-auto">
                  <div className="flex gap-1 items-end min-w-max">
                    {Object.entries(metrics.costs.daily)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([day, cost]) => {
                        const maxCost = Math.max(...Object.values(metrics.costs.daily))
                        const heightPct = maxCost > 0 ? Math.max(4, (cost / maxCost) * 80) : 4
                        return (
                          <div key={day} className="flex flex-col items-center gap-1" title={`${day}: $${cost.toFixed(4)}`}>
                            <div
                              className="w-5 rounded-t bg-red-400/40 hover:bg-red-400/70 transition-colors cursor-default"
                              style={{ height: `${heightPct}px` }}
                            />
                            <span className="text-[9px] font-mono text-text-muted rotate-45 origin-left ml-1">
                              {day.slice(5)}
                            </span>
                          </div>
                        )
                      })}
                  </div>
                </div>
              </div>
            )}

            {/* Profitability note */}
            <div className={`rounded-xl border px-5 py-4 ${
              metrics.profit.inPeriod >= 0
                ? 'border-green-400/20 bg-green-400/5'
                : 'border-red-400/20 bg-red-400/5'
            }`}>
              <p className={`text-sm font-medium ${metrics.profit.inPeriod >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {metrics.profit.inPeriod >= 0
                  ? `Profitable: earning $${metrics.profit.inPeriod.toFixed(2)} more than Claude API costs this period (${metrics.profit.marginPct}% margin)`
                  : `Loss: spending $${Math.abs(metrics.profit.inPeriod).toFixed(2)} more on Claude API than revenue this period`
                }
              </p>
              <p className="text-xs text-text-muted mt-1">
                Revenue estimate = MRR × {days}/30 days. Actual billing depends on Stripe invoices.
              </p>
            </div>
          </>
        )}

        {!metrics && !error && !loading && (
          <div className="text-center py-20 text-text-muted text-sm">No data yet</div>
        )}
      </div>
    </div>
  )
}
