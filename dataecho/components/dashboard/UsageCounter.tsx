'use client'

import { useEffect, useState } from 'react'
import { Zap, Infinity as InfinityIcon } from 'lucide-react'
import Link from 'next/link'

interface UsageData {
  current: number
  limit: number
  plan: string
  remaining: number
}

export function UsageCounter() {
  const [usage, setUsage] = useState<UsageData | null>(null)

  useEffect(() => {
    fetch('/api/usage')
      .then((r) => r.json())
      .then(setUsage)
      .catch(() => null)
  }, [])

  if (!usage) {
    return (
      <div className="h-8 w-48 rounded-full skeleton" />
    )
  }

  if (!Number.isFinite(usage.limit) || usage.plan !== 'FREE') {
    return (
      <div className="flex items-center gap-1.5 rounded-full border border-accent-amber/20 bg-accent-amber/5 px-4 py-1.5 text-sm font-mono">
        <InfinityIcon className="h-3.5 w-3.5 text-accent-amber" />
        <span className="text-accent-amber font-medium capitalize">{usage.plan.toLowerCase()}</span>
        <span className="text-text-muted">— unlimited audits</span>
      </div>
    )
  }

  const pct = (usage.current / usage.limit) * 100
  const isNearLimit = usage.current >= usage.limit - 1
  const isAtLimit = usage.current >= usage.limit

  return (
    <div className="flex items-center gap-3">
      <div
        className={`flex items-center gap-1.5 rounded-full border px-4 py-1.5 text-sm font-mono transition-colors ${
          isAtLimit
            ? 'border-red-400/30 bg-red-400/10 text-red-400'
            : isNearLimit
            ? 'border-amber-400/30 bg-amber-400/10 text-amber-400'
            : 'border-[#1e293b] bg-[#0a0e1a] text-text-secondary'
        }`}
      >
        <Zap className="h-3.5 w-3.5" />
        <span>
          <span className={isAtLimit ? 'text-red-400 font-semibold' : 'text-text-primary font-semibold'}>
            {usage.current}
          </span>
          <span className="text-text-muted"> of {usage.limit} free audits used today</span>
        </span>
      </div>

      {isNearLimit && (
        <Link href="/dashboard?upgrade=true" className="text-xs font-medium text-accent-amber hover:underline">
          Upgrade →
        </Link>
      )}
    </div>
  )
}
