'use client'

import { useState } from 'react'
import { ChevronDown, Lock, ExternalLink, CheckCircle2 } from 'lucide-react'
import { cn, getSensitivityColor } from '@/lib/utils'
import { Button } from '@/components/ui/button'

interface DataCardProps {
  label: string
  summary: string
  detail?: string
  items?: string[]
  tags?: string[]
  platform?: string
  year?: string
  date?: string
  sensitivity?: string
  source?: string
  url?: string
  dataExposed?: string[]
  recordsAffected?: string
  verified?: boolean
  plan: string
  expansionCount: number
  onExpand: () => void
  onPaywall: () => void
  accentColor?: string
}

const PLATFORM_COLORS: Record<string, string> = {
  Reddit: 'text-orange-600 bg-orange-50',
  GitHub: 'text-slate-600 bg-slate-100',
  LinkedIn: 'text-blue-600 bg-blue-50',
  Forum: 'text-violet-600 bg-violet-50',
  News: 'text-emerald-600 bg-emerald-50',
}

export function DataCard({
  label,
  summary,
  detail,
  items,
  tags,
  platform,
  year,
  date,
  sensitivity,
  source,
  url,
  dataExposed,
  recordsAffected,
  verified,
  plan,
  expansionCount,
  onExpand,
  onPaywall,
  accentColor = '#0ea5e9',
}: DataCardProps) {
  const [expanded, setExpanded] = useState(false)

  const canExpand = plan !== 'FREE' || expansionCount < 1

  const handleToggle = () => {
    if (!expanded) {
      if (!canExpand) {
        onPaywall()
        return
      }
      onExpand()
    }
    setExpanded(!expanded)
  }

  const platformStyle = platform ? PLATFORM_COLORS[platform] || 'text-slate-600 bg-slate-100' : ''

  return (
    <div
      className={cn(
        'rounded-xl border border-slate-200 bg-white overflow-hidden transition-all duration-200 hover:border-slate-300 hover:card-shadow',
        expanded && 'border-slate-300 card-shadow'
      )}
    >
      <button
        onClick={handleToggle}
        className="w-full flex items-start gap-3 p-4 text-left"
      >
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <span className="text-sm font-semibold font-mono" style={{ color: accentColor }}>
              {label}
            </span>

            {platform && (
              <span className={cn('text-[10px] font-mono px-2 py-0.5 rounded-full', platformStyle)}>
                {platform}
              </span>
            )}

            {(year || date) && (
              <span className="text-[10px] font-mono text-text-muted">{year || date}</span>
            )}

            {sensitivity && (
              <span className={cn('text-[10px] font-mono', getSensitivityColor(sensitivity))}>
                {sensitivity.toUpperCase()} sensitivity
              </span>
            )}

            {recordsAffected && (
              <span className="text-[10px] font-mono text-red-500">{recordsAffected} records</span>
            )}

            {verified && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700">
                <CheckCircle2 className="h-2.5 w-2.5" />
                VERIFIED
              </span>
            )}
          </div>

          <p className="text-sm text-text-secondary leading-relaxed">{summary}</p>
        </div>

        <div className="flex items-center gap-1.5 flex-shrink-0 mt-0.5">
          {!canExpand && !expanded && (
            <Lock className="h-3.5 w-3.5 text-text-muted" />
          )}
          <ChevronDown
            className={cn(
              'h-4 w-4 text-text-muted transition-transform duration-200',
              expanded && 'rotate-180'
            )}
          />
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 border-t border-slate-100">
          <div className="pt-4 space-y-3">
            {detail && (
              <p className="text-sm text-text-secondary leading-relaxed">{detail}</p>
            )}

            {items && items.length > 0 && (
              <ul className="space-y-1.5">
                {items.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                    <span
                      className="mt-1.5 h-1.5 w-1.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: accentColor }}
                    />
                    {item}
                  </li>
                ))}
              </ul>
            )}

            {dataExposed && dataExposed.length > 0 && (
              <div>
                <p className="text-xs font-mono text-text-muted mb-2">Data exposed:</p>
                <div className="flex flex-wrap gap-1.5">
                  {dataExposed.map((d, i) => (
                    <span
                      key={i}
                      className="text-xs font-mono px-2 py-1 rounded-lg bg-red-50 text-red-700 border border-red-200"
                    >
                      {d}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {tags && tags.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {tags.map((t, i) => (
                  <span key={i} className="text-xs font-mono px-2 py-1 rounded-lg bg-slate-100 text-text-muted">
                    {t}
                  </span>
                ))}
              </div>
            )}

            {(source || url) && (
              <a
                href={url || source}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-xs font-mono text-accent-blue hover:underline"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="h-3 w-3" />
                View source
              </a>
            )}
          </div>
        </div>
      )}

      {!canExpand && expanded && (
        <div className="relative -mt-16 pb-4 px-4">
          <div className="absolute inset-0 backdrop-blur-sm bg-white/70 rounded-b-xl flex flex-col items-center justify-center gap-3 p-4">
            <Lock className="h-6 w-6 text-text-muted" />
            <p className="text-sm font-medium text-text-secondary text-center">
              Upgrade to Pro to expand unlimited cards
            </p>
            <Button size="sm" variant="pro" onClick={onPaywall}>
              Upgrade to Pro
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
