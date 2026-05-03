'use client'

import { useEffect, useState } from 'react'
import { AlertTriangle, Shield, ShieldAlert, ShieldCheck } from 'lucide-react'
import { getRiskBg } from '@/lib/utils'

interface RiskBannerProps {
  riskLevel: string
  riskScore: number
  riskReason: string
}

function AnimatedScore({ target }: { target: number }) {
  const [current, setCurrent] = useState(0)

  useEffect(() => {
    const duration = 1200
    const startTime = Date.now()

    const frame = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setCurrent(Math.round(eased * target))
      if (progress < 1) requestAnimationFrame(frame)
    }

    requestAnimationFrame(frame)
  }, [target])

  return <>{current}</>
}

const riskIcons = {
  low: ShieldCheck,
  medium: Shield,
  high: ShieldAlert,
  critical: AlertTriangle,
}

const riskBg = {
  low: 'from-emerald-50 to-white border-emerald-200',
  medium: 'from-amber-50 to-white border-amber-200',
  high: 'from-orange-50 to-white border-orange-200',
  critical: 'from-red-50 to-white border-red-200',
}

const riskTrack = {
  low: '#d1fae5',
  medium: '#fef3c7',
  high: '#ffedd5',
  critical: '#fee2e2',
}

export function RiskBanner({ riskLevel, riskScore, riskReason }: RiskBannerProps) {
  const level = riskLevel.toLowerCase() as keyof typeof riskIcons
  const Icon = riskIcons[level] || Shield
  const color = getRiskBg(riskLevel)
  const bg = riskBg[level] || riskBg.medium
  const track = riskTrack[level] || riskTrack.medium

  return (
    <div className={`rounded-2xl border bg-gradient-to-r ${bg} p-5`}>
      <div className="flex items-center gap-4">
        <div className="relative flex-shrink-0">
          <svg width="80" height="80" viewBox="0 0 80 80" className="-rotate-90">
            <circle cx="40" cy="40" r="34" fill="none" stroke={track} strokeWidth="6" />
            <circle
              cx="40" cy="40" r="34"
              fill="none"
              stroke={color}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${(riskScore / 100) * 213.6} 213.6`}
              className="transition-all duration-1000"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-mono text-xl font-bold" style={{ color }}>
              <AnimatedScore target={riskScore} />
            </span>
            <span className="text-[9px] font-mono text-text-muted uppercase">/100</span>
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Icon className="h-4 w-4" style={{ color }} />
            <span className="font-mono text-sm font-semibold uppercase tracking-wide" style={{ color }}>
              {riskLevel} Exposure Risk
            </span>
          </div>
          <p className="text-sm text-text-secondary leading-relaxed">{riskReason}</p>
        </div>
      </div>
    </div>
  )
}
