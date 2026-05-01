'use client'

import { useState } from 'react'
import Image from 'next/image'
import { Building2, MapPin, Users, Briefcase } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface CompanyHeaderProps {
  companyName: string
  companyDomain: string | null
  industry?: string | null
  hq?: string | null
  size?: string | null
  riskLevel: string
  engineTier?: string
}

export function CompanyHeader({
  companyName,
  companyDomain,
  industry,
  hq,
  size,
  riskLevel,
  engineTier,
}: CompanyHeaderProps) {
  const [imgState, setImgState] = useState<'loading' | 'loaded' | 'error'>(
    companyDomain ? 'loading' : 'error'
  )

  const riskVariant = riskLevel.toLowerCase() as 'low' | 'medium' | 'high' | 'critical'

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
      {/* Company logo */}
      <div className="relative h-16 w-16 flex-shrink-0">
        {/* Skeleton shown while loading */}
        {imgState === 'loading' && (
          <div className="absolute inset-0 rounded-2xl skeleton" />
        )}

        {/* Letter fallback shown on error */}
        {imgState === 'error' && (
          <div className="h-16 w-16 rounded-2xl flex items-center justify-center bg-[#111827] border border-[#1e293b]">
            <span className="font-display text-3xl font-bold text-text-secondary">
              {companyName.charAt(0).toUpperCase()}
            </span>
          </div>
        )}

        {/* Clearbit logo */}
        {companyDomain && imgState !== 'error' && (
          <div className={`h-16 w-16 rounded-2xl overflow-hidden border border-[#1e293b] bg-white flex items-center justify-center ${imgState === 'loading' ? 'opacity-0' : 'opacity-100'} transition-opacity duration-200`}>
            <Image
              src={`https://logo.clearbit.com/${companyDomain}`}
              alt={companyName}
              width={64}
              height={64}
              className="object-contain"
              onLoad={() => setImgState('loaded')}
              onError={() => setImgState('error')}
            />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-center gap-3 mb-1.5">
          <h1 className="font-display text-2xl md:text-3xl font-bold text-text-primary">
            {companyName}
          </h1>
          <Badge variant={riskVariant} className="text-xs font-mono h-6">
            {riskLevel.toUpperCase()} RISK
          </Badge>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-sm text-text-secondary">
          {industry && (
            <span className="flex items-center gap-1.5">
              <Briefcase className="h-3.5 w-3.5 text-text-muted" />
              {industry}
            </span>
          )}
          {hq && (
            <span className="flex items-center gap-1.5">
              <MapPin className="h-3.5 w-3.5 text-text-muted" />
              {hq}
            </span>
          )}
          {size && (
            <span className="flex items-center gap-1.5">
              <Users className="h-3.5 w-3.5 text-text-muted" />
              ~{size} employees
            </span>
          )}
          {companyDomain && (
            <span className="flex items-center gap-1.5 font-mono text-xs text-text-muted">
              <Building2 className="h-3.5 w-3.5" />
              {companyDomain}
            </span>
          )}
          {engineTier && (
            <span className="text-[11px] font-mono text-text-muted opacity-60">
              {engineTier} Engine
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
