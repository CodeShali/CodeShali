'use client'

import { useState } from 'react'
import Image from 'next/image'
import { Building2, MapPin, Users, Briefcase } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { getRiskColor } from '@/lib/utils'

interface CompanyHeaderProps {
  companyName: string
  companyDomain: string | null
  industry?: string | null
  hq?: string | null
  size?: string | null
  riskLevel: string
}

export function CompanyHeader({
  companyName,
  companyDomain,
  industry,
  hq,
  size,
  riskLevel,
}: CompanyHeaderProps) {
  const [imgError, setImgError] = useState(false)

  const riskVariant = riskLevel.toLowerCase() as 'low' | 'medium' | 'high' | 'critical'

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
      {/* Company logo */}
      {companyDomain && !imgError ? (
        <div className="h-16 w-16 rounded-2xl overflow-hidden border border-[#1e293b] bg-white flex items-center justify-center flex-shrink-0">
          <Image
            src={`https://logo.clearbit.com/${companyDomain}`}
            alt={companyName}
            width={64}
            height={64}
            className="object-contain"
            onError={() => setImgError(true)}
          />
        </div>
      ) : (
        <div className="h-16 w-16 rounded-2xl flex items-center justify-center bg-[#111827] border border-[#1e293b] flex-shrink-0">
          <span className="font-display text-3xl font-bold text-text-secondary">
            {companyName.charAt(0).toUpperCase()}
          </span>
        </div>
      )}

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
        </div>
      </div>
    </div>
  )
}
