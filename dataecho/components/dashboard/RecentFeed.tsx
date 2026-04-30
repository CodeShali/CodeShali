'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { formatDate, getRiskColor, slugify } from '@/lib/utils'
import { Clock, ChevronRight, Building2 } from 'lucide-react'

interface AuditSummary {
  id: string
  companyName: string
  companyDomain: string | null
  riskLevel: string
  industry: string | null
  createdAt: string
}

function CompanyLogo({ domain, name }: { domain: string | null; name: string }) {
  const [imgState, setImgState] = useState<'loading' | 'loaded' | 'error'>(
    domain ? 'loading' : 'error'
  )

  return (
    <div className="relative h-10 w-10 flex-shrink-0">
      {imgState === 'loading' && (
        <div className="absolute inset-0 rounded-xl skeleton" />
      )}
      {imgState === 'error' && (
        <div className="h-10 w-10 rounded-xl flex items-center justify-center bg-[#111827] border border-[#1e293b]">
          <span className="font-display text-lg font-bold text-text-secondary">
            {name.charAt(0).toUpperCase()}
          </span>
        </div>
      )}
      {domain && imgState !== 'error' && (
        <div className={`h-10 w-10 rounded-xl overflow-hidden border border-[#1e293b] bg-white flex items-center justify-center ${imgState === 'loading' ? 'opacity-0' : 'opacity-100'} transition-opacity duration-200`}>
          <Image
            src={`https://logo.clearbit.com/${domain}`}
            alt={name}
            width={40}
            height={40}
            className="object-contain"
            onLoad={() => setImgState('loaded')}
            onError={() => setImgState('error')}
          />
        </div>
      )}
    </div>
  )
}

function AuditSkeletonCard() {
  return (
    <div className="flex items-center gap-4 rounded-xl border border-[#111827] bg-[#0f1829] p-4">
      <div className="h-10 w-10 rounded-xl skeleton flex-shrink-0" />
      <div className="flex-1 space-y-2">
        <div className="h-4 w-32 rounded skeleton" />
        <div className="h-3 w-20 rounded skeleton" />
      </div>
      <div className="h-6 w-16 rounded-full skeleton" />
    </div>
  )
}

export function RecentFeed() {
  const [audits, setAudits] = useState<AuditSummary[] | null>(null)

  useEffect(() => {
    fetch('/api/audit')
      .then((r) => r.json())
      .then(setAudits)
      .catch(() => setAudits([]))
  }, [])

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <Clock className="h-4 w-4 text-text-muted" />
        <h2 className="text-sm font-mono font-medium text-text-muted uppercase tracking-wider">
          Recent Audits
        </h2>
      </div>

      {audits === null ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => <AuditSkeletonCard key={i} />)}
        </div>
      ) : audits.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-[#111827] bg-[#0a0e1a] py-12 text-center">
          <Building2 className="h-10 w-10 text-text-muted mb-3" />
          <p className="text-sm font-medium text-text-secondary">No audits yet</p>
          <p className="text-xs text-text-muted mt-1">Search for a company above to get started</p>
        </div>
      ) : (
        <div className="space-y-3">
          {audits.map((audit) => (
            <Link
              key={audit.id}
              href={`/audit/${slugify(audit.companyName)}`}
              className="group flex items-center gap-4 rounded-xl border border-[#111827] bg-[#0f1829] p-4 hover:border-[#1e293b] hover:bg-[#111827] transition-all"
            >
              <CompanyLogo domain={audit.companyDomain} name={audit.companyName} />

              <div className="flex-1 min-w-0">
                <div className="font-display font-semibold text-sm text-text-primary truncate">
                  {audit.companyName}
                </div>
                <div className="text-xs text-text-muted mt-0.5 flex items-center gap-1.5">
                  {audit.industry && <span>{audit.industry}</span>}
                  {audit.industry && <span>·</span>}
                  <span>{formatDate(audit.createdAt)}</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className={`text-xs font-mono font-semibold px-2.5 py-1 rounded-full border ${getRiskColor(audit.riskLevel)}`}>
                  {audit.riskLevel.toUpperCase()}
                </span>
                <ChevronRight className="h-4 w-4 text-text-muted group-hover:text-text-secondary transition-colors" />
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
