'use client'

import { useEffect, useState } from 'react'
import { useSession } from 'next-auth/react'
import { useParams, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, Share2, Download, RefreshCw, AlertCircle } from 'lucide-react'
import { deslugify } from '@/lib/utils'
import { CompanyHeader } from '@/components/audit/CompanyHeader'
import { RiskBanner } from '@/components/audit/RiskBanner'
import { LayerSection } from '@/components/audit/LayerSection'
import { AuditSkeleton } from '@/components/audit/AuditSkeleton'
import { ChatWidget } from '@/components/audit/ChatWidget'
import { PaywallModal } from '@/components/shared/PaywallModal'
import { Button } from '@/components/ui/button'
import { useToast } from '@/components/ui/use-toast'

interface AuditResult {
  auditId?: string
  companyDomain?: string
  riskLevel: string
  riskScore: number
  riskReason: string
  llmKnowledge?: {
    summary: string
    categories: Array<{
      label: string
      summary: string
      items: string[]
      detail: string
    }>
  }
  breaches?: Array<{
    year: string
    title: string
    summary: string
    detail: string
    dataExposed: string[]
    recordsAffected: string
    source: string
  }>
  userSharedData?: Array<{
    platform: string
    date: string
    summary: string
    detail: string
    sensitivity: string
    url: string
  }>
  publicDataSources?: string[]
}

export default function AuditPage() {
  const params = useParams()
  const searchParams = useSearchParams()
  const { data: session } = useSession()
  const { toast } = useToast()

  const companySlug = params.company as string
  const companyName = deslugify(companySlug)
  const industry = searchParams.get('industry') || ''
  const hq = searchParams.get('hq') || ''
  const size = searchParams.get('size') || ''

  const [auditData, setAuditData] = useState<AuditResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showPaywall, setShowPaywall] = useState(false)

  const plan = session?.user?.plan || 'FREE'

  const runAudit = async () => {
    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/api/audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company: companyName, industry, hq, size }),
      })

      const data = await res.json()

      if (!res.ok) {
        if (data.error === 'DAILY_LIMIT_REACHED') {
          setError('daily_limit')
          toast({ title: 'Daily limit reached', description: 'Upgrade to Pro for more audits.', variant: 'destructive' })
        } else {
          setError(data.error || 'Failed to run audit')
        }
        return
      }

      setAuditData(data)
      toast({ title: 'Audit complete', description: `${companyName} audit finished.` })
    } catch (e) {
      setError('Network error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    runAudit()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [companySlug])

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href)
    toast({ title: 'Link copied', description: 'Shareable audit link copied to clipboard.' })
  }

  if (loading) return <AuditSkeleton />

  if (error === 'daily_limit') {
    return (
      <div className="max-w-2xl mx-auto px-4 py-20 text-center">
        <div className="text-5xl mb-4">⚡</div>
        <h2 className="font-display text-2xl font-bold text-text-primary mb-2">Daily limit reached</h2>
        <p className="text-text-secondary mb-6">You've used all your free audits for today. Upgrade for more.</p>
        <div className="flex gap-3 justify-center">
          <Button variant="outline" asChild>
            <Link href="/dashboard"><ArrowLeft className="h-4 w-4 mr-1.5" />Back</Link>
          </Button>
          <Button variant="pro" onClick={() => setShowPaywall(true)}>
            Upgrade to Pro
          </Button>
        </div>
        {showPaywall && <PaywallModal defaultOpen onClose={() => setShowPaywall(false)} />}
      </div>
    )
  }

  if (error || !auditData) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-20 text-center">
        <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
        <h2 className="font-display text-2xl font-bold text-text-primary mb-2">Audit failed</h2>
        <p className="text-text-secondary mb-6">{error || 'Something went wrong running the audit.'}</p>
        <div className="flex gap-3 justify-center">
          <Button variant="outline" asChild>
            <Link href="/dashboard"><ArrowLeft className="h-4 w-4 mr-1.5" />Back</Link>
          </Button>
          <Button onClick={runAudit}>
            <RefreshCw className="h-4 w-4 mr-1.5" />
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  return (
    <>
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
        {/* Back + actions */}
        <div className="flex items-center justify-between">
          <Link
            href="/dashboard"
            className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text-secondary transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleShare} className="gap-1.5 text-xs">
              <Share2 className="h-3.5 w-3.5" />
              Share
            </Button>
            {plan !== 'FREE' && (
              <Button variant="ghost" size="sm" className="gap-1.5 text-xs">
                <Download className="h-3.5 w-3.5" />
                PDF
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={runAudit} className="gap-1.5 text-xs">
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh
            </Button>
          </div>
        </div>

        {/* Company header */}
        <div className="rounded-2xl border border-[#111827] bg-[#0f1829] p-6">
          <CompanyHeader
            companyName={companyName}
            companyDomain={auditData.companyDomain || null}
            industry={industry}
            hq={hq}
            size={size}
            riskLevel={auditData.riskLevel}
          />
        </div>

        {/* Risk banner */}
        <RiskBanner
          riskLevel={auditData.riskLevel}
          riskScore={auditData.riskScore}
          riskReason={auditData.riskReason}
        />

        {/* Three-layer sections */}
        <div className="rounded-2xl border border-[#111827] bg-[#0f1829] p-6">
          <LayerSection
            data={auditData}
            plan={plan}
            onPaywall={() => setShowPaywall(true)}
          />
        </div>
      </div>

      {/* Floating chat widget */}
      <ChatWidget
        auditContext={{ ...auditData, companyName }}
        companyName={companyName}
        plan={plan}
        onPaywall={() => setShowPaywall(true)}
      />

      {/* Paywall modal */}
      {showPaywall && <PaywallModal defaultOpen onClose={() => setShowPaywall(false)} />}
    </>
  )
}
