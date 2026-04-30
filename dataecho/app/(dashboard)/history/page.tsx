import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import Link from 'next/link'
import Image from 'next/image'
import { formatDate, getRiskColor, slugify } from '@/lib/utils'
import { ChevronRight, Clock } from 'lucide-react'

export default async function HistoryPage() {
  const session = await getServerSession(authOptions)
  if (!session?.user?.id) redirect('/login')

  const plan = session.user.plan
  const daysLimit = plan === 'FREE' ? 7 : plan === 'PRO' ? 90 : null

  const createdAfter = daysLimit
    ? new Date(Date.now() - daysLimit * 24 * 60 * 60 * 1000)
    : undefined

  const audits = await prisma.audit.findMany({
    where: {
      userId: session.user.id,
      ...(createdAfter ? { createdAt: { gte: createdAfter } } : {}),
    },
    orderBy: { createdAt: 'desc' },
    take: plan === 'FREE' ? 10 : undefined,
  })

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <div className="flex items-center gap-2 mb-6">
        <Clock className="h-5 w-5 text-text-muted" />
        <h1 className="font-display text-xl font-semibold text-text-primary">Audit History</h1>
        {daysLimit && (
          <span className="text-xs font-mono text-text-muted ml-2">
            (last {daysLimit} days)
          </span>
        )}
      </div>

      {audits.length === 0 ? (
        <div className="text-center py-16 text-text-muted">No audits in history yet.</div>
      ) : (
        <div className="space-y-3">
          {audits.map((audit) => (
            <Link
              key={audit.id}
              href={`/audit/${slugify(audit.companyName)}`}
              className="group flex items-center gap-4 rounded-xl border border-[#111827] bg-[#0f1829] p-4 hover:border-[#1e293b] transition-all"
            >
              <div className="h-10 w-10 rounded-xl border border-[#1e293b] bg-white flex items-center justify-center flex-shrink-0 overflow-hidden">
                {audit.companyDomain ? (
                  <Image
                    src={`https://logo.clearbit.com/${audit.companyDomain}`}
                    alt={audit.companyName}
                    width={40}
                    height={40}
                    className="object-contain"
                    onError={undefined}
                  />
                ) : (
                  <span className="font-display text-lg font-bold text-text-secondary">
                    {audit.companyName.charAt(0)}
                  </span>
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="font-semibold text-sm text-text-primary">{audit.companyName}</div>
                <div className="text-xs text-text-muted mt-0.5">
                  {audit.industry && `${audit.industry} · `}
                  {formatDate(audit.createdAt.toISOString())}
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className={`text-xs font-mono font-semibold px-2.5 py-1 rounded-full border ${getRiskColor(audit.riskLevel)}`}>
                  {audit.riskLevel.toUpperCase()}
                </span>
                <ChevronRight className="h-4 w-4 text-text-muted group-hover:text-text-secondary" />
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
