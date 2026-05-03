import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { SearchBar } from '@/components/dashboard/SearchBar'
import { UsageCounter } from '@/components/dashboard/UsageCounter'
import { RecentFeed } from '@/components/dashboard/RecentFeed'
import { PaywallModal } from '@/components/shared/PaywallModal'

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: { upgrade?: string; upgraded?: string; canceled?: string }
}) {
  const session = await getServerSession(authOptions)
  const showUpgrade = searchParams.upgrade === 'true'
  const justUpgraded = searchParams.upgraded === 'true'

  return (
    <div className="max-w-3xl mx-auto px-4 py-10 md:py-16">
      {/* Hero search */}
      <div className="mb-10 text-center">
        <h1 className="font-display text-3xl md:text-4xl font-bold text-text-primary mb-2">
          What does AI know about{' '}
          <span className="text-accent-amber">your company?</span>
        </h1>
        <p className="text-text-secondary mb-8 text-sm md:text-base">
          Search any company to see LLM training data, known breaches, and recently shared information.
        </p>
        <SearchBar />
      </div>

      {/* Usage pill */}
      <div className="flex justify-center mb-10">
        <UsageCounter />
      </div>

      {/* Success/cancel toasts */}
      {justUpgraded && (
        <div className="mb-6 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700 flex items-center gap-2">
          <span>🎉</span>
          <span>Your plan has been upgraded successfully!</span>
        </div>
      )}

      {/* Recent feed */}
      <RecentFeed />

      {/* Paywall modal if ?upgrade=true */}
      {showUpgrade && <PaywallModal defaultOpen />}
    </div>
  )
}
