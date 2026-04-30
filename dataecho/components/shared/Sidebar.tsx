'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useSession } from 'next-auth/react'
import { Home, Clock, Zap, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

const navItems = [
  { href: '/dashboard', label: 'Home', icon: Home },
  { href: '/dashboard/history', label: 'History', icon: Clock },
]

export function Sidebar() {
  const pathname = usePathname()
  const { data: session } = useSession()
  const plan = session?.user?.plan || 'FREE'

  return (
    <aside className="hidden lg:flex flex-col w-60 min-h-screen border-r border-[#111827] bg-[#060a12]">
      <div className="flex-1 py-6 px-3">
        <nav className="space-y-1">
          {navItems.map(({ href, label, icon: Icon }) => {
            const active = pathname === href
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  'flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors',
                  active
                    ? 'bg-accent-amber/10 text-accent-amber border border-accent-amber/20'
                    : 'text-text-secondary hover:bg-[#0f1829] hover:text-text-primary'
                )}
              >
                <Icon className="h-4 w-4 flex-shrink-0" />
                {label}
                {active && <ChevronRight className="ml-auto h-3.5 w-3.5" />}
              </Link>
            )
          })}
        </nav>
      </div>

      {/* Plan upgrade CTA */}
      {plan === 'FREE' && (
        <div className="p-3 pb-6">
          <div className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="h-4 w-4 text-accent-amber" />
              <span className="text-sm font-semibold text-text-primary">Upgrade to Pro</span>
            </div>
            <p className="text-xs text-text-secondary mb-3">
              Get 20 audits/day, unlimited expansions, and full details.
            </p>
            <Link href="/dashboard?upgrade=true">
              <Button size="sm" variant="pro" className="w-full text-xs">
                View Plans
              </Button>
            </Link>
          </div>
        </div>
      )}

      {plan !== 'FREE' && (
        <div className="p-3 pb-6">
          <div className="flex items-center gap-2 px-3 py-2">
            <Badge variant={plan === 'PRO' ? 'pro' : 'enterprise'} className="text-xs">
              {plan}
            </Badge>
            <span className="text-xs text-text-muted">plan active</span>
          </div>
        </div>
      )}
    </aside>
  )
}
