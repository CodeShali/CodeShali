'use client'

import { useSession, signOut } from 'next-auth/react'
import Image from 'next/image'
import Link from 'next/link'
import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ChevronDown, LogOut, Settings, CreditCard, Zap, Menu, X, Home, Clock } from 'lucide-react'

export function Navbar() {
  const { data: session } = useSession()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const plan = session?.user?.plan || 'FREE'
  const planBadgeVariant = plan === 'PRO' ? 'pro' : plan === 'ENTERPRISE' ? 'enterprise' : 'free'

  const handleBillingPortal = async () => {
    const res = await fetch('/api/stripe/portal', { method: 'POST' })
    const { url } = await res.json()
    if (url) window.location.href = url
  }

  return (
    <>
      <nav className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-[#111827] bg-[#060a12]/80 px-4 backdrop-blur-md md:px-6">
        <div className="flex items-center gap-3">
          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-2 rounded-lg hover:bg-[#0f1829] text-text-muted hover:text-text-secondary transition-colors"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>

          <Link href="/dashboard" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-amber/10 border border-accent-amber/20">
              <span className="font-display text-sm font-bold text-accent-amber">D</span>
            </div>
            <span className="font-display text-lg font-bold text-text-primary hidden sm:block">DataEcho</span>
          </Link>
        </div>

        <div className="flex items-center gap-3">
          {plan === 'FREE' && (
            <Link href="/dashboard?upgrade=true">
              <Button size="sm" variant="pro" className="hidden sm:flex gap-1.5 text-xs">
                <Zap className="h-3.5 w-3.5" />
                Upgrade to Pro
              </Button>
            </Link>
          )}

          {session?.user && (
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center gap-2 rounded-xl px-3 py-2 hover:bg-[#0f1829] transition-colors"
              >
                {session.user.image ? (
                  <Image src={session.user.image} alt="Avatar" width={28} height={28} className="rounded-full" />
                ) : (
                  <div className="h-7 w-7 rounded-full bg-accent-amber/20 flex items-center justify-center">
                    <span className="text-xs font-semibold text-accent-amber">
                      {session.user.name?.charAt(0) || session.user.email?.charAt(0)}
                    </span>
                  </div>
                )}
                <div className="hidden md:block text-left">
                  <div className="text-xs font-medium text-text-primary leading-none">
                    {session.user.name || session.user.email}
                  </div>
                  <Badge variant={planBadgeVariant as 'pro' | 'enterprise' | 'free'} className="mt-1 text-[10px] h-4 px-1.5">
                    {plan}
                  </Badge>
                </div>
                <ChevronDown className="h-3.5 w-3.5 text-text-muted" />
              </button>

              {dropdownOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setDropdownOpen(false)} />
                  <div className="absolute right-0 top-full mt-1 z-50 w-48 rounded-xl border border-[#1e293b] bg-[#0a0e1a] py-1 shadow-2xl">
                    {plan !== 'FREE' && (
                      <button
                        onClick={handleBillingPortal}
                        className="flex w-full items-center gap-2 px-3 py-2 text-sm text-text-secondary hover:bg-[#0f1829] hover:text-text-primary transition-colors"
                      >
                        <CreditCard className="h-4 w-4" />
                        Manage Billing
                      </button>
                    )}
                    <button className="flex w-full items-center gap-2 px-3 py-2 text-sm text-text-secondary hover:bg-[#0f1829] hover:text-text-primary transition-colors">
                      <Settings className="h-4 w-4" />
                      Settings
                    </button>
                    <div className="my-1 border-t border-[#111827]" />
                    <button
                      onClick={() => signOut({ callbackUrl: '/login' })}
                      className="flex w-full items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-400/10 transition-colors"
                    >
                      <LogOut className="h-4 w-4" />
                      Sign Out
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </nav>

      {/* Mobile slide-down menu */}
      {mobileMenuOpen && (
        <>
          <div
            className="fixed inset-0 z-30 bg-black/40 lg:hidden"
            onClick={() => setMobileMenuOpen(false)}
          />
          <div className="fixed top-16 left-0 right-0 z-30 border-b border-[#111827] bg-[#060a12] lg:hidden">
            <div className="px-4 py-4 space-y-1">
              <Link
                href="/dashboard"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-text-secondary hover:bg-[#0f1829] hover:text-text-primary transition-colors"
              >
                <Home className="h-4 w-4" />
                Home
              </Link>
              <Link
                href="/dashboard/history"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-text-secondary hover:bg-[#0f1829] hover:text-text-primary transition-colors"
              >
                <Clock className="h-4 w-4" />
                History
              </Link>

              {plan === 'FREE' && (
                <div className="pt-2 border-t border-[#111827]">
                  <Link href="/dashboard?upgrade=true" onClick={() => setMobileMenuOpen(false)}>
                    <Button variant="pro" className="w-full mt-2 gap-1.5 text-sm">
                      <Zap className="h-4 w-4" />
                      Upgrade to Pro — $12/mo
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  )
}
