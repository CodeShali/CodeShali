'use client'

import { useState } from 'react'
import { Check, Zap, Crown, X, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PLANS } from '@/lib/stripe'

interface PaywallModalProps {
  defaultOpen?: boolean
  trigger?: React.ReactNode
  onClose?: () => void
}

type BillingInterval = 'monthly' | 'yearly'

export function PaywallModal({ defaultOpen = false, trigger, onClose }: PaywallModalProps) {
  const [open, setOpen] = useState(defaultOpen)

  const handleClose = () => {
    setOpen(false)
    onClose?.()
  }
  const [interval, setInterval] = useState<BillingInterval>('monthly')
  const [loading, setLoading] = useState<string | null>(null)

  const handleCheckout = async (planKey: string, priceId: string) => {
    setLoading(planKey)
    try {
      const res = await fetch('/api/stripe/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priceId }),
      })
      const data = await res.json()
      if (data.url) {
        window.location.href = data.url
      }
    } catch {
      setLoading(null)
    }
  }

  const plans = [
    {
      key: 'PRO',
      name: 'Pro',
      icon: Zap,
      iconColor: 'text-amber-400',
      price: interval === 'monthly' ? PLANS.PRO.monthly.price : PLANS.PRO.yearly.price,
      priceId: interval === 'monthly' ? PLANS.PRO.monthly.priceId : PLANS.PRO.yearly.priceId,
      perMonth: interval === 'yearly' ? Math.round(PLANS.PRO.yearly.price / 12) : PLANS.PRO.monthly.price,
      features: PLANS.PRO.features,
      accent: 'border-amber-400/30 bg-amber-400/5',
      buttonVariant: 'pro' as const,
    },
    {
      key: 'ENTERPRISE',
      name: 'Enterprise',
      icon: Crown,
      iconColor: 'text-purple-400',
      price: interval === 'monthly' ? PLANS.ENTERPRISE.monthly.price : PLANS.ENTERPRISE.yearly.price,
      priceId: interval === 'monthly' ? PLANS.ENTERPRISE.monthly.priceId : PLANS.ENTERPRISE.yearly.priceId,
      perMonth: interval === 'yearly' ? Math.round(PLANS.ENTERPRISE.yearly.price / 12) : PLANS.ENTERPRISE.monthly.price,
      features: PLANS.ENTERPRISE.features,
      accent: 'border-purple-400/30 bg-purple-400/5',
      buttonVariant: 'outline' as const,
    },
  ]

  if (!open && trigger) {
    return <div onClick={() => setOpen(true)}>{trigger}</div>
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-2xl border border-[#1e293b] bg-[#0a0e1a] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="relative px-6 pt-6 pb-4 text-center border-b border-[#111827]">
          <button
            onClick={handleClose}
            className="absolute right-4 top-4 p-1.5 rounded-lg hover:bg-[#0f1829] text-text-muted hover:text-text-secondary transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-amber/10 border border-accent-amber/20 mb-3">
            <Zap className="h-6 w-6 text-accent-amber" />
          </div>
          <h2 className="font-display text-2xl font-bold text-text-primary mb-1">
            Unlock Full Access
          </h2>
          <p className="text-sm text-text-secondary">
            Get unlimited audits, full details, and unrestricted chat.
          </p>

          {/* Billing toggle */}
          <div className="flex items-center justify-center gap-3 mt-4">
            <button
              onClick={() => setInterval('monthly')}
              className={`text-sm font-medium px-4 py-1.5 rounded-full transition-colors ${
                interval === 'monthly'
                  ? 'bg-[#0f1829] text-text-primary border border-[#1e293b]'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setInterval('yearly')}
              className={`text-sm font-medium px-4 py-1.5 rounded-full transition-colors flex items-center gap-1.5 ${
                interval === 'yearly'
                  ? 'bg-[#0f1829] text-text-primary border border-[#1e293b]'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              Yearly
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-green-400/20 text-green-400">
                -30%
              </span>
            </button>
          </div>
        </div>

        {/* Plan cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-6">
          {plans.map((plan) => {
            const Icon = plan.icon
            return (
              <div
                key={plan.key}
                className={`rounded-xl border p-5 ${plan.accent}`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <Icon className={`h-5 w-5 ${plan.iconColor}`} />
                  <span className="font-display font-semibold text-text-primary">{plan.name}</span>
                </div>

                <div className="mb-4">
                  <span className="font-mono text-3xl font-bold text-text-primary">
                    ${plan.perMonth}
                  </span>
                  <span className="text-sm text-text-muted">/mo</span>
                  {interval === 'yearly' && (
                    <div className="text-xs text-text-muted mt-0.5">
                      billed ${plan.price}/year
                    </div>
                  )}
                </div>

                <ul className="space-y-2 mb-5">
                  {plan.features.map((f, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                      <Check className="h-3.5 w-3.5 flex-shrink-0 mt-0.5 text-green-400" />
                      {f}
                    </li>
                  ))}
                </ul>

                <Button
                  variant={plan.buttonVariant}
                  className="w-full"
                  disabled={loading !== null}
                  onClick={() => handleCheckout(plan.key, plan.priceId)}
                >
                  {loading === plan.key ? (
                    <><Loader2 className="h-4 w-4 animate-spin mr-2" />Processing…</>
                  ) : (
                    `Get ${plan.name}`
                  )}
                </Button>
              </div>
            )
          })}
        </div>

        <div className="px-6 pb-4 text-center">
          <p className="text-xs text-text-muted">
            Cancel anytime · Secure payments via Stripe · 30-day money-back guarantee
          </p>
        </div>
      </div>
    </div>
  )
}
