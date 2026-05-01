import Stripe from 'stripe'

let _stripe: Stripe | null = null

export function getStripe(): Stripe {
  if (!_stripe) {
    if (!process.env.STRIPE_SECRET_KEY) throw new Error('STRIPE_SECRET_KEY is not set')
    _stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: '2025-02-24.acacia' })
  }
  return _stripe
}

// Backwards-compatible named export used by existing route handlers
export const stripe = new Proxy({} as Stripe, {
  get(_target, prop) {
    return (getStripe() as unknown as Record<string | symbol, unknown>)[prop]
  },
})

export const PLANS = {
  PRO: {
    name: 'Pro',
    monthly: {
      priceId: process.env.STRIPE_PRO_MONTHLY_PRICE_ID!,
      price: 12,
      interval: 'month' as const,
    },
    yearly: {
      priceId: process.env.STRIPE_PRO_YEARLY_PRICE_ID!,
      price: 99,
      interval: 'year' as const,
    },
    features: [
      '20 audits/day',
      'Unlimited card expansions',
      'Full detail all 3 layers',
      'Unlimited chat per audit',
      'PDF report export',
      'Audit history (90 days)',
    ],
  },
  ENTERPRISE: {
    name: 'Enterprise',
    monthly: {
      priceId: process.env.STRIPE_ENTERPRISE_MONTHLY_PRICE_ID!,
      price: 49,
      interval: 'month' as const,
    },
    yearly: {
      priceId: process.env.STRIPE_ENTERPRISE_YEARLY_PRICE_ID!,
      price: 399,
      interval: 'year' as const,
    },
    features: [
      'Unlimited audits',
      'Bulk company scanning',
      'API access',
      '5 team seats',
      'Priority support',
      'Unlimited audit history',
    ],
  },
}
