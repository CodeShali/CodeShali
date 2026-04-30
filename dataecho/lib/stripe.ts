import Stripe from 'stripe'

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2024-12-18.acacia',
  typescript: true,
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
