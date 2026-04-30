import { NextRequest, NextResponse } from 'next/server'
import { stripe } from '@/lib/stripe'
import { prisma } from '@/lib/prisma'
import Stripe from 'stripe'

export async function POST(req: NextRequest) {
  const body = await req.text()
  const sig = req.headers.get('stripe-signature')!

  let event: Stripe.Event
  try {
    event = stripe.webhooks.constructEvent(body, sig, process.env.STRIPE_WEBHOOK_SECRET!)
  } catch (err) {
    return NextResponse.json({ error: 'Invalid signature' }, { status: 400 })
  }

  try {
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session
        if (session.mode === 'subscription' && session.customer && session.subscription) {
          const subscription = await stripe.subscriptions.retrieve(session.subscription as string)
          const priceId = subscription.items.data[0]?.price.id
          const plan = getPlanFromPriceId(priceId)
          await prisma.user.updateMany({
            where: { stripeCustomerId: session.customer as string },
            data: { plan, stripeSubscriptionId: session.subscription as string },
          })
        }
        break
      }
      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription
        const priceId = subscription.items.data[0]?.price.id
        const plan = getPlanFromPriceId(priceId)
        const status = subscription.status
        await prisma.user.updateMany({
          where: { stripeSubscriptionId: subscription.id },
          data: { plan: status === 'active' ? plan : 'FREE' },
        })
        break
      }
      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription
        await prisma.user.updateMany({
          where: { stripeSubscriptionId: subscription.id },
          data: { plan: 'FREE', stripeSubscriptionId: null },
        })
        break
      }
    }
  } catch (err) {
    console.error('Stripe webhook error:', err)
    return NextResponse.json({ error: 'Webhook handler failed' }, { status: 500 })
  }

  return NextResponse.json({ received: true })
}

function getPlanFromPriceId(priceId: string): 'FREE' | 'PRO' | 'ENTERPRISE' {
  const proIds = [
    process.env.STRIPE_PRO_MONTHLY_PRICE_ID,
    process.env.STRIPE_PRO_YEARLY_PRICE_ID,
  ]
  const enterpriseIds = [
    process.env.STRIPE_ENTERPRISE_MONTHLY_PRICE_ID,
    process.env.STRIPE_ENTERPRISE_YEARLY_PRICE_ID,
  ]
  if (proIds.includes(priceId)) return 'PRO'
  if (enterpriseIds.includes(priceId)) return 'ENTERPRISE'
  return 'FREE'
}
