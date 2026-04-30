import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'

const ADMIN_EMAILS = (process.env.ADMIN_EMAILS || '').split(',').map((e) => e.trim()).filter(Boolean)

// Monthly revenue per plan (USD)
const PLAN_REVENUE: Record<string, number> = {
  FREE: 0,
  PRO: 12,
  ENTERPRISE: 49,
}

export async function GET(req: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session?.user?.email || !ADMIN_EMAILS.includes(session.user.email)) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }

  const { searchParams } = new URL(req.url)
  const days = parseInt(searchParams.get('days') || '30', 10)
  const since = new Date()
  since.setDate(since.getDate() - days)

  // All audits in period with cost data
  const audits = await prisma.audit.findMany({
    where: { createdAt: { gte: since } },
    select: {
      id: true,
      plan: true,
      costUsd: true,
      inputTokens: true,
      outputTokens: true,
      modelUsed: true,
      createdAt: true,
    },
    orderBy: { createdAt: 'asc' },
  })

  // User plan breakdown
  const users = await prisma.user.findMany({
    select: { plan: true },
  })

  const planCounts = users.reduce<Record<string, number>>((acc, u) => {
    acc[u.plan] = (acc[u.plan] || 0) + 1
    return acc
  }, {})

  // Monthly recurring revenue estimate
  const mrr = Object.entries(planCounts).reduce((sum, [plan, count]) => {
    return sum + (PLAN_REVENUE[plan] || 0) * count
  }, 0)

  // Cost aggregation by plan
  const costByPlan: Record<string, { audits: number; cost: number; inputTokens: number; outputTokens: number }> = {}
  let totalCost = 0

  for (const audit of audits) {
    const plan = audit.plan || 'FREE'
    if (!costByPlan[plan]) {
      costByPlan[plan] = { audits: 0, cost: 0, inputTokens: 0, outputTokens: 0 }
    }
    costByPlan[plan].audits += 1
    costByPlan[plan].cost += audit.costUsd || 0
    costByPlan[plan].inputTokens += audit.inputTokens || 0
    costByPlan[plan].outputTokens += audit.outputTokens || 0
    totalCost += audit.costUsd || 0
  }

  // Daily cost breakdown (last N days)
  const dailyCosts: Record<string, number> = {}
  for (const audit of audits) {
    const day = audit.createdAt.toISOString().split('T')[0]
    dailyCosts[day] = (dailyCosts[day] || 0) + (audit.costUsd || 0)
  }

  // Model usage breakdown
  const modelBreakdown: Record<string, { audits: number; cost: number }> = {}
  for (const audit of audits) {
    const model = audit.modelUsed || 'unknown'
    if (!modelBreakdown[model]) modelBreakdown[model] = { audits: 0, cost: 0 }
    modelBreakdown[model].audits += 1
    modelBreakdown[model].cost += audit.costUsd || 0
  }

  // Revenue in period (rough estimate: MRR * days / 30)
  const revenueInPeriod = Math.round(mrr * days / 30 * 100) / 100

  const profitInPeriod = Math.round((revenueInPeriod - totalCost) * 100) / 100
  const profitMarginPct = revenueInPeriod > 0
    ? Math.round((profitInPeriod / revenueInPeriod) * 10000) / 100
    : 0

  return NextResponse.json({
    period: { days, since: since.toISOString() },
    users: { total: users.length, byPlan: planCounts },
    revenue: { mrr, inPeriod: revenueInPeriod },
    costs: {
      totalUsd: Math.round(totalCost * 10000) / 10000,
      byPlan: costByPlan,
      byModel: modelBreakdown,
      daily: dailyCosts,
    },
    profit: {
      inPeriod: profitInPeriod,
      marginPct: profitMarginPct,
    },
    auditCount: audits.length,
  })
}
