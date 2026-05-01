import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'

async function checkAndIncrementUsage(userId: string, plan: string) {
  const today = new Date().toISOString().split('T')[0]
  const limits: Record<string, number> = { FREE: 3, PRO: 20, ENTERPRISE: Infinity }
  const limit = limits[plan] ?? 3

  const usage = await prisma.dailyUsage.findUnique({
    where: { userId_date: { userId, date: today } },
  })

  if (limit !== Infinity && (usage?.count ?? 0) >= limit) {
    return { allowed: false, current: usage?.count ?? 0, limit }
  }

  await prisma.dailyUsage.upsert({
    where: { userId_date: { userId, date: today } },
    update: { count: { increment: 1 } },
    create: { userId, date: today, count: 1 },
  })

  return { allowed: true, current: (usage?.count ?? 0) + 1, limit }
}

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { company, industry, hq, size, companyDomain } = await req.json()
  if (!company) {
    return NextResponse.json({ error: 'Company name required' }, { status: 400 })
  }

  const plan = session.user.plan || 'FREE'
  const ENGINE_TIER: Record<string, string> = { FREE: 'Standard', PRO: 'Advanced', ENTERPRISE: 'Premium' }
  const engineTier = ENGINE_TIER[plan] || 'Standard'

  const usageCheck = await checkAndIncrementUsage(session.user.id, plan)
  if (!usageCheck.allowed) {
    return NextResponse.json(
      { error: 'DAILY_LIMIT_REACHED', limit: usageCheck.limit, current: usageCheck.current },
      { status: 429 }
    )
  }

  try {
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    const backendRes = await fetch(`${backendUrl}/audit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ company, domain: companyDomain || '', industry, hq, size, userId: session.user.id, plan }),
    })

    if (!backendRes.ok) {
      const err = await backendRes.json().catch(() => ({}))
      return NextResponse.json(
        { error: err.detail || 'Audit service error' },
        { status: backendRes.status }
      )
    }

    const payload = await backendRes.json()
    // payload = { result: {...}, usage: { inputTokens, outputTokens, costUsd, model } }
    const auditResult = payload.result ?? payload
    const usage = payload.usage ?? {}

    const audit = await prisma.audit.create({
      data: {
        userId: session.user.id,
        companyName: company,
        companyDomain: auditResult.companyDomain,
        industry: industry || null,
        hq: hq || null,
        size: size || null,
        riskLevel: auditResult.riskLevel || 'Unknown',
        result: auditResult,
        inputTokens: usage.inputTokens ?? null,
        outputTokens: usage.outputTokens ?? null,
        costUsd: usage.costUsd ?? null,
        modelUsed: usage.model ?? null,
        plan,
      },
    })

    return NextResponse.json({ auditId: audit.id, engineTier, ...auditResult })
  } catch (err) {
    console.error('Audit error:', err)
    return NextResponse.json({ error: 'Failed to run audit' }, { status: 500 })
  }
}

export async function GET(req: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { searchParams } = new URL(req.url)
  const company = searchParams.get('company')

  if (company) {
    const audit = await prisma.audit.findFirst({
      where: { userId: session.user.id, companyName: { contains: company, mode: 'insensitive' } },
      orderBy: { createdAt: 'desc' },
    })
    return NextResponse.json(audit ? { ...(audit.result as Record<string, unknown>), auditId: audit.id } : null)
  }

  const audits = await prisma.audit.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: 'desc' },
    take: 10,
    select: {
      id: true, companyName: true, companyDomain: true,
      riskLevel: true, industry: true, createdAt: true,
    },
  })
  return NextResponse.json(audits)
}
