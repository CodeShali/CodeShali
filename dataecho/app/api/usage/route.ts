import { NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { PLAN_LIMITS } from '@/lib/utils'

export async function GET() {
  const session = await getServerSession(authOptions)
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const today = new Date().toISOString().split('T')[0]
  const usage = await prisma.dailyUsage.findUnique({
    where: { userId_date: { userId: session.user.id, date: today } },
  })

  const plan = (session.user.plan as keyof typeof PLAN_LIMITS) || 'FREE'
  const limits = PLAN_LIMITS[plan]

  return NextResponse.json({
    current: usage?.count ?? 0,
    limit: limits.auditsPerDay,
    plan,
    remaining: limits.auditsPerDay === Infinity ? Infinity : Math.max(0, limits.auditsPerDay - (usage?.count ?? 0)),
  })
}
