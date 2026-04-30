import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { PLAN_LIMITS } from '@/lib/utils'

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { message, context, messageCount } = await req.json()
  const plan = (session.user.plan as keyof typeof PLAN_LIMITS) || 'FREE'
  const chatLimit = PLAN_LIMITS[plan].chatMessages

  if (chatLimit !== Infinity && messageCount >= chatLimit) {
    return NextResponse.json({ error: 'CHAT_LIMIT_REACHED' }, { status: 429 })
  }

  try {
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    const res = await fetch(`${backendUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, context }),
    })

    if (!res.ok) {
      return NextResponse.json({ error: 'Chat service error' }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json({ error: 'Chat service unavailable' }, { status: 503 })
  }
}
