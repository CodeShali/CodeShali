import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'
import { authOptions } from '@/lib/auth'
import { LoginForm } from '@/components/auth/LoginForm'

export default async function LoginPage() {
  const session = await getServerSession(authOptions)
  if (session) redirect('/dashboard')

  return (
    <main className="min-h-screen flex items-center justify-center bg-[#060a12] px-4">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_rgba(245,158,11,0.04)_0%,_transparent_70%)]" />
      <LoginForm />
    </main>
  )
}
