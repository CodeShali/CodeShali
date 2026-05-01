import { Navbar } from '@/components/shared/Navbar'

export default function PublicLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-[#060a12]">
      <Navbar />
      <main>{children}</main>
    </div>
  )
}
