'use client'

import { useState, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { Search, Loader2, Building2, MapPin, Users, Briefcase } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { slugify } from '@/lib/utils'

interface SearchModalProps {
  company: string
  onClose: () => void
  onConfirm: (data: { industry: string; hq: string; size: string }) => void
}

function SearchModal({ company, onClose, onConfirm }: SearchModalProps) {
  const [industry, setIndustry] = useState('')
  const [hq, setHq] = useState('')
  const [size, setSize] = useState('')

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl border border-[#1e293b] bg-[#0a0e1a] p-6 shadow-2xl">
        <div className="mb-5">
          <div className="flex items-center gap-2 mb-1">
            <Building2 className="h-5 w-5 text-accent-amber" />
            <h2 className="font-display text-xl font-semibold text-text-primary">
              Auditing <span className="text-accent-amber">{company}</span>
            </h2>
          </div>
          <p className="text-sm text-text-secondary">Provide details to improve the audit quality.</p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="flex items-center gap-1.5 mb-1.5 text-xs font-mono text-text-muted">
              <Briefcase className="h-3.5 w-3.5" /> Industry
            </label>
            <Input
              placeholder="e.g. Technology, Healthcare, Finance..."
              value={industry}
              onChange={(e) => setIndustry(e.target.value)}
            />
          </div>
          <div>
            <label className="flex items-center gap-1.5 mb-1.5 text-xs font-mono text-text-muted">
              <MapPin className="h-3.5 w-3.5" /> Headquarters
            </label>
            <Input
              placeholder="e.g. San Francisco, CA"
              value={hq}
              onChange={(e) => setHq(e.target.value)}
            />
          </div>
          <div>
            <label className="flex items-center gap-1.5 mb-1.5 text-xs font-mono text-text-muted">
              <Users className="h-3.5 w-3.5" /> Approx. Employees
            </label>
            <Input
              placeholder="e.g. 500-1000, 10000+"
              value={size}
              onChange={(e) => setSize(e.target.value)}
            />
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <Button variant="outline" className="flex-1" onClick={onClose}>
            Cancel
          </Button>
          <Button className="flex-1" onClick={() => onConfirm({ industry, hq, size })}>
            Run Audit
          </Button>
        </div>
      </div>
    </div>
  )
}

export function SearchBar() {
  const [query, setQuery] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const inputRef = useRef<HTMLInputElement>(null)

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setShowModal(true)
  }

  const handleConfirm = (details: { industry: string; hq: string; size: string }) => {
    setShowModal(false)
    setLoading(true)
    const slug = slugify(query.trim())
    const params = new URLSearchParams({
      ...(details.industry && { industry: details.industry }),
      ...(details.hq && { hq: details.hq }),
      ...(details.size && { size: details.size }),
    })
    router.push(`/audit/${slug}${params.toString() ? `?${params}` : ''}`)
  }

  return (
    <>
      <form onSubmit={handleSearch} className="relative">
        <div className="relative flex items-center">
          <Search className="absolute left-4 h-5 w-5 text-text-muted pointer-events-none z-10" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search any company…"
            className="w-full h-14 pl-12 pr-36 rounded-2xl border border-[#1e293b] bg-[#0a0e1a] text-text-primary placeholder:text-text-muted text-base focus:outline-none focus:ring-2 focus:ring-accent-amber focus:border-transparent transition-all hover:border-[#1e293b] glow-amber"
          />
          <button
            type="submit"
            disabled={!query.trim() || loading}
            className="absolute right-2 flex items-center gap-2 rounded-xl bg-accent-amber px-5 py-2.5 text-sm font-semibold text-bg-primary transition-all hover:bg-amber-400 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Audit
          </button>
        </div>
      </form>

      {showModal && (
        <SearchModal
          company={query}
          onClose={() => setShowModal(false)}
          onConfirm={handleConfirm}
        />
      )}
    </>
  )
}
