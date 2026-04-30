export function AuditSkeleton() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-6 animate-pulse">
      {/* Header */}
      <div className="flex items-center gap-5">
        <div className="h-16 w-16 rounded-2xl skeleton flex-shrink-0" />
        <div className="flex-1 space-y-3">
          <div className="h-7 w-48 rounded-lg skeleton" />
          <div className="h-4 w-72 rounded skeleton" />
        </div>
      </div>

      {/* Risk banner */}
      <div className="h-24 rounded-2xl skeleton" />

      {/* Tabs */}
      <div className="flex gap-2">
        {[1, 2, 3].map((i) => <div key={i} className="h-10 w-36 rounded-xl skeleton" />)}
      </div>

      {/* Cards */}
      <div className="space-y-3">
        {[1, 2, 3, 4].map((i) => <div key={i} className="h-20 rounded-xl skeleton" />)}
      </div>
    </div>
  )
}
