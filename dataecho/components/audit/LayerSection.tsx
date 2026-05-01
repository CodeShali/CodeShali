'use client'

import { useState } from 'react'
import { Brain, ShieldOff, Globe, Server, CheckCircle2 } from 'lucide-react'
import { DataCard } from './DataCard'

interface LLMCategory {
  label: string
  summary: string
  items: string[]
  detail: string
}

interface Breach {
  year: string
  title: string
  summary: string
  detail: string
  dataExposed: string[]
  recordsAffected: string
  source: string
}

interface VerifiedBreach {
  name: string
  title: string
  date: string
  recordCount: number
  dataClasses: string[]
  isVerified: boolean
  description: string
}

interface SharedData {
  platform: string
  date: string
  summary: string
  detail: string
  sensitivity: string
  url: string
}

interface GitHubLeak {
  repo: string
  file: string
  url: string
  repoUrl: string
}

interface InfraExposure {
  ip?: string
  openPorts?: number[]
  knownVulns?: string[]
  services?: string[]
  hostnames?: string[]
}

interface AuditData {
  companyDomain?: string
  llmKnowledge?: { summary: string; categories: LLMCategory[] }
  breaches?: Breach[]
  verifiedBreaches?: VerifiedBreach[]
  userSharedData?: SharedData[]
  githubLeaks?: GitHubLeak[]
  exposedSubdomains?: string[]
  infraExposure?: InfraExposure
  publicDataSources?: string[]
}

interface LayerSectionProps {
  data: AuditData
  plan: string
  onPaywall: () => void
}

const LAYER_CONFIG = [
  {
    id: 'llm',
    label: 'LLM Knowledge',
    sublabel: 'What AI knows from training data',
    icon: Brain,
    color: '#38bdf8',
    border: 'border-sky-500/30',
    bg: 'bg-sky-500/5',
  },
  {
    id: 'breaches',
    label: 'Breach Database',
    sublabel: 'Known data breaches and incidents',
    icon: ShieldOff,
    color: '#f87171',
    border: 'border-red-500/30',
    bg: 'bg-red-500/5',
  },
  {
    id: 'shared',
    label: 'Leaked Data',
    sublabel: 'Publicly shared or leaked information',
    icon: Globe,
    color: '#a78bfa',
    border: 'border-purple-500/30',
    bg: 'bg-purple-500/5',
  },
  {
    id: 'infra',
    label: 'Infrastructure',
    sublabel: 'Exposed services and subdomains',
    icon: Server,
    color: '#fb923c',
    border: 'border-orange-500/30',
    bg: 'bg-orange-500/5',
  },
]

function VerifiedBadge() {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-green-400/10 border border-green-400/20 text-green-400">
      <CheckCircle2 className="h-2.5 w-2.5" />
      VERIFIED
    </span>
  )
}

function InferredBadge() {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-sky-400/10 border border-sky-400/20 text-sky-400">
      AI INFERRED
    </span>
  )
}

export function LayerSection({ data, plan, onPaywall }: LayerSectionProps) {
  const [activeTab, setActiveTab] = useState('llm')
  const [expansionCounts, setExpansionCounts] = useState<Record<string, number>>({})

  const handleExpand = (cardId: string) => {
    setExpansionCounts((prev) => ({
      ...prev,
      [cardId]: (prev[cardId] ?? 0) + 1,
    }))
  }

  const getTotalExpansions = () =>
    Object.values(expansionCounts).reduce((a, b) => a + b, 0)

  const verifiedBreachCount = (data.verifiedBreaches || []).length
  const hasInfra =
    (data.exposedSubdomains || []).length > 0 ||
    (data.infraExposure?.openPorts || []).length > 0 ||
    (data.infraExposure?.knownVulns || []).length > 0

  return (
    <div>
      {/* Tab headers */}
      <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-hide pb-1">
        {LAYER_CONFIG.map((layer) => {
          const Icon = layer.icon
          const isActive = activeTab === layer.id

          let badge: number | null = null
          if (layer.id === 'breaches' && verifiedBreachCount > 0) badge = verifiedBreachCount
          if (layer.id === 'infra' && (data.exposedSubdomains || []).length > 0)
            badge = (data.exposedSubdomains || []).length

          return (
            <button
              key={layer.id}
              onClick={() => setActiveTab(layer.id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium whitespace-nowrap transition-all border ${
                isActive
                  ? `${layer.bg} ${layer.border} text-white`
                  : 'border-[#111827] text-text-muted hover:text-text-secondary hover:bg-[#0f1829]'
              }`}
            >
              <Icon className="h-4 w-4" style={isActive ? { color: layer.color } : {}} />
              <span>{layer.label}</span>
              {badge !== null && badge > 0 && (
                <span
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded-full ml-1"
                  style={{
                    backgroundColor: `${layer.color}20`,
                    color: layer.color,
                    border: `1px solid ${layer.color}40`,
                  }}
                >
                  {badge}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* Layer content */}
      <div className="animate-fade-in">

        {/* LLM Knowledge */}
        {activeTab === 'llm' && data.llmKnowledge && (
          <div className="space-y-3">
            {data.llmKnowledge.summary && (
              <p className="text-sm text-text-secondary leading-relaxed px-1 mb-4">
                {data.llmKnowledge.summary}
              </p>
            )}
            {(data.llmKnowledge.categories || []).map((cat, i) => (
              <div
                key={i}
                className="animate-fade-in"
                style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'both' }}
              >
                <DataCard
                  label={cat.label}
                  summary={cat.summary}
                  detail={cat.detail}
                  items={cat.items}
                  plan={plan}
                  expansionCount={getTotalExpansions()}
                  onExpand={() => handleExpand(`llm-${i}`)}
                  onPaywall={onPaywall}
                  accentColor="#38bdf8"
                />
              </div>
            ))}
            {(!data.llmKnowledge.categories || data.llmKnowledge.categories.length === 0) && (
              <EmptyState message="No LLM knowledge data found for this company." />
            )}
          </div>
        )}

        {/* Breach Database */}
        {activeTab === 'breaches' && (
          <div className="space-y-4">
            {/* Verified HIBP breaches */}
            {verifiedBreachCount > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 px-1">
                  <VerifiedBadge />
                  <span className="text-xs text-text-muted">From Have I Been Pwned — authoritative breach database</span>
                </div>
                {(data.verifiedBreaches || []).map((breach, i) => (
                  <div
                    key={i}
                    className="animate-fade-in"
                    style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'both' }}
                  >
                    <DataCard
                      label={breach.title || breach.name}
                      summary={`${breach.recordCount.toLocaleString()} records exposed on ${breach.date}`}
                      detail={breach.description}
                      dataExposed={breach.dataClasses}
                      year={breach.date?.split('-')[0]}
                      plan={plan}
                      expansionCount={getTotalExpansions()}
                      onExpand={() => handleExpand(`verified-breach-${i}`)}
                      onPaywall={onPaywall}
                      accentColor="#4ade80"
                    />
                  </div>
                ))}
              </div>
            )}

            {/* AI-inferred breaches */}
            {(data.breaches || []).length > 0 && (
              <div className="space-y-3">
                {verifiedBreachCount > 0 && (
                  <div className="flex items-center gap-2 px-1 pt-2 border-t border-[#111827]">
                    <InferredBadge />
                    <span className="text-xs text-text-muted">Additional context from AI web search</span>
                  </div>
                )}
                {(data.breaches || []).map((breach, i) => (
                  <div
                    key={i}
                    className="animate-fade-in"
                    style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'both' }}
                  >
                    <DataCard
                      label={breach.title}
                      summary={breach.summary}
                      detail={breach.detail}
                      year={breach.year}
                      dataExposed={breach.dataExposed}
                      recordsAffected={breach.recordsAffected}
                      source={breach.source}
                      plan={plan}
                      expansionCount={getTotalExpansions()}
                      onExpand={() => handleExpand(`breach-${i}`)}
                      onPaywall={onPaywall}
                      accentColor="#f87171"
                    />
                  </div>
                ))}
              </div>
            )}

            {verifiedBreachCount === 0 && (data.breaches || []).length === 0 && (
              <EmptyState message="No known data breaches found for this company." icon="✅" positive />
            )}
          </div>
        )}

        {/* Leaked / Shared Data */}
        {activeTab === 'shared' && (
          <div className="space-y-4">
            {/* Verified GitHub leaks */}
            {(data.githubLeaks || []).length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 px-1">
                  <VerifiedBadge />
                  <span className="text-xs text-text-muted">Public GitHub repositories referencing this company</span>
                </div>
                {(data.githubLeaks || []).map((leak, i) => (
                  <div
                    key={i}
                    className="animate-fade-in rounded-xl border border-[#111827] bg-[#0f1829] p-4"
                    style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'both' }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-sm font-mono font-medium text-slate-300 truncate">{leak.repo}</p>
                        <p className="text-xs text-text-muted mt-0.5 truncate">{leak.file}</p>
                      </div>
                      {leak.url && (
                        <a
                          href={leak.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="shrink-0 text-xs font-mono text-sky-400 hover:underline"
                        >
                          View →
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* AI-inferred shared data */}
            {(data.userSharedData || []).length > 0 && (
              <div className="space-y-3">
                {(data.githubLeaks || []).length > 0 && (
                  <div className="flex items-center gap-2 px-1 pt-2 border-t border-[#111827]">
                    <InferredBadge />
                    <span className="text-xs text-text-muted">Additional findings from AI web search</span>
                  </div>
                )}
                {(data.userSharedData || []).map((item, i) => (
                  <div
                    key={i}
                    className="animate-fade-in"
                    style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'both' }}
                  >
                    <DataCard
                      label={item.platform}
                      summary={item.summary}
                      detail={item.detail}
                      date={item.date}
                      platform={item.platform}
                      sensitivity={item.sensitivity}
                      url={item.url}
                      plan={plan}
                      expansionCount={getTotalExpansions()}
                      onExpand={() => handleExpand(`shared-${i}`)}
                      onPaywall={onPaywall}
                      accentColor="#a78bfa"
                    />
                  </div>
                ))}
              </div>
            )}

            {(data.githubLeaks || []).length === 0 && (data.userSharedData || []).length === 0 && (
              <EmptyState message="No significant publicly shared data found." icon="✅" positive />
            )}
          </div>
        )}

        {/* Infrastructure */}
        {activeTab === 'infra' && (
          <div className="space-y-4">
            {/* Shodan exposure */}
            {data.infraExposure &&
              ((data.infraExposure.openPorts || []).length > 0 ||
                (data.infraExposure.knownVulns || []).length > 0) && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 px-1">
                  <VerifiedBadge />
                  <span className="text-xs text-text-muted">
                    Shodan InternetDB — {data.infraExposure.ip}
                  </span>
                </div>
                {(data.infraExposure.openPorts || []).length > 0 && (
                  <div className="rounded-xl border border-[#111827] bg-[#0f1829] p-4">
                    <p className="text-xs font-mono text-text-muted mb-3">OPEN PORTS</p>
                    <div className="flex flex-wrap gap-2">
                      {(data.infraExposure.openPorts || []).map((port) => (
                        <span
                          key={port}
                          className="text-xs font-mono px-2 py-1 rounded-lg bg-orange-400/10 border border-orange-400/20 text-orange-400"
                        >
                          :{port}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {(data.infraExposure.knownVulns || []).length > 0 && (
                  <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
                    <p className="text-xs font-mono text-red-400 mb-3">KNOWN CVEs</p>
                    <div className="flex flex-wrap gap-2">
                      {(data.infraExposure.knownVulns || []).map((cve) => (
                        <a
                          key={cve}
                          href={`https://nvd.nist.gov/vuln/detail/${cve}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs font-mono px-2 py-1 rounded-lg bg-red-400/10 border border-red-400/20 text-red-400 hover:bg-red-400/20 transition-colors"
                        >
                          {cve}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
                {(data.infraExposure.services || []).length > 0 && (
                  <div className="rounded-xl border border-[#111827] bg-[#0f1829] p-4">
                    <p className="text-xs font-mono text-text-muted mb-3">DETECTED SERVICES (CPE)</p>
                    <div className="flex flex-wrap gap-2">
                      {(data.infraExposure.services || []).map((svc, i) => (
                        <span key={i} className="text-xs font-mono px-2 py-1 rounded-lg bg-[#111827] text-text-muted border border-[#1e293b]">
                          {svc}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* crt.sh subdomains */}
            {(data.exposedSubdomains || []).length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 px-1 pt-2">
                  <VerifiedBadge />
                  <span className="text-xs text-text-muted">
                    Certificate Transparency (crt.sh) — {data.exposedSubdomains!.length} subdomains discovered
                  </span>
                </div>
                <div className="rounded-xl border border-[#111827] bg-[#0f1829] p-4">
                  <p className="text-xs font-mono text-text-muted mb-3">EXPOSED SUBDOMAINS</p>
                  <div className="flex flex-wrap gap-2">
                    {(data.exposedSubdomains || []).map((sub, i) => (
                      <span
                        key={i}
                        className="text-xs font-mono px-2 py-1 rounded-lg bg-orange-400/10 border border-orange-400/20 text-orange-300"
                      >
                        {sub}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {!hasInfra && (
              <EmptyState
                message="No infrastructure exposure data found. A company domain is required for infrastructure scanning."
                icon="🔒"
                positive
              />
            )}
          </div>
        )}
      </div>

      {/* Data sources */}
      {data.publicDataSources && data.publicDataSources.length > 0 && (
        <div className="mt-6 pt-4 border-t border-[#111827]">
          <p className="text-xs font-mono text-text-muted mb-2">Sources consulted:</p>
          <div className="flex flex-wrap gap-1.5">
            {data.publicDataSources.map((src, i) => (
              <span key={i} className="text-xs font-mono px-2 py-1 rounded-lg bg-[#0f1829] border border-[#111827] text-text-muted">
                {src}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function EmptyState({
  message,
  icon = '🔍',
  positive = false,
}: {
  message: string
  icon?: string
  positive?: boolean
}) {
  return (
    <div className={`flex flex-col items-center justify-center rounded-xl border ${positive ? 'border-green-400/20 bg-green-400/5' : 'border-[#111827] bg-[#0a0e1a]'} py-10 text-center px-4`}>
      <span className="text-3xl mb-3">{icon}</span>
      <p className={`text-sm ${positive ? 'text-green-400' : 'text-text-secondary'}`}>{message}</p>
    </div>
  )
}
