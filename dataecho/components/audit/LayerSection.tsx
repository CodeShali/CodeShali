'use client'

import { useState } from 'react'
import { Brain, ShieldOff, Globe } from 'lucide-react'
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

interface SharedData {
  platform: string
  date: string
  summary: string
  detail: string
  sensitivity: string
  url: string
}

interface AuditData {
  companyDomain?: string
  llmKnowledge?: { summary: string; categories: LLMCategory[] }
  breaches?: Breach[]
  userSharedData?: SharedData[]
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
    label: 'User-Shared Data',
    sublabel: 'Publicly shared or leaked information',
    icon: Globe,
    color: '#a78bfa',
    border: 'border-purple-500/30',
    bg: 'bg-purple-500/5',
  },
]

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

  const currentLayer = LAYER_CONFIG.find((l) => l.id === activeTab)!

  return (
    <div>
      {/* Tab headers */}
      <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-hide pb-1">
        {LAYER_CONFIG.map((layer) => {
          const Icon = layer.icon
          const isActive = activeTab === layer.id
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
              <span className={isActive ? '' : ''}>{layer.label}</span>
            </button>
          )
        })}
      </div>

      {/* Layer content */}
      <div className="animate-fade-in">
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

        {activeTab === 'breaches' && (
          <div className="space-y-3">
            {(data.breaches || []).length === 0 ? (
              <EmptyState message="No known data breaches found for this company." icon="✅" positive />
            ) : (
              (data.breaches || []).map((breach, i) => (
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
              ))
            )}
          </div>
        )}

        {activeTab === 'shared' && (
          <div className="space-y-3">
            {(data.userSharedData || []).length === 0 ? (
              <EmptyState message="No significant publicly shared data found." icon="✅" positive />
            ) : (
              (data.userSharedData || []).map((item, i) => (
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
              ))
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
