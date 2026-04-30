'use client'

import { useState, useRef, useEffect } from 'react'
import { MessageSquare, X, Minimize2, Send, Loader2, Bot, User } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatWidgetProps {
  auditContext: Record<string, unknown>
  companyName: string
  plan: string
  onPaywall: () => void
}

const FREE_LIMIT = 3

export function ChatWidget({ auditContext, companyName, plan, onPaywall }: ChatWidgetProps) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: `Hi! I'm your DataEcho assistant. Ask me anything about the **${companyName}** audit — breaches, privacy risks, what AI knows, and more.`,
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const userMessageCount = messages.filter((m) => m.role === 'user').length
  const isAtLimit = plan === 'FREE' && userMessageCount >= FREE_LIMIT

  useEffect(() => {
    if (open) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, open])

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    if (isAtLimit) {
      onPaywall()
      return
    }

    const userMsg: Message = { role: 'user', content: input.trim() }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg.content,
          context: auditContext,
          messageCount: userMessageCount + 1,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        if (err.error === 'CHAT_LIMIT_REACHED') {
          onPaywall()
          setMessages((prev) => prev.slice(0, -1))
          return
        }
        throw new Error(err.error)
      }

      const data = await res.json()
      setMessages((prev) => [...prev, { role: 'assistant', content: data.reply }])
    } catch {
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: "Sorry, I couldn't process that request. Please try again.",
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <>
      {/* Floating trigger button */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex h-14 w-14 items-center justify-center rounded-full bg-accent-amber text-bg-primary shadow-2xl shadow-amber-500/30 animate-pulse-amber hover:bg-amber-400 transition-colors"
          aria-label="Open chat"
        >
          <MessageSquare className="h-6 w-6" />
        </button>
      )}

      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-6 right-6 z-40 flex w-[360px] max-w-[calc(100vw-48px)] flex-col rounded-2xl border border-[#1e293b] bg-[#0a0e1a]/95 backdrop-blur-xl shadow-2xl shadow-black/50 overflow-hidden"
          style={{ height: '500px' }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#111827] bg-[#060a12]/60">
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-lg bg-accent-amber/20 flex items-center justify-center">
                <Bot className="h-4 w-4 text-accent-amber" />
              </div>
              <div>
                <div className="text-sm font-semibold text-text-primary">DataEcho AI</div>
                <div className="text-[10px] font-mono text-text-muted">
                  {companyName} audit assistant
                  {plan === 'FREE' && (
                    <> · <span className="text-accent-amber">{FREE_LIMIT - userMessageCount} msgs left</span></>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setOpen(false)}
                className="p-1.5 rounded-lg hover:bg-[#0f1829] text-text-muted hover:text-text-secondary transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-hide">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={cn(
                  'flex gap-2.5 animate-fade-in',
                  msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                )}
              >
                <div className={cn(
                  'flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full',
                  msg.role === 'user'
                    ? 'bg-accent-amber/20 text-accent-amber'
                    : 'bg-[#111827] text-text-muted'
                )}>
                  {msg.role === 'user'
                    ? <User className="h-3.5 w-3.5" />
                    : <Bot className="h-3.5 w-3.5" />
                  }
                </div>
                <div
                  className={cn(
                    'max-w-[80%] rounded-2xl px-3 py-2 text-sm leading-relaxed',
                    msg.role === 'user'
                      ? 'bg-accent-amber/15 text-text-primary rounded-tr-sm'
                      : 'bg-[#111827] text-text-secondary rounded-tl-sm'
                  )}
                >
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex gap-2.5">
                <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-[#111827] text-text-muted">
                  <Bot className="h-3.5 w-3.5" />
                </div>
                <div className="bg-[#111827] rounded-2xl rounded-tl-sm px-4 py-3">
                  <div className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={i}
                        className="h-1.5 w-1.5 rounded-full bg-text-muted animate-bounce"
                        style={{ animationDelay: `${i * 0.15}s` }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}

            {isAtLimit && (
              <div className="rounded-xl border border-amber-400/20 bg-amber-400/10 p-3 text-center">
                <p className="text-xs text-amber-400 mb-2">Free chat limit reached</p>
                <Button size="sm" variant="pro" className="text-xs" onClick={onPaywall}>
                  Upgrade for unlimited chat
                </Button>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="p-3 border-t border-[#111827]">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={isAtLimit ? 'Upgrade to continue chatting…' : 'Ask about this company…'}
                disabled={isAtLimit || loading}
                className="flex-1 h-9 rounded-xl border border-[#1e293b] bg-[#060a12] px-3 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-accent-amber disabled:opacity-50 transition-colors"
              />
              <button
                onClick={isAtLimit ? onPaywall : sendMessage}
                disabled={(!input.trim() && !isAtLimit) || loading}
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-amber text-bg-primary hover:bg-amber-400 disabled:opacity-50 transition-colors flex-shrink-0"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
