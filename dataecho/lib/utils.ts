import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function slugify(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

export function deslugify(slug: string) {
  return slug
    .split('-')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

export function getRiskColor(level: string) {
  switch (level?.toLowerCase()) {
    case 'low':      return 'text-green-400 bg-green-400/10 border-green-400/20'
    case 'medium':   return 'text-amber-400 bg-amber-400/10 border-amber-400/20'
    case 'high':     return 'text-orange-400 bg-orange-400/10 border-orange-400/20'
    case 'critical': return 'text-red-400 bg-red-400/10 border-red-400/20'
    default:         return 'text-slate-400 bg-slate-400/10 border-slate-400/20'
  }
}

export function getRiskBg(level: string) {
  switch (level?.toLowerCase()) {
    case 'low':      return '#4ade80'
    case 'medium':   return '#fbbf24'
    case 'high':     return '#fb923c'
    case 'critical': return '#f87171'
    default:         return '#94a3b8'
  }
}

export function getSensitivityColor(sensitivity: string) {
  switch (sensitivity?.toLowerCase()) {
    case 'low':    return 'text-green-400'
    case 'medium': return 'text-amber-400'
    case 'high':   return 'text-red-400'
    default:       return 'text-slate-400'
  }
}

export function formatDate(dateStr: string) {
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
    })
  } catch {
    return dateStr
  }
}

export const PLAN_LIMITS = {
  FREE:       { auditsPerDay: 3, chatMessages: 3, cardExpansions: 1 },
  PRO:        { auditsPerDay: 20, chatMessages: Infinity, cardExpansions: Infinity },
  ENTERPRISE: { auditsPerDay: Infinity, chatMessages: Infinity, cardExpansions: Infinity },
}
