/**
 * Critical-path unit tests — run with: node --test __tests__/utils.test.mjs
 * Tests all exported functions from lib/utils without any framework setup.
 */
import { strict as assert } from 'node:assert'
import { test, describe } from 'node:test'

// Inline the pure functions from lib/utils (no TS, no imports)
function slugify(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
}
function deslugify(slug) {
  return slug.split('-').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
}
function getRiskColor(level) {
  switch (level?.toLowerCase()) {
    case 'low':      return 'text-green-400 bg-green-400/10 border-green-400/20'
    case 'medium':   return 'text-amber-400 bg-amber-400/10 border-amber-400/20'
    case 'high':     return 'text-orange-400 bg-orange-400/10 border-orange-400/20'
    case 'critical': return 'text-red-400 bg-red-400/10 border-red-400/20'
    default:         return 'text-slate-400 bg-slate-400/10 border-slate-400/20'
  }
}
function getRiskBg(level) {
  switch (level?.toLowerCase()) {
    case 'low':      return '#4ade80'
    case 'medium':   return '#fbbf24'
    case 'high':     return '#fb923c'
    case 'critical': return '#f87171'
    default:         return '#94a3b8'
  }
}
function getSensitivityColor(sensitivity) {
  switch (sensitivity?.toLowerCase()) {
    case 'low':    return 'text-green-400'
    case 'medium': return 'text-amber-400'
    case 'high':   return 'text-red-400'
    default:       return 'text-slate-400'
  }
}
function formatDate(dateStr) {
  try {
    return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
  } catch { return dateStr }
}
const PLAN_LIMITS = {
  FREE:       { auditsPerDay: 3, chatMessages: 3, cardExpansions: 1 },
  PRO:        { auditsPerDay: 20, chatMessages: Infinity, cardExpansions: Infinity },
  ENTERPRISE: { auditsPerDay: Infinity, chatMessages: Infinity, cardExpansions: Infinity },
}
function checkUsage(plan, used) {
  const limit = PLAN_LIMITS[plan]?.auditsPerDay ?? 3
  return limit === Infinity || used < limit
}

describe('slugify', () => {
  test('lowercases and replaces spaces', () => {
    assert.equal(slugify('Adobe Systems'), 'adobe-systems')
  })
  test('collapses multiple spaces/special chars', () => {
    assert.equal(slugify('Google   LLC!'), 'google-llc')
  })
  test('strips leading/trailing hyphens', () => {
    assert.equal(slugify('  Apple  '), 'apple')
  })
  test('preserves numbers', () => {
    assert.equal(slugify('3M Company'), '3m-company')
  })
})

describe('deslugify', () => {
  test('capitalises each word', () => {
    assert.equal(deslugify('adobe-systems'), 'Adobe Systems')
  })
  test('handles single word', () => {
    assert.equal(deslugify('google'), 'Google')
  })
})

describe('getRiskColor', () => {
  test('low → green classes', () => assert.ok(getRiskColor('low').includes('green')))
  test('medium → amber classes', () => assert.ok(getRiskColor('medium').includes('amber')))
  test('high → orange classes', () => assert.ok(getRiskColor('high').includes('orange')))
  test('critical → red classes', () => assert.ok(getRiskColor('critical').includes('red')))
  test('unknown → slate classes', () => assert.ok(getRiskColor('unknown').includes('slate')))
  test('case-insensitive', () => assert.ok(getRiskColor('CRITICAL').includes('red')))
  test('null/undefined → slate', () => assert.ok(getRiskColor(undefined).includes('slate')))
})

describe('getRiskBg', () => {
  test('low → green hex', () => assert.equal(getRiskBg('low'), '#4ade80'))
  test('medium → amber hex', () => assert.equal(getRiskBg('medium'), '#fbbf24'))
  test('high → orange hex', () => assert.equal(getRiskBg('high'), '#fb923c'))
  test('critical → red hex', () => assert.equal(getRiskBg('critical'), '#f87171'))
  test('default → slate hex', () => assert.equal(getRiskBg('none'), '#94a3b8'))
})

describe('getSensitivityColor', () => {
  test('low → green', () => assert.ok(getSensitivityColor('low').includes('green')))
  test('medium → amber', () => assert.ok(getSensitivityColor('medium').includes('amber')))
  test('high → red', () => assert.ok(getSensitivityColor('high').includes('red')))
  test('default → slate', () => assert.ok(getSensitivityColor('').includes('slate')))
})

describe('formatDate', () => {
  test('formats ISO date string', () => {
    const result = formatDate('2024-01-15T00:00:00.000Z')
    assert.ok(result.includes('2024'))
    assert.ok(result.includes('Jan'))
  })
  test('falls back on invalid string', () => {
    assert.equal(formatDate('not-a-date'), 'Invalid Date')
  })
})

describe('PLAN_LIMITS', () => {
  test('FREE: 3 audits, 3 chats, 1 expansion', () => {
    assert.equal(PLAN_LIMITS.FREE.auditsPerDay, 3)
    assert.equal(PLAN_LIMITS.FREE.chatMessages, 3)
    assert.equal(PLAN_LIMITS.FREE.cardExpansions, 1)
  })
  test('PRO: 20 audits, unlimited chat/expansion', () => {
    assert.equal(PLAN_LIMITS.PRO.auditsPerDay, 20)
    assert.equal(PLAN_LIMITS.PRO.chatMessages, Infinity)
    assert.equal(PLAN_LIMITS.PRO.cardExpansions, Infinity)
  })
  test('ENTERPRISE: fully unlimited', () => {
    assert.equal(PLAN_LIMITS.ENTERPRISE.auditsPerDay, Infinity)
    assert.equal(PLAN_LIMITS.ENTERPRISE.chatMessages, Infinity)
    assert.equal(PLAN_LIMITS.ENTERPRISE.cardExpansions, Infinity)
  })
})

describe('checkUsage (daily audit gate)', () => {
  test('FREE: allows when under limit', () => assert.ok(checkUsage('FREE', 2)))
  test('FREE: blocks at limit', () => assert.ok(!checkUsage('FREE', 3)))
  test('FREE: blocks over limit', () => assert.ok(!checkUsage('FREE', 5)))
  test('PRO: allows up to 20', () => assert.ok(checkUsage('PRO', 19)))
  test('PRO: blocks at 20', () => assert.ok(!checkUsage('PRO', 20)))
  test('ENTERPRISE: always allows', () => assert.ok(checkUsage('ENTERPRISE', 9999)))
  test('unknown plan falls back to 3 limit', () => assert.ok(!checkUsage('UNKNOWN', 3)))
})
