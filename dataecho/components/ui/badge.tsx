import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-mono font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default: 'border-amber-200 bg-amber-50 text-amber-700',
        secondary: 'border-slate-200 bg-slate-100 text-slate-600',
        destructive: 'border-red-200 bg-red-50 text-red-700',
        outline: 'border-slate-300 text-slate-700',
        low: 'border-emerald-200 bg-emerald-50 text-emerald-700',
        medium: 'border-amber-200 bg-amber-50 text-amber-700',
        high: 'border-orange-200 bg-orange-50 text-orange-700',
        critical: 'border-red-200 bg-red-50 text-red-700',
        pro: 'border-amber-300 bg-gradient-to-r from-amber-50 to-orange-50 text-amber-700',
        enterprise: 'border-violet-200 bg-violet-50 text-violet-700',
        free: 'border-slate-200 bg-slate-100 text-slate-500',
      },
    },
    defaultVariants: { variant: 'default' },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
