import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-mono font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-accent-amber/20 text-accent-amber',
        secondary: 'border-transparent bg-secondary text-secondary-foreground',
        destructive: 'border-transparent bg-destructive/20 text-destructive-foreground',
        outline: 'text-foreground',
        low: 'border-green-400/20 bg-green-400/10 text-green-400',
        medium: 'border-amber-400/20 bg-amber-400/10 text-amber-400',
        high: 'border-orange-400/20 bg-orange-400/10 text-orange-400',
        critical: 'border-red-400/20 bg-red-400/10 text-red-400',
        pro: 'border-amber-400/30 bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-400',
        enterprise: 'border-purple-400/30 bg-purple-400/10 text-purple-400',
        free: 'border-slate-600/30 bg-slate-700/20 text-slate-400',
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
