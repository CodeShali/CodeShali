'use client'

import { useState } from 'react'
import { Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PaywallModal } from './PaywallModal'

interface UpgradeButtonProps {
  size?: 'sm' | 'default' | 'lg'
  className?: string
  label?: string
}

export function UpgradeButton({ size = 'default', className, label = 'Upgrade to Pro' }: UpgradeButtonProps) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <Button variant="pro" size={size} className={className} onClick={() => setOpen(true)}>
        <Zap className="h-4 w-4 mr-1.5" />
        {label}
      </Button>
      {open && <PaywallModal defaultOpen onClose={() => setOpen(false)} />}
    </>
  )
}
