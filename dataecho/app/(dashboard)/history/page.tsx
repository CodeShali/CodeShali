import { permanentRedirect } from 'next/navigation'

// History moved to /dashboard/history
export default function OldHistory() {
  permanentRedirect('/dashboard/history')
}
