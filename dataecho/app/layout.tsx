import type { Metadata } from 'next'
import './globals.css'
import { Providers } from './providers'

export const metadata: Metadata = {
  title: 'DataEcho — See what AI knows about you',
  description: 'Search any company and see what LLMs know, known data breaches, and recently shared information.',
  metadataBase: new URL(process.env.NEXTAUTH_URL || 'https://dataecho.ai'),
  openGraph: {
    title: 'DataEcho',
    description: 'See what AI knows about you.',
    siteName: 'DataEcho',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
