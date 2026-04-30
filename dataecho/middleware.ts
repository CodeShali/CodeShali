export { default } from 'next-auth/middleware'

export const config = {
  // Only protect /dashboard routes — /audit routes are public for shareable links
  matcher: ['/dashboard/:path*'],
}
