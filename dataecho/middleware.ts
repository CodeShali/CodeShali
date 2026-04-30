export { default } from 'next-auth/middleware'

export const config = {
  // /dashboard requires login; /admin auth checked in the page/API route itself
  matcher: ['/dashboard/:path*', '/admin/:path*'],
}
