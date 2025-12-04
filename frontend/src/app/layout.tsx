import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'EmoTune Chat',
  description: 'AI-powered music recommendations based on your emotions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-chat-bg text-gray-100 antialiased">
        {children}
      </body>
    </html>
  )
}
