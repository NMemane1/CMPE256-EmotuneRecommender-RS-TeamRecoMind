/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'chat-bg': '#1a1a1a',
        'chat-sidebar': '#0d0d0d',
        'chat-input': '#2a2a2a',
        'chat-user': '#3b3b3b',
        'chat-assistant': '#1e1e1e',
        'accent': '#d97706',
      },
    },
  },
  plugins: [],
}
