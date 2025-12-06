/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Paragon brand colors
        paragon: {
          primary: '#3B82F6',
          secondary: '#8B5CF6',
          success: '#10B981',
          warning: '#F59E0B',
          danger: '#EF4444',
          dark: '#1F2937',
          light: '#F9FAFB',
        },
        // Graph layer colors
        layer: {
          1: '#EF4444',
          2: '#F59E0B',
          3: '#F59E0B',
          4: '#10B981',
          5: '#3B82F6',
          6: '#6366F1',
          7: '#8B5CF6',
          8: '#EC4899',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
