/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Warm paper surfaces
        bone: { DEFAULT: '#F2EEE6', 50: '#FBF9F4', 100: '#F6F3EC', 200: '#ECE6DA', 300: '#DED6C6' },
        // Ink scale (text) with a hint of green
        ink: { DEFAULT: '#141613', 2: '#2B2E29', 3: '#55584F', 4: '#8A8B80', 5: '#B7B5AA' },
        // Deep moss for immersive sections
        moss: { DEFAULT: '#0E1A15', 2: '#14241D', 3: '#1D3128', 4: '#2C4438', line: '#2A3B33' },
        // Single signal accent
        signal: { DEFAULT: '#E5482B', soft: '#F07A5F', ink: '#B5361E' },
      },
      fontFamily: {
        display: ['"Inter Tight"', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        serif: ['Newsreader', 'ui-serif', 'Georgia', 'serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      letterSpacing: { tightest: '-0.045em' },
      keyframes: {
        rise: { '0%': { opacity: 0, transform: 'translateY(14px)' }, '100%': { opacity: 1, transform: 'none' } },
        marquee: { '0%': { transform: 'translateX(0)' }, '100%': { transform: 'translateX(-50%)' } },
        shimmer: { '0%': { backgroundPosition: '-400px 0' }, '100%': { backgroundPosition: '400px 0' } },
      },
      animation: {
        rise: 'rise .7s cubic-bezier(.2,.7,.2,1) both',
        marquee: 'marquee 60s linear infinite',
        shimmer: 'shimmer 1.4s linear infinite',
      },
    },
  },
  plugins: [],
}
