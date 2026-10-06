/** @type {import('tailwindcss').Config} */
// BioAtlas: a printed scientific atlas. Ivory paper, ink, and the two stains of histology (H&E).
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        paper: { DEFAULT: '#F7F4EE', 2: '#EFEAE0', 3: '#E6DFD2' },
        ink: { DEFAULT: '#1B1916', 2: '#45413B', 3: '#77716A', 4: '#A39C92' },
        rule: { DEFAULT: '#D8D1C4', strong: '#B9B0A1' },
        hema: { DEFAULT: '#3E2F73', light: '#6A5AA3', wash: '#E7E3F1' }, // hematoxylin violet
        eosin: { DEFAULT: '#C9476E', light: '#E07C9A', wash: '#F6E1E7' }, // eosin pink
      },
      fontFamily: {
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        ink: { '0%': { opacity: 0, transform: 'translateY(8px)' }, '100%': { opacity: 1, transform: 'none' } },
      },
      animation: { ink: 'ink .6s cubic-bezier(.2,.7,.2,1) both' },
    },
  },
  plugins: [],
}
