import React, { useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { Menu, X } from 'lucide-react'

export const HelixGlyph = ({ className = 'h-6 w-6' }) => (
  <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
    <path d="M10 3c0 9 12 8 12 13s-12 4-12 13" fill="none" stroke="#3E2F73" strokeWidth="2.2" strokeLinecap="round" />
    <path d="M22 3c0 9-12 8-12 13s12 4 12 13" fill="none" stroke="#C9476E" strokeWidth="2.2" strokeLinecap="round" />
    {[8, 13, 19, 24].map((y) => <line key={y} x1="12.5" x2="19.5" y1={y} y2={y} stroke="#1B1916" strokeOpacity=".25" strokeWidth="1" />)}
  </svg>
)

const LINKS = [
  ['/search', 'Search'],
  ['/brief', 'Briefs'],
  ['/atlas', 'Atlas'],
  ['/topics', 'Research fronts'],
  ['/trends', 'Trends'],
]

export function Nav() {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => setOpen(false), [pathname])
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])

  return (
    <header className={`fixed inset-x-0 top-0 z-50 bg-paper/90 backdrop-blur-sm transition-[border-color] ${scrolled || open ? 'border-b border-rule' : 'border-b border-transparent'}`}>
      <div className="page flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5" aria-label="BioAtlas home">
          <HelixGlyph />
          <span className="font-serif text-[22px] font-semibold tracking-[-0.01em]">BioAtlas</span>
        </Link>
        <nav className="hidden items-center gap-7 md:flex">
          {LINKS.map(([to, label]) => (
            <NavLink key={to} to={to} className={({ isActive }) =>
              `font-sans text-[14px] transition-colors ${isActive ? 'text-ink underline decoration-eosin decoration-2 underline-offset-[6px]' : 'text-ink-3 hover:text-ink'}`}>
              {label}
            </NavLink>
          ))}
        </nav>
        <button className="p-2 md:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu">
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>
      {open && (
        <nav className="page ruled pb-4 md:hidden">
          {LINKS.map(([to, label]) => <NavLink key={to} to={to} className="block py-3 font-serif text-lg">{label}</NavLink>)}
        </nav>
      )}
    </header>
  )
}

export function Footer() {
  return (
    <footer className="mt-24 border-t border-ink">
      <div className="page grid gap-10 py-12 md:grid-cols-[1.3fr_1fr_1fr_1.3fr]">
        <div>
          <div className="flex items-center gap-2.5"><HelixGlyph /><span className="font-serif text-xl font-semibold">BioAtlas</span></div>
          <p className="meta mt-4 max-w-xs leading-6">An open atlas of biomedical research, built from PubMed abstracts and a language model trained on them.</p>
        </div>
        <div>
          <p className="font-sans text-[13px] font-semibold text-ink">Instruments</p>
          <ul className="mt-3 space-y-2">{LINKS.map(([to, label]) => <li key={to}><Link className="meta hover:text-ink" to={to}>{label}</Link></li>)}</ul>
        </div>
        <div>
          <p className="font-sans text-[13px] font-semibold text-ink">Sources</p>
          <ul className="mt-3 space-y-2 meta">
            <li><a className="hover:text-ink" href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank" rel="noreferrer">NCBI PubMed</a></li>
            <li><a className="hover:text-ink" href="https://github.com/DarainHyder/BioMed_ResearchHelper" target="_blank" rel="noreferrer">Source code</a></li>
            <li><a className="hover:text-ink" href="https://sawabedarain-biomed-ai-backend.hf.space/docs" target="_blank" rel="noreferrer">API reference</a></li>
          </ul>
        </div>
        <div>
          <p className="font-sans text-[13px] font-semibold text-ink">Notice</p>
          <p className="meta mt-3 leading-6">
            A portfolio project, not a clinical tool or medical advice. The API runs on a free tier and may take about 30 seconds to
            wake. Abstracts remain the property of their publishers.
          </p>
        </div>
      </div>
    </footer>
  )
}
