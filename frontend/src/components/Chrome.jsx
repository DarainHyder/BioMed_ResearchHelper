import React, { useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { ArrowUpRight, Menu, X } from 'lucide-react'

export const LogoMark = ({ className = 'h-7 w-7', dark = false }) => (
  <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
    <rect width="32" height="32" rx="9" fill={dark ? '#F2EEE6' : '#0E1A15'} />
    <circle cx="11" cy="12" r="2.6" fill={dark ? '#0E1A15' : '#F2EEE6'} />
    <circle cx="21" cy="10" r="1.8" fill={dark ? '#0E1A15' : '#F2EEE6'} opacity=".7" />
    <circle cx="19" cy="20" r="3.4" fill="#E5482B" />
    <circle cx="10" cy="21" r="1.6" fill={dark ? '#0E1A15' : '#F2EEE6'} opacity=".5" />
  </svg>
)

const LINKS = [
  ['/search', 'Search'],
  ['/brief', 'Briefs'],
  ['/atlas', 'Atlas'],
  ['/topics', 'Topics'],
  ['/trends', 'Trends'],
]

export function Nav() {
  const [dark, setDark] = useState(false)
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()

  useEffect(() => setOpen(false), [pathname])
  useEffect(() => {
    // Invert whenever the nav sits over an element marked data-nav-dark
    const check = () => {
      const darkEls = document.querySelectorAll('[data-nav-dark]')
      setDark([...darkEls].some((el) => {
        const r = el.getBoundingClientRect()
        return r.top <= 40 && r.bottom >= 40
      }))
    }
    check()
    window.addEventListener('scroll', check, { passive: true })
    window.addEventListener('resize', check)
    const id = setInterval(check, 500)
    return () => { window.removeEventListener('scroll', check); window.removeEventListener('resize', check); clearInterval(id) }
  }, [pathname])

  const shell = dark
    ? 'bg-moss-2/70 border-white/10 text-bone'
    : 'bg-bone-50/75 border-ink/10 text-ink'

  return (
    <header className="fixed inset-x-0 top-4 z-50 px-4">
      <nav className={`mx-auto flex max-w-[1100px] items-center justify-between rounded-full border px-3 py-2 backdrop-blur-xl transition-colors duration-500 ${shell}`}>
        <Link to="/" className="flex items-center gap-2.5 pl-1.5" aria-label="BioAtlas home">
          <LogoMark className="h-7 w-7" dark={dark} />
          <span className="font-display text-[17px] font-semibold tracking-tight">BioAtlas</span>
        </Link>
        <div className="hidden items-center gap-1 md:flex">
          {LINKS.map(([to, label]) => (
            <NavLink key={to} to={to} className={({ isActive }) =>
              `rounded-full px-3.5 py-1.5 text-sm transition-colors ${isActive
                ? (dark ? 'bg-white/10' : 'bg-ink/[0.06]')
                : (dark ? 'text-bone/70 hover:text-bone' : 'text-ink-3 hover:text-ink')}`}>
              {label}
            </NavLink>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Link to="/brief" className={`hidden sm:inline-flex btn py-2 ${dark ? 'bg-bone text-moss hover:bg-white' : 'bg-ink text-bone-50 hover:bg-ink-2'}`}>
            Ask a question <ArrowUpRight className="h-4 w-4" />
          </Link>
          <button className="rounded-full p-2 md:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu">
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </nav>
      {open && (
        <div className="mx-auto mt-2 max-w-[1100px] rounded-3xl border border-ink/10 bg-bone-50/95 p-3 backdrop-blur-xl md:hidden">
          {LINKS.map(([to, label]) => (
            <NavLink key={to} to={to} className="block rounded-2xl px-4 py-3 text-ink hover:bg-ink/5">{label}</NavLink>
          ))}
        </div>
      )}
    </header>
  )
}

export function Footer() {
  return (
    <footer className="relative overflow-hidden bg-moss text-bone grain" data-nav-dark>
      <div className="container-x relative py-20">
        <div className="grid gap-12 md:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <div className="flex items-center gap-3">
              <LogoMark className="h-9 w-9" dark />
              <span className="font-display text-2xl font-semibold tracking-tight">BioAtlas</span>
            </div>
            <p className="mt-5 max-w-sm text-bone/60">
              A living map of biomedical research. Built on PubMed abstracts, a fine-tuned biomedical
              embedding model and hybrid retrieval.
            </p>
          </div>
          <div>
            <p className="eyebrow text-bone/40">Product</p>
            <ul className="mt-4 space-y-2.5 text-bone/80">
              {LINKS.map(([to, label]) => <li key={to}><Link to={to} className="hover:text-bone">{label}</Link></li>)}
            </ul>
          </div>
          <div>
            <p className="eyebrow text-bone/40">Project</p>
            <ul className="mt-4 space-y-2.5 text-bone/80">
              <li><a className="hover:text-bone" href="https://github.com/DarainHyder/BioMed_ResearchHelper" target="_blank" rel="noreferrer">Source on GitHub</a></li>
              <li><a className="hover:text-bone" href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank" rel="noreferrer">Data: NCBI PubMed</a></li>
              <li><a className="hover:text-bone" href="https://sawabedarain-biomed-ai-backend.hf.space/docs" target="_blank" rel="noreferrer">API reference</a></li>
            </ul>
          </div>
        </div>
        <div className="mt-16 flex flex-col gap-4 border-t border-white/10 pt-8 text-sm text-bone/45 md:flex-row md:items-center md:justify-between">
          <p>Portfolio project, not a production service or medical advice. The API runs on a free tier and may take ~30s to wake.</p>
          <p className="font-mono text-xs">Abstracts © their publishers via NCBI E-utilities</p>
        </div>
      </div>
      <div className="pointer-events-none select-none text-center font-display text-[22vw] font-semibold leading-[0.8] tracking-tightest text-white/[0.035]">BioAtlas</div>
    </footer>
  )
}
