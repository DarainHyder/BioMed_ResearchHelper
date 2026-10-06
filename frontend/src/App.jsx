import React, { Suspense, lazy, useEffect } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import Lenis from 'lenis'
import { Footer, Nav } from './components/Chrome'
import { warmUp } from './lib/api'

const Home = lazy(() => import('./pages/Home'))
const Search = lazy(() => import('./pages/Search'))
const Brief = lazy(() => import('./pages/Brief'))
const Atlas = lazy(() => import('./pages/Atlas'))
const Topics = lazy(() => import('./pages/Topics'))
const Trends = lazy(() => import('./pages/Trends'))

const Fallback = () => <div className="min-h-screen bg-bone" />

export default function App() {
  const { pathname } = useLocation()

  useEffect(() => {
    warmUp()
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const lenis = new Lenis({ lerp: 0.09, smoothWheel: true })
    let raf
    const loop = (t) => { lenis.raf(t); raf = requestAnimationFrame(loop) }
    raf = requestAnimationFrame(loop)
    window.__lenis = lenis
    return () => { cancelAnimationFrame(raf); lenis.destroy() }
  }, [])

  useEffect(() => {
    window.__lenis ? window.__lenis.scrollTo(0, { immediate: true }) : window.scrollTo(0, 0)
  }, [pathname])

  return (
    <>
      <Nav />
      <Suspense fallback={<Fallback />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/search" element={<Search />} />
          <Route path="/brief" element={<Brief />} />
          <Route path="/atlas" element={<Atlas />} />
          <Route path="/topics" element={<Topics />} />
          <Route path="/trends" element={<Trends />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </Suspense>
      {pathname !== '/atlas' && <Footer />}
    </>
  )
}
