const BASE = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? 'https://sawabedarain-biomed-ai-backend.hf.space' : 'http://localhost:7860')

async function request(path, { method = 'GET', body, params, signal } = {}) {
  const url = new URL(BASE + path)
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v === undefined || v === null || v === '') return
      if (Array.isArray(v)) v.forEach((x) => url.searchParams.append(k, x))
      else url.searchParams.set(k, v)
    })
  }
  const res = await fetch(url, {
    method, signal,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail?.[0]?.msg || err.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  health: () => request('/api/health'),
  stats: () => request('/api/stats'),
  search: (params, signal) => request('/api/search', { params, signal }),
  paper: (pmid) => request(`/api/papers/${pmid}`),
  brief: (body) => request('/api/brief', { method: 'POST', body }),
  topics: (params) => request('/api/topics', { params }),
  topic: (id) => request(`/api/topics/${id}`),
  trends: () => request('/api/trends'),
}

// Wake the free-tier backend early (it sleeps after inactivity)
let warmed = false
export const warmUp = () => {
  if (warmed) return
  warmed = true
  api.health().catch(() => { warmed = false })
}
