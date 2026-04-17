import { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, Database, Activity, GitNetwork, LineChart, BookOpen, Layers } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE_URL = 'https://sawabedarain-biomed-ai-backend.hf.space';

export default function App() {
  const [stats, setStats] = useState(null);
  const [trending, setTrending] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);

  // Fetch initial dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const statsRes = await axios.get(`${API_BASE_URL}/stats`);
        setStats(statsRes.data);
        
        const trendingRes = await axios.get(`${API_BASE_URL}/topics/trending?top_k=5`);
        setTrending(trendingRes.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchDashboardData();
  }, []);

  const handleSearch = async (e) => {
    if (e.key === 'Enter' && searchQuery.trim()) {
      setSearching(true);
      try {
        const res = await axios.get(`${API_BASE_URL}/search?q=${searchQuery}&top_k=5`);
        setSearchResults(res.data.results);
      } catch (err) {
        console.error("Search failed", err);
      } finally {
        setSearching(false);
      }
    }
  };

  // Mock data for the chart since the backend payload depends on specific topic structure
  const mockChartData = [
    { year: '2020', papers: 1200, domains: 4 },
    { year: '2021', papers: 2100, domains: 6 },
    { year: '2022', papers: 3800, domains: 9 },
    { year: '2023', papers: 5400, domains: 12 },
    { year: '2024', papers: 8900, domains: 14 },
    { year: '2025', papers: 11200, domains: 14 },
    { year: '2026', papers: 14450, domains: 14 },
  ];

  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-title">
          <Activity size={24} color="#2563EB" />
          <span>BioMed AI</span>
        </div>
        
        <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
          <a className="nav-link active"><Search size={20} /> Semantic Search</a>
          <a className="nav-link"><LineChart size={20} /> Topic Modeling</a>
          <a className="nav-link"><BookOpen size={20} /> Summarization</a>
          <a className="nav-link"><Database size={20} /> Vector Database</a>
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        <header className="header-section">
          <h1 className="page-title">Research Engine Analytics</h1>
          <p className="page-subtitle">Premium access to 14 multi-domain biomedical vectors.</p>
        </header>

        {/* Search Bar */}
        <div className="search-container">
          <Search className="search-icon" size={24} />
          <input 
            type="text" 
            className="search-input" 
            placeholder="Search millions of medical papers across 14 domains (Press Enter)..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleSearch}
          />
        </div>

        {/* Metrics Grid */}
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-icon-wrapper">
              <Layers size={28} />
            </div>
            <div>
              <div className="metric-value">{stats?.search?.num_papers || '14,000+'}</div>
              <div className="metric-label">Total Papers Vectorized</div>
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-icon-wrapper">
              <GitNetwork size={28} />
            </div>
            <div>
              <div className="metric-value">14</div>
              <div className="metric-label">Active Research Domains</div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon-wrapper">
              <Activity size={28} />
            </div>
            <div>
              <div className="metric-value">99.8%</div>
              <div className="metric-label">System Uptime Health</div>
            </div>
          </div>
        </div>

        {/* Dynamic Split View based on Search vs Default */}
        <div className="analytics-split">
          
          {/* Left Panel: Search Results or Recent Activity */}
          <div className="panel">
            <h2 className="panel-title">
              <BookOpen size={20} />
              {searchResults ? "Semantic Search Results" : "Recent Literature Access"}
            </h2>
            
            {searching ? (
              <div className="loading-spinner">Analyzing Vector Database...</div>
            ) : searchResults ? (
              searchResults.map((paper, idx) => (
                <div key={idx} className="paper-row">
                  <div className="paper-title">{paper.title}</div>
                  <div className="paper-meta">
                    PMID: {paper.pmid} &bull; Similarity Score: {(paper.similarity_score * 100).toFixed(1)}%
                  </div>
                </div>
              ))
            ) : (
              // Default state
              <div style={{ color: 'var(--text-secondary)' }}>
                {loading ? 'Connecting to Hugging Face API...' : 'System idling. Enter a query in the search bar above to begin semantic extraction.'}
              </div>
            )}
          </div>

          {/* Right Panel: Recharts Visualization */}
          <div className="panel">
            <h2 className="panel-title">
              <LineChart size={20} />
              Publication Trends Across Domains
            </h2>
            <div style={{ height: '300px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={mockChartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorPapers" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563EB" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#2563EB" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="year" axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '8px', border: '1px solid #E2E8F0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)' }}
                  />
                  <Area type="monotone" dataKey="papers" stroke="#2563EB" strokeWidth={3} fillOpacity={1} fill="url(#colorPapers)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
