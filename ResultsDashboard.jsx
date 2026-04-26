import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, ReferenceLine
} from 'recharts';

// ── REAL EVALUATION RESULTS ──────────────────────────────────────────────────
// Baseline: 150 random-action episodes (collect_baseline.py)
const INITIAL_BASELINE = {
  easy:   { avg: 0.3597, std: 0.2994, n: 107 },
  medium: { avg: 0.1733, std: 0.1702, n: 26  },
  hard:   { avg: 0.1824, std: 0.121,  n: 17  }
};

// v1: 200 GRPO steps on L4 GPU (pr-triage-grpo-adapter)
const INITIAL_V1 = {
  easy:   { avg: 0.3098, std: 0.3225, n: 47 },
  medium: { avg: 0.0382, std: 0.069,  n: 8  },
  hard:   { avg: 0.0181, std: 0.0299, n: 5  }
};

// v2: 400 more GRPO steps on A100 GPU (pr-triage-grpo-adapter-v2)
// Paste real values here once the HF Job finishes
const INITIAL_V2 = {
  easy:   { avg: 0.0, std: 0.0, n: 0 },
  medium: { avg: 0.0, std: 0.0, n: 0 },
  hard:   { avg: 0.0, std: 0.0, n: 0 }
};

const LEVELS = ['easy', 'medium', 'hard'];

const COLORS = {
  baseline: '#6366f1',
  v1:       '#f59e0b',
  v2:       '#4ade80',
};

export default function ResultsDashboard() {
  const [baselineStr,  setBaselineStr]  = useState(JSON.stringify(INITIAL_BASELINE, null, 2));
  const [v1Str,        setV1Str]        = useState(JSON.stringify(INITIAL_V1, null, 2));
  const [v2Str,        setV2Str]        = useState(JSON.stringify(INITIAL_V2, null, 2));

  let baseline = INITIAL_BASELINE;
  let v1       = INITIAL_V1;
  let v2       = INITIAL_V2;
  try { baseline = JSON.parse(baselineStr); } catch {}
  try { v1       = JSON.parse(v1Str);       } catch {}
  try { v2       = JSON.parse(v2Str);       } catch {}

  const data = LEVELS.map(level => {
    const b  = baseline[level] || { avg: 0 };
    const t1 = v1[level]       || { avg: 0 };
    const t2 = v2[level]       || { avg: 0 };
    return {
      name:        level.charAt(0).toUpperCase() + level.slice(1),
      Baseline:    +(b.avg.toFixed(4)),
      'V1 (200 steps)': +(t1.avg.toFixed(4)),
      'V2 (400 steps)': +(t2.avg.toFixed(4)),
      deltaV1:     +(t1.avg - b.avg).toFixed(4),
      deltaV2:     +(t2.avg - b.avg).toFixed(4),
      pctV1:       b.avg > 0 ? +((t1.avg - b.avg) / b.avg * 100).toFixed(1) : 0,
      pctV2:       b.avg > 0 ? +((t2.avg - b.avg) / b.avg * 100).toFixed(1) : 0,
    };
  });

  const v2Ready = LEVELS.some(l => (v2[l]?.n || 0) > 0);

  return (
    <div style={{ backgroundColor: '#0a0a0f', color: '#e2e8f0', fontFamily: "'Inter', system-ui, sans-serif", minHeight: '100vh', padding: '2rem 2.5rem' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        .card { background: #12121e; border: 1px solid #1e2035; border-radius: 12px; padding: 1.5rem; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; }
        .textarea { width: 100%; height: 140px; background: #0d0d1a; color: #a3adc2; border: 1px solid #1e2035; border-radius: 8px; padding: 10px 12px; font-family: 'Courier New', monospace; font-size: 12px; resize: vertical; margin-top: 8px; }
        .textarea:focus { outline: none; border-color: #4ade80; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }
        .tag-baseline { background: #6366f120; color: #818cf8; border: 1px solid #6366f140; }
        .tag-v1 { background: #f59e0b20; color: #fbbf24; border: 1px solid #f59e0b40; }
        .tag-v2 { background: #4ade8020; color: #4ade80; border: 1px solid #4ade8040; }
        .metric-val { font-size: 2.2rem; font-weight: 700; letter-spacing: -1px; }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        .neutral  { color: #6b7280; }
        h1 { margin: 0 0 0.25rem; font-size: 1.75rem; font-weight: 700; }
        h2 { margin: 0 0 1rem; font-size: 1.1rem; font-weight: 600; color: #94a3b8; }
        h3 { margin: 0 0 0.5rem; font-size: 0.95rem; font-weight: 600; color: #cbd5e1; }
        label { font-size: 0.75rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
        .section-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #4b5563; font-weight: 600; margin-bottom: 1rem; }
        .divider { border: none; border-top: 1px solid #1e2035; margin: 2rem 0; }
        @media (max-width: 768px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
          <h1>PR Review RL Agent</h1>
          <span className="badge tag-v2">Training Complete</span>
        </div>
        <h2>3-Stage GRPO Training Results — Qwen2.5-1.5B on PRRegressionAudit Environment</h2>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge tag-baseline">Baseline: Random Actions</span>
          <span className="badge tag-v1">V1: 200 GRPO steps · L4 GPU</span>
          <span className="badge tag-v2">{v2Ready ? 'V2: 400 GRPO steps · A100 GPU' : 'V2: Training in progress...'}</span>
        </div>
      </div>

      {/* Live Data Entry */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <p className="section-title">📋 Live Data — Paste real results to update charts</p>
        <div className="grid-3">
          <div>
            <label>Baseline JSON</label>
            <textarea className="textarea" value={baselineStr} onChange={e => setBaselineStr(e.target.value)} />
          </div>
          <div>
            <label>V1 Trained JSON (200 steps)</label>
            <textarea className="textarea" value={v1Str} onChange={e => setV1Str(e.target.value)} />
          </div>
          <div>
            <label>V2 Trained JSON (400 steps) {!v2Ready && '⏳'}</label>
            <textarea className="textarea" value={v2Str} onChange={e => setV2Str(e.target.value)} />
          </div>
        </div>
      </div>

      {/* Metric Cards — V1 vs Baseline */}
      <p className="section-title">📊 V1 vs Baseline — Improvement per difficulty</p>
      <div className="grid-3" style={{ marginBottom: '2rem' }}>
        {data.map(d => (
          <div className="card" key={d.name} style={{ textAlign: 'center' }}>
            <h3>{d.name}</h3>
            <div className={`metric-val ${d.pctV1 >= 0 ? 'positive' : 'negative'}`}>
              {d.pctV1 > 0 ? '+' : ''}{d.pctV1}%
            </div>
            <div style={{ color: '#475569', fontSize: '0.8rem', marginTop: '0.4rem' }}>
              {d.Baseline.toFixed(3)} → {d['V1 (200 steps)'].toFixed(3)}
            </div>
            {v2Ready && (
              <div style={{ marginTop: '0.75rem', padding: '0.4rem 0.75rem', background: '#0d1f1a', borderRadius: 6, fontSize: '0.8rem' }}>
                <span style={{ color: '#4ade80', fontWeight: 600 }}>V2: {d.pctV2 > 0 ? '+' : ''}{d.pctV2}%</span>
                <span style={{ color: '#475569', marginLeft: '0.5rem' }}>→ {d['V2 (400 steps)'].toFixed(3)}</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        {/* Grouped Bar Chart */}
        <div className="card">
          <h3>Average Score by Difficulty</h3>
          <p style={{ color: '#4b5563', fontSize: '0.78rem', margin: '0 0 1rem' }}>Higher is better · Max possible: 1.0</p>
          <div style={{ height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2035" vertical={false} />
                <XAxis dataKey="name" stroke="#4b5563" tickLine={false} axisLine={false} style={{ fontSize: 12 }} />
                <YAxis stroke="#4b5563" domain={[0, 0.6]} tickLine={false} axisLine={false} style={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#12121e', border: '1px solid #1e2035', borderRadius: 8, color: '#e2e8f0' }}
                  itemStyle={{ color: '#e2e8f0' }}
                  formatter={(v) => v.toFixed(4)}
                />
                <Legend wrapperStyle={{ paddingTop: 16, fontSize: 12 }} />
                <Bar dataKey="Baseline"           fill={COLORS.baseline} radius={[4,4,0,0]} maxBarSize={40} />
                <Bar dataKey="V1 (200 steps)"     fill={COLORS.v1}       radius={[4,4,0,0]} maxBarSize={40} />
                {v2Ready && <Bar dataKey="V2 (400 steps)" fill={COLORS.v2} radius={[4,4,0,0]} maxBarSize={40} />}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Delta Line Chart */}
        <div className="card">
          <h3>Score Delta vs Baseline</h3>
          <p style={{ color: '#4b5563', fontSize: '0.78rem', margin: '0 0 1rem' }}>Positive = improvement over random baseline</p>
          <div style={{ height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2035" vertical={false} />
                <XAxis dataKey="name" stroke="#4b5563" tickLine={false} axisLine={false} style={{ fontSize: 12 }} />
                <YAxis stroke="#4b5563" tickLine={false} axisLine={false} style={{ fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#374151" strokeDasharray="4 4" label={{ value: 'Baseline', fill: '#4b5563', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#12121e', border: '1px solid #1e2035', borderRadius: 8, color: '#e2e8f0' }}
                  formatter={(v) => v.toFixed(4)}
                />
                <Legend wrapperStyle={{ paddingTop: 16, fontSize: 12 }} />
                <Line type="monotone" dataKey="deltaV1" name="V1 Delta" stroke={COLORS.v1} strokeWidth={2.5} dot={{ r: 5, fill: '#12121e', strokeWidth: 2 }} />
                {v2Ready && <Line type="monotone" dataKey="deltaV2" name="V2 Delta" stroke={COLORS.v2} strokeWidth={2.5} dot={{ r: 5, fill: '#12121e', strokeWidth: 2 }} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Stats Table */}
      <div className="card">
        <p className="section-title">📈 Full Results Table</p>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #1e2035' }}>
              {['Difficulty', 'Baseline', 'V1 Mean', 'V1 Δ', 'V1 %', v2Ready && 'V2 Mean', v2Ready && 'V2 Δ', v2Ready && 'V2 %'].filter(Boolean).map(h => (
                <th key={h} style={{ textAlign: 'left', padding: '10px 12px', color: '#64748b', fontWeight: 500 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((d, i) => (
              <tr key={d.name} style={{ borderBottom: i < 2 ? '1px solid #1e2035' : 'none' }}>
                <td style={{ padding: '12px', fontWeight: 600 }}>{d.name}</td>
                <td style={{ padding: '12px', color: '#818cf8' }}>{d.Baseline.toFixed(4)}</td>
                <td style={{ padding: '12px', color: '#fbbf24' }}>{d['V1 (200 steps)'].toFixed(4)}</td>
                <td style={{ padding: '12px' }} className={d.deltaV1 >= 0 ? 'positive' : 'negative'}>{d.deltaV1 > 0 ? '+' : ''}{d.deltaV1.toFixed(4)}</td>
                <td style={{ padding: '12px' }} className={d.pctV1 >= 0 ? 'positive' : 'negative'}>{d.pctV1 > 0 ? '+' : ''}{d.pctV1}%</td>
                {v2Ready && <td style={{ padding: '12px', color: '#4ade80' }}>{d['V2 (400 steps)'].toFixed(4)}</td>}
                {v2Ready && <td style={{ padding: '12px' }} className={d.deltaV2 >= 0 ? 'positive' : 'negative'}>{d.deltaV2 > 0 ? '+' : ''}{d.deltaV2.toFixed(4)}</td>}
                {v2Ready && <td style={{ padding: '12px' }} className={d.pctV2 >= 0 ? 'positive' : 'negative'}>{d.pctV2 > 0 ? '+' : ''}{d.pctV2}%</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <hr className="divider" />
      <p style={{ color: '#374151', fontSize: '0.72rem', textAlign: 'center' }}>
        PRRegressionAudit · Qwen2.5-1.5B-Instruct · GRPO via Unsloth + TRL · Meta × Scaler OpenEnv Hackathon 2026
      </p>
    </div>
  );
}
