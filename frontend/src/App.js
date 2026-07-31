import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const BIOPOLYMERS = ['PLA', 'PCL', 'PGA', 'PLGA', 'Chitosan', 'Gelatin'];
const PRINTING_METHODS = ['FDM', 'SLA', 'SLS', 'salt leaching', 'freeze drying', 'electrospinning'];

// compressive_strength/elastic_modulus come from a physics model (Gibson-Ashby)
// calibrated to real bulk material properties. The other 3 have no experimental
// data behind them yet and are literature-informed heuristic estimates only.
const HEURISTIC_PROPS = new Set(['degradation_rate', 'cell_viability', 'bone_regeneration_score']);
const MODELS = [
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'neural_network', label: 'Neural Network' },
];

const MODEL_COLORS = {
  random_forest: '#2ecc71',
  xgboost: '#3498db',
  lightgbm: '#e67e22',
  neural_network: '#9b59b6',
};

function App() {
  const [form, setForm] = useState({
    biopolymer_type: 'PLA',
    pore_size: 300,
    porosity: 70,
    printing_method: 'FDM',
    layer_thickness: 200,
    material_composition: 0.5,
    density: 1.0,
  });
  const [selectedModel, setSelectedModel] = useState('random_forest');
  const [result, setResult] = useState(null);
  const [allResults, setAllResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [modelInfo, setModelInfo] = useState([]);

  useEffect(() => {
    axios.get(`${API_BASE}/models`).then(r => setModelInfo(r.data.models)).catch(() => {});
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setAllResults(null);
    try {
      const payload = { ...form, pore_size: Number(form.pore_size), porosity: Number(form.porosity), layer_thickness: Number(form.layer_thickness), material_composition: Number(form.material_composition), density: Number(form.density) };
      if (!showAll) {
        const res = await axios.post(`${API_BASE}/api/predict?model=${selectedModel}`, payload);
        setResult(res.data);
      } else {
        const res = await axios.post(`${API_BASE}/api/predict/all`, payload);
        setAllResults(res.data.results);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const ResultCard = ({ label, value, unit, color, heuristic }) => (
    <div className="result-card" style={{ borderLeftColor: color || '#3498db' }}>
      <div className="result-label">
        {label}
        {heuristic && <span className="heuristic-badge" title="Literature-informed estimate, not experimentally validated">estimate</span>}
      </div>
      <div className="result-value">{value}</div>
      <div className="result-unit">{unit}</div>
    </div>
  );

  return (
    <div className="app">
      <header className="header">
        <h1>Bone Scaffold Property Predictor</h1>
        <p className="subtitle">AI-powered prediction of scaffold mechanical and biological properties</p>
      </header>

      <div className="container">
        <div className="form-section">
          <form onSubmit={handlePredict}>
            <div className="form-grid">
              <div className="form-group">
                <label>Biopolymer Type</label>
                <select name="biopolymer_type" value={form.biopolymer_type} onChange={handleChange}>
                  {BIOPOLYMERS.map(b => <option key={b} value={b}>{b}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label>Pore Size (μm)</label>
                <input type="number" name="pore_size" value={form.pore_size} onChange={handleChange} min={10} max={1000} step={10} />
                <input type="range" name="pore_size" value={form.pore_size} onChange={handleChange} min={10} max={1000} />
              </div>

              <div className="form-group">
                <label>Porosity (%)</label>
                <input type="number" name="porosity" value={form.porosity} onChange={handleChange} min={10} max={100} step={1} />
                <input type="range" name="porosity" value={form.porosity} onChange={handleChange} min={10} max={100} />
              </div>

              <div className="form-group">
                <label>Printing Method</label>
                <select name="printing_method" value={form.printing_method} onChange={handleChange}>
                  {PRINTING_METHODS.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label>Layer Thickness (μm)</label>
                <input type="number" name="layer_thickness" value={form.layer_thickness} onChange={handleChange} min={10} max={500} step={10} />
                <input type="range" name="layer_thickness" value={form.layer_thickness} onChange={handleChange} min={10} max={500} />
              </div>

              <div className="form-group">
                <label>Material Composition</label>
                <input type="number" name="material_composition" value={form.material_composition} onChange={handleChange} min={0} max={1} step={0.05} />
                <input type="range" name="material_composition" value={form.material_composition} onChange={handleChange} min={0} max={1} step={0.05} />
              </div>

              <div className="form-group">
                <label>Density (g/cm³)</label>
                <input type="number" name="density" value={form.density} onChange={handleChange} min={0.01} max={3.0} step={0.05} />
                <input type="range" name="density" value={form.density} onChange={handleChange} min={0.01} max={3.0} step={0.05} />
              </div>
            </div>

            <div className="model-selector">
              <label>AI Model:</label>
              <div className="model-buttons">
                {MODELS.map(m => (
                  <button
                    key={m.value}
                    type="button"
                    className={`model-btn ${selectedModel === m.value ? 'active' : ''}`}
                    style={selectedModel === m.value ? { borderColor: MODEL_COLORS[m.value], background: MODEL_COLORS[m.value] + '20' } : {}}
                    onClick={() => setSelectedModel(m.value)}
                    disabled={showAll}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="compare-toggle">
              <label>
                <input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)} />
                Compare all models
              </label>
            </div>

            <button type="submit" className="predict-btn" disabled={loading}>
              {loading ? 'Predicting...' : 'Predict Properties'}
            </button>
          </form>

          {error && <div className="error">{error}</div>}
        </div>

        <div className="results-section">
          {result && !showAll && (
            <div className="results">
              <h2>Prediction Results</h2>
              <p className="model-info">Model: {MODELS.find(m => m.value === result.model_used)?.label} | Confidence (R²): {result.model_confidence}</p>
              <div className="results-grid">
                <ResultCard label="Compressive Strength" value={`${result.compressive_strength} MPa`} unit="MPa" color="#e74c3c" />
                <ResultCard label="Elastic Modulus" value={`${result.elastic_modulus} MPa`} unit="MPa" color="#3498db" />
                <ResultCard label="Degradation Rate" value={`${result.degradation_rate}`} unit="months" color="#f39c12" heuristic />
                <ResultCard label="Cell Viability" value={`${result.cell_viability}%`} unit="%" color="#2ecc71" heuristic />
                <ResultCard label="Bone Regeneration Score" value={result.bone_regeneration_score} unit="/10" color="#9b59b6" heuristic />
              </div>
              <p className="heuristic-note">
                "estimate" properties have no experimental training data yet — they're literature-informed
                heuristic formulas, not validated predictions. Compressive strength and elastic modulus come
                from a physics model (Gibson-Ashby) calibrated to real bulk material properties.
              </p>
            </div>
          )}

          {allResults && showAll && (
            <div className="results">
              <h2>Model Comparison</h2>
              <div className="comparison-table-wrapper">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Property</th>
                      {Object.keys(allResults).map(m => (
                        <th key={m} style={{ color: MODEL_COLORS[m] }}>{MODELS.find(x => x.value === m)?.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { key: 'compressive_strength', label: 'Compressive Strength (MPa)' },
                      { key: 'elastic_modulus', label: 'Elastic Modulus (MPa)' },
                      { key: 'degradation_rate', label: 'Degradation Rate (months)' },
                      { key: 'cell_viability', label: 'Cell Viability (%)' },
                      { key: 'bone_regeneration_score', label: 'Bone Regeneration Score' },
                    ].map(prop => (
                      <tr key={prop.key}>
                        <td>{prop.label}{HEURISTIC_PROPS.has(prop.key) && <span className="heuristic-badge" title="Literature-informed estimate, not experimentally validated">estimate</span>}</td>
                        {Object.keys(allResults).map(m => (
                          <td key={m}>{allResults[m][prop.key]?.toFixed(2) ?? '-'}</td>
                        ))}
                      </tr>
                    ))}
                    <tr>
                      <td><strong>Confidence (R²)</strong></td>
                      {Object.keys(allResults).map(m => (
                        <td key={m}>{(allResults[m].confidence * 100).toFixed(1)}%</td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="model-stats">
            <h3>Model Performance</h3>
            <div className="stats-grid">
              {modelInfo.map(m => (
                <div key={m.name} className="stat-card" style={{ borderTopColor: MODEL_COLORS[m.name] }}>
                  <div className="stat-name">{m.label}</div>
                  <div className="stat-value">R²: {m.r2_score?.toFixed(4) ?? 'N/A'}</div>
                  <div className="stat-value small">RMSE: {m.rmse?.toFixed(4) ?? 'N/A'}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
