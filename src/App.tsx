import React, { useState, useMemo, useEffect } from 'react';
import { 
  BarChart3, 
  Play, 
  RefreshCw, 
  Settings2, 
  Zap, 
  ArrowRight, 
  Activity,
  BookOpen
} from 'lucide-react';
import { 
  ComposedChart,
  Bar, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Area,
  ReferenceDot,
  ReferenceLine,
  ReferenceArea,
  Label
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { MathJax, MathJaxContext } from 'better-react-mathjax';

// --- Types ---
interface Parameter {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

interface Distribution {
  id: string;
  name: string;
  params: Parameter[];
  pdf: (x: number, params: Record<string, number>) => number;
  range: (params: Record<string, number>) => [number, number];
  mean: (params: Record<string, number>) => number | string;
  variance: (params: Record<string, number>) => number | string;
  formulas: {
    pdf: (params: Record<string, number>) => string;
    mean: (params: Record<string, number>) => string;
    variance: (params: Record<string, number>) => string;
  };
}

// --- Constants & Distributions ---
const distributions: Distribution[] = [
  {
    id: 'exponential',
    name: 'Exponential',
    params: [
      { id: 'lambda', label: 'Lambda (λ)', min: 0.1, max: 5, step: 0.1, default: 1 }
    ],
    pdf: (x, p) => x < 0 ? 0 : p.lambda * Math.exp(-p.lambda * x),
    range: (p) => [0, Math.max(5, 5 / p.lambda)],
    mean: (p) => 1 / p.lambda,
    variance: (p) => 1 / (p.lambda * p.lambda),
    formulas: {
      pdf: (p) => `f(x; \\lambda) = \\lambda e^{-\\lambda x}, x \\ge 0`,
      mean: (p) => `E[X] = \\frac{1}{\\lambda} = ${(1/p.lambda).toFixed(4)}`,
      variance: (p) => `Var(X) = \\frac{1}{\\lambda^2} = ${(1/(p.lambda*p.lambda)).toFixed(4)}`
    }
  },
  {
    id: 'normal',
    name: 'Normal (Gaussian)',
    params: [
      { id: 'mu', label: 'Mean (μ)', min: -5, max: 5, step: 0.1, default: 0 },
      { id: 'sigma', label: 'Std Dev (σ)', min: 0.1, max: 3, step: 0.1, default: 1 }
    ],
    pdf: (x, p) => {
      const exponent = -Math.pow(x - p.mu, 2) / (2 * Math.pow(p.sigma, 2));
      return (1 / (p.sigma * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
    },
    range: (p) => [p.mu - 4 * p.sigma, p.mu + 4 * p.sigma],
    mean: (p) => p.mu,
    variance: (p) => p.sigma * p.sigma,
    formulas: {
      pdf: (p) => `f(x; \\mu, \\sigma) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}`,
      mean: (p) => `E[X] = \\mu = ${p.mu}`,
      variance: (p) => `Var(X) = \\sigma^2 = ${(p.sigma * p.sigma).toFixed(4)}`
    }
  },
  {
    id: 'weibull',
    name: 'Weibull',
    params: [
      { id: 'lambda', label: 'Scale (λ)', min: 0.1, max: 5, step: 0.1, default: 1 },
      { id: 'k', label: 'Shape (k)', min: 0.1, max: 5, step: 0.1, default: 1.5 }
    ],
    pdf: (x, p) => {
      if (x < 0) return 0;
      return (p.k / p.lambda) * Math.pow(x / p.lambda, p.k - 1) * Math.exp(-Math.pow(x / p.lambda, p.k));
    },
    range: (p) => [0, p.lambda * 3],
    mean: (p) => {
      const gamma = (n: number) => {
        // Simple approximation for Gamma function
        const g = 7;
        const C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
          771.32342877765313, -176.61502916214059, 12.507343278686905,
          -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (n < 0.5) return Math.PI / (Math.sin(Math.PI * n) * gamma(1 - n));
        n -= 1;
        let x = C[0];
        for (let i = 1; i < g + 2; i++) x += C[i] / (n + i);
        const t = n + g + 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, n + 0.5) * Math.exp(-t) * x;
      };
      return p.lambda * gamma(1 + 1 / p.k);
    },
    variance: (p) => {
      const gamma = (n: number) => {
        const g = 7;
        const C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
          771.32342877765313, -176.61502916214059, 12.507343278686905,
          -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (n < 0.5) return Math.PI / (Math.sin(Math.PI * n) * gamma(1 - n));
        n -= 1;
        let x = C[0];
        for (let i = 1; i < g + 2; i++) x += C[i] / (n + i);
        const t = n + g + 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, n + 0.5) * Math.exp(-t) * x;
      };
      const m1 = gamma(1 + 1 / p.k);
      const m2 = gamma(1 + 2 / p.k);
      return p.lambda * p.lambda * (m2 - m1 * m1);
    },
    formulas: {
      pdf: (p) => `f(x; \\lambda, k) = \\frac{k}{\\lambda}\\left(\\frac{x}{\\lambda}\\right)^{k-1}e^{-(x/\\lambda)^k}`,
      mean: (p) => `E[X] = \\lambda\\Gamma(1+1/k)`,
      variance: (p) => `Var(X) = \\lambda^2[\\Gamma(1+2/k) - (\\Gamma(1+1/k))^2]`
    }
  },
  {
    id: 'beta',
    name: 'Beta',
    params: [
      { id: 'alpha', label: 'Alpha (α)', min: 0.1, max: 10, step: 0.1, default: 2 },
      { id: 'beta', label: 'Beta (β)', min: 0.1, max: 10, step: 0.1, default: 5 }
    ],
    pdf: (x, p) => {
      if (x < 0 || x > 1) return 0;
      const gamma = (n: number) => {
        const g = 7;
        const C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
          771.32342877765313, -176.61502916214059, 12.507343278686905,
          -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (n < 0.5) return Math.PI / (Math.sin(Math.PI * n) * gamma(1 - n));
        n -= 1;
        let x = C[0];
        for (let i = 1; i < g + 2; i++) x += C[i] / (n + i);
        const t = n + g + 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, n + 0.5) * Math.exp(-t) * x;
      };
      const betaFunc = (gamma(p.alpha) * gamma(p.beta)) / gamma(p.alpha + p.beta);
      return (Math.pow(x, p.alpha - 1) * Math.pow(1 - x, p.beta - 1)) / betaFunc;
    },
    range: (p) => [0, 1],
    mean: (p) => p.alpha / (p.alpha + p.beta),
    variance: (p) => (p.alpha * p.beta) / (Math.pow(p.alpha + p.beta, 2) * (p.alpha + p.beta + 1)),
    formulas: {
      pdf: (p) => `f(x; \\alpha, \\beta) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha, \\beta)}`,
      mean: (p) => `E[X] = \\frac{\\alpha}{\\alpha+\\beta} = ${(p.alpha / (p.alpha + p.beta)).toFixed(4)}`,
      variance: (p) => `Var(X) = \\frac{\\alpha\\beta}{(\\alpha+\\beta)^2(\\alpha+\\beta+1)}`
    }
  },
  {
    id: 'gamma',
    name: 'Gamma',
    params: [
      { id: 'k', label: 'Shape (k)', min: 0.1, max: 10, step: 0.1, default: 2 },
      { id: 'theta', label: 'Scale (θ)', min: 0.1, max: 5, step: 0.1, default: 2 }
    ],
    pdf: (x, p) => {
      if (x < 0) return 0;
      const gamma = (n: number) => {
        const g = 7;
        const C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
          771.32342877765313, -176.61502916214059, 12.507343278686905,
          -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (n < 0.5) return Math.PI / (Math.sin(Math.PI * n) * gamma(1 - n));
        n -= 1;
        let x = C[0];
        for (let i = 1; i < g + 2; i++) x += C[i] / (n + i);
        const t = n + g + 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, n + 0.5) * Math.exp(-t) * x;
      };
      return (1 / (gamma(p.k) * Math.pow(p.theta, p.k))) * Math.pow(x, p.k - 1) * Math.exp(-x / p.theta);
    },
    range: (p) => [0, p.k * p.theta * 2.5],
    mean: (p) => p.k * p.theta,
    variance: (p) => p.k * p.theta * p.theta,
    formulas: {
      pdf: (p) => `f(x; k, \\theta) = \\frac{1}{\\Gamma(k)\\theta^k} x^{k-1} e^{-x/\\theta}`,
      mean: (p) => `E[X] = k\\theta = ${(p.k * p.theta).toFixed(4)}`,
      variance: (p) => `Var(X) = k\\theta^2 = ${(p.k * p.theta * p.theta).toFixed(4)}`
    }
  }
];

const Translations = {
  en: {
    title: 'Rejection',
    subtitle: 'Sampler',
    description: 'A visual tool for generating samples from givent probability distributions with known PDF using the Rejection Sampling method.',
    distParams: 'Distribution & Parameters',
    simulation: 'Simulation Controls',
    gen1: 'Generate 1 Point',
    clear: 'Clear All',
    transProcess: 'Rejection Process',
    resultsTitle: 'Final Distribution',
    statsTitle: 'Statistics',
    formulasTitle: 'Mathematical Basis',
    step1: 'Proposed X',
    step2: 'Uniform Y',
    targetValue: 'Target Value (x)',
    cumProb: 'Probability Density f(x)',
    value: 'Value',
    density: 'Density',
    noSamples: 'No samples generated yet. Click "Generate" to start.',
    totalSamples: 'Total Accepted Samples',
    rejectedCount: 'Rejected Points',
    totalGenerated: 'Total Generated',
    acceptanceRatio: 'Acceptance Ratio',
    mean: 'Mean',
    variance: 'Variance',
    actual: 'Actual',
    theoretical: 'Theoretical',
    legendEmpirical: 'Empirical Histogram',
    legendTheoretical: 'Theoretical PDF',
    autoRange: 'Auto Range',
    fixedRange: 'Fixed Range',
    howItWorks: 'How it works',
    howItWorksDesc: 'Rejection sampling works by generating a random point (X, Y) within a bounding box. If Y is less than the probability density f(X), the point is accepted. Otherwise, it is rejected.',
    mathFormula: 'Acceptance Criterion',
    copyright: "© Richard Lipka, ZČU, lipka@kiv.zcu.cz",
    pdfLabel: 'Probability Density Function',
    meanLabel: 'Expected Value',
    varianceLabel: 'Variance',
    count: 'Count',
    bin: 'Bin',
    empiricalDensity: 'Empirical Density',
    theoreticalPdf: 'Theoretical PDF',
    paramNote: 'Current parameters:'
  },
  cs: {
    title: 'Rejection',
    subtitle: 'Sampler',
    description: 'Vizuální nástroj pro generování vzorků ze zadaných pravděpodobnostních rozdělení se známou fukncí hustoty pravděpodobnosti (PDF) metodou zamítání vzorků.',
    distParams: 'Rozdělení a parametry',
    simulation: 'Ovládání simulace',
    gen1: 'Generovat 1 bod',
    clear: 'Vymazat vše',
    transProcess: 'Proces zamítání',
    resultsTitle: 'Výsledné rozdělení',
    statsTitle: 'Statistiky',
    formulasTitle: 'Matematický základ',
    step1: 'Navržené X',
    step2: 'Uniformní Y',
    targetValue: 'Cílová hodnota (x)',
    cumProb: 'Hustota pravděpodobnosti f(x)',
    value: 'Hodnota',
    density: 'Hustota',
    noSamples: 'Zatím nebyly generovány žádné vzorky. Klikněte na "Generovat".',
    totalSamples: 'Celkem přijatých vzorků',
    rejectedCount: 'Zamítnutých bodů',
    totalGenerated: 'Celkem generováno',
    acceptanceRatio: 'Poměr přijetí',
    mean: 'Průměr',
    variance: 'Rozptyl',
    actual: 'Skutečný',
    theoretical: 'Teoretický',
    legendEmpirical: 'Empirický histogram',
    legendTheoretical: 'Teoretická PDF',
    autoRange: 'Auto rozsah',
    fixedRange: 'Fixní rozsah',
    howItWorks: 'Jak to funguje',
    howItWorksDesc: 'Zamítání vzorků funguje tak, že generuje náhodný bod (X, Y) v ohraničujícím obdélníku jako dvě náhodná čísla s uniformním rozdělním na délce hrany. Pokud je Y menší než hustota pravděpodobnosti f(X), bod je přijat. V opačném případě je zamítnut.',
    mathFormula: 'Kritérium přijetí',
    copyright: "© Richard Lipka, ZČU, lipka@kiv.zcu.cz",
    pdfLabel: 'Hustota pravděpodobnosti',
    meanLabel: 'Střední hodnota',
    varianceLabel: 'Rozptyl',
    count: 'Počet',
    bin: 'Bin',
    empiricalDensity: 'Empirická hustota',
    theoreticalPdf: 'Teoretická PDF',
    paramNote: 'Aktuální parametry:'
  }
};

const config = {
  loader: { load: ['input/tex', 'output/chtml'] }, 
};

// --- Main Component ---
export default function App() {
  const [lang, setLang] = useState<'en' | 'cs'>('en');
  const [selectedDistId, setSelectedDistId] = useState('exponential');
  const [paramValues, setParamValues] = useState<Record<string, Record<string, number>>>(() => {
    const initial: Record<string, Record<string, number>> = {};
    distributions.forEach(d => {
      initial[d.id] = {};
      d.params.forEach(p => {
        initial[d.id][p.id] = p.default;
      });
    });
    return initial;
  });
  const [samples, setSamples] = useState<number[]>([]);
  const [rejectionHistory, setRejectionHistory] = useState<{x: number, y: number, accepted: boolean}[]>([]);
  const [totalRejected, setTotalRejected] = useState(0);
  const [totalGenerated, setTotalGenerated] = useState(0);
  const [useFixedRange, setUseFixedRange] = useState(false);
  
  // Bounding Box State
  const [manualBox, setManualBox] = useState(false);
  const [boxXMin, setBoxXMin] = useState(-5);
  const [boxXMax, setBoxXMax] = useState(5);
  const [boxYMax, setBoxYMax] = useState(0.5);

  const t = (key: keyof typeof Translations['en']) => Translations[lang][key];

  const selectedDist = useMemo(() => 
    distributions.find(d => d.id === selectedDistId) || distributions[0], 
  [selectedDistId]);

  const currentParams = paramValues[selectedDistId];

  const getPdf = (x: number) => {
    return selectedDist.pdf(x, currentParams);
  };

  const getRange = () => {
    if (manualBox) return [boxXMin, boxXMax] as [number, number];
    return selectedDist.range(currentParams);
  };

  // Find max PDF value for bounding box
  const autoMaxPdf = useMemo(() => {
    const [min, max] = getRange();
    let maxVal = 0;
    const steps = 200;
    for (let i = 0; i <= steps; i++) {
      const x = min + (i / steps) * (max - min);
      maxVal = Math.max(maxVal, getPdf(x));
    }
    return maxVal * 1.1; // Add 10% buffer
  }, [selectedDistId, currentParams, manualBox, boxXMin, boxXMax]);

  const currentMaxPdf = manualBox ? boxYMax : autoMaxPdf;

  const generatePoints = (count: number) => {
    const [min, max] = getRange();
    const newSamples: number[] = [];
    const newHistory: {x: number, y: number, accepted: boolean}[] = [];
    let rejected = 0;
    let generated = 0;

    while (newSamples.length < count && generated < 10000) {
      const x = min + Math.random() * (max - min);
      const y = Math.random() * currentMaxPdf;
      const pdfVal = getPdf(x);
      const accepted = y <= pdfVal;

      generated++;
      if (accepted) {
        newSamples.push(x);
        if (count === 1) newHistory.push({x, y, accepted: true});
      } else {
        rejected++;
        if (count === 1) newHistory.push({x, y, accepted: false});
      }
    }

    setSamples(prev => [...prev, ...newSamples]);
    setTotalRejected(prev => prev + rejected);
    setTotalGenerated(prev => prev + generated);
    if (count === 1) setRejectionHistory(newHistory);
    else setRejectionHistory([]);
  };

  const clearSamples = () => {
    setSamples([]);
    setRejectionHistory([]);
    setTotalRejected(0);
    setTotalGenerated(0);
  };

  const updateParam = (distId: string, paramId: string, value: number) => {
    setParamValues(prev => ({
      ...prev,
      [distId]: {
        ...prev[distId],
        [paramId]: value
      }
    }));
    clearSamples();
  };

  const pdfData = useMemo(() => {
    const [min, max] = getRange();
    const points = 100;
    return Array.from({ length: points + 1 }, (_, i) => {
      const x = min + (i / points) * (max - min);
      return { x, y: getPdf(x) };
    });
  }, [selectedDistId, currentParams]);

  const stats = useMemo(() => {
    if (samples.length === 0) return null;
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / samples.length;
    return { mean, variance };
  }, [samples]);

  const histogramData = useMemo(() => {
    if (samples.length === 0) return [];
    const [min, max] = getRange();
    
    const binCount = 30;
    const binWidth = (max - min) / binCount;
    
    const bins = Array.from({ length: binCount }, (_, i) => ({
      x: min + (i + 0.5) * binWidth,
      count: 0,
      range: `${(min + i * binWidth).toFixed(2)} - ${(min + (i + 1) * binWidth).toFixed(2)}`
    }));

    samples.forEach(s => {
      const binIdx = Math.min(binCount - 1, Math.floor((s - min) / binWidth));
      if (binIdx >= 0) bins[binIdx].count++;
    });

    const total = samples.length;
    const pdf = getPdf;

    return bins.map(b => {
      const x = b.x;
      const count = b.count;
      return {
        ...b,
        density: count / (total * binWidth),
        expected: pdf(x),
      };
    });
  }, [samples, selectedDist, currentParams, selectedDistId]);

  return (
    <MathJaxContext version={3} config={config}>
      <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-4 md:p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <header className="mb-8 flex flex-col md:flex-row md:items-start justify-between gap-4">
            <div>
              <h1 className="text-4xl font-bold tracking-tight text-blue-900 mb-2">
                {t('title')} <span className="text-blue-600">{t('subtitle')}</span>
              </h1>
              <p className="text-slate-600 max-w-2xl">
                {t('description')}
              </p>
            </div>
            <div className="flex flex-col items-end gap-3">
              <div className="flex items-center gap-1 bg-white p-1 rounded-xl shadow-sm border border-slate-200">
                <button 
                  onClick={() => setLang('en')}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${lang === 'en' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:bg-slate-50'}`}
                >
                  en
                </button>
                <button 
                  onClick={() => setLang('cs')}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${lang === 'cs' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:bg-slate-50'}`}
                >
                  cz
                </button>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Controls Panel */}
            <aside className="lg:col-span-3 space-y-6">
              <section className="bg-white p-6 rounded-3xl shadow-sm border border-blue-100 flex flex-col">
                <h2 className="text-lg font-bold text-blue-900 mb-4 flex items-center gap-2">
                  <Settings2 className="w-5 h-5" /> {t('distParams')}
                </h2>
                <div className="space-y-3 max-h-[420px] overflow-y-auto pr-2 custom-scrollbar">
                  {distributions.map(dist => (
                    <div key={dist.id} className="flex flex-col mb-2 last:mb-0">
                      <button
                        onClick={() => {
                          if (selectedDistId !== dist.id) {
                            setSelectedDistId(dist.id);
                            setManualBox(false);
                            clearSamples();
                          }
                        }}
                        className={`w-full text-left px-4 py-3 rounded-xl transition-all duration-200 border ${
                          selectedDistId === dist.id 
                            ? 'bg-blue-600 text-white border-blue-600 shadow-md' 
                            : 'bg-white text-slate-700 border-slate-200 hover:border-blue-300 hover:bg-blue-50'
                        }`}
                      >
                        <div className="font-semibold text-sm">{dist.name}</div>
                      </button>
                      
                      <AnimatePresence>
                        {selectedDistId === dist.id && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden bg-blue-50/50 rounded-b-xl border-x border-b border-blue-100 px-4 py-4 space-y-4"
                          >
                            {dist.params.map(p => (
                              <div key={p.id} className="space-y-1">
                                <div className="flex justify-between text-xs font-bold text-blue-800">
                                  <span>{p.label}</span>
                                  <span>{paramValues[dist.id][p.id]}</span>
                                </div>
                                <input
                                  type="range"
                                  min={p.min}
                                  max={p.max}
                                  step={p.step}
                                  value={paramValues[dist.id][p.id]}
                                  onChange={(e) => updateParam(dist.id, p.id, parseFloat(e.target.value))}
                                  className="w-full h-1.5 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                />
                              </div>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  ))}
                </div>
              </section>

              <section className="bg-white p-6 rounded-3xl shadow-sm border border-blue-100 sticky top-8">
                <h2 className="text-lg font-bold text-blue-900 mb-4 flex items-center gap-2">
                  <Zap className="w-5 h-5" /> {t('simulation')}
                </h2>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => generatePoints(1)}
                    className="col-span-2 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-xl transition-all shadow-lg active:scale-95"
                  >
                    <Play className="w-4 h-4" /> {t('gen1')}
                  </button>
                  {[10, 100, 1000].map(count => (
                    <button
                      key={count}
                      onClick={() => generatePoints(count)}
                      className="flex items-center justify-center bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold py-2 px-4 rounded-xl border border-blue-200 transition-all active:scale-95"
                    >
                      +{count}
                    </button>
                  ))}
                  <button
                    onClick={clearSamples}
                    className="flex items-center justify-center bg-slate-100 hover:bg-slate-200 text-slate-600 font-semibold py-2 px-4 rounded-xl transition-all active:scale-95"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" /> {t('clear')}
                  </button>
                </div>
              </section>

              <section className="bg-white p-6 rounded-3xl shadow-sm border border-blue-100">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-bold text-blue-900 flex items-center gap-2">
                    <Settings2 className="w-5 h-5" /> Bounding Box
                  </h2>
                  <button 
                    onClick={() => setManualBox(!manualBox)}
                    className={`text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wider transition-all ${manualBox ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-400'}`}
                  >
                    {manualBox ? 'Manual' : 'Auto'}
                  </button>
                </div>
                
                <div className={`space-y-4 transition-opacity ${manualBox ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-bold text-blue-800">
                      <span>X Min</span>
                      <span>{boxXMin.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="-10"
                      max={boxXMax - 0.1}
                      step="0.1"
                      value={boxXMin}
                      onChange={(e) => setBoxXMin(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-bold text-blue-800">
                      <span>X Max</span>
                      <span>{boxXMax.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min={boxXMin + 0.1}
                      max="20"
                      step="0.1"
                      value={boxXMax}
                      onChange={(e) => setBoxXMax(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-bold text-blue-800">
                      <span>Y Max (Height)</span>
                      <span>{boxYMax.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.01"
                      max="2"
                      step="0.01"
                      value={boxYMax}
                      onChange={(e) => setBoxYMax(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                  </div>
                </div>
              </section>
            </aside>

            {/* Main Visualization Area */}
            <main className="lg:col-span-9 space-y-8">
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                {/* Step 1: Rejection Process */}
                <section className="bg-white p-6 rounded-3xl shadow-sm border border-blue-100">
                  <h2 className="text-xl font-bold text-blue-900 mb-6 flex items-center gap-2">
                    <ArrowRight className="w-6 h-6 text-blue-500" /> {t('transProcess')}
                  </h2>
                  
                  <div className="grid grid-cols-3 gap-4 mb-8">
                    <div className="bg-blue-50 p-3 rounded-2xl border border-blue-100">
                      <div className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-1">{t('step1')}</div>
                      <div className="text-lg font-mono font-bold text-blue-900">
                        X = {rejectionHistory.length > 0 ? rejectionHistory[rejectionHistory.length - 1].x.toFixed(4) : '—'}
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-2xl border border-blue-100">
                      <div className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-1">{t('step2')}</div>
                      <div className="text-lg font-mono font-bold text-blue-900">
                        Y = {rejectionHistory.length > 0 ? rejectionHistory[rejectionHistory.length - 1].y.toFixed(4) : '—'}
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-2xl border border-blue-100">
                      <div className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-1">Status</div>
                      <div className={`text-lg font-bold ${rejectionHistory.length > 0 ? (rejectionHistory[rejectionHistory.length - 1].accepted ? 'text-emerald-600' : 'text-rose-600') : 'text-blue-900'}`}>
                        {rejectionHistory.length > 0 ? (rejectionHistory[rejectionHistory.length - 1].accepted ? 'OK' : 'FAIL') : '—'}
                      </div>
                    </div>
                  </div>

                  <div className="h-[320px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={pdfData} margin={{ top: 10, right: 30, left: 0, bottom: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis 
                          dataKey="x" 
                          type="number" 
                          domain={useFixedRange ? [-20, 20] : getRange()} 
                          label={{ value: t('targetValue'), position: 'insideBottom', offset: -20, fontSize: 12, fontWeight: 600, fill: '#64748b' }}
                        />
                        <YAxis 
                          domain={[0, Math.max(currentMaxPdf, 0.1)]} 
                          label={{ value: t('cumProb'), angle: -90, position: 'insideLeft', fontSize: 12 }}
                        />
                        <Tooltip 
                          contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                          formatter={(value: number) => [value.toFixed(4), 'PDF(x)']}
                        />
                         <Area type="monotone" dataKey="y" stroke="#2563eb" fill="#dbeafe" strokeWidth={2} />
                        
                        {/* Bounding Box Visualization */}
                        <ReferenceArea 
                          {...({
                            x1: getRange()[0],
                            x2: getRange()[1],
                            y1: 0,
                            y2: currentMaxPdf,
                            stroke: "#3b82f6",
                            strokeDasharray: "5 5",
                            strokeWidth: 1,
                            fill: "none"
                          } as any)}
                        />

                        {rejectionHistory.length > 0 && (
                          <>
                            {/* Projection Lines for all points in the current generation attempt */}
                            {rejectionHistory.map((pt, idx) => (
                              <React.Fragment key={`lines-${idx}`}>
                                <ReferenceLine 
                                  x={pt.x} 
                                  stroke={pt.accepted ? "#10b981" : "#f43f5e"} 
                                  strokeWidth={0.5} 
                                  strokeDasharray="3 3"
                                  opacity={idx === rejectionHistory.length - 1 ? 1 : 0.4}
                                >
                                  <Label 
                                    value={pt.x.toFixed(2)} 
                                    position="insideBottom" 
                                    fill={pt.accepted ? "#065f46" : "#9f1239"} 
                                    fontSize={8} 
                                    fontWeight={idx === rejectionHistory.length - 1 ? "bold" : "normal"}
                                    offset={idx * 2} // Slight offset to avoid perfect overlap if values are very close
                                  />
                                </ReferenceLine>
                                <ReferenceLine 
                                  y={pt.y} 
                                  stroke={pt.accepted ? "#10b981" : "#f43f5e"} 
                                  strokeWidth={0.5} 
                                  strokeDasharray="3 3"
                                  opacity={idx === rejectionHistory.length - 1 ? 1 : 0.4}
                                >
                                  <Label 
                                    value={pt.y.toFixed(2)} 
                                    position="insideLeft" 
                                    fill={pt.accepted ? "#065f46" : "#9f1239"} 
                                    fontSize={8} 
                                    fontWeight={idx === rejectionHistory.length - 1 ? "bold" : "normal"}
                                  />
                                </ReferenceLine>
                              </React.Fragment>
                            ))}
                          </>
                        )}

                        {rejectionHistory.map((pt, idx) => (
                          <ReferenceDot 
                            key={idx}
                            x={pt.x}
                            y={pt.y}
                            r={4}
                            fill={pt.accepted ? "#10b981" : "#f43f5e"}
                            stroke="none"
                          />
                        ))}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-4 flex justify-end">
                    <button
                      onClick={() => setUseFixedRange(!useFixedRange)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all border ${
                        useFixedRange 
                          ? 'bg-blue-600 text-white border-blue-600 shadow-sm' 
                          : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'
                      }`}
                    >
                      <Activity className="w-4 h-4" />
                      {useFixedRange ? t('fixedRange') : t('autoRange')}
                    </button>
                  </div>
                </section>

                {/* Step 2: Distribution Results */}
                <section className="bg-white p-6 rounded-3xl shadow-sm border border-blue-100">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold text-blue-900 flex items-center gap-2">
                      <BarChart3 className="w-6 h-6 text-blue-500" /> {t('resultsTitle')}
                    </h2>
                  </div>

                  {samples.length > 0 ? (
                    <div className="h-[320px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={histogramData} margin={{ top: 10, right: 30, left: 0, bottom: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis dataKey="x" label={{ value: t('value'), position: 'insideBottom', offset: -20, fontSize: 12, fontWeight: 600, fill: '#64748b' }} />
                          <YAxis label={{ value: t('density'), angle: -90, position: 'insideLeft' }} />
                          <Tooltip 
                            contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)', padding: '12px' }}
                            labelStyle={{ fontWeight: 'bold', color: '#1e3a8a', marginBottom: '4px' }}
                            labelFormatter={(label, payload) => {
                              if (payload && payload.length > 0) {
                                return `${t('bin')}: ${payload[0].payload.range}`;
                              }
                              return label;
                            }}
                            formatter={(value: number, name: string, props) => {
                              if (name === 'density') {
                                return [
                                  <div key="empirical" className="space-y-1">
                                    <div className="text-blue-600 font-bold">{value.toFixed(4)}</div>
                                    <div className="text-[10px] text-slate-400 uppercase tracking-wider">{t('empiricalDensity')}</div>
                                    <div className="text-xs text-slate-500 mt-1">{t('count')}: {props.payload.count}</div>
                                  </div>,
                                  ''
                                ];
                              }
                              if (name === 'expected') {
                                return [
                                  <div key="theoretical" className="space-y-1">
                                    <div className="text-indigo-900 font-bold">{value.toFixed(4)}</div>
                                    <div className="text-[10px] text-slate-400 uppercase tracking-wider">{t('theoreticalPdf')}</div>
                                  </div>,
                                  ''
                                ];
                              }
                              return [value, name];
                            }}
                          />
                          <Bar dataKey="density" fill="#3b82f6" radius={[4, 4, 0, 0]} opacity={0.7} name={t('empiricalDensity')} />
                          <Line 
                            type="monotone" 
                            dataKey="expected" 
                            stroke="#1e3a8a" 
                            strokeWidth={3} 
                            dot={false} 
                            name={t('theoreticalPdf')}
                          />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[320px] flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-100 rounded-2xl">
                      <BarChart3 className="w-12 h-12 mb-2 opacity-20" />
                      <p>{t('noSamples')}</p>
                    </div>
                  )}
                  <div className="flex justify-center gap-6 mt-4">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-blue-500 opacity-70 rounded"></div>
                      <span className="text-sm font-medium text-slate-600">{t('legendEmpirical')}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-1 bg-blue-900 rounded"></div>
                      <span className="text-sm font-medium text-slate-600">{t('legendTheoretical')}</span>
                    </div>
                  </div>
                </section>
              </div>

              {/* Statistics Section */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div className="lg:col-span-12 space-y-6">
                  <section className="bg-white p-6 rounded-3xl shadow-sm border border-blue-100">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Distribution Statistics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-slate-50 p-3 rounded-2xl border border-slate-100">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{t('actual')} {t('mean')}</div>
                        <div className="text-lg font-mono font-bold text-blue-900">{stats ? stats.mean.toFixed(4) : '—'}</div>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-2xl border border-slate-100">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{t('theoretical')} {t('mean')}</div>
                        <div className="text-lg font-mono font-bold text-slate-600">
                          {typeof selectedDist.mean(currentParams) === 'number' 
                            ? (selectedDist.mean(currentParams) as number).toFixed(4) 
                            : selectedDist.mean(currentParams)}
                        </div>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-2xl border border-slate-100">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{t('actual')} {t('variance')}</div>
                        <div className="text-lg font-mono font-bold text-blue-900">{stats ? stats.variance.toFixed(4) : '—'}</div>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-2xl border border-slate-100">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{t('theoretical')} {t('variance')}</div>
                        <div className="text-lg font-mono font-bold text-slate-600">
                          {typeof selectedDist.variance(currentParams) === 'number' 
                            ? (selectedDist.variance(currentParams) as number).toFixed(4) 
                            : selectedDist.variance(currentParams)}
                        </div>
                      </div>
                    </div>
                  </section>

                  <section className="bg-white p-6 rounded-3xl shadow-sm border border-emerald-100">
                    <h3 className="text-sm font-bold text-emerald-400 uppercase tracking-wider mb-4">Acceptance Statistics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-emerald-50/50 p-3 rounded-2xl border border-emerald-100">
                        <div className="text-[10px] font-bold text-emerald-600/60 uppercase tracking-wider mb-1">{t('totalSamples')}</div>
                        <div className="text-lg font-mono font-bold text-emerald-700">{samples.length.toLocaleString()}</div>
                      </div>
                      <div className="bg-rose-50/50 p-3 rounded-2xl border border-rose-100">
                        <div className="text-[10px] font-bold text-rose-600/60 uppercase tracking-wider mb-1">{t('rejectedCount')}</div>
                        <div className="text-lg font-mono font-bold text-rose-700">{totalRejected.toLocaleString()}</div>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-2xl border border-slate-100">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{t('totalGenerated')}</div>
                        <div className="text-lg font-mono font-bold text-slate-700">{totalGenerated.toLocaleString()}</div>
                      </div>
                      <div className="bg-blue-50/50 p-3 rounded-2xl border border-blue-100">
                        <div className="text-[10px] font-bold text-blue-600/60 uppercase tracking-wider mb-1">{t('acceptanceRatio')}</div>
                        <div className="text-lg font-mono font-bold text-blue-700">
                          {totalGenerated > 0 ? ((samples.length / totalGenerated) * 100).toFixed(1) + '%' : '—'}
                        </div>
                      </div>
                    </div>
                  </section>

                  <section className="bg-white p-6 rounded-3xl shadow-sm border border-blue-100">
                    <h2 className="text-lg font-bold text-blue-900 mb-4 flex items-center gap-2">
                      <BookOpen className="w-5 h-5" /> {t('formulasTitle')}
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div>
                        <h3 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-1">{t('pdfLabel')}</h3>
                        <div className="bg-blue-50 p-3 rounded-xl border border-blue-100 font-mono text-xs text-blue-900 break-words">
                          <MathJax dynamic inline>{"\\(" + selectedDist.formulas.pdf(currentParams) + "\\)"}</MathJax>
                        </div>
                      </div>
                      <div>
                        <h3 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-1">{t('meanLabel')}</h3>
                        <div className="bg-blue-50 p-3 rounded-xl border border-blue-100 font-mono text-xs text-blue-900 break-words">
                          <MathJax dynamic inline>{"\\(" + selectedDist.formulas.mean(currentParams) + "\\)"}</MathJax>
                        </div>
                      </div>
                      <div>
                        <h3 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-1">{t('varianceLabel')}</h3>
                        <div className="bg-blue-50 p-3 rounded-xl border border-blue-100 font-mono text-xs text-blue-900 break-words">
                          <MathJax dynamic inline>{"\\(" + selectedDist.formulas.variance(currentParams) + "\\)"}</MathJax>
                        </div>
                      </div>
                    </div>
                  </section>
                </div>
              </div>
            </main>
          </div>
          
          <footer className="mt-12 pt-8 border-t border-slate-200 text-center text-slate-400 text-xs">
            <p className="mb-2">{t('copyright')}</p>
            <div className="flex justify-center gap-4">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                <span>{t('actual')}</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-blue-900"></div>
                <span>{t('theoretical')}</span>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </MathJaxContext>
  );
}
