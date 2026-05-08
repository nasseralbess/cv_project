import { useState, useEffect, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Upload, Loader2, RefreshCw } from 'lucide-react';
import './index.css';

interface Node {
  id: string;
  block: string;
  filter_index: number;
  activation: number;
  img: string;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

interface Link {
  source: string;
  target: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface AllData {
  [key: string]: GraphData;
}

const API_BASE = 'http://localhost:8000';

function App() {
  const [data, setData] = useState<AllData | null>(null);
  const [method, setMethod] = useState<string>('');
  const [imgCache, setImgCache] = useState<Record<string, HTMLImageElement>>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [selectedLabel, setSelectedLabel] = useState<number>(6);
  const [selectedModel, setSelectedModel] = useState<string>("simple_cnn");
  const [models, setModels] = useState<Record<string, any>>({});
  const [classes, setClasses] = useState<string[]>([]);
  const fgRef = useRef<any>();
  const pollInterval = useRef<any>(null);

  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(res => res.json())
      .then(setModels)
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (models[selectedModel]) {
      fetch(`${API_BASE}/classes/${models[selectedModel].dataset}`)
        .then(res => res.json())
        .then(setClasses)
        .catch(console.error);
    }
  }, [selectedModel, models]);

  useEffect(() => {
    if (classes.length > 0 && selectedLabel >= classes.length) {
      setSelectedLabel(0);
    }
  }, [classes, selectedLabel]);

  const processGraphData = (json: AllData) => {
    const methods = Object.keys(json);
    methods.forEach(m => {
      const nodes = json[m].nodes;
      const blocks = Array.from(new Set(nodes.map((n: any) => n.block)));
      const blockCounts: Record<string, number> = {};
      const blockIndices: Record<string, number> = {};
      
      nodes.forEach((n: any) => {
        blockCounts[n.block] = (blockCounts[n.block] || 0) + 1;
      });

      nodes.forEach((n: any) => {
        const blockIdx = blockIndices[n.block] || 0;
        blockIndices[n.block] = blockIdx + 1;
        
        n.fx = (blocks.indexOf(n.block) - (blocks.length-1)/2) * 250;
        n.fy = (blockIdx - (blockCounts[n.block] - 1) / 2) * 45;
      });
    });
    setData(json);
    if (methods.length > 0) setMethod(methods[0]);
    setImgCache({});
  };

  const pollStatus = async (taskId: string) => {
    try {
      const res = await fetch(`${API_BASE}/status/${taskId}`);
      const json = await res.json();
      
      if (json.status === 'completed') {
        if (pollInterval.current) clearInterval(pollInterval.current);
        processGraphData(json.result);
        setLoading(false);
        setProgress(0);
      } else if (json.status === 'failed') {
        if (pollInterval.current) clearInterval(pollInterval.current);
        alert("Processing failed: " + json.error);
        setLoading(false);
        setProgress(0);
      } else {
        setProgress(json.progress || 0);
      }
    } catch (e) {
      console.error("Polling error", e);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setProgress(0);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', selectedLabel.toString());
    formData.append('model_type', selectedModel);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const json = await res.json();
      
      if (json.status === 'completed') {
        processGraphData(json.result);
        setLoading(false);
      } else if (json.task_id) {
        pollInterval.current = setInterval(() => pollStatus(json.task_id), 1000);
      } else if (json.error) {
        alert(json.error);
        setLoading(false);
      }
    } catch (err) {
      alert("Upload failed: " + err);
      setLoading(false);
    }
  };

  const nodePaint = useCallback((node: any, ctx: CanvasRenderingContext2D) => {
    const size = 18;
    const img = imgCache[node.img];

    if (img) {
      ctx.drawImage(img, node.x - size / 2, node.y - size / 2, size, size);
    } else {
      ctx.fillStyle = '#646cff';
      ctx.beginPath();
      ctx.arc(node.x, node.y, 4, 0, 2 * Math.PI, false);
      ctx.fill();
    }

    ctx.strokeStyle = '#444';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(node.x - size / 2, node.y - size / 2, size, size);
  }, [imgCache]);

  useEffect(() => {
    if (!data || !method) return;
    
    const nodes = data[method].nodes;
    nodes.forEach(node => {
      if (node.img && !imgCache[node.img]) {
        const img = new Image();
        img.src = `${API_BASE}/images/${node.img}?t=${Date.now()}`;
        img.onload = () => {
          setImgCache(prev => ({ ...prev, [node.img]: img }));
        };
      }
    });
  }, [data, method, imgCache]);

  return (
    <div className="container">
      <header className="header">
        <h1>Activation Maximization Explorer</h1>
        <div className="controls">
          <div className="method-switcher">
            {data && Object.keys(data).map(m => (
              <button 
                key={m} 
                className={method === m ? 'active' : ''} 
                onClick={() => setMethod(m)}
              >
                {m.replace('_', ' ')}
              </button>
            ))}
          </div>
          
          <div className="upload-group">
            <select 
              className="class-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loading}
            >
              {Object.keys(models).map(m => (
                <option key={m} value={m}>{m.replace('_', ' ').toUpperCase()}</option>
              ))}
            </select>
            <select 
              className="class-select"
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(parseInt(e.target.value))}
              disabled={loading}
            >
              {classes.map((c, i) => (
                <option key={i} value={i}>{c}</option>
              ))}
            </select>
            <label className={`upload-btn ${loading ? 'disabled' : ''}`}>
              {loading ? <Loader2 className="animate-spin" /> : <Upload />}
              <span>{loading ? 'Processing...' : 'Upload Image'}</span>
              <input 
                type="file" 
                hidden 
                accept="image/*" 
                onChange={handleFileUpload}
                disabled={loading}
              />
            </label>
          </div>
        </div>
      </header>

      <main className="graph-container">
        {loading && (
          <div className="overlay">
            <div className="loading-modal">
              <Loader2 className="animate-spin" size={48} />
              <h2>Processing {selectedModel.replace('_', ' ').toUpperCase()}</h2>
              <p>Analyzing image and generating filter visualizations...</p>
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
              </div>
              <p className="progress-text">{progress}% complete</p>
            </div>
          </div>
        )}
        {data && method && (
          <ForceGraph2D
            ref={fgRef}
            graphData={data[method]}
            nodeCanvasObject={nodePaint}
            nodeLabel={(node: any) => `
              <strong>${node.block} | Filter ${node.filter_index}</strong><br/>
              Activation: ${node.activation.toFixed(4)}
            `}
            linkColor={() => '#444'}
            linkDirectionalArrowLength={3}
            linkDirectionalArrowRelPos={1}
            dagMode="lr"
            dagLevelDistance={250}
          />
        )}
        {!data && !loading && (
          <div className="empty-state">
            <RefreshCw size={64} />
            <p>Select model and label, then upload an image to begin.</p>
          </div>
        )}
      </main>
    </div>
  );
}
export default App;
