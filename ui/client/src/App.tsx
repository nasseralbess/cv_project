import { useState, useEffect, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Upload, Loader2, RefreshCw, AlertTriangle } from 'lucide-react';
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
const CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

function App() {
  const [data, setData] = useState<AllData | null>(null);
  const [method, setMethod] = useState<string>('');
  const [imgCache, setImgCache] = useState<Record<string, HTMLImageElement>>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedLabel, setSelectedLabel] = useState<number>(6); // Default to frog
  const fgRef = useRef<any>();

  const blockMap: Record<string, number> = { 'block1': 0, 'block2': 1, 'block3': 2 };

  const fetchGraph = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/graph`);
      const json = await res.json();
      if (json.error) return;
      
      const methods = Object.keys(json);
      methods.forEach(m => {
        const nodes = json[m].nodes;
        const blockCounts: Record<string, number> = {};
        const blockIndices: Record<string, number> = {};
        nodes.forEach((n: any) => {
          blockCounts[n.block] = (blockCounts[n.block] || 0) + 1;
        });
        nodes.forEach((n: any) => {
          const blockIdx = blockIndices[n.block] || 0;
          blockIndices[n.block] = blockIdx + 1;
          n.fx = (blockMap[n.block] - 1) * 200;
          n.fy = (blockIdx - (blockCounts[n.block] - 1) / 2) * 40;
        });
      });

      setData(json);
      if (methods.length > 0 && !method) setMethod(methods[0]);
    } catch (e) {
      console.error("Failed to fetch graph", e);
    }
  }, [method]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

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
  }, [data, method]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', selectedLabel.toString());

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const json = await res.json();
      if (json.error) {
        alert(json.error);
      } else {
        const methods = Object.keys(json);
        methods.forEach(m => {
          const nodes = json[m].nodes;
          const blockCounts: Record<string, number> = {};
          const blockIndices: Record<string, number> = {};
          nodes.forEach((n: any) => {
            blockCounts[n.block] = (blockCounts[n.block] || 0) + 1;
          });
          nodes.forEach((n: any) => {
            const blockIdx = blockIndices[n.block] || 0;
            blockIndices[n.block] = blockIdx + 1;
            n.fx = (blockMap[n.block] - 1) * 200;
            n.fy = (blockIdx - (blockCounts[n.block] - 1) / 2) * 40;
          });
        });
        setData(json);
        setImgCache({});
      }
    } catch (err) {
      alert("Upload failed: " + err);
    } finally {
      setLoading(false);
    }
  };

  const nodePaint = useCallback((node: any, ctx: CanvasRenderingContext2D) => {
    const size = 16;
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
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(parseInt(e.target.value))}
              disabled={loading}
            >
              {CLASSES.map((c, i) => (
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
              <h2>Validating & Generating</h2>
              <p>We are checking if the model correctly identifies the image as a <strong>{CLASSES[selectedLabel]}</strong> before generating visualizations.</p>
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
            dagLevelDistance={200}
          />
        )}
        {!data && !loading && (
          <div className="empty-state">
            <RefreshCw size={64} />
            <p>No data found. Select a label and upload an image to start.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
