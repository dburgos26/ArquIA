import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import "../styles/mermaidChart.css";

const MermaidChart = ({ chart }) => {
  const mermaidRef = useRef(null);
  const chartId = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
  const [rendered, setRendered] = useState(false);

  useEffect(() => {
    // Only initialize mermaid once
    if (!window.mermaidInitialized) {
      mermaid.initialize({
        startOnLoad: false, // Changed to false to have more control
        theme: 'dark',
        securityLevel: 'loose',
        fontFamily: 'monospace',
        fontSize: 16,
      });
      window.mermaidInitialized = true;
    }
    
    // Delay rendering slightly to ensure the DOM is fully prepared
    const renderTimer = setTimeout(() => {
      if (chart && mermaidRef.current) {
        try {
          mermaid.render(chartId, chart).then(({ svg }) => {
            if (mermaidRef.current) {
              mermaidRef.current.innerHTML = svg;
              setRendered(true);
            }
          });
        } catch (error) {
          console.error("Failed to render mermaid chart:", error);
          if (mermaidRef.current) {
            mermaidRef.current.innerHTML = `<div class="error-container">
              <p>Error rendering diagram</p>
              <pre>${error.message}</pre>
            </div>`;
          }
        }
      }
    }, 100);
    
    return () => clearTimeout(renderTimer);
  }, [chart, chartId]);

  return (
    <div className="mermaid-container" style={{ minHeight: '100px' }}>
      <div ref={mermaidRef} className="mermaid-chart" data-processed={rendered.toString()}></div>
    </div>
  );
};

export default MermaidChart;