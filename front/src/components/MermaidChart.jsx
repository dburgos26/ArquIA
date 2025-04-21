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
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        fontFamily: 'monospace',
        fontSize: 16,
        themeVariables: {
          // Customize theme to ensure text is always readable against backgrounds
          primaryTextColor: '#ffffff',
          primaryColor: '#434857',
          primaryBorderColor: '#ffffff',
          lineColor: '#d3d3d3',
          secondaryColor: '#2a3052',
          tertiaryColor: '#1a1a2e',
          // Ensure text has dark color on light backgrounds
          nodeBorder: '#ffffff',
          clusterBkg: '#23283d',
          clusterBorder: '#ffffff',
          defaultLinkColor: '#d3d3d3',
          titleColor: '#ffffff',
          edgeLabelBackground: '#23283d',
          // Ensure node text is dark on light backgrounds
          nodeTextColor: '#1a1a1a'
        }
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