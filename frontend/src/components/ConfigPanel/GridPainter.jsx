import React, { useState, useRef } from 'react';
import { useConfigStore } from '../../store/configStore';
import Button from '../shared/Button';

export default function GridPainter() {
  const { config, paintedCells, toggleCell, clearGrid } = useConfigStore();
  const { area, cell_size } = config;
  const [paintMode, setPaintMode] = useState('building'); // 'building', 'low_priority', 'intersection', 'road'
  const isMouseDown = useRef(false);

  const cols = Math.floor(area.width / cell_size) || 0;
  const rows = Math.floor(area.height / cell_size) || 0;
  const totalCells = cols * rows;

  const handleCellInteraction = (col, row) => {
    if (paintMode === 'road') {
      const cellKey = `${col},${row}`;
      if (paintedCells[cellKey]) {
        toggleCell(col, row, paintedCells[cellKey]);
      }
    } else {
      toggleCell(col, row, paintMode);
    }
  };

  const handleMouseDown = (col, row) => {
    isMouseDown.current = true;
    handleCellInteraction(col, row);
  };

  const handleMouseEnter = (col, row) => {
    if (isMouseDown.current) {
      handleCellInteraction(col, row);
    }
  };

  const handleMouseUp = () => {
    isMouseDown.current = false;
  };

  React.useEffect(() => {
    const handleGlobalMouseUp = () => {
      isMouseDown.current = false;
    };
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  const maxCellsToRender = 2500;
  const isGridTooLarge = totalCells > maxCellsToRender;

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', userSelect: 'none' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
        <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '1.1rem', fontWeight: 600 }}>
          City Grid Painter
        </h3>
        <Button variant="secondary" onClick={clearGrid} style={{ padding: '2px 8px', fontSize: '0.75rem' }}>
          Clear All
        </Button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem' }}>
        <button
          type="button"
          onClick={() => setPaintMode('building')}
          style={{
            padding: '6px 10px',
            fontSize: '0.75rem',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            cursor: 'pointer',
            background: (paintMode === 'building' || paintMode === 'restricted') ? '#1e293b' : 'rgba(30, 41, 59, 0.4)',
            color: '#fff',
            fontWeight: 600,
            transition: 'all var(--transition-fast)',
            boxShadow: (paintMode === 'building' || paintMode === 'restricted') ? '0 0 0 1px #64748b' : 'none'
          }}
        >
          Building (Dark)
        </button>
        <button
          type="button"
          onClick={() => setPaintMode('low_priority')}
          style={{
            padding: '6px 10px',
            fontSize: '0.75rem',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            cursor: 'pointer',
            background: (paintMode === 'low_priority' || paintMode === 'non_critical') ? '#15803d' : 'rgba(34, 197, 94, 0.2)',
            color: '#fff',
            fontWeight: 600,
            transition: 'all var(--transition-fast)',
            boxShadow: (paintMode === 'low_priority' || paintMode === 'non_critical') ? '0 0 0 1px #22c55e' : 'none'
          }}
        >
          Low-Priority (Green)
        </button>
        <button
          type="button"
          onClick={() => setPaintMode('intersection')}
          style={{
            padding: '6px 10px',
            fontSize: '0.75rem',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            cursor: 'pointer',
            background: paintMode === 'intersection' ? '#b45309' : 'rgba(245, 158, 11, 0.2)',
            color: '#fff',
            fontWeight: 600,
            transition: 'all var(--transition-fast)',
            boxShadow: paintMode === 'intersection' ? '0 0 0 1px #f59e0b' : 'none'
          }}
        >
          Intersection (Amber)
        </button>
        <button
          type="button"
          onClick={() => setPaintMode('road')}
          style={{
            padding: '6px 10px',
            fontSize: '0.75rem',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            cursor: 'pointer',
            background: paintMode === 'road' ? '#475569' : 'rgba(100, 116, 139, 0.2)',
            color: '#fff',
            fontWeight: 600,
            transition: 'all var(--transition-fast)',
            boxShadow: paintMode === 'road' ? '0 0 0 1px #94a3b8' : 'none'
          }}
        >
          Road Cell (Eraser)
        </button>
      </div>

      {isGridTooLarge ? (
        <div style={{
          padding: '1rem',
          background: 'rgba(251, 191, 36, 0.08)',
          border: '1px solid rgba(251, 191, 36, 0.2)',
          borderRadius: 'var(--radius-sm)',
          fontSize: '0.85rem',
          textAlign: 'center',
          color: 'var(--color-warning)'
        }}>
          Grid size ({cols} &times; {rows} = {totalCells} cells) is too large to paint interactively.
          Please increase the Grid Resolution (e.g. 5.0m) to enable painting.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${cols}, 1fr)`,
              gap: '2px',
              width: '100%',
              maxWidth: '320px',
              aspectRatio: `${cols} / ${rows}`,
              background: 'rgba(0,0,0,0.4)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-sm)',
              padding: '4px',
              cursor: 'crosshair'
            }}
          >
            {Array.from({ length: rows }).map((_, r) => {
              const rowIdx = rows - 1 - r;
              return Array.from({ length: cols }).map((__, colIdx) => {
                const key = `${colIdx},${rowIdx}`;
                const cellType = paintedCells[key];

                let bg = 'rgba(255,255,255,0.04)'; // Default Road cell (gray)
                if (cellType === 'building' || cellType === 'restricted') {
                  bg = '#1e293b'; // Building (dark)
                } else if (cellType === 'low_priority' || cellType === 'non_critical') {
                  bg = 'rgba(34, 197, 94, 0.45)'; // Low-priority (light green)
                } else if (cellType === 'intersection') {
                  bg = 'rgba(245, 158, 11, 0.65)'; // Intersection (amber/gold)
                }

                return (
                  <div
                    key={key}
                    data-testid={`cell-${colIdx}-${rowIdx}`}
                    onMouseDown={() => handleMouseDown(colIdx, rowIdx)}
                    onMouseEnter={() => handleMouseEnter(colIdx, rowIdx)}
                    style={{
                      background: bg,
                      borderRadius: '1px',
                      transition: 'background-color 0.1s ease',
                      border: '0.5px solid rgba(255,255,255,0.02)',
                    }}
                  />
                );
              });
            })}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Drag mouse over grid cells to set city cell types.
          </span>
        </div>
      )}
    </div>
  );
}
