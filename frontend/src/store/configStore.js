import { create } from 'zustand';

const DEFAULT_CONFIG = {
  area: { width: 200, height: 200 },
  num_nodes: 100,
  sensing_radius: 15,
  comm_radius: 30,
  initial_energy: 1.0,
  weights: { w1: 0.7, w2: 0.15, w3: 0.15 },
  pso_params: {
    swarm_size: 30,
    iterations: 500,
    inertia: 0.7,
    c1: 1.5,
    c2: 1.5,
  },
  use_gpu: false,
  use_vdcoa: false,
  seed: 42,
  restricted_areas: [],
  non_critical_areas: [],
  strategy: 'pso',
  cell_size: 5.0,
};

export const useConfigStore = create((set, get) => ({
  config: { ...DEFAULT_CONFIG },
  paintedCells: {},

  updateField: (field, value) => {
    set((state) => ({
      config: {
        ...state.config,
        [field]: value,
      },
    }));
    // If we updated cell_size, re-sync bounds
    if (field === 'cell_size') {
      get().syncGridBounds();
    }
  },

  updateNestedField: (parentField, field, value) => {
    set((state) => ({
      config: {
        ...state.config,
        [parentField]: {
          ...state.config[parentField],
          [field]: value,
        },
      },
    }));
    // If we updated width/height in area, re-sync bounds
    if (parentField === 'area') {
      get().syncGridBounds();
    }
  },

  toggleCell: (col, row, type) => {
    set((state) => {
      const cellKey = `${col},${row}`;
      const currentType = state.paintedCells[cellKey];
      const newPaintedCells = { ...state.paintedCells };

      if (currentType === type || type === 'road') {
        delete newPaintedCells[cellKey];
      } else {
        newPaintedCells[cellKey] = type;
      }

      const { cell_size } = state.config;
      const restricted = [];
      const nonCritical = [];
      const intersections = [];

      Object.entries(newPaintedCells).forEach(([key, val]) => {
        const [c, r] = key.split(',').map(Number);
        const rect = {
          x1: c * cell_size,
          y1: r * cell_size,
          x2: (c + 1) * cell_size,
          y2: (r + 1) * cell_size,
        };
        if (val === 'restricted' || val === 'building') {
          restricted.push(rect);
        } else if (val === 'non_critical' || val === 'low_priority') {
          nonCritical.push(rect);
        } else if (val === 'intersection') {
          intersections.push(rect);
        }
      });

      return {
        paintedCells: newPaintedCells,
        config: {
          ...state.config,
          restricted_areas: restricted,
          non_critical_areas: nonCritical,
          intersections: intersections,
        },
      };
    });
  },

  clearGrid: () => {
    set((state) => ({
      paintedCells: {},
      config: {
        ...state.config,
        restricted_areas: [],
        non_critical_areas: [],
        intersections: [],
      },
    }));
  },

  syncGridBounds: () => {
    set((state) => {
      const { width, height } = state.config.area;
      const { cell_size } = state.config;
      const cols = Math.floor(width / cell_size);
      const rows = Math.floor(height / cell_size);

      const newPaintedCells = {};
      Object.entries(state.paintedCells).forEach(([key, val]) => {
        const [c, r] = key.split(',').map(Number);
        if (c < cols && r < rows) {
          newPaintedCells[key] = val;
        }
      });

      const restricted = [];
      const nonCritical = [];
      const intersections = [];

      Object.entries(newPaintedCells).forEach(([key, val]) => {
        const [c, r] = key.split(',').map(Number);
        const rect = {
          x1: c * cell_size,
          y1: r * cell_size,
          x2: (c + 1) * cell_size,
          y2: (r + 1) * cell_size,
        };
        if (val === 'restricted' || val === 'building') {
          restricted.push(rect);
        } else if (val === 'non_critical' || val === 'low_priority') {
          nonCritical.push(rect);
        } else if (val === 'intersection') {
          intersections.push(rect);
        }
      });

      return {
        paintedCells: newPaintedCells,
        config: {
          ...state.config,
          restricted_areas: restricted,
          non_critical_areas: nonCritical,
          intersections: intersections,
        },
      };
    });
  },

  resetConfig: () =>
    set({
      config: { ...DEFAULT_CONFIG },
      paintedCells: {},
    }),
}));
