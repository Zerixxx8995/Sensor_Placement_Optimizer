import { useState, useEffect, useRef, useCallback } from 'react';
import { getPPOTrainingProgressURL, trainPPOAgent } from '../api/ppoService';

export function usePPOTraining() {
  const [isTraining, setIsTraining] = useState(false);
  const [history, setHistory] = useState([]);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [latestReward, setLatestReward] = useState(0);
  const [latestCoverage, setLatestCoverage] = useState(0);
  const [bestCoverage, setBestCoverage] = useState(0);
  const eventSourceRef = useRef(null);

  const stopTraining = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsTraining(false);
  }, []);

  const startTraining = useCallback(async (episodes = 50) => {
    setHistory([]);
    setIsTraining(true);
    try {
      await trainPPOAgent(episodes);

      const url = getPPOTrainingProgressURL();
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.status === 'completed') {
            stopTraining();
            return;
          }
          if (data.episode) {
            setCurrentEpisode(data.episode);
            setLatestReward(data.reward);
            setLatestCoverage(data.coverage);
            setBestCoverage(data.best_coverage);
            setHistory((prev) => [...prev, data]);
          }
        } catch (e) {
          console.error('Error parsing SSE PPO training data', e);
        }
      };

      es.onerror = () => {
        stopTraining();
      };
    } catch (err) {
      console.error('Failed to start PPO training:', err);
      setIsTraining(false);
    }
  }, [stopTraining]);

  useEffect(() => {
    return () => {
      stopTraining();
    };
  }, [stopTraining]);

  return {
    isTraining,
    history,
    currentEpisode,
    latestReward,
    latestCoverage,
    bestCoverage,
    startTraining,
    stopTraining,
  };
}
