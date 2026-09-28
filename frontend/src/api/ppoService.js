const BASE_URL = import.meta.env?.VITE_API_URL || 'http://localhost:8000/api/v1';

export async function getPPOStatus() {
  const resp = await fetch(`${BASE_URL}/ppo/status`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch PPO status: ${resp.statusText}`);
  }
  return resp.json();
}

export async function trainPPOAgent(episodes = 50) {
  const resp = await fetch(`${BASE_URL}/ppo/train?episodes=${episodes}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!resp.ok) {
    throw new Error(`Failed to start PPO training: ${resp.statusText}`);
  }
  return resp.json();
}

export function getPPOTrainingProgressURL() {
  return `${BASE_URL}/ppo/training-progress`;
}
