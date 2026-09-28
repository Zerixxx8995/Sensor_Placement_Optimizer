import { httpClient } from './httpClient';

export async function redeploySensors(payload) {
  return httpClient.post('/redeploy', payload);
}

export async function getRedeployStatus() {
  return httpClient.get('/redeploy/status');
}

export async function trainRedeployAgent(episodes = 500) {
  return httpClient.post(`/redeploy/train?episodes=${episodes}`);
}
