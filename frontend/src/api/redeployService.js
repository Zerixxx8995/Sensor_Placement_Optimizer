import httpClient from './httpClient';

export async function redeploySensors(payload) {
  const response = await httpClient.post('/redeploy', payload);
  return response.data;
}

export async function getRedeployStatus() {
  const response = await httpClient.get('/redeploy/status');
  return response.data;
}

export async function trainRedeployAgent(episodes = 500) {
  const response = await httpClient.post(`/redeploy/train?episodes=${episodes}`);
  return response.data;
}
