import httpClient from './httpClient';

export async function getSurrogateStatus() {
  const response = await httpClient.get('/surrogate/status');
  return response.data;
}

export async function trainSurrogate(epochs = 50) {
  const response = await httpClient.post(`/surrogate/train?epochs=${epochs}`);
  return response.data;
}
