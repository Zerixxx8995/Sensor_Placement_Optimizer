import { httpClient } from './httpClient';

export async function getSurrogateStatus() {
  return httpClient.get('/surrogate/status');
}

export async function trainSurrogate(epochs = 50) {
  return httpClient.post(`/surrogate/train?epochs=${epochs}`);
}
