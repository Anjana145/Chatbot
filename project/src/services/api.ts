import axios from 'axios';

const API_URL = 'http://localhost:5000';

export interface Message {
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export interface TrainingData {
  tag: string;
  pattern: string;
  response: string;
}

// Send a message to the chatbot
export const sendMessage = async (message: string): Promise<string> => {
  try {
    const response = await axios.post(`${API_URL}/api/message`, { message });
    return response.data.response;
  } catch (error) {
    console.error('Error sending message:', error);
    throw new Error('Failed to send message');
  }
};

// Get conversation history
export const getConversationHistory = async (): Promise<Message[]> => {
  try {
    const response = await axios.get(`${API_URL}/api/history`);
    return response.data.map((item: { user: string; bot: string }) => [
      { text: item.user, sender: 'user', timestamp: new Date() },
      { text: item.bot, sender: 'bot', timestamp: new Date() }
    ]).flat();
  } catch (error) {
    console.error('Error getting conversation history:', error);
    return [];
  }
};

// Train the chatbot with new data
export const trainChatbot = async (data: TrainingData): Promise<void> => {
  try {
    await axios.post(`${API_URL}/api/train`, data);
  } catch (error) {
    console.error('Error training chatbot:', error);
    throw new Error('Failed to train chatbot');
  }
};