export type Mood = "Positive" | "Negative" | "Neutral" | "Mixed" | "Confused" | "Unknown";
export type Energy = "High Energy" | "Low Energy" | "High Stress" | "Calm" | "Unknown";

export interface JournalEntry {
  id: number;
  timestamp: string;
  text: string;
  mood: Mood;
  energy: Energy;
}

export interface CreateEntryRequest {
  text: string;
}

// Use relative path to leverage Vite proxy defined in vite.config.ts
const API_BASE_URL = '';

export const api = {
  getEntries: async (limit: number = 50): Promise<JournalEntry[]> => {
    const response = await fetch(`${API_BASE_URL}/entries?limit=${limit}`);
    if (!response.ok) {
      throw new Error('Failed to fetch entries');
    }
    return response.json();
  },

  createEntry: async (text: string): Promise<JournalEntry> => {
    const response = await fetch(`${API_BASE_URL}/entries`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new Error('Failed to create entry');
    }
    return response.json();
  }
};
