import { useState, useEffect } from 'react';
import { JournalLayout } from './components/JournalLayout';
import { EntryComposer } from './components/EntryComposer';
import { HistoryList } from './components/HistoryList';
import { api, JournalEntry } from './api';
import { Loader2 } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState<'write' | 'history'>('write');
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch entries on mount
  useEffect(() => {
    const fetchEntries = async () => {
      try {
        const data = await api.getEntries();
        setEntries(data);
      } catch (error) {
        console.error('Failed to fetch entries:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchEntries();
  }, []);

  const handleEntryCreated = (newEntry: JournalEntry) => {
    // Add new entry to the top of the list
    setEntries(prev => [newEntry, ...prev]);
  };

  return (
    <JournalLayout activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === 'write' ? (
        <EntryComposer onEntryCreated={handleEntryCreated} />
      ) : (
        <>
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <Loader2 className="w-8 h-8 animate-spin mb-2" />
              <p>Loading your journal...</p>
            </div>
          ) : (
            <HistoryList entries={entries} />
          )}
        </>
      )}
    </JournalLayout>
  );
}

export default App;
