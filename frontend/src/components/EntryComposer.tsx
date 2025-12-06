import React, { useState } from 'react';
import { api, JournalEntry } from '../api';
import { MoodAvatar } from './MoodAvatar';
import { Loader2, Send } from 'lucide-react';

interface EntryComposerProps {
  onEntryCreated: (entry: JournalEntry) => void;
}

export const EntryComposer: React.FC<EntryComposerProps> = ({ onEntryCreated }) => {
  const [text, setText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastEntry, setLastEntry] = useState<JournalEntry | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const newEntry = await api.createEntry(text);
      setLastEntry(newEntry);
      onEntryCreated(newEntry);
      setText('');
    } catch (err) {
      setError('Couldn’t save your entry. Please try again.');
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col md:flex-row gap-8 h-full">
      {/* Left Side: Avatar Panel */}
      <div className="w-full md:w-1/3 flex flex-col items-center justify-center p-6 bg-white/50 rounded-2xl border border-white/60 shadow-sm backdrop-blur-sm">
        <h3 className="text-slate-500 uppercase tracking-wider text-xs font-semibold mb-6">Current State</h3>
        
        <MoodAvatar 
          mood={lastEntry?.mood ?? 'Unknown'} 
          energy={lastEntry?.energy ?? 'Unknown'} 
          size="lg" 
        />

        {lastEntry && (
          <div className="mt-6 flex flex-wrap gap-2 justify-center">
            <span className="px-3 py-1 rounded-full text-xs font-medium bg-indigo-100 text-indigo-700">
              {lastEntry.mood}
            </span>
            <span className="px-3 py-1 rounded-full text-xs font-medium bg-pink-100 text-pink-700">
              {lastEntry.energy}
            </span>
          </div>
        )}
        
        {!lastEntry && (
          <p className="mt-4 text-sm text-slate-400 text-center italic">
            Write an entry to see your mood avatar update...
          </p>
        )}
      </div>

      {/* Right Side: Composer */}
      <div className="w-full md:w-2/3 flex flex-col">
        <form onSubmit={handleSubmit} className="flex flex-col h-full bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
          <div className="flex-1 p-4">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="How are you feeling right now? What's on your mind?"
              className="w-full h-full resize-none outline-none text-slate-700 placeholder:text-slate-300 text-lg leading-relaxed font-serif bg-transparent p-2"
              disabled={isSubmitting}
            />
          </div>
          
          {error && (
            <div className="px-6 py-2 bg-red-50 text-red-600 text-sm border-t border-red-100">
              {error}
            </div>
          )}

          <div className="p-4 bg-slate-50 border-t border-slate-100 flex justify-between items-center">
            <div className="text-xs text-slate-400">
              {text.length > 0 ? `${text.length} characters` : 'Ready to listen'}
            </div>
            <button
              type="submit"
              disabled={isSubmitting || !text.trim()}
              className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-full font-medium transition-colors shadow-sm"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Saving...</span>
                </>
              ) : (
                <>
                  <span>Save Entry</span>
                  <Send className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
