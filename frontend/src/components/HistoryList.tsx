import React, { useState, useMemo } from 'react';
import { JournalEntry, Mood } from '../api';
import { MoodAvatar } from './MoodAvatar';
import { format } from 'date-fns';
import { Search, Filter } from 'lucide-react';

interface HistoryListProps {
  entries: JournalEntry[];
}

export const HistoryList: React.FC<HistoryListProps> = ({ entries }) => {
  const [filterMood, setFilterMood] = useState<Mood | 'All'>('All');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredEntries = useMemo(() => {
    return entries.filter(entry => {
      const matchesMood = filterMood === 'All' || entry.mood === filterMood;
      const matchesSearch = entry.text.toLowerCase().includes(searchTerm.toLowerCase());
      return matchesMood && matchesSearch;
    });
  }, [entries, filterMood, searchTerm]);

  const moods: (Mood | 'All')[] = ['All', 'Positive', 'Negative', 'Mixed', 'Neutral', 'Confused'];

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-slate-100">
        <div className="flex items-center gap-2 overflow-x-auto w-full sm:w-auto pb-2 sm:pb-0 no-scrollbar">
          <Filter className="w-4 h-4 text-slate-400 shrink-0" />
          {moods.map(mood => (
            <button
              key={mood}
              onClick={() => setFilterMood(mood)}
              className={`px-3 py-1.5 rounded-full text-sm font-medium whitespace-nowrap transition-colors ${
                filterMood === mood
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'bg-slate-50 text-slate-600 hover:bg-slate-100'
              }`}
            >
              {mood}
            </button>
          ))}
        </div>
        
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search entries..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-slate-50 border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 text-sm"
          />
        </div>
      </div>

      {/* List */}
      <div className="space-y-4">
        {filteredEntries.length === 0 ? (
          <div className="text-center py-12 text-slate-400">
            <p>No entries found matching your criteria.</p>
          </div>
        ) : (
          filteredEntries.map(entry => (
            <div key={entry.id} className="bg-white p-5 rounded-xl shadow-sm border border-slate-100 hover:shadow-md transition-shadow flex gap-4">
              <div className="shrink-0">
                <MoodAvatar mood={entry.mood} energy={entry.energy} size="sm" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex gap-2 items-center">
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                      {format(new Date(entry.timestamp), 'MMM d, yyyy · h:mm a')}
                    </span>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-100 text-slate-600">
                      {entry.mood}
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-100 text-slate-600">
                      {entry.energy}
                    </span>
                  </div>
                </div>
                <p className="text-slate-700 font-serif leading-relaxed line-clamp-3">
                  {entry.text}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
