import React, { ReactNode } from 'react';
import { BookHeart, History, PenLine } from 'lucide-react';

interface JournalLayoutProps {
  children: ReactNode;
  activeTab: 'write' | 'history';
  onTabChange: (tab: 'write' | 'history') => void;
}

export const JournalLayout: React.FC<JournalLayoutProps> = ({ children, activeTab, onTabChange }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 text-slate-800 font-sans">
      <div className="max-w-4xl mx-auto px-4 py-8 min-h-screen flex flex-col">
        
        {/* Header */}
        <header className="mb-8 flex flex-col items-center text-center">
          <div className="w-16 h-16 bg-white rounded-2xl shadow-lg flex items-center justify-center mb-4 text-indigo-600">
            <BookHeart size={32} />
          </div>
          <h1 className="text-3xl font-serif font-bold text-slate-900 mb-2">Mood Journal</h1>
          <p className="text-slate-500 max-w-md">
            Record your thoughts, track your feelings, and reflect on your journey.
          </p>
        </header>

        {/* Navigation Tabs */}
        <nav className="flex justify-center mb-8">
          <div className="bg-white/60 backdrop-blur-md p-1 rounded-full shadow-sm border border-white/50 flex gap-1">
            <button
              onClick={() => onTabChange('write')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                activeTab === 'write'
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'
              }`}
            >
              <PenLine size={16} />
              Write Entry
            </button>
            <button
              onClick={() => onTabChange('history')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                activeTab === 'history'
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'
              }`}
            >
              <History size={16} />
              History
            </button>
          </div>
        </nav>

        {/* Main Content Area */}
        <main className="flex-1 relative">
          <div className="absolute inset-0 bg-white/30 backdrop-blur-xl rounded-3xl shadow-xl border border-white/40 -z-10 transform rotate-1 scale-[1.02] opacity-50"></div>
          <div className="bg-white/40 backdrop-blur-md rounded-3xl shadow-lg border border-white/60 p-6 md:p-8 min-h-[500px]">
            {children}
          </div>
        </main>

        {/* Footer */}
        <footer className="mt-12 text-center text-slate-400 text-sm">
          <p>© 2025 PANW Mood Journal. All thoughts are private.</p>
        </footer>
      </div>
    </div>
  );
};
