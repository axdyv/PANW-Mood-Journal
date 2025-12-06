import React from 'react';
import { Mood, Energy } from '../api';
import { clsx } from 'clsx';

interface MoodAvatarProps {
  mood: Mood;
  energy: Energy;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

interface AvatarConfig {
  emoji: string;
  label: string;
  color: string;
  bg: string;
}

const getAvatarConfig = (mood: Mood, energy: Energy): AvatarConfig => {
  // Default fallback
  let config: AvatarConfig = {
    emoji: '🌀',
    label: 'Unclear',
    color: 'text-slate-500',
    bg: 'bg-slate-100'
  };

  if (mood === 'Positive') {
    if (energy === 'High Energy') {
      config = { emoji: '😄', label: 'Bright & Energized', color: 'text-yellow-600', bg: 'bg-yellow-100' };
    } else if (energy === 'Calm') {
      config = { emoji: '😊', label: 'Peaceful', color: 'text-green-600', bg: 'bg-green-100' };
    } else {
      config = { emoji: '🙂', label: 'Good', color: 'text-emerald-600', bg: 'bg-emerald-100' };
    }
  } else if (mood === 'Negative') {
    if (energy === 'High Stress') {
      config = { emoji: '😰', label: 'Stressed', color: 'text-red-600', bg: 'bg-red-100' };
    } else if (energy === 'Low Energy') {
      config = { emoji: '😔', label: 'Drained', color: 'text-blue-600', bg: 'bg-blue-100' };
    } else {
      config = { emoji: '🙁', label: 'Down', color: 'text-indigo-600', bg: 'bg-indigo-100' };
    }
  } else if (mood === 'Mixed') {
    config = { emoji: '🤔', label: 'Mixed Feelings', color: 'text-purple-600', bg: 'bg-purple-100' };
  } else if (mood === 'Confused') {
    config = { emoji: '😵‍💫', label: 'Confused', color: 'text-orange-600', bg: 'bg-orange-100' };
  } else if (mood === 'Neutral') {
    config = { emoji: '😐', label: 'Neutral', color: 'text-gray-600', bg: 'bg-gray-100' };
  }

  return config;
};

export const MoodAvatar: React.FC<MoodAvatarProps> = ({ mood, energy, size = 'md', className }) => {
  const { emoji, label, color, bg } = getAvatarConfig(mood, energy);

  const sizeClasses = {
    sm: 'w-10 h-10 text-xl',
    md: 'w-24 h-24 text-5xl',
    lg: 'w-40 h-40 text-7xl',
  };

  return (
    <div className={clsx("flex flex-col items-center gap-2", className)}>
      <div 
        className={clsx(
          "rounded-full flex items-center justify-center shadow-sm transition-all duration-300",
          sizeClasses[size],
          bg
        )}
        title={label}
      >
        <span role="img" aria-label={label}>{emoji}</span>
      </div>
      {size !== 'sm' && (
        <span className={clsx("font-medium text-center", color, size === 'lg' ? 'text-lg' : 'text-sm')}>
          {label}
        </span>
      )}
    </div>
  );
};
