import React from 'react';
import { AmbiguityMarker } from '../types/dialectic.types';
import { useUIStore } from '../stores/uiStore';

interface AmbiguityCardProps {
  ambiguity: AmbiguityMarker;
  index: number;
  onAcceptSuggested: (index: number, answer: string) => void;
  onProvideOwn: (index: number) => void;
  answer?: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  low: 'bg-blue-500',
  medium: 'bg-yellow-500',
  high: 'bg-orange-500',
  critical: 'bg-red-500',
};

export function AmbiguityCard({ ambiguity, index, onAcceptSuggested, onProvideOwn, answer }: AmbiguityCardProps) {
  const { theme } = useUIStore();

  const severityColor = SEVERITY_COLORS[ambiguity.severity] || SEVERITY_COLORS.medium;
  const cardClass = theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200';
  const subClass = theme === 'dark' ? 'text-gray-400' : 'text-gray-500';
  const textBgClass = theme === 'dark' ? 'bg-gray-900' : 'bg-gray-50';
  const textClass = theme === 'dark' ? 'text-gray-300' : 'text-gray-700';
  const answerClass = theme === 'dark' ? 'bg-green-900 text-green-300' : 'bg-green-50 text-green-700';
  const primaryBtnClass = theme === 'dark' ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-500 hover:bg-blue-600 text-white';
  const secondaryBtnClass = theme === 'dark' ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-gray-200 hover:bg-gray-300 text-gray-800';

  return (
    <div className={'p-3 rounded-lg border mb-3 ' + cardClass}>
      <div className="flex items-start gap-2 mb-2">
        <span className={'px-2 py-0.5 rounded text-xs text-white font-bold ' + severityColor}>
          {ambiguity.type.toUpperCase()}
        </span>
        <span className={'text-xs ' + subClass}>{ambiguity.severity}</span>
      </div>

      <div className={'text-sm mb-2 p-2 rounded ' + textBgClass}>"{ambiguity.text}"</div>

      {ambiguity.position && (
        <div className={'text-xs mb-2 ' + subClass}>
          {ambiguity.position.context && <span>Context: {ambiguity.position.context}</span>}
          {ambiguity.position.line && <span className="ml-2">Line {ambiguity.position.line}</span>}
        </div>
      )}

      {ambiguity.suggested_clarifications && ambiguity.suggested_clarifications.length > 0 && (
        <div className="mb-3">
          <div className={'text-xs mb-1 ' + subClass}>Suggested:</div>
          <div className={'text-sm italic ' + textClass}>{ambiguity.suggested_clarifications[0]}</div>
        </div>
      )}

      {answer && (
        <div className={'text-xs p-2 rounded mb-2 ' + answerClass}>Answered: {answer}</div>
      )}

      {!answer && (
        <div className="flex gap-2">
          {ambiguity.suggested_clarifications && ambiguity.suggested_clarifications.length > 0 && (
            <button
              onClick={() => onAcceptSuggested(index, ambiguity.suggested_clarifications![0])}
              className={'flex-1 px-3 py-1.5 text-xs rounded ' + primaryBtnClass}
            >
              Accept Suggested
            </button>
          )}
          <button
            onClick={() => onProvideOwn(index)}
            className={'flex-1 px-3 py-1.5 text-xs rounded ' + secondaryBtnClass}
          >
            Provide Own Answer
          </button>
        </div>
      )}
    </div>
  );
}
