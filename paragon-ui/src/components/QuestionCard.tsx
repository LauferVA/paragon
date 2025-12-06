import React, { useState } from 'react';
import { ClarificationQuestion } from '../types/dialectic.types';
import { useUIStore } from '../stores/uiStore';

interface QuestionCardProps {
  question: ClarificationQuestion;
  index: number;
  answer?: string;
  onAnswerChange: (index: number, answer: string) => void;
  disabled?: boolean;
}

export function QuestionCard({ question, index, answer, onAnswerChange, disabled }: QuestionCardProps) {
  const { theme } = useUIStore();
  const [customMode, setCustomMode] = useState(false);
  const [inputValue, setInputValue] = useState('');

  const cardClass = theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200';
  const subClass = theme === 'dark' ? 'text-gray-400' : 'text-gray-500';
  const inputClass = theme === 'dark' 
    ? 'bg-gray-900 border-gray-600 text-white placeholder-gray-500' 
    : 'bg-white border-gray-300 text-gray-900 placeholder-gray-400';
  const btnClass = theme === 'dark' ? 'bg-blue-600 hover:bg-blue-500' : 'bg-blue-500 hover:bg-blue-600';
  const optionClass = theme === 'dark' ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200';
  const selectedClass = theme === 'dark' ? 'bg-blue-700 ring-2 ring-blue-500' : 'bg-blue-100 ring-2 ring-blue-500';

  const priorityColors: Record<string, string> = {
    high: 'text-red-500',
    medium: 'text-yellow-500',
    low: 'text-green-500',
  };

  const handleOptionClick = (option: string) => {
    if (!disabled) {
      onAnswerChange(index, option);
    }
  };

  const handleCustomSubmit = () => {
    if (inputValue.trim() && !disabled) {
      onAnswerChange(index, inputValue.trim());
      setCustomMode(false);
    }
  };

  return (
    <div className={'p-3 rounded-lg border mb-3 ' + cardClass}>
      <div className="flex items-start justify-between mb-2">
        <span className={'text-xs font-medium ' + priorityColors[question.priority]}>
          {question.priority.toUpperCase()} PRIORITY
        </span>
        <span className={'text-xs ' + subClass}>Q{index + 1}</span>
      </div>

      <div className="font-medium text-sm mb-2">{question.question}</div>

      {question.context && (
        <div className={'text-xs mb-3 ' + subClass}>Context: {question.context}</div>
      )}

      {answer ? (
        <div className={theme === 'dark' ? 'bg-green-900 text-green-300 p-2 rounded text-sm' : 'bg-green-50 text-green-700 p-2 rounded text-sm'}>
          Selected: {answer}
        </div>
      ) : customMode ? (
        <div className="space-y-2">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter your answer..."
            className={'w-full p-2 rounded border text-sm resize-none ' + inputClass}
            rows={3}
            disabled={disabled}
          />
          <div className="flex gap-2">
            <button
              onClick={handleCustomSubmit}
              disabled={!inputValue.trim() || disabled}
              className={'flex-1 px-3 py-1.5 text-xs rounded text-white disabled:opacity-50 ' + btnClass}
            >
              Submit
            </button>
            <button
              onClick={() => setCustomMode(false)}
              className={'px-3 py-1.5 text-xs rounded ' + optionClass}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {question.options.map((option, optIdx) => (
            <button
              key={optIdx}
              onClick={() => handleOptionClick(option)}
              disabled={disabled}
              className={'w-full text-left p-2 rounded text-sm disabled:opacity-50 ' + (answer === option ? selectedClass : optionClass)}
            >
              {option}
            </button>
          ))}
          {question.requires_freeform && (
            <button
              onClick={() => setCustomMode(true)}
              disabled={disabled}
              className={'w-full text-left p-2 rounded text-sm disabled:opacity-50 ' + optionClass}
            >
              + Provide custom answer
            </button>
          )}
        </div>
      )}
    </div>
  );
}
