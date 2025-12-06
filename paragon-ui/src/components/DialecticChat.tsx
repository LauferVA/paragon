import React, { useCallback } from 'react';
import { useDialecticStore } from '../stores/dialecticStore';
import { useUIStore } from '../stores/uiStore';
import { AmbiguityCard } from './AmbiguityCard';
import { QuestionCard } from './QuestionCard';

interface DialecticChatProps {
  sessionId?: string;
  onComplete?: () => void;
}

export function DialecticChat({ sessionId, onComplete }: DialecticChatProps) {
  const { theme } = useUIStore();
  const {
    phase,
    ambiguities,
    questions,
    answers,
    pendingSubmit,
    setAnswer,
    acceptSuggested,
  } = useDialecticStore();

  const bgClass = theme === 'dark' ? 'bg-gray-900' : 'bg-gray-50';
  const headerClass = theme === 'dark' ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white';
  const subClass = theme === 'dark' ? 'text-gray-400' : 'text-gray-500';

  const phaseColors: Record<string, string> = {
    IDLE: 'bg-gray-500',
    DIALECTIC: 'bg-blue-500',
    CLARIFICATION: 'bg-yellow-500',
    RESEARCH: 'bg-green-500',
  };

  const handleAcceptSuggested = useCallback((index: number, answer: string) => {
    acceptSuggested(index, answer);
  }, [acceptSuggested]);

  const handleProvideOwn = useCallback((index: number) => {
    // This would typically open a modal or inline input
    // For now, just log
    console.log('Provide own answer for index:', index);
  }, []);

  const handleAnswerChange = useCallback((index: number, answer: string) => {
    setAnswer(index, answer);
  }, [setAnswer]);

  const allAnswered = questions.length > 0 && 
    questions.every((_, idx) => answers.has(idx));

  return (
    <div className={'h-full flex flex-col ' + bgClass}>
      <div className={'p-3 border-b ' + headerClass}>
        <div className="flex items-center justify-between">
          <h2 className="font-bold">Dialectic</h2>
          <span className={'text-xs px-2 py-0.5 rounded text-white ' + phaseColors[phase]}>
            {phase}
          </span>
        </div>
        {sessionId && (
          <div className={'text-xs mt-1 ' + subClass}>Session: {sessionId.slice(0, 8)}...</div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {phase === 'IDLE' && (
          <div className={'text-center py-8 ' + subClass}>
            <p>No active session</p>
            <p className="text-xs mt-2">Start a new requirement analysis to begin</p>
          </div>
        )}

        {phase === 'DIALECTIC' && ambiguities.length > 0 && (
          <div>
            <div className={'text-xs font-medium mb-3 ' + subClass}>
              DETECTED AMBIGUITIES ({ambiguities.length})
            </div>
            {ambiguities.map((amb, idx) => (
              <AmbiguityCard
                key={amb.id || idx}
                ambiguity={amb}
                index={idx}
                onAcceptSuggested={handleAcceptSuggested}
                onProvideOwn={handleProvideOwn}
                answer={answers.get(idx)}
              />
            ))}
          </div>
        )}

        {phase === 'CLARIFICATION' && questions.length > 0 && (
          <div>
            <div className={'text-xs font-medium mb-3 ' + subClass}>
              CLARIFICATION QUESTIONS ({questions.length})
            </div>
            {questions.map((q, idx) => (
              <QuestionCard
                key={idx}
                question={q}
                index={idx}
                answer={answers.get(idx)}
                onAnswerChange={handleAnswerChange}
                disabled={pendingSubmit}
              />
            ))}
          </div>
        )}

        {phase === 'RESEARCH' && (
          <div className={'text-center py-8 ' + subClass}>
            <p className="text-green-500 font-medium">Clarification Complete</p>
            <p className="text-xs mt-2">Proceeding to research phase...</p>
          </div>
        )}
      </div>

      {(phase === 'DIALECTIC' || phase === 'CLARIFICATION') && (
        <div className={'p-3 border-t ' + headerClass}>
          <button
            onClick={onComplete}
            disabled={!allAnswered || pendingSubmit}
            className={
              'w-full py-2 rounded font-medium text-sm disabled:opacity-50 ' +
              (theme === 'dark' ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-500 hover:bg-blue-600 text-white')
            }
          >
            {pendingSubmit ? 'Submitting...' : allAnswered ? 'Submit All Answers' : 'Answer All Questions to Continue'}
          </button>
        </div>
      )}
    </div>
  );
}
