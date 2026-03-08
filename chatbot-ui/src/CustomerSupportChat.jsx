import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

const CustomerSupportChat = () => {
  const [userId, setUserId] = useState('');
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const startSession = async () => {
    if (!userId.trim()) {
      setError('Please enter a valid user ID');
      return;
    }

    setError('');
    setIsLoading(true);

    try {
      console.log('→ Sending session start request for user:', userId);  // ADD THIS
      
      const response = await fetch('/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId
        })
      });

      console.log('→ Response status:', response.status);  // ADD THIS
      console.log('→ Response:', await response.clone().json());  // ADD THIS
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      setIsSessionActive(true);
      setMessages([{
        role: 'assistant',
        content: `Welcome! I'm your customer support assistant. How can I help you today?`,
        timestamp: new Date()
      }]);
    } catch (err) {
      console.error('→ Error starting session:', err);  // ADD THIS
      setError('Failed to start session. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Replace this with your actual API call to the backend
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          message: inputMessage
        })
      });

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.reply || 'I apologize, but I encountered an issue processing your request.',
        confidence: data.confidence,
        usedLLM: data.used_llm,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resetSession = () => {
    setIsSessionActive(false);
    setMessages([]);
    setUserId('');
    setError('');
  };

  if (!isSessionActive) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center p-4">
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@300;400;600;700&family=DM+Sans:wght@400;500;600&display=swap');
          
          * {
            font-family: 'DM Sans', sans-serif;
          }
          
          .brand-title {
            font-family: 'Fraunces', serif;
          }
          
          @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(2deg); }
          }
          
          .float-animation {
            animation: float 6s ease-in-out infinite;
          }
          
          @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
          }
          
          .shimmer {
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            background-size: 1000px 100%;
            animation: shimmer 3s infinite;
          }
        `}</style>
        
        <div className="w-full max-w-md">
          <div className="text-center mb-12 float-animation">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl mb-6 shadow-2xl relative overflow-hidden">
              <div className="shimmer absolute inset-0"></div>
              <Bot className="w-10 h-10 text-white relative z-10" strokeWidth={1.5} />
            </div>
            <h1 className="brand-title text-5xl font-bold text-slate-800 mb-3">
              Support Hub
            </h1>
            <p className="text-slate-600 text-lg">Your intelligent customer assistant</p>
          </div>

          <div className="bg-white/70 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 p-8">
            <div className="mb-6">
              <label className="block text-sm font-semibold text-slate-700 mb-3">
                Enter Your User ID
              </label>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && startSession()}
                placeholder="e.g., 12345"
                className="w-full px-5 py-4 bg-white border-2 border-slate-200 rounded-2xl focus:outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100 transition-all text-lg"
                disabled={isLoading}
              />
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}

            <button
              onClick={startSession}
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-4 rounded-2xl font-semibold text-lg hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 active:translate-y-0 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Starting Session...
                </>
              ) : (
                'Start Chat'
              )}
            </button>
          </div>

          <div className="mt-8 text-center text-sm text-slate-500">
            Powered by SLM + LLM Hybrid Architecture
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@300;400;600;700&family=DM+Sans:wght@400;500;600&display=swap');
        
        * {
          font-family: 'DM Sans', sans-serif;
        }
        
        .brand-title {
          font-family: 'Fraunces', serif;
        }
        
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .message-animate {
          animation: slideIn 0.3s ease-out;
        }
        
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
          50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.5); }
        }
        
        .typing-indicator {
          animation: pulse-glow 2s ease-in-out infinite;
        }
      `}</style>

      {/* Header */}
      <div className="bg-white/70 backdrop-blur-xl border-b border-white/20 shadow-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg">
              <Bot className="w-6 h-6 text-white" strokeWidth={2} />
            </div>
            <div>
              <h1 className="brand-title text-2xl font-bold text-slate-800">Support Hub</h1>
              <p className="text-sm text-slate-500">User ID: {userId}</p>
            </div>
          </div>
          <button
            onClick={resetSession}
            className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-xl transition-colors"
          >
            End Session
          </button>
        </div>
      </div>

      {/* Messages Container */}
      <div className="max-w-4xl mx-auto px-4 py-6 h-[calc(100vh-180px)] overflow-y-auto">
        <div className="space-y-6">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex gap-4 message-animate ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg">
                  <Bot className="w-5 h-5 text-white" strokeWidth={2} />
                </div>
              )}
              
              <div className={`max-w-[75%] ${msg.role === 'user' ? 'order-1' : ''}`}>
                <div
                  className={`rounded-2xl px-5 py-3.5 shadow-md ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                      : 'bg-white border border-slate-200'
                  }`}
                >
                  <p className={`leading-relaxed ${msg.role === 'user' ? 'text-white' : 'text-slate-800'}`}>
                    {msg.content}
                  </p>
                  
                  {msg.role === 'assistant' && msg.usedLLM && (
                    <div className="mt-3 pt-3 border-t border-slate-200 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-amber-600" />
                      <span className="text-xs text-amber-700 font-medium">Enhanced with LLM</span>
                    </div>
                  )}
                </div>
                
              </div>

              {msg.role === 'user' && (
                <div className="w-10 h-10 bg-gradient-to-br from-slate-600 to-slate-800 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg">
                  <User className="w-5 h-5 text-white" strokeWidth={2} />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-4 message-animate">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg">
                <Bot className="w-5 h-5 text-white" strokeWidth={2} />
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-md typing-indicator">
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/70 backdrop-blur-xl border-t border-white/20 shadow-2xl">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex gap-3">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault(); // Prevent new line
                  sendMessage();
                }
                // Shift + Enter naturally inserts a new line
              }}
              placeholder="Type your message..."
              disabled={isLoading}
              rows={1}
              className="flex-1 px-5 py-3.5 bg-white border-2 border-slate-200 rounded-2xl focus:outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100 disabled:opacity-50 disabled:cursor-not-allowed transition-all resize-none overflow-hidden"
              style={{ height: 'auto' }}
            />

            <button
              onClick={sendMessage}
              disabled={isLoading || !inputMessage.trim()}
              className="px-6 py-3.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-2xl font-semibold hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 active:translate-y-0 flex items-center gap-2"
            >
              <Send className="w-5 h-5" strokeWidth={2} />
            </button>
          </div>
          <p className="text-xs text-slate-500 mt-3 text-center">
            Press Enter to send • Shift + Enter for new line
          </p>
        </div>
      </div>

    </div>
  );
};

export default CustomerSupportChat;