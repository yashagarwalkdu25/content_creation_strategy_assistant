import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

interface ApiResponse {
  success: boolean;
  data: {
    ai_message: string;
    current_stage: string;
    profile_complete: boolean;
    selected_idea: string | null;
    retrieved_ideas: any | null;
    generated_content_type: string | null;
  };
  error_message: string | null;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your AI assistant. First, we will build your profile. What is the main topic or niche you focus on in your content',
      sender: 'ai',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://0.0.0.0:5002/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: input }),
      });

      if (response.ok) {
        const data: ApiResponse = await response.json();
        if (data.success) {
          const aiMessage: Message = {
            id: (Date.now() + 1).toString(),
            text: data.data.ai_message,
            sender: 'ai',
            timestamp: new Date()
          };
          setMessages(prev => [...prev, aiMessage]);
        } else {
          throw new Error(data.error_message || 'Failed to get response');
        }
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I\'m having trouble connecting right now. Please try again later.',
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((message, index) => (
          <div
            key={message.id}
            className={`flex gap-4 animate-fade-in-up ${
              message.sender === 'user' ? 'justify-end' : 'justify-start'
            }`}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            {message.sender === 'ai' && (
              <div className="w-8 h-8 bg-cyber-neon rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-black" />
              </div>
            )}
            
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-cyber-neon bg-opacity-20 text-cyber-neon neon-border'
                  : 'glass-dark text-white'
              }`}
            >
              <p className="text-sm">{message.text}</p>
              <p className="text-xs opacity-70 mt-1">
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>

            {message.sender === 'user' && (
              <div className="w-8 h-8 bg-cyber-gray-light rounded-full flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-cyber-neon" />
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="flex gap-4 justify-start">
            <div className="w-8 h-8 bg-cyber-neon rounded-full flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4 text-black" />
            </div>
            <div className="glass-dark rounded-lg px-4 py-2">
              <Loader2 className="w-4 h-4 animate-spin text-cyber-neon" />
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="p-6 border-t border-cyber-gray-light">
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about content creation..."
            className="flex-1 cyber-input"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="cyber-button disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
