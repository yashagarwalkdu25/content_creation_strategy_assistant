
import React from 'react';
import ChatInterface from '../components/ChatInterface';

const Chat = () => {
  return (
    <div className="h-screen flex flex-col">
      <header className="p-6 border-b border-cyber-gray-light">
        <h1 className="text-2xl font-bold text-cyber-neon">AI Chat Assistant</h1>
        <p className="text-gray-400 mt-2">Your intelligent companion for content creation</p>
      </header>
      <div className="flex-1">
        <ChatInterface />
      </div>
    </div>
  );
};

export default Chat;
