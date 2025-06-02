
import React, { useState, useEffect } from 'react';
import { Brain, MessageCircle, TrendingUp, Zap, BarChart3, Users, Target } from 'lucide-react';
import { Link } from 'react-router-dom';

const Dashboard = () => {
  const [recentChats, setRecentChats] = useState(3);
  const [trendsAnalyzed, setTrendsAnalyzed] = useState(12);
  const [contentGenerated, setContentGenerated] = useState(7);

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setRecentChats(prev => prev + Math.floor(Math.random() * 2));
      setTrendsAnalyzed(prev => prev + Math.floor(Math.random() * 3));
      setContentGenerated(prev => prev + Math.floor(Math.random() * 2));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const stats = [
    { label: 'Recent Chats', value: recentChats, icon: MessageCircle, change: '+2.3%' },
    { label: 'Trends Analyzed', value: trendsAnalyzed, icon: TrendingUp, change: '+12.5%' },
    { label: 'Content Generated', value: contentGenerated, icon: Target, change: '+8.1%' },
    { label: 'AI Insights', value: 24, icon: Zap, change: '+15.2%' },
  ];

  const quickActions = [
    { title: 'Start New Chat', description: 'Begin a conversation with AI', icon: MessageCircle, link: '/chat', color: 'cyber-neon' },
    { title: 'View Trends', description: 'Explore trending topics', icon: TrendingUp, link: '/trending', color: 'cyber-neon-blue' },
    { title: 'Analytics', description: 'Check your performance', icon: BarChart3, link: '/analytics', color: 'cyber-neon-purple' },
  ];

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="animate-fade-in-up">
        <h1 className="text-4xl font-bold text-cyber-neon mb-2">Neural Dashboard</h1>
        <p className="text-gray-400">Welcome back! Your AI assistant is ready to help.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <div
            key={stat.label}
            className="glass-card animate-fade-in-up"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="flex items-center justify-between mb-4">
              <stat.icon className="w-8 h-8 text-cyber-neon" />
              <span className="text-cyber-neon text-sm font-medium">{stat.change}</span>
            </div>
            <div>
              <p className="text-2xl font-bold text-white mb-1">{stat.value}</p>
              <p className="text-gray-400 text-sm">{stat.label}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
        <h2 className="text-2xl font-bold text-white mb-6">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {quickActions.map((action, index) => (
            <Link
              key={action.title}
              to={action.link}
              className="glass-card hover:scale-105 transition-all duration-300 group"
              style={{ animationDelay: `${(index + 4) * 0.1}s` }}
            >
              <action.icon className={`w-12 h-12 text-${action.color} mb-4 group-hover:animate-pulse`} />
              <h3 className="text-xl font-bold text-white mb-2">{action.title}</h3>
              <p className="text-gray-400">{action.description}</p>
            </Link>
          ))}
        </div>
      </div>

      {/* AI Status */}
      <div className="animate-fade-in-up" style={{ animationDelay: '0.7s' }}>
        <div className="glass-card">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 bg-cyber-neon rounded-xl flex items-center justify-center animate-glow-pulse">
              <Brain className="w-8 h-8 text-black" />
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-cyber-neon mb-2">AI Neural Network Status</h3>
              <p className="text-gray-300 mb-3">
                Your AI assistant is operating at optimal capacity with real-time learning enabled.
              </p>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-cyber-neon rounded-full animate-pulse"></div>
                <span className="text-cyber-neon text-sm font-medium">Online & Processing</span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-cyber-neon">99.9%</p>
              <p className="text-gray-400 text-sm">Uptime</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
