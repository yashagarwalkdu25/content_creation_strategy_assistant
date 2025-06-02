
import React from 'react';
import { Brain, MessageCircle, TrendingUp, Settings, Zap } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const Sidebar = () => {
  const navItems = [
    // { icon: Brain, label: 'Dashboard', path: '/' },
    // { icon: MessageCircle, label: 'Chat', path: '/chat' },
    { icon: TrendingUp, label: 'Trending', path: '/trending' },
    // { icon: Zap, label: 'Analytics', path: '/analytics' },
    // { icon: Settings, label: 'Settings', path: '/settings' },
  ];

  return (
    <div className="w-64 h-screen bg-cyber-dark glass-dark border-r border-cyber-gray-light animate-slide-in-left">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-cyber-neon rounded-lg flex items-center justify-center animate-glow-pulse">
            <Brain className="w-6 h-6 text-black" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-cyber-neon">NeuralMind</h1>
            <p className="text-xs text-gray-400">AI Second Brain</p>
          </div>
        </div>

        <nav className="space-y-2">
          {navItems.map((item, index) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-300 ${
                  isActive
                    ? 'bg-cyber-neon bg-opacity-20 text-cyber-neon neon-border'
                    : 'text-gray-300 hover:text-cyber-neon hover:bg-cyber-gray-light'
                }`
              }
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="absolute bottom-6 left-6 right-6">
        <div className="glass-card p-4">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-cyber-neon rounded-full animate-pulse"></div>
            <div>
              <p className="text-sm font-medium text-cyber-neon">AI Status</p>
              <p className="text-xs text-gray-400">Online & Learning</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
