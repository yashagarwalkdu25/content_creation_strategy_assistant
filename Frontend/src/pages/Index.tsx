
import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import Dashboard from './Dashboard';

const Index = () => {
  return (
    <div className="flex h-screen bg-cyber-black overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Dashboard />
      </main>
    </div>
  );
};

export default Index;
