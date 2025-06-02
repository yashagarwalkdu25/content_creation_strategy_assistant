import React, { useState, useEffect } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import TrendingCard from '../components/TrendingCard';

interface TrendingTopic {
  Title: string;
  Info: string;
  Type: string;
  Region: string;
  Metadata: {
    rank: number;
    collected_at: string;
    url?: string;
    related_news?: Array<{
      title: string;
      source: string;
      url: string;
    }>;
  };
}

interface TrendingResponse {
  report_generated_at: string;
  trending_topics: TrendingTopic[];
}

const Trending = () => {
  const [data, setData] = useState<TrendingResponse | null>(() => {
    const savedData = localStorage.getItem('trendingData');
    return savedData ? JSON.parse(savedData) : null;
  });
  const [loading, setLoading] = useState(!data); // Only show loading if no initial data
  const [error, setError] = useState<string | null>(null);

  const fetchTrendingTopics = async (forceRefresh = false) => {
    if (!forceRefresh && data) {
      return; // Don't fetch if we have data and refresh not forced
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5001/api/trending-topics/india?use_rag=true');
      if (!response.ok) {
        throw new Error('Failed to fetch trending topics');
      }
      const result = await response.json();
      setData(result);
      localStorage.setItem('trendingData', JSON.stringify(result));
    } catch (err) {
      console.error('Error fetching trending topics:', err);
      setError('Failed to load trending topics. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrendingTopics();
  }, []);

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-cyber-neon mx-auto mb-4" />
          <p className="text-gray-400">Loading trending topics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <button onClick={fetchTrendingTopics} className="cyber-button">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-cyber-neon mb-2">Trending Topics</h1>
          <p className="text-gray-400">
            Report generated: {data ? new Date(data.report_generated_at).toLocaleString() : 'Unknown'}
          </p>
        </div>
        <button onClick={() => fetchTrendingTopics(true)} className="cyber-button">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {data?.trending_topics.map((topic, index) => (
          <TrendingCard key={`${topic.Title}-${index}`} topic={topic} index={index} />
        ))}
      </div>
    </div>
  );
};

export default Trending;
