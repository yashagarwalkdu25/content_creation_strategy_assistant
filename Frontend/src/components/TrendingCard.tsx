
import React from 'react';
import { ExternalLink, Clock, MapPin } from 'lucide-react';

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

interface TrendingCardProps {
  topic: TrendingTopic;
  index: number;
}

const TrendingCard: React.FC<TrendingCardProps> = ({ topic, index }) => {
  return (
    <div 
      className="glass-card hover:scale-105 hover:neon-glow cursor-pointer animate-fade-in-up"
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-2">
          <span className="text-cyber-neon font-bold text-lg">#{topic.Metadata.rank}</span>
          <span className="px-2 py-1 bg-cyber-neon bg-opacity-20 text-cyber-neon rounded-full text-xs">
            {topic.Type}
          </span>
        </div>
        {topic.Metadata.url && (
          <a 
            href={topic.Metadata.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyber-neon hover:text-cyber-neon-blue transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>

      <h3 className="text-white font-bold text-lg mb-3 leading-tight">
        {topic.Title}
      </h3>

      <p className="text-gray-300 text-sm mb-4 line-clamp-3">
        {topic.Info}
      </p>

      <div className="flex items-center justify-between text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <MapPin className="w-3 h-3" />
          <span>{topic.Region}</span>
        </div>
        <div className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          <span>{new Date(topic.Metadata.collected_at).toLocaleDateString()}</span>
        </div>
      </div>

      {topic.Metadata.related_news && topic.Metadata.related_news.length > 0 && (
        <div className="mt-4 pt-4 border-t border-cyber-gray-light">
          <p className="text-cyber-neon text-xs mb-2">Related News:</p>
          <div className="space-y-1">
            {topic.Metadata.related_news.slice(0, 2).map((news, idx) => (
              <a
                key={idx}
                href={news.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block text-xs text-gray-400 hover:text-cyber-neon transition-colors truncate"
              >
                {news.title}
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TrendingCard;
