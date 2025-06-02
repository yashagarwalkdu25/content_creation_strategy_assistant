
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Chat from "./pages/Chat";
import Trending from "./pages/Trending";
import Sidebar from "./components/Sidebar";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const Layout = ({ children }: { children: React.ReactNode }) => (
  <div className="flex h-screen bg-cyber-black overflow-hidden">
    <Sidebar />
    <main className="flex-1 overflow-auto">
      {children}
    </main>
  </div>
);

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/chat" element={<Layout><Chat /></Layout>} />
          <Route path="/trending" element={<Layout><Trending /></Layout>} />
          <Route path="/analytics" element={<Layout><div className="p-6"><h1 className="text-3xl font-bold text-cyber-neon">Analytics Coming Soon</h1></div></Layout>} />
          <Route path="/settings" element={<Layout><div className="p-6"><h1 className="text-3xl font-bold text-cyber-neon">Settings Coming Soon</h1></div></Layout>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
