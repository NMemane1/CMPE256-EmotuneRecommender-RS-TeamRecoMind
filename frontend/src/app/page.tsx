'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Music, Smile, X, Loader2, Sparkles, Brain, Database, GitCompare, ChevronRight, Zap, Radio, MessageSquare, Link2, Play, Pause, SkipForward, SkipBack, Volume2 } from 'lucide-react';
import clsx from 'clsx';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  audioFile?: File;
  mood?: string;
  recommendations?: Song[];
  isLoading?: boolean;
  requestType?: 'audio' | 'text' | 'mood' | 'similar';
}

interface Song {
  track_id?: string;
  track_name: string;
  track_artist: string;
  similarity?: number;
  explanation?: string;
  valence?: number;
  energy?: number;
}

const MOODS = ['happy', 'sad', 'energetic', 'calm', 'angry', 'romantic', 'mellow'];

const AUDIO_FEATURES = [
  { name: 'Danceability', desc: 'How suitable for dancing', icon: '💃' },
  { name: 'Energy', desc: 'Intensity and activity', icon: '⚡' },
  { name: 'Valence', desc: 'Musical positiveness', icon: '😊' },
  { name: 'Tempo', desc: 'Beats per minute', icon: '🥁' },
  { name: 'Acousticness', desc: 'Acoustic vs electronic', icon: '🎸' },
  { name: 'Instrumentalness', desc: 'Vocal vs instrumental', icon: '🎹' },
];

const PIPELINE_STEPS = {
  similar: [
    { id: 1, title: 'Input Received', desc: 'Song name or Spotify link', icon: Link2, color: 'text-blue-400' },
    { id: 2, title: 'LLM Processing', desc: 'Claude extracts song name', icon: Brain, color: 'text-purple-400' },
    { id: 3, title: 'Database Search', desc: 'Fuzzy match in 30K+ songs', icon: Database, color: 'text-green-400' },
    { id: 4, title: 'Feature Extraction', desc: '9 audio features extracted', icon: Radio, color: 'text-yellow-400' },
    { id: 5, title: 'Cosine Similarity', desc: 'Compare with all songs', icon: GitCompare, color: 'text-pink-400' },
    { id: 6, title: 'Top Matches', desc: 'Return most similar tracks', icon: Sparkles, color: 'text-accent' },
  ],
  mood: [
    { id: 1, title: 'Mood Selected', desc: 'User picks a mood', icon: Smile, color: 'text-blue-400' },
    { id: 2, title: 'Target Mapping', desc: 'Mood → (valence, energy)', icon: Brain, color: 'text-purple-400' },
    { id: 3, title: 'Distance Calc', desc: 'Euclidean distance to target', icon: GitCompare, color: 'text-green-400' },
    { id: 4, title: 'Rank Songs', desc: 'Sort by mood distance', icon: Database, color: 'text-yellow-400' },
    { id: 5, title: 'Top Matches', desc: 'Return closest songs', icon: Sparkles, color: 'text-accent' },
  ],
  text: [
    { id: 1, title: 'Text Input', desc: 'User describes feelings', icon: MessageSquare, color: 'text-blue-400' },
    { id: 2, title: 'Emotion Detection', desc: 'Keywords → emotion', icon: Brain, color: 'text-purple-400' },
    { id: 3, title: 'Mood Mapping', desc: 'Emotion → mood category', icon: Zap, color: 'text-green-400' },
    { id: 4, title: 'Recommendation', desc: 'Mood-based matching', icon: Database, color: 'text-yellow-400' },
    { id: 5, title: 'Results', desc: 'Personalized songs', icon: Sparkles, color: 'text-accent' },
  ],
  audio: [
    { id: 1, title: 'Audio Upload', desc: 'Voice/music file', icon: Radio, color: 'text-blue-400' },
    { id: 2, title: 'Hume AI', desc: 'Prosody analysis', icon: Brain, color: 'text-purple-400' },
    { id: 3, title: 'Emotion Scores', desc: '48 emotions detected', icon: Zap, color: 'text-green-400' },
    { id: 4, title: 'Top Emotion', desc: 'Highest confidence', icon: Sparkles, color: 'text-yellow-400' },
    { id: 5, title: 'Mood Mapping', desc: 'Emotion → mood', icon: GitCompare, color: 'text-pink-400' },
    { id: 6, title: 'Recommendations', desc: 'Songs for your mood', icon: Music, color: 'text-accent' },
  ],
};

const MOOD_TARGETS: Record<string, { valence: number; energy: number }> = {
  happy: { valence: 0.85, energy: 0.70 },
  sad: { valence: 0.20, energy: 0.30 },
  energetic: { valence: 0.75, energy: 0.90 },
  calm: { valence: 0.55, energy: 0.25 },
  angry: { valence: 0.25, energy: 0.85 },
  romantic: { valence: 0.80, energy: 0.55 },
  mellow: { valence: 0.55, energy: 0.45 },
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: `# 🎵 Welcome to EmoTune!

I'm your AI music assistant. I can help you discover music based on your emotions. Here's what I can do:

- **🎤 Voice Analysis**: Upload an audio file and I'll detect emotions from your voice to recommend songs
- **💬 Text Analysis**: Tell me how you're feeling and I'll find music that matches
- **😊 Mood Selection**: Pick a mood directly and get instant recommendations
- **🎸 Similar Songs**: Give me a song name or Spotify link and I'll find similar tracks
- **▶️ Play in Spotify**: Click play on any recommendation to listen instantly

How can I help you today?`,
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [showMoodPicker, setShowMoodPicker] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [pipelineType, setPipelineType] = useState<'similar' | 'mood' | 'text' | 'audio'>('similar');
  const [isAnimating, setIsAnimating] = useState(false);
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null);
  const [spotifyAvailable, setSpotifyAvailable] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check Spotify availability on mount
  useEffect(() => {
    fetch('/api/spotify/play')
      .then(res => res.json())
      .then(data => setSpotifyAvailable(data.available))
      .catch(() => setSpotifyAvailable(false));
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateId = () => Math.random().toString(36).substring(7);

  // Play song in Spotify
  const playInSpotify = async (song: Song) => {
    try {
      setPlayingTrackId(song.track_id || song.track_name);
      
      const response = await fetch('/api/spotify/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trackId: song.track_id,
          trackName: song.track_name,
          trackArtist: song.track_artist,
          action: song.track_id ? 'play' : 'search',
        }),
      });

      const data = await response.json();
      
      if (!data.success) {
        console.error('Spotify error:', data.error);
        alert(data.error || 'Failed to play in Spotify');
        setPlayingTrackId(null);
      }
    } catch (error) {
      console.error('Failed to play:', error);
      setPlayingTrackId(null);
    }
  };

  // Spotify control actions
  const spotifyControl = async (action: 'pause' | 'next' | 'previous') => {
    try {
      await fetch('/api/spotify/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      
      if (action === 'pause') {
        setPlayingTrackId(null);
      }
    } catch (error) {
      console.error('Spotify control error:', error);
    }
  };

  // Animate through pipeline steps when loading
  useEffect(() => {
    if (isLoading && isAnimating) {
      const steps = PIPELINE_STEPS[pipelineType];
      const interval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev < steps.length - 1) return prev + 1;
          return prev;
        });
      }, 600);
      return () => clearInterval(interval);
    }
  }, [isLoading, isAnimating, pipelineType]);

  const handleSend = async () => {
    if (!input.trim() && !audioFile && !selectedMood) return;

    // Determine pipeline type
    let type: 'similar' | 'mood' | 'text' | 'audio' = 'text';
    if (audioFile) type = 'audio';
    else if (selectedMood) type = 'mood';
    else if (input.toLowerCase().includes('similar') || input.includes('spotify.com') || input.toLowerCase().includes('like')) type = 'similar';

    setPipelineType(type);
    setCurrentStep(0);
    setIsAnimating(true);

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: audioFile
        ? `[Uploaded audio: ${audioFile.name}]`
        : selectedMood
        ? `Recommend songs for mood: ${selectedMood}`
        : input,
      audioFile: audioFile || undefined,
      mood: selectedMood || undefined,
      requestType: type,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const loadingId = generateId();
    setMessages((prev) => [
      ...prev,
      { id: loadingId, role: 'assistant', content: '', isLoading: true },
    ]);

    try {
      let response: Response;

      if (audioFile) {
        const formData = new FormData();
        formData.append('file', audioFile);
        formData.append('userMessage', input || 'Analyze this audio and recommend songs');

        response = await fetch('/api/chat', {
          method: 'POST',
          body: formData,
        });
      } else {
        response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: input,
            mood: selectedMood,
            history: messages.filter((m) => !m.isLoading).slice(-10),
          }),
        });
      }

      const data = await response.json();

      // Complete animation
      setCurrentStep(PIPELINE_STEPS[type].length - 1);
      await new Promise(resolve => setTimeout(resolve, 500));

      setMessages((prev) =>
        prev.filter((m) => m.id !== loadingId).concat({
          id: generateId(),
          role: 'assistant',
          content: data.message || data.error || 'Something went wrong',
          recommendations: data.recommendations,
          requestType: type,
        })
      );
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) =>
        prev.filter((m) => m.id !== loadingId).concat({
          id: generateId(),
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please make sure the backend server is running.',
        })
      );
    } finally {
      setIsLoading(false);
      setIsAnimating(false);
      setAudioFile(null);
      setSelectedMood(null);
      setShowMoodPicker(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAudioFile(file);
    }
  };

  const renderRecommendations = (recommendations: Song[]) => {
    const isPlaying = (song: Song) => playingTrackId === (song.track_id || song.track_name);
    
    return (
      <div className="mt-4 space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-gray-400">🎵 Recommended Songs:</h4>
          {spotifyAvailable && playingTrackId && (
            <div className="flex items-center gap-1">
              <button
                onClick={() => spotifyControl('previous')}
                className="p-1 text-gray-400 hover:text-white transition-colors"
                title="Previous track"
              >
                <SkipBack className="w-4 h-4" />
              </button>
              <button
                onClick={() => spotifyControl('pause')}
                className="p-1.5 bg-accent rounded-full text-white hover:bg-accent/80 transition-colors"
                title="Pause"
              >
                <Pause className="w-3 h-3" />
              </button>
              <button
                onClick={() => spotifyControl('next')}
                className="p-1 text-gray-400 hover:text-white transition-colors"
                title="Next track"
              >
                <SkipForward className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
        <div className="grid gap-2">
          {recommendations.slice(0, 10).map((song, idx) => (
            <div
              key={idx}
              className={clsx(
                "bg-chat-input rounded-lg p-3 border animate-fade-in group transition-all",
                isPlaying(song) 
                  ? "border-accent bg-accent/10" 
                  : "border-gray-700 hover:border-gray-600"
              )}
              style={{ animationDelay: `${idx * 100}ms` }}
            >
              <div className="flex items-center gap-3">
                {/* Play Button */}
                {spotifyAvailable !== false && (
                  <button
                    onClick={() => playInSpotify(song)}
                    className={clsx(
                      "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center transition-all",
                      isPlaying(song)
                        ? "bg-accent text-white animate-pulse"
                        : "bg-gray-700 text-gray-400 hover:bg-accent hover:text-white group-hover:scale-105"
                    )}
                    title={isPlaying(song) ? "Now playing" : "Play in Spotify"}
                  >
                    {isPlaying(song) ? (
                      <Volume2 className="w-5 h-5 animate-pulse" />
                    ) : (
                      <Play className="w-5 h-5 ml-0.5" />
                    )}
                  </button>
                )}
                
                <div className="flex-1 min-w-0">
                  <p className={clsx(
                    "font-medium truncate",
                    isPlaying(song) ? "text-accent" : "text-white"
                  )}>
                    {song.track_name}
                  </p>
                  <p className="text-sm text-gray-400 truncate">{song.track_artist}</p>
                </div>
                
                {song.similarity !== undefined && (
                  <span className="flex-shrink-0 text-xs bg-accent/20 text-accent px-2 py-1 rounded">
                    {(song.similarity * 100).toFixed(0)}% match
                  </span>
                )}
              </div>
              {song.explanation && (
                <p className="text-xs text-gray-500 mt-2 ml-13">{song.explanation}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const steps = PIPELINE_STEPS[pipelineType];

  return (
    <div className="flex h-screen bg-chat-bg">
      {/* Left Sidebar */}
      <div className="hidden lg:flex w-56 bg-chat-sidebar flex-col border-r border-gray-800">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Music className="w-6 h-6 text-accent" />
            EmoTune
          </h1>
        </div>
        <div className="flex-1 p-4">
          <p className="text-sm text-gray-500">
            AI-powered music recommendations based on your emotions
          </p>
        </div>
        <div className="p-4 border-t border-gray-800">
          <p className="text-xs text-gray-600">Powered by OpenRouter + Hume AI</p>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="p-4 border-b border-gray-800 flex items-center justify-between lg:hidden">
          <h1 className="text-lg font-bold flex items-center gap-2">
            <Music className="w-5 h-5 text-accent" />
            EmoTune
          </h1>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={clsx(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={clsx(
                  'max-w-[85%] rounded-2xl px-4 py-3',
                  message.role === 'user'
                    ? 'bg-chat-user text-white'
                    : 'bg-chat-assistant'
                )}
              >
                {message.isLoading ? (
                  <div className="flex items-center gap-1">
                    <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                    <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                    <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                  </div>
                ) : (
                  <>
                    <div
                      className="prose prose-invert max-w-none text-sm"
                      dangerouslySetInnerHTML={{
                        __html: message.content
                          .replace(/\n/g, '<br>')
                          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          .replace(/\*(.*?)\*/g, '<em>$1</em>')
                          .replace(/^# (.*?)$/gm, '<h1 class="text-xl font-bold">$1</h1>')
                          .replace(/^## (.*?)$/gm, '<h2 class="text-lg font-semibold">$1</h2>')
                          .replace(/^- (.*?)$/gm, '<li>$1</li>'),
                      }}
                    />
                    {message.recommendations && message.recommendations.length > 0 && (
                      renderRecommendations(message.recommendations)
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-gray-800">
          {audioFile && (
            <div className="mb-2 flex items-center gap-2 bg-chat-input rounded-lg px-3 py-2">
              <Music className="w-4 h-4 text-accent" />
              <span className="text-sm flex-1 truncate">{audioFile.name}</span>
              <button onClick={() => setAudioFile(null)} className="text-gray-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {showMoodPicker && (
            <div className="mb-2 flex flex-wrap gap-2">
              {MOODS.map((mood) => (
                <button
                  key={mood}
                  onClick={() => {
                    setSelectedMood(mood);
                    setShowMoodPicker(false);
                  }}
                  className={clsx(
                    'px-3 py-1.5 rounded-full text-sm capitalize transition-colors',
                    selectedMood === mood
                      ? 'bg-accent text-white'
                      : 'bg-chat-input hover:bg-gray-700 text-gray-300'
                  )}
                >
                  {mood}
                </button>
              ))}
            </div>
          )}

          {selectedMood && !showMoodPicker && (
            <div className="mb-2 flex items-center gap-2 bg-accent/20 rounded-lg px-3 py-2">
              <Smile className="w-4 h-4 text-accent" />
              <span className="text-sm flex-1 capitalize">Mood: {selectedMood}</span>
              <button onClick={() => setSelectedMood(null)} className="text-gray-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          <div className="flex items-end gap-2 bg-chat-input rounded-2xl p-2">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="audio/*"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700 transition-colors"
              title="Upload audio file"
            >
              <Paperclip className="w-5 h-5" />
            </button>
            <button
              onClick={() => setShowMoodPicker(!showMoodPicker)}
              className={clsx(
                'p-2 rounded-lg transition-colors',
                showMoodPicker ? 'text-accent bg-accent/20' : 'text-gray-400 hover:text-white hover:bg-gray-700'
              )}
              title="Pick a mood"
            >
              <Smile className="w-5 h-5" />
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe how you're feeling, or paste a Spotify link..."
              className="flex-1 bg-transparent border-none outline-none resize-none text-white placeholder-gray-500 min-h-[24px] max-h-[120px] py-2"
              rows={1}
            />
            <button
              onClick={handleSend}
              disabled={isLoading || (!input.trim() && !audioFile && !selectedMood)}
              className={clsx(
                'p-2 rounded-lg transition-colors',
                isLoading || (!input.trim() && !audioFile && !selectedMood)
                  ? 'text-gray-600 cursor-not-allowed'
                  : 'text-accent hover:bg-accent/20'
              )}
            >
              {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Right Panel - How It Works */}
      <div className="hidden xl:flex w-80 bg-chat-sidebar flex-col border-l border-gray-800 overflow-hidden">
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-lg font-bold flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-accent" />
            How It Works
          </h2>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Pipeline Type Selector */}
          <div className="space-y-2">
            <p className="text-xs text-gray-500 uppercase tracking-wider">Pipeline Type</p>
            <div className="grid grid-cols-2 gap-2">
              {(['similar', 'mood', 'text', 'audio'] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => { setPipelineType(type); setCurrentStep(0); }}
                  className={clsx(
                    'px-3 py-2 rounded-lg text-xs font-medium transition-all',
                    pipelineType === type
                      ? 'bg-accent text-white'
                      : 'bg-chat-input text-gray-400 hover:text-white'
                  )}
                >
                  {type === 'similar' && '🎸 Similar'}
                  {type === 'mood' && '😊 Mood'}
                  {type === 'text' && '💬 Text'}
                  {type === 'audio' && '🎤 Audio'}
                </button>
              ))}
            </div>
          </div>

          {/* Pipeline Steps */}
          <div className="space-y-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider">Pipeline Steps</p>
            <div className="relative">
              {/* Connecting Line */}
              <div className="absolute left-5 top-8 bottom-4 w-0.5 bg-gray-700" />
              
              {steps.map((step, idx) => {
                const Icon = step.icon;
                const isActive = idx === currentStep && isAnimating;
                const isComplete = idx < currentStep || (!isAnimating && idx <= currentStep);
                
                return (
                  <div
                    key={step.id}
                    className={clsx(
                      'relative flex items-start gap-3 p-2 rounded-lg transition-all duration-500',
                      isActive && 'bg-accent/10 scale-105',
                      isComplete && !isActive && 'opacity-100',
                      !isComplete && !isActive && 'opacity-40'
                    )}
                  >
                    <div
                      className={clsx(
                        'relative z-10 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-500',
                        isActive && 'bg-accent animate-pulse shadow-lg shadow-accent/50',
                        isComplete && !isActive && 'bg-gray-700',
                        !isComplete && !isActive && 'bg-gray-800'
                      )}
                    >
                      <Icon className={clsx('w-5 h-5', isActive ? 'text-white' : step.color)} />
                    </div>
                    <div className="flex-1 pt-1">
                      <p className={clsx(
                        'font-medium text-sm transition-colors',
                        isActive ? 'text-white' : 'text-gray-300'
                      )}>
                        {step.title}
                      </p>
                      <p className="text-xs text-gray-500">{step.desc}</p>
                    </div>
                    {isActive && (
                      <ChevronRight className="w-4 h-4 text-accent animate-bounce-x mt-2" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Audio Features */}
          {pipelineType === 'similar' && (
            <div className="space-y-3">
              <p className="text-xs text-gray-500 uppercase tracking-wider">Audio Features Used</p>
              <div className="grid grid-cols-2 gap-2">
                {AUDIO_FEATURES.map((feature, idx) => (
                  <div
                    key={feature.name}
                    className="bg-chat-input rounded-lg p-2 animate-fade-in"
                    style={{ animationDelay: `${idx * 100}ms` }}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{feature.icon}</span>
                      <div>
                        <p className="text-xs font-medium text-gray-300">{feature.name}</p>
                        <p className="text-[10px] text-gray-500">{feature.desc}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Mood Targets */}
          {pipelineType === 'mood' && (
            <div className="space-y-3">
              <p className="text-xs text-gray-500 uppercase tracking-wider">Mood Target Points</p>
              <div className="bg-chat-input rounded-lg p-3">
                <div className="relative w-full h-48">
                  {/* Grid */}
                  <div className="absolute inset-0 grid grid-cols-5 grid-rows-5 opacity-20">
                    {[...Array(25)].map((_, i) => (
                      <div key={i} className="border border-gray-600" />
                    ))}
                  </div>
                  
                  {/* Axis Labels */}
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-[10px] text-gray-500">
                    Valence →
                  </div>
                  <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-gray-500">
                    Energy →
                  </div>
                  
                  {/* Mood Points */}
                  {Object.entries(MOOD_TARGETS).map(([mood, target], idx) => (
                    <div
                      key={mood}
                      className="absolute w-3 h-3 rounded-full animate-pop-in"
                      style={{
                        left: `${target.valence * 100}%`,
                        bottom: `${target.energy * 100}%`,
                        transform: 'translate(-50%, 50%)',
                        animationDelay: `${idx * 150}ms`,
                        backgroundColor: mood === 'happy' ? '#22c55e' :
                                        mood === 'sad' ? '#3b82f6' :
                                        mood === 'energetic' ? '#f59e0b' :
                                        mood === 'calm' ? '#06b6d4' :
                                        mood === 'angry' ? '#ef4444' :
                                        mood === 'romantic' ? '#ec4899' : '#8b5cf6',
                      }}
                      title={`${mood}: valence=${target.valence}, energy=${target.energy}`}
                    >
                      <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] text-gray-400 whitespace-nowrap capitalize">
                        {mood}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Cosine Similarity Explanation */}
          {pipelineType === 'similar' && (
            <div className="space-y-3">
              <p className="text-xs text-gray-500 uppercase tracking-wider">Similarity Calculation</p>
              <div className="bg-chat-input rounded-lg p-3 space-y-2">
                <div className="font-mono text-xs text-accent">
                  cos(θ) = A·B / (||A|| × ||B||)
                </div>
                <p className="text-xs text-gray-500">
                  Each song is a 9-dimensional vector. We find songs pointing in the same direction in feature space.
                </p>
                <div className="flex items-center gap-2 pt-2">
                  <div className="w-8 h-8 rounded-full bg-accent/20 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-accent animate-ping-slow" />
                  </div>
                  <div className="flex-1 h-1 bg-gradient-to-r from-accent to-transparent rounded" />
                  <span className="text-xs text-gray-500">Similar vectors</span>
                </div>
              </div>
            </div>
          )}

          {/* Spotify Integration Status */}
          <div className="space-y-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider">Spotify Integration</p>
            <div className="bg-chat-input rounded-lg p-3">
              <div className="flex items-center gap-3">
                <div className={clsx(
                  "w-10 h-10 rounded-full flex items-center justify-center",
                  spotifyAvailable ? "bg-green-500/20" : "bg-gray-700"
                )}>
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill={spotifyAvailable ? "#1DB954" : "#666"}>
                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                  </svg>
                </div>
                <div className="flex-1">
                  <p className={clsx(
                    "text-sm font-medium",
                    spotifyAvailable ? "text-green-400" : "text-gray-400"
                  )}>
                    {spotifyAvailable === null 
                      ? "Checking..." 
                      : spotifyAvailable 
                        ? "Connected" 
                        : "Not available"}
                  </p>
                  <p className="text-xs text-gray-500">
                    {spotifyAvailable 
                      ? "Click play on any song" 
                      : "macOS + Spotify required"}
                  </p>
                </div>
                {playingTrackId && (
                  <div className="flex items-center gap-1">
                    <span className="w-1 h-3 bg-green-500 rounded-full animate-pulse" />
                    <span className="w-1 h-4 bg-green-500 rounded-full animate-pulse animation-delay-100" />
                    <span className="w-1 h-2 bg-green-500 rounded-full animate-pulse animation-delay-200" />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-gray-800 text-center">
          <p className="text-xs text-gray-600">
            30,000+ songs • 9 audio features • Real-time matching
          </p>
        </div>
      </div>
    </div>
  );
}
