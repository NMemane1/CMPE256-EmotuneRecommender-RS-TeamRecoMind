export const runtime = 'nodejs';

import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const OPENROUTER_BASE_URL = process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'anthropic/claude-3-haiku';

const SITE_URL = process.env.SITE_URL || 'http://localhost:3000';
const APP_NAME = process.env.APP_NAME || 'EmoTune';

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

const openai = new OpenAI({
  baseURL: OPENROUTER_BASE_URL,
  apiKey: OPENROUTER_API_KEY,
  defaultHeaders: {
    // OpenRouter recommended headers (helps with auth/attribution + fewer weird 401s)
    'HTTP-Referer': SITE_URL,
    'X-Title': APP_NAME,
  },
});

// --------------------
// ✅ Option A helper: if backend returns 0 recs (e.g., "hi"), show professional prompt
// --------------------
function respondNoRecs(userMessage: string) {
  void userMessage;
  return NextResponse.json({
    message: "👋 Tell me how you're feeling (happy, sad, angry, calm, etc.) and I’ll recommend music that fits your mood 🎵",
    recommendations: [],
  });
}

// --------------------
// ✅ Small-talk guard (prevents hanging OpenRouter calls on “hi”)
// --------------------
function isSmallTalkOrTooShort(text: string): boolean {
  const t = String(text || '').toLowerCase().trim();
  if (!t) return true;

  const smalltalk = new Set([
    'hi', 'hey', 'hello', 'yo', 'sup', 'ok', 'okay', 'k', 'kk', 'hii', 'heyy', 'hola', 'test',
  ]);

  if (smalltalk.has(t)) return true;

  // single token like "hmm", "huh", "bro", etc. -> ask for feeling instead of recommending
  const tokens = t.split(/\s+/).filter(Boolean);
  if (tokens.length <= 1) return true;

  return false;
}

// --------------------
// ✅ Fear UX helper: if user mentions fear/anxiety, use a comforting message
// --------------------
function containsFearLanguage(text: string): boolean {
  const fearWords = ['scared', 'afraid', 'fear', 'anxious', 'anxiety', 'nervous', 'panic', 'panicking', 'terrified'];
  const t = String(text || '').toLowerCase();
  return fearWords.some((w) => t.includes(w));
}

function fearRecoMessage(): string {
  return 'That sounds scary. 💙 Here are some songs you can listen to that may help you feel calmer and more at ease.';
}

function textRecoMessage(userMessage: string): string {
  return containsFearLanguage(userMessage)
    ? fearRecoMessage()
    : "I analyzed your message and found songs that match how you're feeling:";
}

// --------------------
// ✅ Timeout helpers (prevents infinite “…” loading)
// --------------------
async function fetchWithTimeout(url: string, init: RequestInit = {}, ms = 12000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function openRouterWithTimeout<T>(fn: (signal: AbortSignal) => Promise<T>, ms = 12000): Promise<T> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  try {
    return await fn(controller.signal);
  } finally {
    clearTimeout(id);
  }
}

// Helper to extract Spotify track ID from various URL formats or bare track IDs
function extractSpotifyTrackId(input: string): string | null {
  const urlMatch = input.match(/spotify\.com\/track\/([a-zA-Z0-9]+)/);
  if (urlMatch) return urlMatch[1];

  const uriMatch = input.match(/spotify:track:([a-zA-Z0-9]+)/);
  if (uriMatch) return uriMatch[1];

  const bareIdMatch = input.trim().match(/^([a-zA-Z0-9]{22})$/);
  if (bareIdMatch) return bareIdMatch[1];

  const embeddedIdMatch = input.match(/\b([a-zA-Z0-9]{22})\b/);
  if (embeddedIdMatch) return embeddedIdMatch[1];

  return null;
}

// Tool definitions for function calling
const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'recommend_by_mood',
      description:
        'Get song recommendations based on a mood/emotion. Use this when the user mentions a mood like happy, sad, energetic, calm, angry, romantic, mellow.',
      parameters: {
        type: 'object',
        properties: {
          mood: { type: 'string' },
          count: { type: 'number' },
        },
        required: ['mood'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'analyze_text_emotion',
      description:
        'Analyze the emotion in a text message and return song recommendations. Use this when the user describes their feelings or situation.',
      parameters: {
        type: 'object',
        properties: {
          text: { type: 'string' },
        },
        required: ['text'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_similar_songs',
      description:
        'Find songs similar to a given song. Use this when the user mentions they like a specific song or want songs like a particular track.',
      parameters: {
        type: 'object',
        properties: {
          song_name: { type: 'string' },
          count: { type: 'number' },
        },
        required: ['song_name'],
      },
    },
  },
];

// Tool execution functions
async function recommendByMood(mood: string, count = 5) {
  try {
    const response = await fetchWithTimeout(
      `${BACKEND_URL}/api/recommend/text`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: `I am feeling ${mood}`, top_n: count }),
      },
      12000
    );

    if (!response.ok) {
      const errorText = await response.text();
      return { error: errorText || 'Failed to get recommendations' };
    }

    return await response.json();
  } catch (e) {
    return { error: `Backend server is not available or timed out. (${BACKEND_URL})` };
  }
}

async function analyzeTextEmotion(text: string) {
  try {
    const response = await fetchWithTimeout(
      `${BACKEND_URL}/api/recommend/text`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, top_n: 5 }),
      },
      12000
    );

    if (!response.ok) {
      const errorText = await response.text();
      return { error: errorText || 'Failed to analyze text' };
    }

    return await response.json();
  } catch (e) {
    return { error: `Backend server is not available or timed out. (${BACKEND_URL})` };
  }
}

async function getSimilarSongs(songName: string, count = 5) {
  try {
    const response = await fetchWithTimeout(
      `${BACKEND_URL}/api/recommend/similar`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song_name: songName, top_n: count }),
      },
      12000
    );

    if (!response.ok) {
      const errorText = await response.text();
      return { error: errorText || 'Failed to find similar songs' };
    }

    const result = await response.json();

    if (!result.found) {
      return {
        error: `Song "${songName}" not found in the database. Try a different song name.`,
        recommendations: [],
      };
    }

    return {
      recommendations: result.recommendations,
      message: `Songs similar to "${songName}"`,
    };
  } catch (e) {
    return { error: `Backend server is not available or timed out. (${BACKEND_URL})` };
  }
}

async function getSimilarSongsBySpotifyId(trackId: string, count = 5) {
  try {
    const response = await fetchWithTimeout(
      `${BACKEND_URL}/api/recommend/spotify`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track_id: trackId, top_n: count }),
      },
      12000
    );

    if (!response.ok) {
      const errorText = await response.text();
      return { error: errorText || 'Failed to find song from Spotify link' };
    }

    const result = await response.json();

    if (!result.found) {
      return {
        error: `This song (Spotify ID: ${trackId}) is not in our database. Try sharing a different song!`,
        recommendations: [],
        notFound: true,
      };
    }

    return {
      recommendations: result.recommendations,
      trackName: result.track_name,
      trackArtist: result.track_artist,
      message: `Songs similar to "${result.track_name}" by ${result.track_artist}`,
    };
  } catch (e) {
    return { error: `Backend server is not available or timed out. (${BACKEND_URL})` };
  }
}

async function analyzeAudio(formData: FormData) {
  try {
    const response = await fetchWithTimeout(
      `${BACKEND_URL}/api/recommend/audio`,
      {
        method: 'POST',
        body: formData,
      },
      20000
    );

    if (!response.ok) {
      const errorText = await response.text();
      return { error: errorText || 'Failed to analyze audio' };
    }

    return await response.json();
  } catch (e) {
    return { error: `Backend server is not available or timed out. (${BACKEND_URL})` };
  }
}

// Execute tool calls
async function executeTool(name: string, args: Record<string, unknown>) {
  switch (name) {
    case 'recommend_by_mood':
      return await recommendByMood(args.mood as string, (args.count as number) ?? 5);
    case 'analyze_text_emotion':
      return await analyzeTextEmotion(args.text as string);
    case 'get_similar_songs':
      return await getSimilarSongs(args.song_name as string, (args.count as number) ?? 5);
    default:
      return { error: `Unknown tool: ${name}` };
  }
}

function looksLikeAuthError(err: unknown): boolean {
  const s = String(err || '');
  return s.includes('401') || s.toLowerCase().includes('unauthorized') || s.toLowerCase().includes('user not found');
}

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type') || '';

    // Handle audio file upload
    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      const file = formData.get('file') as File;

      if (!file) {
        return NextResponse.json({ error: 'No audio file provided' }, { status: 400 });
      }

      const backendFormData = new FormData();
      backendFormData.append('file', file);

      const result = await analyzeAudio(backendFormData);

      if ((result as any).error) {
        return NextResponse.json({
          message: `❌ Error: ${(result as any).error}`,
          recommendations: [],
        });
      }

      const emotions = (result as any).emotions || [];
      const topEmotion = emotions[0];
      const recommendations = (result as any).recommendations || [];

      let message = `## 🎤 Audio Analysis Complete!\n\n`;
      if (topEmotion) {
        message += `I detected **${topEmotion.name}** (${(topEmotion.score * 100).toFixed(1)}% confidence) in your voice.\n\n`;
        message += `Based on this emotion, here are some songs that match your mood:`;
      } else {
        message += `I analyzed your audio and found some songs you might enjoy:`;
      }

      if (!recommendations || recommendations.length === 0) {
        return respondNoRecs('(audio)');
      }

      return NextResponse.json({
        message,
        recommendations,
        emotions,
      });
    }

    // Handle JSON request (text/mood)
    const body = await request.json();
    const { message = '', mood, history = [] } = body;

    // ✅ IMPORTANT: prevent “…” hanging for small-talk by returning immediately
    if (isSmallTalkOrTooShort(message) && !mood) {
      return respondNoRecs(message);
    }

    // Spotify link shortcut (only if message exists)
    const spotifyTrackId = message ? extractSpotifyTrackId(message) : null;
    if (spotifyTrackId) {
      const result = await getSimilarSongsBySpotifyId(spotifyTrackId, 5);

      if ((result as any).error) {
        return NextResponse.json({
          message: (result as any).notFound
            ? `😔 Sorry, I couldn't find that song in our database. The Spotify track ID \`${spotifyTrackId}\` isn't in our music collection. Try sharing a different song or search by song name instead!`
            : `❌ Error: ${(result as any).error}`,
          recommendations: [],
        });
      }

      const recs = (result as any).recommendations || [];
      if (recs.length === 0) return respondNoRecs(message);

      return NextResponse.json({
        message: `## 🎵 Found it!\n\nI found **"${(result as any).trackName}"** by **${(result as any).trackArtist}** from your Spotify link.\n\nHere are some similar songs based on audio features like danceability, energy, valence, and tempo:`,
        recommendations: recs,
        sourceSong: {
          name: (result as any).trackName,
          artist: (result as any).trackArtist,
        },
      });
    }

    // Mood buttons -> backend
    if (mood) {
      const result = await recommendByMood(mood, 5);

      if ((result as any).error) {
        return NextResponse.json({
          message: `❌ Error: ${(result as any).error}`,
          recommendations: [],
        });
      }

      const recs = (result as any).recommendations || [];
      if (recs.length === 0) return respondNoRecs(String(mood));

      const moodMsg = containsFearLanguage(String(mood))
        ? fearRecoMessage()
        : `## 😊 Mood: ${mood.charAt(0).toUpperCase() + mood.slice(1)}\n\nHere are some songs that match your **${mood}** mood:`;

      return NextResponse.json({
        message: moodMsg,
        recommendations: recs,
      });
    }

    // If no OpenRouter key -> backend only
    if (!OPENROUTER_API_KEY) {
      const result = await analyzeTextEmotion(message);

      if ((result as any).error) {
        return NextResponse.json({
          message: `❌ Error: ${(result as any).error}`,
          recommendations: [],
        });
      }

      const recs = (result as any).recommendations || [];
      if (recs.length === 0) return respondNoRecs(message);

      return NextResponse.json({
        message: textRecoMessage(message),
        recommendations: recs,
      });
    }

    // Build conversation history for the LLM
    interface HistoryMessage {
      role: 'user' | 'assistant';
      content: string;
    }

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: `You are EmoTune, an AI music recommendation assistant. You help users discover music based on their emotions.

## YOUR ONLY JOB
You MUST use the provided tools to get song recommendations. The tools query a real database of 30,000+ songs.

## AVAILABLE TOOLS
1. recommend_by_mood - Use when user mentions a mood (happy, sad, energetic, calm, angry, romantic, mellow)
2. analyze_text_emotion - Use when user describes feelings or a situation  
3. get_similar_songs - Use when user mentions a song name they like

## ABSOLUTE RULES - VIOLATION = FAILURE
1. NEVER write song names, artist names, or album names in your response
2. NEVER suggest songs from your own knowledge - you don't know any songs
3. NEVER list songs in any format (bullet points, numbered lists, etc.)
4. ALWAYS call a tool when the user wants music recommendations
5. After calling a tool, write ONLY a brief friendly message (1-2 sentences) - the actual songs will be displayed as interactive cards by the UI
6. If a tool returns an error or "not found", tell the user the song isn't in our database - DO NOT suggest alternatives
7. If you cannot help with music, just say so - never make up songs

Remember: You have NO knowledge of songs. The tools are your ONLY source of music recommendations.`,
      },
      ...history.map((h: HistoryMessage) => ({
        role: h.role as 'user' | 'assistant',
        content: h.content,
      })),
      { role: 'user', content: message },
    ];

    // First LLM call with tools - FORCE tool usage (with timeout)
    let assistantMessage: OpenAI.Chat.Completions.ChatCompletionMessage;
    try {
      const completion = await openRouterWithTimeout(
        (signal) =>
          openai.chat.completions.create({
            model: DEFAULT_MODEL,
            messages,
            tools,
            tool_choice: 'required',
            // OpenAI SDK supports abort via signal (runtime safe)
            signal: signal as any,
          } as any),
        12000
      );

      assistantMessage = completion.choices[0].message;
    } catch (err) {
      // If OpenRouter auth OR OpenRouter timed out -> fall back to backend instead of hanging/500
      const isTimeout = String(err || '').toLowerCase().includes('aborted') || String(err || '').toLowerCase().includes('timeout');

      if (looksLikeAuthError(err) || isTimeout) {
        const result = await analyzeTextEmotion(message);
        if ((result as any).error) {
          return NextResponse.json(
            { message: `❌ OpenRouter failed and backend fallback also failed: ${(result as any).error}`, recommendations: [] },
            { status: 200 }
          );
        }

        const recs = (result as any).recommendations || [];
        if (recs.length === 0) return respondNoRecs(message);

        return NextResponse.json({ message: textRecoMessage(message), recommendations: recs }, { status: 200 });
      }

      throw err;
    }

    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      const toolResults: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
      let allRecommendations: unknown[] = [];

      for (const toolCall of assistantMessage.tool_calls) {
        const args = JSON.parse(toolCall.function.arguments);
        const result = await executeTool(toolCall.function.name, args);

        toolResults.push({
          tool_call_id: toolCall.id,
          role: 'tool',
          content: JSON.stringify(result),
        });

        if ((result as any).recommendations) {
          allRecommendations = [...allRecommendations, ...(result as any).recommendations];
        }
      }

      if (!allRecommendations || allRecommendations.length === 0) {
        return respondNoRecs(message);
      }

      const finalCompletion = await openRouterWithTimeout(
        (signal) =>
          openai.chat.completions.create({
            model: DEFAULT_MODEL,
            messages: [...messages, assistantMessage, ...toolResults],
            signal: signal as any,
          } as any),
        12000
      );

      const finalMsg = containsFearLanguage(message)
        ? fearRecoMessage()
        : String(finalCompletion.choices[0].message.content || "Here are some recommendations for you:");

      return NextResponse.json({
        message: finalMsg,
        recommendations: allRecommendations,
      });
    }

    return NextResponse.json({
      message:
        "I can help you discover music! Try telling me your mood (happy, sad, energetic, calm) or share a song name you like, and I'll find recommendations from our database.",
      recommendations: [],
    });
  } catch (error) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      {
        message: 'Sorry, I encountered an error processing your request. Please try again.',
        error: String(error),
      },
      { status: 500 }
    );
  }
}