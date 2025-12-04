import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY,
});

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Helper to extract Spotify track ID from various URL formats or bare track IDs
function extractSpotifyTrackId(input: string): string | null {
  // Match patterns like:
  // https://open.spotify.com/track/0RHPoliIwT6ddbPugZNitX
  // https://open.spotify.com/track/0RHPoliIwT6ddbPugZNitX?si=6827d49204c048b7
  // spotify:track:0RHPoliIwT6ddbPugZNitX
  const urlMatch = input.match(/spotify\.com\/track\/([a-zA-Z0-9]+)/);
  if (urlMatch) return urlMatch[1];
  
  const uriMatch = input.match(/spotify:track:([a-zA-Z0-9]+)/);
  if (uriMatch) return uriMatch[1];
  
  // Also match bare Spotify track IDs (22 characters, alphanumeric)
  // Spotify track IDs are base62 encoded, 22 chars long
  const bareIdMatch = input.trim().match(/^([a-zA-Z0-9]{22})$/);
  if (bareIdMatch) return bareIdMatch[1];
  
  // Check if the message contains a bare track ID anywhere
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
      description: 'Get song recommendations based on a mood/emotion. Use this when the user mentions a mood like happy, sad, energetic, calm, etc.',
      parameters: {
        type: 'object',
        properties: {
          mood: {
            type: 'string',
            description: 'The mood to get recommendations for (e.g., happy, sad, energetic, calm, angry, romantic, mellow)',
          },
          count: {
            type: 'number',
            description: 'Number of songs to recommend (default 5)',
          },
        },
        required: ['mood'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'analyze_text_emotion',
      description: 'Analyze the emotion in a text message and return song recommendations. Use this when the user describes their feelings or situation.',
      parameters: {
        type: 'object',
        properties: {
          text: {
            type: 'string',
            description: 'The text to analyze for emotion',
          },
        },
        required: ['text'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_similar_songs',
      description: 'Find songs similar to a given song. Use this when the user mentions they like a specific song or want songs like a particular track.',
      parameters: {
        type: 'object',
        properties: {
          song_name: {
            type: 'string',
            description: 'The name of the song to find similar tracks for',
          },
          count: {
            type: 'number',
            description: 'Number of similar songs to return (default 5)',
          },
        },
        required: ['song_name'],
      },
    },
  },
];

// Tool execution functions
async function recommendByMood(mood: string, count: number = 5) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/recommend/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: `I am feeling ${mood}`, top_n: count }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || 'Failed to get recommendations' };
    }
    
    return await response.json();
  } catch (error) {
    return { error: 'Backend server is not available. Please make sure it is running on ' + BACKEND_URL };
  }
}

async function analyzeTextEmotion(text: string) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/recommend/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, top_n: 5 }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || 'Failed to analyze text' };
    }
    
    return await response.json();
  } catch (error) {
    return { error: 'Backend server is not available. Please make sure it is running on ' + BACKEND_URL };
  }
}

async function getSimilarSongs(songName: string, count: number = 5) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/recommend/similar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        song_name: songName, 
        top_n: count 
      }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || 'Failed to find similar songs' };
    }
    
    const result = await response.json();
    
    if (!result.found) {
      return { 
        error: `Song "${songName}" not found in the database. Try a different song name.`,
        recommendations: []
      };
    }
    
    return {
      recommendations: result.recommendations,
      message: `Songs similar to "${songName}"`,
    };
  } catch (error) {
    return { error: 'Backend server is not available. Please make sure it is running on ' + BACKEND_URL };
  }
}

async function getSimilarSongsBySpotifyId(trackId: string, count: number = 5) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/recommend/spotify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        track_id: trackId, 
        top_n: count 
      }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || 'Failed to find song from Spotify link' };
    }
    
    const result = await response.json();
    
    if (!result.found) {
      return { 
        error: `This song (Spotify ID: ${trackId}) is not in our database. Try sharing a different song!`,
        recommendations: [],
        notFound: true
      };
    }
    
    return {
      recommendations: result.recommendations,
      trackName: result.track_name,
      trackArtist: result.track_artist,
      message: `Songs similar to "${result.track_name}" by ${result.track_artist}`,
    };
  } catch (error) {
    return { error: 'Backend server is not available. Please make sure it is running on ' + BACKEND_URL };
  }
}

async function analyzeAudio(formData: FormData) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/recommend/audio`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || 'Failed to analyze audio' };
    }
    
    return await response.json();
  } catch (error) {
    return { error: 'Backend server is not available. Please make sure it is running on ' + BACKEND_URL };
  }
}

// Execute tool calls
async function executeTool(name: string, args: Record<string, unknown>) {
  switch (name) {
    case 'recommend_by_mood':
      return await recommendByMood(args.mood as string, args.count as number);
    case 'analyze_text_emotion':
      return await analyzeTextEmotion(args.text as string);
    case 'get_similar_songs':
      return await getSimilarSongs(args.song_name as string, args.count as number);
    default:
      return { error: `Unknown tool: ${name}` };
  }
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
      
      // Forward to backend
      const backendFormData = new FormData();
      backendFormData.append('file', file);
      
      const result = await analyzeAudio(backendFormData);
      
      if (result.error) {
        return NextResponse.json({ 
          message: `❌ Error: ${result.error}`,
          recommendations: [] 
        });
      }
      
      const emotions = result.emotions || [];
      const topEmotion = emotions[0];
      const recommendations = result.recommendations || [];
      
      let message = `## 🎤 Audio Analysis Complete!\n\n`;
      if (topEmotion) {
        message += `I detected **${topEmotion.name}** (${(topEmotion.score * 100).toFixed(1)}% confidence) in your voice.\n\n`;
        message += `Based on this emotion, here are some songs that match your mood:`;
      } else {
        message += `I analyzed your audio and found some songs you might enjoy:`;
      }
      
      return NextResponse.json({ 
        message,
        recommendations,
        emotions,
      });
    }
    
    // Handle JSON request (text/mood)
    const body = await request.json();
    const { message, mood, history = [] } = body;
    
    // Check if the message contains a Spotify link - handle it directly
    const spotifyTrackId = extractSpotifyTrackId(message);
    if (spotifyTrackId) {
      const result = await getSimilarSongsBySpotifyId(spotifyTrackId, 5);
      
      if (result.error) {
        return NextResponse.json({ 
          message: result.notFound 
            ? `😔 Sorry, I couldn't find that song in our database. The Spotify track ID \`${spotifyTrackId}\` isn't in our music collection. Try sharing a different song or search by song name instead!`
            : `❌ Error: ${result.error}`,
          recommendations: [] 
        });
      }
      
      return NextResponse.json({
        message: `## 🎵 Found it!\n\nI found **"${result.trackName}"** by **${result.trackArtist}** from your Spotify link.\n\nHere are some similar songs based on audio features like danceability, energy, valence, and tempo:`,
        recommendations: result.recommendations || [],
        sourceSong: {
          name: result.trackName,
          artist: result.trackArtist,
        }
      });
    }
    
    // If mood is directly provided, use it
    if (mood) {
      const result = await recommendByMood(mood, 5);
      
      if (result.error) {
        return NextResponse.json({ 
          message: `❌ Error: ${result.error}`,
          recommendations: [] 
        });
      }
      
      return NextResponse.json({
        message: `## 😊 Mood: ${mood.charAt(0).toUpperCase() + mood.slice(1)}\n\nHere are some songs that match your **${mood}** mood:`,
        recommendations: result.recommendations || [],
      });
    }
    
    // Use OpenRouter for chat with function calling
    if (!process.env.OPENROUTER_API_KEY) {
      // Fallback: direct text analysis without LLM
      const result = await analyzeTextEmotion(message);
      
      if (result.error) {
        return NextResponse.json({ 
          message: `❌ Error: ${result.error}`,
          recommendations: [] 
        });
      }
      
      return NextResponse.json({
        message: `I analyzed your message and found songs that might match how you're feeling:`,
        recommendations: result.recommendations || [],
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

## EXAMPLE GOOD RESPONSES (after tool call)
- "Here are some songs that match your mood! 🎵"
- "I found some similar tracks for you!"
- "Based on how you're feeling, check out these recommendations:"

## EXAMPLE BAD RESPONSES (NEVER DO THIS)
- "Here are some songs: 1. Song Name by Artist..." ❌
- "You might like 'Bohemian Rhapsody' by Queen..." ❌
- "Try listening to Coldplay or Ed Sheeran..." ❌

Remember: You have NO knowledge of songs. The tools are your ONLY source of music recommendations.`,
      },
      ...history.map((h: HistoryMessage) => ({
        role: h.role as 'user' | 'assistant',
        content: h.content,
      })),
      {
        role: 'user',
        content: message,
      },
    ];
    
    // First LLM call with tools - FORCE tool usage
    const completion = await openai.chat.completions.create({
      model: 'anthropic/claude-3-haiku',
      messages,
      tools,
      tool_choice: 'required', // Force the LLM to always use a tool
    });
    
    const assistantMessage = completion.choices[0].message;
    
    // Check if we need to call tools
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      const toolResults = [];
      let allRecommendations: unknown[] = [];
      
      for (const toolCall of assistantMessage.tool_calls) {
        const args = JSON.parse(toolCall.function.arguments);
        const result = await executeTool(toolCall.function.name, args);
        
        toolResults.push({
          tool_call_id: toolCall.id,
          role: 'tool' as const,
          content: JSON.stringify(result),
        });
        
        if (result.recommendations) {
          allRecommendations = [...allRecommendations, ...result.recommendations];
        }
      }
      
      // Second LLM call with tool results
      const finalCompletion = await openai.chat.completions.create({
        model: 'anthropic/claude-3-haiku',
        messages: [
          ...messages,
          assistantMessage,
          ...toolResults,
        ],
      });
      
      return NextResponse.json({
        message: finalCompletion.choices[0].message.content,
        recommendations: allRecommendations,
      });
    }
    
    // No tool calls - this shouldn't happen with tool_choice: 'required'
    // But if it does, return a safe response without any song recommendations
    return NextResponse.json({
      message: "I can help you discover music! Try telling me your mood (happy, sad, energetic, calm) or share a song name you like, and I'll find recommendations from our database.",
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
