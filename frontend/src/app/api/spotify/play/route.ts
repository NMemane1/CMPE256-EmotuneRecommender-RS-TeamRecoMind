import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface PlayRequest {
  trackId?: string;
  trackName?: string;
  trackArtist?: string;
  action?: 'play' | 'pause' | 'next' | 'previous' | 'search';
}

// AppleScript to control Spotify
const appleScripts = {
  // Play a specific track by Spotify URI
  playTrack: (trackId: string) => `
    tell application "Spotify"
      activate
      play track "spotify:track:${trackId}"
    end tell
  `,
  
  // Search and play by name
  searchAndPlay: (query: string) => `
    tell application "Spotify"
      activate
      delay 0.5
    end tell
    
    tell application "System Events"
      tell process "Spotify"
        -- Open search with Cmd+L then type query
        keystroke "l" using command down
        delay 0.3
        keystroke "${query.replace(/"/g, '\\"')}"
        delay 0.5
        keystroke return
      end tell
    end tell
  `,
  
  // Play/pause toggle
  playPause: () => `
    tell application "Spotify"
      playpause
    end tell
  `,
  
  // Next track
  nextTrack: () => `
    tell application "Spotify"
      next track
    end tell
  `,
  
  // Previous track
  previousTrack: () => `
    tell application "Spotify"
      previous track
    end tell
  `,
  
  // Get current track info
  getCurrentTrack: () => `
    tell application "Spotify"
      if player state is playing then
        set trackName to name of current track
        set artistName to artist of current track
        return trackName & " - " & artistName
      else
        return "Not playing"
      end if
    end tell
  `,
  
  // Check if Spotify is running
  isRunning: () => `
    tell application "System Events"
      return (name of processes) contains "Spotify"
    end tell
  `,
  
  // Open Spotify
  openSpotify: () => `
    tell application "Spotify"
      activate
    end tell
  `,
};

async function runAppleScript(script: string): Promise<string> {
  try {
    const { stdout, stderr } = await execAsync(`osascript -e '${script.replace(/'/g, "'\"'\"'")}'`);
    if (stderr) {
      console.error('AppleScript stderr:', stderr);
    }
    return stdout.trim();
  } catch (error) {
    console.error('AppleScript error:', error);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: PlayRequest = await request.json();
    const { trackId, trackName, trackArtist, action = 'play' } = body;

    // Check if running on macOS
    if (process.platform !== 'darwin') {
      return NextResponse.json(
        { error: 'Spotify AppleScript integration is only available on macOS' },
        { status: 400 }
      );
    }

    let result: string;

    switch (action) {
      case 'play':
        if (trackId) {
          // Play by track ID (most reliable)
          await runAppleScript(appleScripts.openSpotify());
          await new Promise(resolve => setTimeout(resolve, 500));
          result = await runAppleScript(appleScripts.playTrack(trackId));
        } else if (trackName) {
          // Search and play by name
          const query = trackArtist ? `${trackName} ${trackArtist}` : trackName;
          result = await runAppleScript(appleScripts.searchAndPlay(query));
        } else {
          // Just play/pause
          result = await runAppleScript(appleScripts.playPause());
        }
        break;
        
      case 'pause':
        result = await runAppleScript(appleScripts.playPause());
        break;
        
      case 'next':
        result = await runAppleScript(appleScripts.nextTrack());
        break;
        
      case 'previous':
        result = await runAppleScript(appleScripts.previousTrack());
        break;
        
      case 'search':
        if (trackName) {
          const query = trackArtist ? `${trackName} ${trackArtist}` : trackName;
          result = await runAppleScript(appleScripts.searchAndPlay(query));
        } else {
          return NextResponse.json({ error: 'Track name required for search' }, { status: 400 });
        }
        break;
        
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }

    return NextResponse.json({ 
      success: true, 
      message: `Spotify: ${action} command sent`,
      result 
    });

  } catch (error) {
    console.error('Spotify control error:', error);
    return NextResponse.json(
      { 
        error: 'Failed to control Spotify. Make sure Spotify is installed.',
        details: String(error)
      },
      { status: 500 }
    );
  }
}

// GET endpoint to check Spotify status
export async function GET() {
  try {
    if (process.platform !== 'darwin') {
      return NextResponse.json({ 
        available: false, 
        reason: 'macOS only' 
      });
    }

    const isRunning = await runAppleScript(appleScripts.isRunning());
    const running = isRunning.toLowerCase() === 'true';
    
    let currentTrack = null;
    if (running) {
      try {
        currentTrack = await runAppleScript(appleScripts.getCurrentTrack());
      } catch {
        // Spotify might be running but not playing
      }
    }

    return NextResponse.json({ 
      available: true,
      running,
      currentTrack
    });

  } catch (error) {
    return NextResponse.json({ 
      available: false, 
      reason: String(error)
    });
  }
}
