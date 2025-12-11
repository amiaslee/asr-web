import React, { useEffect, useState, useMemo, useRef } from 'react';
import { LyricPlayer as LyricPlayerReact } from '@applemusic-like-lyrics/react';
import { LyricPlayer as LyricPlayerCore } from '@applemusic-like-lyrics/core';
import '@applemusic-like-lyrics/core/style.css';

/**
 * LyricsView component
 */
const LyricsView = ({ segments, currentTime }) => {
    const [parsedLines, setParsedLines] = useState([]);
    const playerRef = useRef(null);

    // Directly construct LyricLine objects from segments
    // This avoids parsing issues and ensures accurate timestamp mapping
    const lines = useMemo(() => {
        if (!segments || segments.length === 0) return [];

        return segments.map(seg => {
            const lineStartTime = Math.floor(seg.start * 1000);
            const lineEndTime = Math.floor(seg.end * 1000);

            let lyricWords = [];

            // If we have word-level timestamps, construct LyricWords
            if (seg.words && seg.words.length > 0) {
                lyricWords = seg.words.map(w => ({
                    startTime: Math.floor(w.start * 1000),
                    endTime: Math.floor(w.end * 1000),
                    word: w.word,
                    romanWord: ''
                }));
            } else {
                // Fallback: Treat the whole line as one "word"
                lyricWords = [{
                    startTime: lineStartTime,
                    endTime: lineEndTime,
                    word: seg.text,
                    romanWord: ''
                }];
            }

            return {
                words: lyricWords,
                translatedLyric: '',
                romanLyric: '',
                isBG: false,
                isDuet: false,
                startTime: lineStartTime,
                endTime: lineEndTime
            };
        });
    }, [segments]);

    useEffect(() => {
        setParsedLines(lines);
    }, [lines]);

    return (
        <div style={{ height: '500px', width: '100%', position: 'relative', background: '#222', borderRadius: '12px', overflow: 'hidden' }}>
            {parsedLines.length > 0 ? (
                <LyricPlayerReact
                    ref={playerRef}
                    lyricLines={parsedLines}
                    currentTime={Math.floor((currentTime || 0) * 1000)}
                    playing={true}
                    // Pass dimensions via STYLE, not attributes (wrapper is a DIV)
                    style={{ width: '100%', height: '100%' }}
                    align="center"
                    isSeeking={false}
                    enableSpring={true}
                    enableBlur={true}
                    lyricPlayer={LyricPlayerCore} // Explicitly pass core class (Required!)
                />
            ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#fff' }}>
                    {segments?.length > 0 ? "Loading lyrics..." : "No lyrics available"}
                </div>
            )}
        </div>
    );
};

export default LyricsView;
