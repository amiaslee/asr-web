import React, { useState, useEffect, useRef } from 'react';
import {
  ConfigProvider,
  Layout,
  Typography,
  Switch,
  Card,
  Upload,
  Button,
  message,
  Input,
  Space,
  theme,
  Tag,
  Row,
  Col,
  Slider,
  Select,
  Segmented,
  Tabs,
  Table,
  Menu,
  Dropdown,
  Progress,
  Alert
} from 'antd';
import {
  InboxOutlined,
  MoonOutlined,
  SunOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  CloseCircleOutlined,
  PlayCircleOutlined,
  DownloadOutlined,
  CopyOutlined,
  AudioOutlined
} from '@ant-design/icons';
import './App.css';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import { useWebSocketTranscription } from './hooks/useWebSocketTranscription';
import { useMediaRecorder } from './hooks/useMediaRecorder';

import LyricsView from './components/LyricsView';

const { Header, Content, Footer } = Layout;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Dragger } = Upload;
const { TextArea } = Input;

function App() {
  // Check localStorage first, then fall back to system preference
  const getInitialTheme = () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme !== null) {
      return savedTheme === 'dark';
    }
    // Fall back to system preference
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  };

  const [isDarkMode, setIsDarkMode] = useState(getInitialTheme);
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [fileType, setFileType] = useState(null); // 'audio' or 'video'
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState({ text: 'Checking...', color: 'default' });
  const [maxTokens, setMaxTokens] = useState(2048);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [currentTime, setCurrentTime] = useState(0); // Add currentTime state

  // Refs
  const mediaRef = useRef(null);
  const scrollRef = useRef(null);

  // Time update handler for media elements
  const handleTimeUpdate = (e) => {
    setCurrentTime(e.target.currentTime);
  };

  // Multi-model states
  const [selectedModel, setSelectedModel] = useState('glm-asr');
  const [availableModels, setAvailableModels] = useState([]);
  const [language, setLanguage] = useState('auto');
  const [timestampLevel, setTimestampLevel] = useState('none');
  const [segments, setSegments] = useState([]);

  // Real-time mode state
  const [mode, setMode] = useState('file'); // 'file', 'realtime', or 'recording'

  // Real-time recording hooks - must be initialized before useEffect
  const audioRecorder = useAudioRecorder(); // For real-time WebSocket streaming
  const wsTranscription = useWebSocketTranscription();

  // Recording mode hook - for file recording (separate from real-time)
  const mediaRecorder = useMediaRecorder();

  // Pagination state
  const [segmentPageSize, setSegmentPageSize] = useState(10);

  // Progress state
  const [uploadProgress, setUploadProgress] = useState(0);

  // Ref to store recording chunks
  const recordingChunksRef = useRef([]);

  // Check backend health periodically
  useEffect(() => {
    checkHealth();
    // Poll every 10 seconds instead of 5 to reduce server load
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Apply theme class to body for CSS variables
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
    }
  }, [isDarkMode]);

  // Sync real-time transcriptions to result
  useEffect(() => {
    if (wsTranscription.transcriptions.length > 0) {
      const fullText = wsTranscription.transcriptions
        .map(t => t.text)
        .join(' ')
        .trim();
      setResult(fullText);
    }
  }, [wsTranscription.transcriptions]);

  const checkHealth = async () => {
    try {
      const res = await fetch('/health');
      const data = await res.json();
      if (data.status === 'ok') {
        const allLoaded = data.all_loaded || false;
        setBackendStatus({
          text: allLoaded ? 'System Ready' : 'Models Loading...',
          color: allLoaded ? 'success' : 'processing'
        });
      } else {
        setBackendStatus({ text: 'System Error', color: 'error' });
      }
    } catch (e) {
      setBackendStatus({ text: 'Offline', color: 'error' });
    }
  };

  // Fetch available models
  useEffect(() => {
    let mounted = true;

    const fetchModels = async () => {
      try {
        const res = await fetch('/models');
        const data = await res.json();

        if (!mounted) return false;

        setAvailableModels(data.models);

        // Only set selected model if user hasn't chosen one yet
        if (data.default && !selectedModel) {
          setSelectedModel(data.default);
        }

        // Return true if all models are loaded
        const allLoaded = !data.models.some(m => !m.loaded);
        return allLoaded;
      } catch (e) {
        console.error('Error fetching models:', e);
        return false;
      }
    };

    fetchModels();

    // Poll for model status updates every 10 seconds (slower)
    // Stop polling once all models are loaded
    const pollInterval = setInterval(async () => {
      const allLoaded = await fetchModels();
      if (allLoaded) {
        console.log('All models loaded, stopping poll');
        clearInterval(pollInterval);
      }
    }, 10000);

    return () => {
      mounted = false;
      clearInterval(pollInterval);
    };
  }, []); // Note: selectedModel is NOT in dependencies to avoid resetting

  // Theme toggle handler with localStorage
  const toggleTheme = () => {
    setIsDarkMode(prev => {
      const newTheme = !prev;
      localStorage.setItem('theme', newTheme ? 'dark' : 'light');
      return newTheme;
    });
  };

  const handleTranscribe = async (fileOverride = null) => {
    const fileToUse = fileOverride instanceof File ? fileOverride : file;

    if (!fileToUse) {
      console.warn("No file to transcribe");
      return;
    }

    setLoading(true);
    setUploadProgress(0);
    setResult('');
    setSegments([]);

    const formData = new FormData();
    formData.append('file', fileToUse);

    // Build URL with params
    let url = `/transcribe-stream?model=${selectedModel}`;
    if (language && language !== 'auto') {
      url += `&language=${language}`;
    }
    if (timestampLevel && timestampLevel !== 'none') {
      url += `&timestamp_level=${timestampLevel}`;
    }
    if (selectedModel === 'glm-asr') {
      url += `&max_tokens=${maxTokens}`;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Read Server-Sent Events stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split('\n');
        // Keep the last line in the buffer as it might be incomplete
        buffer = lines.pop();

        for (const line of lines) {
          if (line.trim().startsWith('data: ')) {
            try {
              const jsonStr = line.trim().slice(6);
              if (jsonStr === '[DONE]') continue; // Handle potential end stream marker if used

              const data = JSON.parse(jsonStr);

              if (data.error) {
                throw new Error(data.error);
              }

              if (data.progress !== undefined) {
                setUploadProgress(data.progress);
              }

              if (data.result) {
                // Final result received
                setResult(data.result.text || '');
                setSegments(data.result.segments || []);
                setUploadProgress(100);

                // Show success message
                message.success('Transcription completed successfully!');
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      console.error('Transcription error:', error);
      message.error(`Transcription failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Export functions
  const generateSRT = () => {
    if (!segments || segments.length === 0) return '';

    return segments.map((seg, idx) => {
      const start = formatSRTTime(seg.start);
      const end = formatSRTTime(seg.end);
      return `${idx + 1}\n${start} --> ${end}\n${seg.text}\n`;
    }).join('\n');
  };

  const generateVTT = () => {
    if (!segments || segments.length === 0) return '';

    const content = segments.map((seg) => {
      const start = formatVTTTime(seg.start);
      const end = formatVTTTime(seg.end);
      return `${start} --> ${end}\n${seg.text}`;
    }).join('\n\n');

    return `WEBVTT\n\n${content}`;
  };

  const generateLRC = () => {
    if (!segments || segments.length === 0) return '';

    return segments.map((seg) => {
      const time = formatLRCTime(seg.start);
      return `[${time}]${seg.text}`;
    }).join('\n');
  };

  const formatSRTTime = (seconds) => {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    const ms = Math.floor((seconds % 1) * 1000).toString().padStart(3, '0');
    return `${h}:${m}:${s},${ms}`;
  };

  const formatVTTTime = (seconds) => {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    const ms = Math.floor((seconds % 1) * 1000).toString().padStart(3, '0');
    return `${h}:${m}:${s}.${ms}`;
  };

  const formatLRCTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    const ms = Math.floor((seconds % 1) * 1000).toString().padStart(3, '0');
    return `${m}:${s}.${ms}`;
  };

  const handleExport = (format) => {
    let content = '';
    let filename = '';
    let mimeType = '';

    switch (format) {
      case 'txt':
        content = result;
        filename = 'transcription.txt';
        mimeType = 'text/plain';
        break;
      case 'srt':
        content = generateSRT();
        filename = 'transcription.srt';
        mimeType = 'text/srt';
        break;
      case 'vtt':
        content = generateVTT();
        filename = 'transcription.vtt';
        mimeType = 'text/vtt';
        break;
      case 'lrc':
        content = generateLRC();
        filename = 'transcription.lrc';
        mimeType = 'text/plain';
        break;
      case 'json':
        content = JSON.stringify({ text: result, segments, metadata: { exported_at: new Date().toISOString() } }, null, 2);
        filename = 'transcription.json';
        mimeType = 'application/json';
        break;
      default:
        return;
    }

    if (!content) {
      message.warning('No content to export');
      return;
    }

    // Create blob and download
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    message.success(`Exported as ${filename}`);
  };

  // Real-time recording handlers
  const handleStartRealtime = async () => {
    try {
      // Connect WebSocket first
      wsTranscription.connect(selectedModel);

      // Start recording and send audio chunks
      await audioRecorder.startRecording((audioBlob) => {
        wsTranscription.sendAudio(audioBlob);
      });
    } catch (err) {
      message.error(`Failed to start recording: ${err.message}`);
    }
  };

  const handleStopRealtime = () => {
    audioRecorder.stopRecording();
    wsTranscription.disconnect();
  };

  const handleClearResults = () => {
    setResult('');
    setSegments([]);
    wsTranscription.clearTranscriptions();
  };

  // Non-realtime recording handlers
  const handleStartRecording = async () => {
    try {
      await mediaRecorder.startRecording();
    } catch (err) {
      message.error(`Failed to start recording: ${err.message}`);
    }
  };

  const handleStopRecording = async () => {
    try {
      const blob = await mediaRecorder.stopRecording();

      if (blob && blob.size > 0) {
        setRecordedBlob(blob);

        // Create URL for playback
        const url = URL.createObjectURL(blob);
        setRecordedUrl(url);

        message.success(`Recording saved! (${(blob.size / 1024).toFixed(1)} KB)`);
      } else {
        message.warning('No audio data recorded');
      }
    } catch (err) {
      message.error(`Failed to stop recording: ${err.message}`);
    }
  };

  const handleDownloadRecording = () => {
    if (recordedBlob) {
      const url = URL.createObjectURL(recordedBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `recording-${Date.now()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      message.success('Recording downloaded');
    }
  };

  const handleTranscribeRecording = async () => {
    if (!recordedBlob) {
      message.warning('No recording available to transcribe');
      return;
    }

    try {
      setLoading(true);
      setUploadProgress(0);
      setResult('');
      setSegments([]);

      // Create a File object from the recorded blob
      const file = new File([recordedBlob], 'recording.webm', { type: recordedBlob.type });

      // Use the refactored handleTranscribe function
      await handleTranscribe(file);

      // Success message is shown in handleTranscribe after transcription completes
    } catch (error) {
      console.error('Transcription error:', error);
      message.error(`Transcription failed: ${error.message}`);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: 'audio/*,video/*',
    beforeUpload: (file) => {
      // Determine if it's audio or video
      const isAudio = file.type.startsWith('audio/');
      const isVideo = file.type.startsWith('video/');

      if (!isAudio && !isVideo) {
        message.error('Please upload an audio or video file');
        return false;
      }

      setFile(file);
      setFileType(isVideo ? 'video' : 'audio');
      setResult('');

      // Create URL for playback
      const url = URL.createObjectURL(file);
      setFileUrl(url);

      return false; // Prevent auto upload
    },
    onRemove: () => {
      setFile(null);
      setFileType(null);
      setResult('');
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
        setFileUrl(null);
      }
    },
    fileList: file ? [file] : [],
  };

  return (
    <ConfigProvider
      theme={{
        algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
          colorPrimary: isDarkMode ? '#1e1e1e' : '#1890ff',
          borderRadius: 8,
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          background: isDarkMode ? '#141414' : '#fff',
          borderBottom: `1px solid ${isDarkMode ? '#303030' : '#f0f0f0'}`
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <AudioOutlined style={{ fontSize: 24, color: isDarkMode ? '#fff' : '#000' }} />
            <Title level={4} style={{ margin: 0 }}>GLM-ASR Studio</Title>
          </div>
          <Space>
            <Tag icon={
              backendStatus.color === 'success' ? <CheckCircleOutlined /> :
                backendStatus.color === 'processing' ? <SyncOutlined spin /> :
                  <CloseCircleOutlined />
            } color={backendStatus.color}>
              {backendStatus.text}
            </Tag>
            <Switch
              checkedChildren={<MoonOutlined />}
              unCheckedChildren={<SunOutlined />}
              checked={isDarkMode}
              onChange={toggleTheme}
            />
          </Space>
        </Header>

        <Content style={{ padding: '24px' }}>
          <div style={{ maxWidth: 1400, margin: '0 auto' }}>
            <Row gutter={[24, 24]}>
              {/* Left Column: Upload & Player */}
              <Col xs={24} lg={12}>
                <Space direction="vertical" size="large" style={{ width: '100%' }}>
                  {/* Model Configuration Card */}
                  <Card bordered={false} style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                    <Title level={4} style={{ marginBottom: 16 }}>Model Configuration</Title>

                    {/* Model Selector */}
                    <div style={{ marginBottom: 16 }}>
                      <Text strong style={{ display: 'block', marginBottom: 8 }}>ASR Model</Text>
                      <Select
                        value={selectedModel}
                        onChange={setSelectedModel}
                        style={{ width: '100%' }}
                        size="large"
                      >
                        {availableModels.map(model => (
                          <Select.Option key={model.name} value={model.name}>
                            <Space>
                              <span>
                                {model.name === 'glm-asr' ? '🎯 GLM-ASR Nano' : '⚡ Whisper v3 Turbo'}
                              </span>
                              {!model.loaded && <Tag color="orange">Not loaded</Tag>}
                            </Space>
                          </Select.Option>
                        ))}
                      </Select>
                    </div>

                    {/* Language Selector (Whisper only) */}
                    {selectedModel === 'whisper-turbo' && (
                      <div style={{ marginBottom: 16 }}>
                        <Text strong style={{ display: 'block', marginBottom: 8 }}>Language</Text>
                        <Select
                          value={language}
                          onChange={setLanguage}
                          style={{ width: '100%' }}
                          size="large"
                        >
                          <Select.Option value="auto">🌐 Auto Detect</Select.Option>
                          <Select.Option value="zh">🇨🇳 中文</Select.Option>
                          <Select.Option value="en">🇺🇸 English</Select.Option>
                          <Select.Option value="ja">🇯🇵 日本語</Select.Option>
                          <Select.Option value="ko">🇰🇷 한국어</Select.Option>
                          <Select.Option value="es">🇪🇸 Español</Select.Option>
                          <Select.Option value="fr">🇫🇷 Français</Select.Option>
                        </Select>
                      </div>
                    )}

                    {/* Timestamp Level */}
                    <div style={{ marginBottom: 16 }}>
                      <Text strong style={{ display: 'block', marginBottom: 8 }}>Timestamps</Text>
                      <Select
                        value={timestampLevel}
                        onChange={setTimestampLevel}
                        style={{ width: '100%' }}
                        size="large"
                        disabled={selectedModel === 'glm-asr'}
                      >
                        <Select.Option value="none">None</Select.Option>
                        <Select.Option value="sentence">Sentence Level</Select.Option>
                        <Select.Option value="word">Word Level</Select.Option>
                      </Select>
                      {selectedModel === 'glm-asr' && (
                        <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                          Timestamps not supported by GLM-ASR
                        </Text>
                      )}
                    </div>

                    {/* Max Tokens (GLM-ASR only) */}
                    {selectedModel === 'glm-asr' && (
                      <div>
                        <Text strong>Max Tokens: {maxTokens}</Text>
                        <Slider
                          min={100}
                          max={3000}
                          step={100}
                          value={maxTokens}
                          onChange={setMaxTokens}
                          tooltip={{ formatter: (value) => `${value} tokens` }}
                          style={{
                            // Force visible handle in dark mode
                          }}
                          handleStyle={{
                            borderColor: isDarkMode ? '#1890ff' : '#1890ff',
                            backgroundColor: '#fff',
                            borderWidth: 2
                          }}
                          trackStyle={{
                            backgroundColor: isDarkMode ? '#1890ff' : '#1890ff'
                          }}
                          railStyle={{
                            backgroundColor: isDarkMode ? '#434343' : '#d9d9d9'
                          }}
                        />
                      </div>
                    )}
                  </Card>

                  {/* Mode Switcher */}
                  <Card bordered={false} style={{ marginBottom: 0, boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                    <Segmented
                      size="large"
                      value={mode}
                      onChange={setMode}
                      block
                      options={[
                        {
                          label: (
                            <div style={{ padding: 4 }}>
                              <InboxOutlined />
                              <div>File Upload</div>
                            </div>
                          ),
                          value: 'file'
                        },
                        {
                          label: (
                            <div style={{ padding: 4 }}>
                              <AudioOutlined />
                              <div>Recording</div>
                            </div>
                          ),
                          value: 'recording'
                        },
                        {
                          label: (
                            <div style={{ padding: 4 }}>
                              <PlayCircleOutlined />
                              <div>Real-time</div>
                            </div>
                          ),
                          value: 'realtime'
                        }
                      ]}
                    />
                  </Card>

                  {mode === 'file' ? (
                    <Card bordered={false} style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                      <Title level={2} style={{ textAlign: 'center', fontWeight: 300, marginBottom: 8 }}>
                        Speech to Text
                      </Title>
                      <Paragraph style={{ textAlign: 'center', color: 'gray', marginBottom: 24 }}>
                        Upload audio or video files (MP3, MP4, WAV, AAC, etc.)
                      </Paragraph>

                      <div>
                        <Dragger {...uploadProps} style={{ padding: 40, background: isDarkMode ? '#1f1f1f' : '#fafafa' }}>
                          <p className="ant-upload-drag-icon">
                            <InboxOutlined style={{ color: isDarkMode ? '#444' : '#ccc' }} />
                          </p>
                          <p className="ant-upload-text">Click or drag media file to this area</p>
                          <p className="ant-upload-hint">
                            Support for audio and video formats
                          </p>
                        </Dragger>
                      </div>

                      {/* Media Player */}
                      {fileUrl && (
                        <div style={{ marginTop: 24 }}>
                          <Title level={5} style={{ marginBottom: 12 }}>
                            <PlayCircleOutlined /> Preview
                          </Title>
                          {fileType === 'video' ? (
                            <video
                              ref={mediaRef}
                              src={fileUrl}
                              controls
                              onTimeUpdate={handleTimeUpdate} // Add listener
                              style={{ width: '100%', borderRadius: 8, maxHeight: 400, background: '#000' }}
                            />
                          ) : (
                            <audio
                              ref={mediaRef}
                              src={fileUrl}
                              controls
                              onTimeUpdate={handleTimeUpdate} // Add listener
                              style={{ width: '100%' }}
                            />
                          )}
                        </div>
                      )}

                      <div style={{ marginTop: 24, textAlign: 'center' }}>
                        <Button
                          type="primary"
                          size="large"
                          onClick={handleTranscribe}
                          loading={loading}
                          disabled={!file}
                          style={{
                            minWidth: 200,
                            background: isDarkMode ? '#333' : '#000',
                            width: '100%',
                            color: 'var(--ant-button-primary-color)'
                          }}
                        >
                          {loading ? 'Transcribing...' : 'Start Transcription'}
                        </Button>

                        {/* Progress indicator */}
                        {loading && (
                          <div style={{ marginTop: 16 }}>
                            <Progress
                              percent={Math.round(uploadProgress)}
                              status={uploadProgress === 100 ? 'success' : 'active'}
                              strokeColor={{
                                from: '#108ee9',
                                to: '#87d068',
                              }}
                            />
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {uploadProgress < 50 ? 'Uploading...' : uploadProgress < 95 ? 'Processing...' : 'Finalizing...'}
                            </Text>
                          </div>
                        )}
                      </div>
                    </Card>
                  ) : mode === 'recording' ? (
                    /* Recording Mode Card */
                    <Card
                      title="Audio Recording"
                      bordered={false}
                      style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}
                    >
                      <div style={{ textAlign: 'center', marginBottom: 24 }}>
                        {mediaRecorder.isRecording ? (
                          <div>
                            <div style={{ fontSize: 48, marginBottom: 16, color: '#ff4d4f' }}>
                              <AudioOutlined className="pulse-icon" />
                            </div>
                            <Title level={4}>Recording...</Title>
                            <Text type="secondary">Duration: {Math.floor(mediaRecorder.duration / 60)}:{(mediaRecorder.duration % 60).toString().padStart(2, '0')}</Text>
                          </div>
                        ) : (
                          <div>
                            <div style={{ fontSize: 48, marginBottom: 16, color: '#999' }}>
                              <AudioOutlined />
                            </div>
                            <Paragraph type="secondary">
                              {recordedBlob ? 'Recording saved! Play, download or transcribe below.' : 'Click Start to record audio'}
                            </Paragraph>
                          </div>
                        )}
                      </div>

                      {/* Recording Controls */}
                      <div style={{ textAlign: 'center', marginBottom: 24 }}>
                        {!mediaRecorder.isRecording ? (
                          <Space>
                            <Button
                              type="primary"
                              size="large"
                              icon={<PlayCircleOutlined />}
                              onClick={handleStartRecording}
                              disabled={recordedBlob !== null}
                            >
                              Start Recording
                            </Button>
                            {recordedBlob && (
                              <Button
                                onClick={() => {
                                  setRecordedBlob(null);
                                  setRecordedUrl(null);
                                }}
                              >
                                New Recording
                              </Button>
                            )}
                          </Space>
                        ) : (
                          <Button
                            danger
                            size="large"
                            onClick={handleStopRecording}
                          >
                            Stop Recording
                          </Button>
                        )}
                      </div>

                      {/* Playback and Actions */}
                      {recordedUrl && (
                        <div>
                          <div style={{ marginBottom: 16 }}>
                            <Text strong>Playback:</Text>
                            <audio
                              controls
                              src={recordedUrl}
                              onTimeUpdate={handleTimeUpdate} // Add listener
                              style={{ width: '100%', marginTop: 8 }}
                            />
                          </div>

                          <Space direction="vertical" style={{ width: '100%' }} size="large">
                            <Button
                              block
                              icon={<DownloadOutlined />}
                              onClick={handleDownloadRecording}
                            >
                              Download Recording
                            </Button>
                            <Button
                              block
                              type="primary"
                              loading={loading}
                              onClick={handleTranscribeRecording}
                              style={{ color: '#fff' }}
                            >
                              {loading ? 'Transcribing...' : 'Transcribe Recording'}
                            </Button>
                          </Space>

                          {/* Progress Bar for Recording Mode - Moved outside buttons */}
                          {(loading || uploadProgress > 0) && (
                            <div style={{ marginTop: 24, marginBottom: 24 }}>
                              <Progress
                                percent={uploadProgress}
                                status={loading ? 'active' : 'success'}
                                strokeColor={{
                                  '0%': '#108ee9',
                                  '100%': '#87d068',
                                }}
                              />
                              <div style={{ textAlign: 'center', marginTop: 8, color: isDarkMode ? '#aaa' : '#666' }}>
                                {uploadProgress < 100 ? 'Transcribing...' : 'Transcription Complete'}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Error Display */}
                      {mediaRecorder.error && (
                        <Alert
                          type="error"
                          message={mediaRecorder.error}
                          closable
                          style={{ marginTop: 16 }}
                        />
                      )}
                    </Card>
                  ) : (
                    /* Real-time Recording Card */
                    <Card
                      title="Real-time Recording"
                      bordered={false}
                      style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}
                    >
                      <div style={{ textAlign: 'center', marginBottom: 24 }}>
                        {/* Recording Status */}
                        {audioRecorder.isRecording ? (
                          <div>
                            <div style={{ fontSize: 48, marginBottom: 16, color: '#ff4d4f' }}>
                              <AudioOutlined className="pulse-icon" />
                            </div>
                            <Title level={4}>Recording...</Title>
                            <Text type="secondary">Duration: {Math.floor(audioRecorder.duration / 60)}:{(audioRecorder.duration % 60).toString().padStart(2, '0')}</Text>
                            <div style={{ marginTop: 16 }}>
                              <Tag color="processing">Connected to {selectedModel}</Tag>
                            </div>
                          </div>
                        ) : (
                          <div>
                            <div style={{ fontSize: 48, marginBottom: 16, color: '#999' }}>
                              <AudioOutlined />
                            </div>
                            <Paragraph type="secondary">
                              Click Start to begin recording from your microphone
                            </Paragraph>
                          </div>
                        )}
                      </div>

                      {/* Recording Controls */}
                      <div style={{ textAlign: 'center', marginBottom: 24 }}>
                        {!audioRecorder.isRecording ? (
                          <Button
                            type="primary"
                            size="large"
                            icon={<PlayCircleOutlined />}
                            onClick={handleStartRealtime}
                            disabled={!wsTranscription.isConnected && audioRecorder.isRecording}
                          >
                            Start Recording
                          </Button>
                        ) : (
                          <Button
                            danger
                            size="large"
                            onClick={handleStopRealtime}
                          >
                            Stop Recording
                          </Button>
                        )}
                      </div>

                      {/* Error Display */}
                      {(audioRecorder.error || wsTranscription.error) && (
                        <Alert
                          type="error"
                          message={audioRecorder.error || wsTranscription.error}
                          closable
                          style={{ marginBottom: 16 }}
                        />
                      )}
                    </Card>
                  )}
                </Space>
              </Col>

              {/* Right Column: Results */}
              <Col xs={24} lg={12}>
                <Card
                  title="Transcription Result"
                  bordered={false}
                  style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.05)', minHeight: 500 }}
                  extra={
                    result && (
                      <Space>
                        <Button
                          icon={<CopyOutlined />}
                          onClick={() => {
                            navigator.clipboard.writeText(result);
                            message.success('Copied to clipboard');
                          }}
                        >
                          Copy
                        </Button>
                        <Button
                          danger
                          icon={<CloseCircleOutlined />}
                          onClick={handleClearResults}
                        >
                          Clear
                        </Button>
                        <Dropdown
                          menu={{
                            items: [
                              {
                                key: 'txt',
                                label: 'Export as TXT',
                                icon: <DownloadOutlined />,
                                onClick: () => handleExport('txt')
                              },
                              {
                                key: 'srt',
                                label: 'Export as SRT',
                                icon: <DownloadOutlined />,
                                onClick: () => handleExport('srt'),
                                disabled: !segments || segments.length === 0
                              },
                              {
                                key: 'vtt',
                                label: 'Export as VTT',
                                icon: <DownloadOutlined />,
                                onClick: () => handleExport('vtt'),
                                disabled: !segments || segments.length === 0
                              },
                              {
                                key: 'lrc',
                                label: 'Export as LRC',
                                icon: <DownloadOutlined />,
                                onClick: () => handleExport('lrc'),
                                disabled: !segments || segments.length === 0
                              },
                              {
                                key: 'json',
                                label: 'Export as JSON',
                                icon: <DownloadOutlined />,
                                onClick: () => handleExport('json')
                              }
                            ]
                          }}
                          placement="bottomRight"
                        >
                          <Button type="primary" icon={<DownloadOutlined />}>
                            Export
                          </Button>
                        </Dropdown>
                      </Space>
                    )
                  }
                >
                  {result || (mode === 'realtime' && wsTranscription.transcriptions.length > 0) ? (
                    <Tabs
                      defaultActiveKey="1"
                      items={[
                        {
                          key: '1',
                          label: 'Text',
                          children: (
                            <TextArea
                              value={result || (mode === 'realtime' && wsTranscription.transcriptions.map(t => t.text).join(' ').trim())}
                              readOnly
                              autoSize={{ minRows: 10, maxRows: 30 }}
                              style={{
                                fontSize: 16,
                                lineHeight: 1.8,
                                background: 'transparent',
                                border: 'none',
                                resize: 'none',
                                color: isDarkMode ? '#ddd' : '#333'
                              }}
                            />
                          )
                        },
                        {
                          key: '2',
                          label: `Segments (${segments.length})`,
                          disabled: segments.length === 0,
                          children: (
                            <Table
                              dataSource={segments.map((s, i) => ({ ...s, key: i }))}
                              rowKey="key"
                              pagination={{
                                pageSize: segmentPageSize,
                                showSizeChanger: true,
                                pageSizeOptions: ['5', '10', '20', '50', '100'],
                                showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items`,
                                onChange: (page, pageSize) => {
                                  setSegmentPageSize(pageSize);
                                }
                              }}
                              size="small"
                              expandable={{
                                expandedRowRender: (record) => {
                                  if (!record.words || record.words.length === 0) {
                                    return <p style={{ margin: 0, color: '#999' }}>No word-level timestamps available</p>;
                                  }
                                  return (
                                    <Table
                                      dataSource={record.words}
                                      rowKey={(word, idx) => `${record.id}-${idx}`}
                                      pagination={false}
                                      size="small"
                                      columns={[
                                        {
                                          title: 'Word',
                                          dataIndex: 'word',
                                          key: 'word',
                                        },
                                        {
                                          title: 'Start',
                                          dataIndex: 'start',
                                          key: 'start',
                                          width: 80,
                                          render: (time) => time ? `${time.toFixed(2)}s` : '-'
                                        },
                                        {
                                          title: 'End',
                                          dataIndex: 'end',
                                          key: 'end',
                                          width: 80,
                                          render: (time) => time ? `${time.toFixed(2)}s` : '-'
                                        }
                                      ]}
                                    />
                                  );
                                },
                                rowExpandable: (record) => record.words && record.words.length > 0,
                              }}
                              columns={[
                                {
                                  title: 'Start',
                                  dataIndex: 'start',
                                  key: 'start',
                                  width: 80,
                                  render: (time) => time ? `${time.toFixed(1)}s` : '-'
                                },
                                {
                                  title: 'End',
                                  dataIndex: 'end',
                                  key: 'end',
                                  width: 80,
                                  render: (time) => time ? `${time.toFixed(1)}s` : '-'
                                },
                                {
                                  title: 'Text',
                                  dataIndex: 'text',
                                  key: 'text',
                                  ellipsis: false, // Don't truncate
                                  render: (text) => (
                                    <div style={{
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      maxWidth: '100%'
                                    }}>
                                      {text}
                                    </div>
                                  )
                                }
                              ]}
                            />
                          )
                        },
                        {
                          key: '3',
                          label: 'Lyrics',
                          disabled: segments.length === 0,
                          children: (
                            <LyricsView
                              segments={segments}
                              currentTime={currentTime}
                            />
                          )
                        }
                      ]}
                    />
                  ) : (

                    <div style={{
                      textAlign: 'center',
                      padding: '100px 20px',
                      color: isDarkMode ? '#666' : '#999'
                    }}>
                      <AudioOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                      <Paragraph>Upload a file and click "Start Transcription" to see results here</Paragraph>
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
          </div>
        </Content>

        <Footer style={{ textAlign: 'center', background: 'transparent' }}>
          GLM-ASR Web Interface ©{new Date().getFullYear()}
        </Footer>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
