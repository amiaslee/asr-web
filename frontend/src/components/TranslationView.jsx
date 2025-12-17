import React, { useState, useEffect } from 'react';
import { Card, Button, Modal, Select, Input, Form, message, Space, Typography, Dropdown, Menu, Tabs, Table } from 'antd';
import { TranslationOutlined, SettingOutlined, DownloadOutlined, CopyOutlined, PlayCircleOutlined } from '@ant-design/icons';
import LyricsView from './LyricsView';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

const TranslationView = ({ originalText, segments, isDarkMode, currentTime = 0 }) => {
  const [isSettingsVisible, setIsSettingsVisible] = useState(false);
  const [translatedText, setTranslatedText] = useState('');
  const [translatedSegments, setTranslatedSegments] = useState([]);
  const [isTranslating, setIsTranslating] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState('English');
  const [customLanguage, setCustomLanguage] = useState('');
  const [segmentPageSize, setSegmentPageSize] = useState(10);

  const [settings, setSettings] = useState({
    provider: 'zhipu', // 'zhipu' or 'deepseek'
    zhipuKey: '',
    deepseekKey: '',
    model: '',
    apiUrl: ''
  });

  // Load settings from localStorage
  useEffect(() => {
    const savedSettings = localStorage.getItem('ai_translation_settings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  const handleSaveSettings = (values) => {
    const newSettings = { ...settings, ...values };
    setSettings(newSettings);
    localStorage.setItem('ai_translation_settings', JSON.stringify(newSettings));
    setIsSettingsVisible(false);
    message.success('Settings saved');
  };

  const getProviderConfig = () => {
    if (settings.provider === 'zhipu') {
      return {
        url: settings.apiUrl || 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
        apiKey: settings.zhipuKey,
        defaultModel: 'glm-4'
      };
    } else {
      return {
        url: settings.apiUrl || 'https://api.deepseek.com/v1/chat/completions',
        apiKey: settings.deepseekKey,
        defaultModel: 'deepseek-chat'
      };
    }
  };

  const handleTranslate = async () => {
    if (!originalText && (!segments || segments.length === 0)) {
      message.warning('No text to translate');
      return;
    }

    const config = getProviderConfig();
    if (!config.apiKey) {
      message.error('Please configure API Key in settings');
      setIsSettingsVisible(true);
      return;
    }

    setIsTranslating(true);
    setTranslatedSegments([]);
    setTranslatedText('');

    try {
      const targetLang = customLanguage || targetLanguage;

      // Determine content to translate
      // We prioritize segments for better structure
      const hasSegments = segments && segments.length > 0;

      // If we have segments, we want to translate them while preserving timestamps
      // To save tokens/complexity, we might need to batch them, but for this MVP let's try sending them all
      // or a simplified version.
      // NOTE: Sending too many segments might hit context limits.
      // For now, we assume reasonable length or that the model can handle it.

      const contentToTranslate = hasSegments
        ? JSON.stringify(segments.map(s => ({ start: s.start, end: s.end, text: s.text })))
        : originalText;

      const systemPrompt = hasSegments
        ? `You are a professional translator. Translate the text content of the following JSON segments into ${targetLang}.
           IMPORTANT:
           1. Return EXACTLY the same JSON structure: a list of objects with keys "start", "end", "text".
           2. Do NOT change the "start" or "end" values.
           3. Only translate the "text" value.
           4. Output ONLY the valid JSON, no markdown, no explanations.`
        : `You are a professional translator. Translate the following text into ${targetLang}. Return only the translated text.`;

      const response = await fetch(config.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.apiKey}`
        },
        body: JSON.stringify({
          model: settings.model || config.defaultModel,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: contentToTranslate }
          ],
          stream: false
        })
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error.message || 'API Error');
      }

      const result = data.choices?.[0]?.message?.content || '';

      if (hasSegments) {
        try {
           // Clean up markdown code blocks if present
           const cleanResult = result.replace(/```json/g, '').replace(/```/g, '').trim();
           // Find the first [ and last ]
           const firstBracket = cleanResult.indexOf('[');
           const lastBracket = cleanResult.lastIndexOf(']');

           if (firstBracket !== -1 && lastBracket !== -1) {
             const jsonStr = cleanResult.substring(firstBracket, lastBracket + 1);
             const parsedSegments = JSON.parse(jsonStr);

             // Validate structure
             if (Array.isArray(parsedSegments)) {
               setTranslatedSegments(parsedSegments);
               // Also set full text
               setTranslatedText(parsedSegments.map(s => s.text).join(' '));
             } else {
               throw new Error("Response is not an array");
             }
           } else {
             throw new Error("No JSON array found in response");
           }
        } catch (e) {
           console.warn("Failed to parse JSON response", e);
           message.warning("Translation received but format was not perfect JSON. Showing as text.");
           setTranslatedText(result);
           setTranslatedSegments([]);
        }
      } else {
        setTranslatedText(result);
      }

    } catch (error) {
      console.error('Translation error:', error);
      message.error(`Translation failed: ${error.message}`);
    } finally {
      setIsTranslating(false);
    }
  };

  const handleExport = (format) => {
      let content = '';
      let filename = 'translation';
      let mimeType = 'text/plain';

      if (format === 'json') {
          content = JSON.stringify({
              text: translatedText,
              segments: translatedSegments,
              language: customLanguage || targetLanguage
          }, null, 2);
          filename += '.json';
          mimeType = 'application/json';
      } else if (format === 'txt') {
          content = translatedText;
          filename += '.txt';
      } else if (format === 'srt' && translatedSegments.length > 0) {
          content = translatedSegments.map((seg, idx) => {
              const start = formatSRTTime(seg.start);
              const end = formatSRTTime(seg.end);
              return `${idx + 1}\n${start} --> ${end}\n${seg.text}\n`;
          }).join('\n');
          filename += '.srt';
      } else if (format === 'vtt' && translatedSegments.length > 0) {
          const vttContent = translatedSegments.map((seg) => {
              const start = formatVTTTime(seg.start);
              const end = formatVTTTime(seg.end);
              return `${start} --> ${end}\n${seg.text}`;
          }).join('\n\n');
          content = `WEBVTT\n\n${vttContent}`;
          filename += '.vtt';
          mimeType = 'text/vtt';
      } else if (format === 'lrc' && translatedSegments.length > 0) {
          content = translatedSegments.map((seg) => {
              const time = formatLRCTime(seg.start);
              return `[${time}]${seg.text}`;
          }).join('\n');
          filename += '.lrc';
      }

      if (!content) return;

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
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

  // Header Extra Content
  const renderHeaderExtra = () => (
    <Space>
      <Space.Compact>
        <Select
            value={targetLanguage}
            onChange={setTargetLanguage}
            style={{ width: 120 }}
            options={[
                { value: 'English', label: 'English' },
                { value: 'Chinese', label: 'Chinese' },
                { value: 'Japanese', label: 'Japanese' },
                { value: 'Korean', label: 'Korean' },
                { value: 'Spanish', label: 'Spanish' },
                { value: 'French', label: 'French' },
                { value: 'German', label: 'German' },
                { value: 'Custom', label: 'Custom' },
            ]}
        />
        {targetLanguage === 'Custom' && (
            <Input
                placeholder="Target Lang"
                value={customLanguage}
                onChange={e => setCustomLanguage(e.target.value)}
                style={{ width: 100 }}
            />
        )}
      </Space.Compact>

      <Button
        type="primary"
        onClick={handleTranslate}
        loading={isTranslating}
        icon={<PlayCircleOutlined />}
      >
        Start Translation
      </Button>

      <Button
        icon={<SettingOutlined />}
        onClick={() => setIsSettingsVisible(true)}
      >
        Settings
      </Button>

      {translatedText && (
        <>
            <Button
                icon={<CopyOutlined />}
                onClick={() => {
                    navigator.clipboard.writeText(translatedText);
                    message.success('Copied');
                }}
            >
                Copy
            </Button>
            <Dropdown
              menu={{
                items: [
                  { key: 'txt', label: 'Export TXT', onClick: () => handleExport('txt') },
                  { key: 'json', label: 'Export JSON', onClick: () => handleExport('json') },
                  { key: 'srt', label: 'Export SRT', disabled: !translatedSegments.length, onClick: () => handleExport('srt') },
                  { key: 'vtt', label: 'Export VTT', disabled: !translatedSegments.length, onClick: () => handleExport('vtt') },
                  { key: 'lrc', label: 'Export LRC', disabled: !translatedSegments.length, onClick: () => handleExport('lrc') },
                ]
              }}
            >
                <Button icon={<DownloadOutlined />}>Export</Button>
            </Dropdown>
        </>
      )}
    </Space>
  );

  return (
    <Card
      title={
        <Space>
          <TranslationOutlined />
          <span>AI Translation</span>
        </Space>
      }
      extra={renderHeaderExtra()}
      style={{
        marginTop: 24,
        boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
      }}
    >
      {!translatedText && !isTranslating ? (
        <div style={{
            textAlign: 'center',
            padding: '60px 20px',
            color: isDarkMode ? '#666' : '#999',
            border: `1px dashed ${isDarkMode ? '#444' : '#d9d9d9'}`,
            borderRadius: 8
        }}>
            <TranslationOutlined style={{ fontSize: 32, marginBottom: 16 }} />
            <Paragraph>
                Select a target language and click "Start Translation" to translate the results using AI.
            </Paragraph>
        </div>
      ) : (
        <Tabs
            defaultActiveKey="1"
            items={[
            {
                key: '1',
                label: 'Text',
                children: (
                <TextArea
                    value={translatedText}
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
                label: `Segments (${translatedSegments.length})`,
                disabled: translatedSegments.length === 0,
                children: (
                <Table
                    dataSource={translatedSegments.map((s, i) => ({ ...s, key: i }))}
                    rowKey="key"
                    pagination={{
                        pageSize: segmentPageSize,
                        showSizeChanger: true,
                        pageSizeOptions: ['5', '10', '20', '50', '100'],
                        onChange: (page, pageSize) => setSegmentPageSize(pageSize)
                    }}
                    size="small"
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
                        render: (text) => (
                        <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
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
                disabled: translatedSegments.length === 0,
                children: (
                <LyricsView
                    segments={translatedSegments}
                    currentTime={currentTime}
                />
                )
            }
            ]}
        />
      )}

      <Modal
        title="AI Provider Settings"
        open={isSettingsVisible}
        onCancel={() => setIsSettingsVisible(false)}
        footer={null}
      >
        <Form
          layout="vertical"
          initialValues={settings}
          onFinish={handleSaveSettings}
        >
          <Form.Item label="Provider" name="provider">
            <Select>
              <Option value="zhipu">Zhipu AI (ChatGLM)</Option>
              <Option value="deepseek">DeepSeek</Option>
            </Select>
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prev, current) => prev.provider !== current.provider}
          >
            {({ getFieldValue }) => {
                const isZhipu = getFieldValue('provider') === 'zhipu';
                return (
                    <>
                        <Form.Item
                            label="API Key"
                            name={isZhipu ? "zhipuKey" : "deepseekKey"}
                            rules={[{ required: true, message: 'Please enter API Key' }]}
                        >
                            <Input.Password placeholder={`Enter ${isZhipu ? 'Zhipu' : 'DeepSeek'} API Key`} />
                        </Form.Item>

                        <Form.Item label="API Endpoint" name="apiUrl" help="Optional. Overrides default endpoint.">
                            <Input placeholder={isZhipu ? "https://open.bigmodel.cn/api/paas/v4/chat/completions" : "https://api.deepseek.com/v1/chat/completions"} />
                        </Form.Item>
                    </>
                );
            }}
          </Form.Item>

          <Form.Item label="Model Name" name="model" tooltip="Leave empty for default">
            <Input placeholder="e.g., glm-4-plus or deepseek-chat" />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" block>
              Save Configuration
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export default TranslationView;
