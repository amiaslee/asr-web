import React, { useState, useEffect } from 'react';
import { Card, Button, Modal, Select, Input, Form, message, Space, Typography, Dropdown, Menu } from 'antd';
import { TranslationOutlined, SettingOutlined, DownloadOutlined, CopyOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

const TranslationView = ({ originalText, segments, isDarkMode }) => {
  const [isSettingsVisible, setIsSettingsVisible] = useState(false);
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
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
        url: 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
        apiKey: settings.zhipuKey,
        defaultModel: 'glm-4'
      };
    } else {
      return {
        url: settings.apiUrl || 'https://api.deepseek.com/v1/chat/completions', // Example URL
        apiKey: settings.deepseekKey,
        defaultModel: 'deepseek-chat'
      };
    }
  };

  const handleTranslate = async () => {
    if (!originalText) {
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
    try {
      // Construct the prompt
      // We want to translate the content while keeping structure if possible, or just text
      // The user suggested "translate original format... ensuring model returns correct format... JSON best"

      const contentToTranslate = segments && segments.length > 0
        ? JSON.stringify(segments.map(s => ({ start: s.start, end: s.end, text: s.text })))
        : originalText;

      const systemPrompt = segments && segments.length > 0
        ? `You are a professional translator. Translate the following JSON segments into English (or the target language if specified). Keep the JSON structure exactly the same (keys: start, end, text), only translate the 'text' value. Output VALID JSON only.`
        : `You are a professional translator. Translate the following text into English.`;

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
          stream: false // Simplify for now, maybe stream later
        })
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error.message || 'API Error');
      }

      const result = data.choices?.[0]?.message?.content || '';

      // If it was JSON, try to parse it to pretty print or verify
      if (segments && segments.length > 0) {
        try {
           // Clean up markdown code blocks if present
           const cleanResult = result.replace(/```json/g, '').replace(/```/g, '').trim();
           const parsed = JSON.parse(cleanResult);
           setTranslatedText(JSON.stringify(parsed, null, 2));
        } catch (e) {
           console.warn("Failed to parse JSON response", e);
           setTranslatedText(result);
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

  const handleExport = () => {
      if (!translatedText) return;
      const blob = new Blob([translatedText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'translation.json'; // Default to json as requested
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
  };

  return (
    <Card
      title={
        <Space>
          <TranslationOutlined />
          <span>AI Translation</span>
        </Space>
      }
      extra={
        <Space>
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
                />
                <Button
                    icon={<DownloadOutlined />}
                    onClick={handleExport}
                >
                    Export
                </Button>
            </>
          )}
        </Space>
      }
      style={{
        marginTop: 24,
        boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
      }}
    >
      <div style={{ marginBottom: 16, textAlign: 'center' }}>
        <Button
            type="primary"
            size="large"
            onClick={handleTranslate}
            loading={isTranslating}
            style={{ minWidth: 200 }}
        >
            {isTranslating ? 'Translating...' : 'Start Translation'}
        </Button>
      </div>

      {translatedText ? (
        <TextArea
          value={translatedText}
          readOnly
          autoSize={{ minRows: 6, maxRows: 20 }}
          style={{
            fontFamily: 'monospace',
            background: isDarkMode ? '#1f1f1f' : '#f5f5f5',
            color: isDarkMode ? '#ddd' : '#333'
          }}
        />
      ) : (
        <div style={{
            textAlign: 'center',
            padding: '40px',
            color: isDarkMode ? '#666' : '#999',
            border: `1px dashed ${isDarkMode ? '#444' : '#d9d9d9'}`,
            borderRadius: 8
        }}>
            <TranslationOutlined style={{ fontSize: 32, marginBottom: 8 }} />
            <p>Configure settings and click Start to translate recognition results</p>
        </div>
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
            {({ getFieldValue }) =>
                getFieldValue('provider') === 'zhipu' ? (
                    <Form.Item
                        label="API Key"
                        name="zhipuKey"
                        rules={[{ required: true, message: 'Please enter API Key' }]}
                    >
                        <Input.Password placeholder="Enter Zhipu API Key" />
                    </Form.Item>
                ) : (
                    <>
                        <Form.Item
                            label="API Key"
                            name="deepseekKey"
                            rules={[{ required: true, message: 'Please enter API Key' }]}
                        >
                            <Input.Password placeholder="Enter DeepSeek API Key" />
                        </Form.Item>
                        <Form.Item label="API Endpoint" name="apiUrl">
                            <Input placeholder="https://api.deepseek.com/v1/chat/completions" />
                        </Form.Item>
                    </>
                )
            }
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
