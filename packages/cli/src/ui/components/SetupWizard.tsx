/**
 * @license
 * Copyright 2026 OpenBeard Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useState, useCallback } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { RadioButtonSelect } from './shared/RadioButtonSelect.js';
import type { LoadedSettings } from '../../config/settings.js';
import { SettingScope } from '../../config/settings.js';
import { AuthType } from '@google/gemini-cli-core';
import { useKeypress } from '../hooks/useKeypress.js';
import { Command } from '../key/keyMatchers.js';
import { useKeyMatchers } from '../hooks/useKeyMatchers.js';

interface SetupWizardProps {
  settings: LoadedSettings;
  onComplete: () => void;
  onCancel: () => void;
}

type ProviderPreset = {
  label: string;
  baseUrl: string;
  defaultModel: string;
  needsKey: boolean;
};

const PRESETS: Record<string, ProviderPreset> = {
  openai: {
    label: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    defaultModel: 'gpt-4o',
    needsKey: true,
  },
  ollama: {
    label: 'Ollama (local)',
    baseUrl: 'http://localhost:11434/v1',
    defaultModel: 'llama3',
    needsKey: false,
  },
  custom: {
    label: 'Custom endpoint',
    baseUrl: '',
    defaultModel: '',
    needsKey: true,
  },
};

type WizardStep = 'preset' | 'url' | 'apikey' | 'model' | 'testing' | 'done';

export function SetupWizard({
  settings,
  onComplete,
  onCancel,
}: SetupWizardProps): React.JSX.Element {
  const [step, setStep] = useState<WizardStep>('preset');
  const [preset, setPreset] = useState<string>('openai');
  const [baseUrl, setBaseUrl] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [model, setModel] = useState<string>('');
  const [inputValue, setInputValue] = useState<string>('');
  const [testStatus, setTestStatus] = useState<
    'idle' | 'testing' | 'success' | 'error'
  >('idle');
  const [testError, setTestError] = useState<string>('');
  const keyMatchers = useKeyMatchers();

  const handlePresetSelect = useCallback(
    (value: string) => {
      setPreset(value);
      const p = PRESETS[value];
      if (p) {
        if (value === 'custom') {
          setBaseUrl('');
          setModel('');
        } else {
          setBaseUrl(p.baseUrl);
          setModel(p.defaultModel);
        }
      }
      setStep('url');
      setInputValue('');
    },
    [],
  );

  const submitInput = useCallback(() => {
    if (step === 'url') {
      const url = inputValue.trim();
      if (!url) return;
      setBaseUrl(url);
      const p = PRESETS[preset];
      if (p && !p.needsKey) {
        setApiKey('');
        setStep('model');
      } else {
        setStep('apikey');
      }
      setInputValue('');
    } else if (step === 'apikey') {
      setApiKey(inputValue.trim());
      setStep('model');
      setInputValue('');
    } else if (step === 'model') {
      const m = inputValue.trim();
      if (!m) return;
      setModel(m);
      setStep('testing');
      setTestStatus('testing');
      setTestError('');
    }
  }, [step, inputValue, preset]);

  // Run connection test when entering testing step
  const runTest = useCallback(async () => {
    try {
      const url = `${baseUrl.replace(/\/+$/, '')}/models`;
      const headers: Record<string, string> = {};
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      const response = await fetch(url, {
        method: 'GET',
        headers,
        signal: controller.signal,
      });
      clearTimeout(timeout);
      // Any HTTP response (even 404/401) means the endpoint is reachable.
      // 404 on /models just means that path doesn't exist — the server is up.
      // 401 means auth is required — expected without a valid key.
      if (response.status < 500) {
        setTestStatus('success');
        settings.setValue(SettingScope.User, 'openai.baseUrl', baseUrl);
        if (apiKey) {
          settings.setValue(SettingScope.User, 'openai.apiKey', apiKey);
        }
        settings.setValue(SettingScope.User, 'openai.model', model);
        settings.setValue(
          SettingScope.User,
          'security.auth.selectedType',
          AuthType.OPENAI_COMPATIBLE,
        );
        setTimeout(() => {
          setStep('done');
          onComplete();
        }, 800);
      } else {
        setTestStatus('error');
        setTestError(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (err) {
      setTestStatus('error');
      const message = err instanceof Error ? err.message : String(err);
      setTestError(
        message === 'The operation was aborted'
          ? 'Connection timed out'
          : message,
      );
    }
  }, [baseUrl, apiKey, model, settings, onComplete]);

  // Trigger test when entering testing step
  if (step === 'testing' && testStatus === 'testing' && testError === '') {
    void runTest();
  }

  // Key handling
  useKeypress(
    (key) => {
      const cmd = keyMatchers;

      // Escape goes back
      if (key.name === 'escape') {
        switch (step) {
          case 'url':
            setStep('preset');
            setInputValue('');
            return;
          case 'apikey':
            setStep('url');
            setInputValue(baseUrl);
            return;
          case 'model':
            setStep(PRESETS[preset]?.needsKey ? 'apikey' : 'url');
            setInputValue('');
            return;
          case 'testing':
            setStep('model');
            setTestStatus('idle');
            setTestError('');
            return;
          default:
            onCancel();
            return;
        }
      }

      // Enter submits input
      if (cmd[Command.SUBMIT](key)) {
        submitInput();
        return;
      }

      // Backspace
      if (
        key.name === 'backspace' &&
        (step === 'url' || step === 'apikey' || step === 'model')
      ) {
        setInputValue((prev) => prev.slice(0, -1));
        return;
      }

      // Regular character input
      if (
        key.sequence &&
        !key.ctrl &&
        (step === 'url' || step === 'apikey' || step === 'model')
      ) {
        setInputValue((prev) => prev + key.sequence);
      }
    },
    { isActive: true },
  );

  const stepNumber = {
    preset: 1,
    url: 2,
    apikey: 3,
    model: 4,
    testing: 5,
    done: 5,
  }[step];
  const totalSteps = PRESETS[preset]?.needsKey ? 4 : 3;

  return (
    <Box
      borderStyle="round"
      borderColor={theme.ui.focus}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold color={theme.text.primary}>
        OpenBeard Setup ({stepNumber}/{totalSteps})
      </Text>

      {step === 'preset' && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={theme.text.primary}>
            Choose your API provider:
          </Text>
          <Box marginTop={1}>
            <RadioButtonSelect
              items={Object.entries(PRESETS).map(([key, p]) => ({
                label: p.label,
                value: key,
                key,
              }))}
              initialIndex={0}
              onSelect={handlePresetSelect}
            />
          </Box>
          <Box marginTop={1}>
            <Text color={theme.text.secondary} dimColor>
              (Enter to select, Esc to exit)
            </Text>
          </Box>
        </Box>
      )}

      {step === 'url' && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={theme.text.primary}>API Base URL:</Text>
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>{'> '}</Text>
            <Text color={theme.text.primary}>
              {inputValue}
              <Text color={theme.text.secondary} dimColor>
                {PRESETS[preset]?.baseUrl
                  ? ` (${PRESETS[preset].baseUrl})`
                  : ' https://api.example.com/v1'}
              </Text>
            </Text>
          </Box>
          <Box marginTop={1}>
            <Text color={theme.text.secondary} dimColor>
              (Enter to continue, Esc to go back)
            </Text>
          </Box>
        </Box>
      )}

      {step === 'apikey' && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={theme.text.primary}>API Key:</Text>
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>{'> '}</Text>
            <Text color={theme.text.primary}>
              {apiKey ? '*'.repeat(Math.min(apiKey.length, 20)) : inputValue ? '*'.repeat(inputValue.length) : ''}
              {!inputValue && (
                <Text color={theme.text.secondary} dimColor>
                  {' (press Enter to skip)'}
                </Text>
              )}
            </Text>
          </Box>
          <Box marginTop={1}>
            <Text color={theme.text.secondary} dimColor>
              (Enter to continue, Esc to go back, empty to skip)
            </Text>
          </Box>
        </Box>
      )}

      {step === 'model' && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={theme.text.primary}>Model name:</Text>
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>{'> '}</Text>
            <Text color={theme.text.primary}>
              {inputValue}
              {PRESETS[preset]?.defaultModel && !inputValue && (
                <Text color={theme.text.secondary} dimColor>
                  {' ' + PRESETS[preset].defaultModel}
                </Text>
              )}
            </Text>
          </Box>
          <Box marginTop={1}>
            <Text color={theme.text.secondary} dimColor>
              (Enter to continue, Esc to go back)
            </Text>
          </Box>
        </Box>
      )}

      {step === 'testing' && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={theme.text.primary}>Testing connection...</Text>
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>Endpoint: {baseUrl}</Text>
          </Box>
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>Model: {model}</Text>
          </Box>
          {testStatus === 'testing' && (
            <Box marginTop={1}>
              <Text color={theme.status.warning}>Connecting...</Text>
            </Box>
          )}
          {testStatus === 'success' && (
            <Box marginTop={1}>
              <Text color={theme.status.success}>
                Connected! Saving settings...
              </Text>
            </Box>
          )}
          {testStatus === 'error' && (
            <Box flexDirection="column" marginTop={1}>
              <Text color={theme.status.error}>Connection failed</Text>
              <Text color={theme.status.error}>{testError}</Text>
              <Box marginTop={1}>
                <Text color={theme.text.secondary} dimColor>
                  (Press Esc to go back and fix settings)
                </Text>
              </Box>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}
