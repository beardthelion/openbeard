/**
 * @license
 * Copyright 2026 OpenBeard Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  OpenDialogActionReturn,
  SlashCommand,
} from './types.js';
import { CommandKind } from './types.js';

export const setupCommand: SlashCommand = {
  name: 'setup',
  description: 'Configure API endpoint, key, and model',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: (_context, _args): OpenDialogActionReturn => ({
    type: 'dialog',
    dialog: 'setupWizard',
  }),
};
