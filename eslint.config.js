import eslint from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist/**', 'release/**', 'coverage/**', '.venv*/**'] },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['src/**/*.ts', 'tests/**/*.ts'],
    languageOptions: { globals: { ...globals.browser, chrome: 'readonly' } },
  },
  {
    files: ['scripts/**/*.mjs', 'tools/**/*.mjs'],
    languageOptions: { globals: globals.node },
  },
);
