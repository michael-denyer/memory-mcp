/**
 * Tailwind config for the Memory MCP dashboard's vendored CSS build.
 * Used by scripts/build-dashboard-css.sh. The content globs drive which
 * utility classes survive minification; the theme extend supplies the
 * custom dark palette the templates rely on.
 */
module.exports = {
  darkMode: 'class',
  content: [
    'src/memory_mcp/dashboard/templates/**/*.html',
    'src/memory_mcp/dashboard/app.py',
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          800: '#1e1e2e',
          900: '#11111b',
          950: '#0a0a0f',
        },
      },
    },
  },
};
