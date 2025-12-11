# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static HTML website containing educational materials based on Stephen Wolfram's essay "What Is ChatGPT Doing … and Why Does It Work?" (February 2023). The project presents analysis, chapter summaries, and a comprehensive PRD for creating learning materials about LLM technology.

## Project Structure

- `index.html` - Landing page with navigation to different analysis versions
- `wolfram-chatgpt-analysis.html` - Analysis presentation
- `wolfram-chatgpt-complete.html` - Complete analysis version
- `wolfram-chatgpt-full.html` - Full analysis version
- `sec.html` - Section/chapter view
- `wolfram-chatgpt-analysis-prd.md` - Source markdown with 16-chapter analysis and PRD

## Development

This is a static HTML site with no build process. To develop:

```bash
# Serve locally (any static server works)
python -m http.server 8000
# or
npx serve .
```

Open `index.html` in browser to preview.

## Architecture

- **No framework** - Pure HTML/CSS with inline styles
- **Fonts** - Google Fonts (Material Symbols Outlined)
- **Content source** - `wolfram-chatgpt-analysis-prd.md` contains the canonical content which HTML files render
- **Design system** - CSS custom properties defined in `:root` (primary blue, secondary purple, accent cyan)
