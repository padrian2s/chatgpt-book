# Technical Blueprint: `wolfram-chatgpt-full.html`

A reusable template for creating illustrated book/essay analysis pages.

---

## AI Instructions: Processing Source Text

When given a body of text (book, essay, article), follow this workflow to transform it into the illustrated analysis format:

### Step 1: Analyze Structure

1. **Identify natural divisions** - Find chapters, sections, or major topic shifts
2. **Extract key statistics** - Numbers, metrics, specifications worth highlighting in stats bar
3. **Note image opportunities** - Concepts that benefit from diagrams, charts, or illustrations
4. **Identify vocabulary** - Technical terms requiring glossary definitions

### Step 2: Generate Chapter Content

For each chapter/section:

1. **Write chapter title** - Concise, descriptive (5-8 words max)
2. **Break into paragraphs** - Each `.para` should contain one core idea from the source
3. **Write commentary for each paragraph:**
   - **Analysis heading** - What makes this important
   - **Key insight** - The deeper meaning or implication
   - **Practical takeaway** - Why reader should care
   - **Connections** - Links to other concepts in the text
4. **Create key concept box** - One per chapter summarizing the main takeaway

### Step 3: Generate Supporting Elements

| Element | AI Task |
|---------|---------|
| **Stats Bar** | Extract 4-6 quantitative facts (numbers, percentages, counts) |
| **Image Captions** | Write descriptive captions explaining what each figure shows |
| **Tables** | Convert comparisons/lists into structured table format |
| **Glossary** | Define 8-12 key technical terms in plain language |
| **Specs Box** | Format key specifications as label-value pairs |

### Step 4: Commentary Writing Guidelines

**Tone:** Educational, insightful, conversational but professional

**Structure each commentary:**
```
<h4>{Insight Title}</h4>
<p>{2-3 sentences explaining significance}</p>
<h4>{Secondary Point}</h4>
<ul>
  <li><strong>{Term}:</strong> {explanation}</li>
  <li><strong>{Term}:</strong> {explanation}</li>
</ul>
```

**Commentary types to include:**
- **Why This Matters** - Contextual significance
- **Technical Deep Dive** - Detailed explanation for advanced readers
- **Common Misconception** - What people often get wrong
- **Real-World Application** - Practical examples
- **Connection to...** - Links between concepts

### Step 5: Output Format

Generate complete HTML following the template structure:

```html
<section class="chapter" id="ch1">
    <div class="chapter-header">
        <div class="chapter-number">Chapter 1</div>
        <h2 class="chapter-title">{Derived from source}</h2>
    </div>
    <div class="chapter-content">
        <div class="para">
            <div class="para-text">{Original or paraphrased text}</div>
            <div class="para-hint">Click for commentary</div>
            <div class="commentary">
                <div class="commentary-head">
                    <div class="commentary-icon">💡</div>
                    <div class="commentary-title">Analysis</div>
                </div>
                <h4>{AI-generated insight}</h4>
                <p>{AI-generated explanation}</p>
            </div>
        </div>

        <div class="key-concept">
            <div class="key-concept-label">Key Concept</div>
            <div class="key-concept-text">{AI-generated summary}</div>
        </div>
    </div>
</section>
```

### Quality Checklist

- [ ] Every paragraph has meaningful commentary (not just restating)
- [ ] Commentary adds value beyond the source text
- [ ] Key concepts capture the essential takeaway
- [ ] Technical terms are defined in glossary
- [ ] Stats bar contains accurate, impactful numbers
- [ ] Chapter titles are scannable and informative
- [ ] Image captions explain significance, not just describe

## Overview

Self-contained, single-file HTML educational reading experience (~1857 lines). No external dependencies except Google Fonts. Suitable as a template for other book/essay analyses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          HTML Document                          │
├─────────────────────────────────────────────────────────────────┤
│  <head>                                                         │
│    └── Inline CSS (~637 lines)                                  │
│         ├── CSS Custom Properties (Design Tokens)               │
│         ├── Layout Components                                   │
│         ├── Typography & Colors                                 │
│         └── Responsive Breakpoints                              │
├─────────────────────────────────────────────────────────────────┤
│  <body>                                                         │
│    ├── Progress Bar (fixed, top)                                │
│    ├── Mobile Menu Toggle (fixed, hidden on desktop)            │
│    ├── Sidebar Navigation (fixed, 280px)                        │
│    │     ├── Header (gradient banner)                           │
│    │     └── Nav Links (chapter anchors)                        │
│    ├── Main Content (margin-left: 280px)                        │
│    │     ├── Hero Section (title, meta, stats bar)              │
│    │     ├── Chapters 1-16 (repeating structure)                │
│    │     ├── PRD Section (dark background variant)              │
│    │     └── Footer                                             │
│    ├── Back-to-Top Button (fixed, bottom-right)                 │
│    └── JavaScript (~40 lines, inline)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design System (CSS Custom Properties)

```css
:root {
    --primary: #3b82f6;        /* Blue - main accent */
    --primary-dark: #2563eb;
    --secondary: #8b5cf6;      /* Purple - alternate chapters */
    --accent: #06b6d4;         /* Cyan - highlights */
    --success: #10b981;        /* Green - alternate chapters */
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark: #0f172a;           /* Near-black backgrounds */
    --gray-50 to --gray-900    /* Full grayscale palette */
}
```

---

## Component Inventory

| Component | Class | Purpose |
|-----------|-------|---------|
| **Sidebar** | `.sidebar` | Fixed 280px left navigation |
| **Hero** | `.hero` | Top banner with title, subtitle, meta |
| **Stats Bar** | `.stats-bar` | Key numbers display (175B params, etc.) |
| **Chapter** | `.chapter` | Container for each chapter |
| **Chapter Header** | `.chapter-header` | Gradient banner with chapter number/title |
| **Paragraph** | `.para` | Clickable text block with expandable commentary |
| **Commentary** | `.commentary` | Hidden analysis revealed on click |
| **Image Block** | `.img-block` | Image with dark caption bar |
| **Image Grid** | `.img-grid` | 2-column responsive image layout |
| **Key Concept** | `.key-concept` | Dark highlighted box for main takeaways |
| **Tech Box** | `.tech-box` | Monospace code/formula display |
| **Table** | `.table-wrap > table` | Styled data tables |
| **Specs Box** | `.specs-box` | Key-value specifications display |
| **Glossary** | `.glossary-grid` | Definition cards grid |
| **PRD Section** | `.prd` | Dark-theme variant for PRD content |
| **Progress Bar** | `.progress` | Reading progress indicator |
| **Back to Top** | `.back-top` | Floating scroll button |

---

## Chapter Structure (Repeatable Pattern)

```html
<section class="chapter" id="ch{N}">
    <div class="chapter-header">
        <div class="chapter-number">Chapter {N}</div>
        <h2 class="chapter-title">{Title}</h2>
    </div>
    <div class="chapter-content">
        <!-- Images -->
        <div class="img-block">
            <img src="{URL}" alt="{description}">
            <div class="img-caption"><strong>Figure {N}.{M}:</strong> {caption}</div>
        </div>

        <!-- Paragraph with commentary -->
        <div class="para">
            <div class="para-text">{Original text}</div>
            <div class="para-hint">Click for commentary</div>
            <div class="commentary">
                <div class="commentary-head">
                    <div class="commentary-icon">💡</div>
                    <div class="commentary-title">Analysis</div>
                </div>
                <h4>{Sub-heading}</h4>
                <p>{Commentary text}</p>
                <ul><li>{Points}</li></ul>
            </div>
        </div>

        <!-- Key concept box -->
        <div class="key-concept">
            <div class="key-concept-label">Key Concept</div>
            <div class="key-concept-text">{Summary}</div>
        </div>
    </div>
</section>
```

---

## Chapter Header Color Rotation

```css
/* Default: Blue */
.chapter-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
}

/* 4n+2: Purple */
.chapter:nth-child(4n+2) .chapter-header {
    background: linear-gradient(135deg, var(--secondary) 0%, #7c3aed 100%);
}

/* 4n+3: Cyan */
.chapter:nth-child(4n+3) .chapter-header {
    background: linear-gradient(135deg, #0891b2 0%, var(--accent) 100%);
}

/* 4n+4: Green */
.chapter:nth-child(4n+4) .chapter-header {
    background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
}
```

---

## JavaScript Functionality

```javascript
// 1. Paragraph expand/collapse (event delegation)
document.addEventListener('click', function(e) {
    const para = e.target.closest('.para');
    if (para) para.classList.toggle('expanded');
});

// 2. Mobile sidebar close on link click
// 3. Back-to-top button visibility (scrollY > 500)
// 4. Reading progress bar calculation
// 5. Print mode: auto-expand all commentaries
```

---

## Responsive Breakpoints

```css
@media (max-width: 1024px) {
    .sidebar { transform: translateX(-100%); }  /* Hidden */
    .sidebar.open { transform: translateX(0); } /* Toggle visible */
    .menu-toggle { display: block; }            /* Hamburger shown */
    .main-content { margin-left: 0; }           /* Full width */
}

@media print {
    .sidebar, .menu-toggle, .back-top, .progress { display: none !important; }
    .main-content { margin-left: 0; }
    .chapter { break-inside: avoid; }
    .commentary { display: block !important; }  /* Show all */
}
```

---

## Blueprint Checklist for New Books

### 1. Replace Content Variables

- [ ] Title in `<title>`, `.hero h1`, `.sidebar-header h1`
- [ ] Author/source in `.hero-meta`, `.sidebar-header p`
- [ ] Source URL in footer
- [ ] Stats bar numbers (customize metrics for the content)

### 2. Structure Chapters

- [ ] Create `<section class="chapter" id="ch{N}">` for each chapter
- [ ] Add corresponding sidebar `<a href="#ch{N}">` navigation links
- [ ] Follow paragraph/commentary/image pattern within each chapter

### 3. Add Images

- [ ] Use `.img-block` for single images with captions
- [ ] Use `.img-grid` for 2-column responsive layouts
- [ ] Include numbered figure captions (`Figure {chapter}.{number}`)

### 4. Add Supplementary Sections

- [ ] PRD/appendix content in `.prd` dark section
- [ ] Glossary terms using `.glossary-grid > .glossary-item`
- [ ] Key specifications using `.specs-box > .specs-row`

### 5. Customize Theme (Optional)

- [ ] Modify `:root` CSS variables for different color schemes
- [ ] Adjust chapter header gradient rotation pattern
- [ ] Change fonts in `body` font-family declaration

---

## File Statistics

| Metric | Value |
|--------|-------|
| Total lines | ~1857 |
| CSS lines | ~637 |
| JavaScript lines | ~40 |
| Chapters | 16 |
| External dependencies | Google Fonts only |
