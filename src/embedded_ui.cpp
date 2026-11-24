#include "embedded_ui.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace qc {
namespace ui {

static const char EMBEDDED_HTML[] = R"====rawhtml====(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="icon" href="data:,">
  <title>Bharat AI-SoC PS5 · Unified Metal QC</title>
  <style>
    :root {
      --bg-base: #0F1419;
      --bg-card: #171D25;
      --bg-elevated: #1E2631;
      --border-subtle: #232B36;
      --text-main: #E6EDF3;
      --text-muted: #9AA7B4;
      --accent-teal: #2DD4BF;

      /* Status colors used strictly for semantic meaning */
      --status-pass: #22C55E;
      --status-med: #F59E0B;
      --status-high: #EF4444;

      --font-ui: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: "JetBrains Mono", Consolas, Menlo, monospace;

      --radius-card: 12px;
      --radius-control: 8px;
      --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.22);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    html, body {
      width: 100%;
      height: 100vh;
      overflow: hidden;
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-ui);
      font-size: 13px;
      line-height: 1.45;
    }

    body {
      display: flex;
      flex-direction: column;
    }

    :focus-visible {
      outline: 2px solid #2DD4BF;
      outline-offset: 2px;
    }

    /* -------------------------------------------------------------
       1. TOP BAR (56px)
       ------------------------------------------------------------- */
    .app-header {
      height: 56px;
      min-height: 56px;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      z-index: 100;
    }

    .brand-section {
      display: flex;
      align-items: center;
      min-width: 0;
    }

    .brand-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-main);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .mode-segmented-control {
      display: flex;
      align-items: center;
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
    }

    .segment-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      font-family: var(--font-ui);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }

    .segment-btn:hover {
      color: var(--text-main);
      background: var(--bg-elevated);
    }

    .segment-btn.active {
      background: var(--accent-teal);
      color: #0F1419;
      font-weight: 600;
    }

    .segment-icon {
      width: 15px;
      height: 15px;
      stroke-width: 1.9;
      stroke: currentColor;
      fill: none;
      flex-shrink: 0;
    }

    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 999px;
      padding: 6px 12px;
      font-family: var(--font-ui);
      font-size: 12px;
      font-weight: 500;
      color: var(--text-main);
      white-space: nowrap;
      flex-shrink: 0;
    }

    .status-indicator-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-teal);
      flex-shrink: 0;
    }

    /* -------------------------------------------------------------
       2. MAIN WORKSPACE (Viewport ~65% + Result Panel ~35%)
       ------------------------------------------------------------- */
    .main-workspace {
      flex: 1;
      display: flex;
      position: relative;
      overflow: hidden;
      height: calc(100vh - 56px);
    }

    .viewport-area {
      flex: 1 1 65%;
      position: relative;
      background: var(--bg-base);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      min-width: 0;
    }

    .viewport-canvas-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      padding: 12px 12px 68px 12px;
    }

    /* Top-left overlay chip */
    .corner-chip-left {
      position: absolute;
      top: 12px;
      left: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(23, 29, 37, 0.90);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--text-main);
      z-index: 20;
      max-width: calc(100% - 176px);
      box-shadow: var(--shadow-card);
    }

    #active-source-tag {
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 500;
      color: var(--text-main);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .counter-pill {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      color: var(--accent-teal);
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      flex-shrink: 0;
    }

    .file-select-dropdown {
      background: var(--bg-elevated);
      border: 1px solid rgba(45, 212, 191, 0.45);
      color: var(--text-main);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-family: var(--font-mono);
      font-weight: 500;
      cursor: pointer;
      min-width: 180px;
      max-width: 280px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      pointer-events: auto;
    }

    .file-select-dropdown:hover,
    .file-select-dropdown:focus {
      border-color: var(--accent-teal);
      outline: none;
    }

    /* Top-right temporal lock chip */
    .corner-chip-right {
      position: absolute;
      top: 12px;
      right: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(23, 29, 37, 0.90);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: 999px;
      padding: 6px 12px;
      font-family: var(--font-ui);
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      z-index: 20;
      box-shadow: var(--shadow-card);
      cursor: help;
    }

    .lock-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }

    .lock-pill.lock-pill-locked {
      color: var(--accent-teal);
    }

    .lock-pill.lock-pill-scan {
      color: var(--text-muted);
    }

    .lock-pill.lock-pill-warn {
      color: var(--status-med);
    }

    .hidden-compat {
      display: none !important;
    }

    /* Toast Banner */
    .scada-toast {
      position: absolute;
      top: 60px;
      left: 50%;
      transform: translateX(-50%) translateY(-8px);
      background: var(--bg-elevated);
      border: 1px solid var(--accent-teal);
      border-radius: 10px;
      padding: 8px 16px;
      font-family: var(--font-ui);
      font-size: 13px;
      font-weight: 500;
      color: var(--text-main);
      z-index: 60;
      opacity: 0;
      pointer-events: none;
      box-shadow: var(--shadow-card);
      max-width: 85%;
      text-align: center;
    }

    .scada-toast.visible {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }

    .scada-toast.toast-error {
      border-color: var(--status-high);
      color: #FCA5A5;
    }

    /* Slim Bottom Toolbar */
    .floating-toolbar {
      position: absolute;
      bottom: 12px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(23, 29, 37, 0.92);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-card);
      padding: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      gap: 4px;
      max-width: calc(100% - 24px);
      z-index: 30;
      box-shadow: var(--shadow-card);
    }

    .tool-btn {
      height: 36px;
      padding: 0 12px;
      background: transparent;
      border: 1px solid transparent;
      border-radius: var(--radius-control);
      color: var(--text-main);
      font-size: 13px;
      font-weight: 500;
      font-family: var(--font-ui);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      cursor: pointer;
      white-space: nowrap;
    }

    .tool-btn:hover {
      background: var(--bg-elevated);
      border-color: var(--border-subtle);
    }

    .tool-btn svg {
      width: 15px;
      height: 15px;
      stroke: currentColor;
      fill: none;
      stroke-width: 1.85;
      stroke-linecap: round;
      stroke-linejoin: round;
      flex-shrink: 0;
    }

    .tool-btn.active-pause {
      background: rgba(245, 158, 11, 0.14);
      border-color: rgba(245, 158, 11, 0.55);
      color: var(--status-med);
    }

    .tool-btn-accent {
      border-color: rgba(45, 212, 191, 0.45);
      background: rgba(45, 212, 191, 0.10);
      color: var(--accent-teal);
    }

    .tool-btn-accent:hover {
      background: rgba(45, 212, 191, 0.18);
      border-color: var(--accent-teal);
    }

    /* Drag & Drop Upload Zone */
    .upload-dropzone {
      position: absolute;
      inset: 16px;
      background: rgba(23, 29, 37, 0.94);
      border: 1px dashed var(--border-subtle);
      border-radius: var(--radius-card);
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      z-index: 40;
      cursor: pointer;
    }

    .upload-dropzone.active-drag {
      display: flex;
      border-color: var(--accent-teal);
      background: rgba(23, 29, 37, 0.96);
    }

    .upload-zone-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-main);
    }

    .upload-zone-sub {
      font-size: 13px;
      color: var(--text-muted);
      text-align: center;
      max-width: 380px;
    }

    /* -------------------------------------------------------------
       3. RIGHT RESULT PANEL (~35% width, clean vertical scroll)
       ------------------------------------------------------------- */
    .result-panel {
      width: 35%;
      min-width: 360px;
      max-width: 580px;
      background: var(--bg-base);
      border-left: 1px solid var(--border-subtle);
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: var(--border-subtle) transparent;
      z-index: 10;
    }

    .panel-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-card);
      padding: 14px 16px;
      box-shadow: var(--shadow-card);
      flex-shrink: 0;
    }

    /* a) Sticky Verdict Card */
    .verdict-card {
      position: sticky;
      top: 0;
      z-index: 20;
      background: var(--bg-card);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.36);
    }

    .verdict-top-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 12px;
    }

    .verdict-badge {
      display: inline-flex;
      align-items: center;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid currentColor;
      line-height: 1.2;
      letter-spacing: 0.02em;
    }

    .action-state-pill {
      display: inline-flex;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      padding: 4px 10px;
      border-radius: 999px;
      white-space: nowrap;
    }

    .action-state-pill.state-pass {
      color: var(--status-pass);
      border-color: rgba(34, 197, 94, 0.35);
      background: rgba(34, 197, 94, 0.08);
    }

    .action-state-pill.state-med {
      color: var(--status-med);
      border-color: rgba(245, 158, 11, 0.35);
      background: rgba(245, 158, 11, 0.08);
    }

    .action-state-pill.state-high {
      color: #F87171;
      border-color: rgba(239, 68, 68, 0.35);
      background: rgba(239, 68, 68, 0.09);
    }

    .defect-header-container {
      display: flex;
      align-items: stretch;
      gap: 12px;
      margin-bottom: 12px;
    }

    .defect-accent-bar {
      width: 4px;
      border-radius: 4px;
      background: var(--status-pass);
      flex-shrink: 0;
    }

    .defect-name-text {
      font-size: 28px;
      font-weight: 700;
      color: var(--text-main);
      line-height: 1.18;
      letter-spacing: -0.02em;
      word-break: break-word;
    }

    .verdict-meta-row {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }

    .severity-badge {
      display: inline-flex;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 6px;
      border: 1px solid currentColor;
      line-height: 1.25;
    }

    .badge-pass {
      color: var(--status-pass);
      border-color: rgba(34, 197, 94, 0.45);
      background: rgba(34, 197, 94, 0.09);
    }

    .badge-low {
      color: var(--text-muted);
      border-color: var(--border-subtle);
      background: var(--bg-elevated);
    }

    .badge-med {
      color: var(--status-med);
      border-color: rgba(245, 158, 11, 0.45);
      background: rgba(245, 158, 11, 0.10);
    }

    .badge-high {
      color: #F87171;
      border-color: rgba(239, 68, 68, 0.45);
      background: rgba(239, 68, 68, 0.09);
    }

    .family-label-pill {
      display: inline-flex;
      align-items: center;
      font-size: 12px;
      font-weight: 500;
      padding: 3px 10px;
      border-radius: 6px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
    }

    .family-label-pill.meta-pill-warn {
      color: var(--status-med);
      border-color: rgba(245, 158, 11, 0.4);
      background: rgba(245, 158, 11, 0.10);
    }

    /* b) Key Metrics: 3 stat tiles in one row */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      flex-shrink: 0;
    }

    .stat-tile {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-card);
      padding: 12px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }

    .metric-label {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
    }

    .metric-value {
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      font-size: 18px;
      font-weight: 700;
      color: var(--text-main);
    }

    /* c) Anomaly Check (PatchCore) */
    .card-header-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 12px;
    }

    .card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
    }

    .anomaly-gauge-wrap {
      position: relative;
      height: 10px;
      margin-bottom: 8px;
    }

    .anomaly-gauge-track {
      width: 100%;
      height: 8px;
      margin-top: 1px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 999px;
      overflow: hidden;
    }

    .anomaly-gauge-fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: var(--status-pass);
    }

    .anomaly-gauge-fill.fill-pass {
      background: var(--status-pass);
    }

    .anomaly-gauge-fill.fill-med {
      background: var(--status-med);
    }

    .anomaly-gauge-fill.fill-high {
      background: var(--status-high);
    }

    .anomaly-thresh-marker {
      position: absolute;
      top: -2px;
      width: 2px;
      height: 14px;
      background: var(--text-main);
      border-radius: 2px;
      transform: translateX(-50%);
      pointer-events: none;
    }

    .anomaly-readout {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    .mono-num {
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      color: var(--text-main);
      font-weight: 600;
    }

    .sub-details {
      border-top: 1px solid var(--border-subtle);
      padding-top: 8px;
    }

    .sub-details summary {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      list-style: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      user-select: none;
    }

    .sub-details summary::-webkit-details-marker {
      display: none;
    }

    .sub-details summary::before {
      content: "▸";
      font-size: 12px;
      color: var(--text-muted);
    }

    .sub-details[open] summary::before {
      content: "▾";
    }

    .sub-details-body {
      margin-top: 8px;
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* d) AI Advisory (Llama) */
    .advisory-card {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .advisory-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .source-chip-value {
      font-family: var(--font-ui);
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      padding: 2px 8px;
      border-radius: 999px;
      max-width: 210px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .advisory-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .advisory-label {
      display: inline-flex;
      align-items: center;
      justify-content: flex-start;
      gap: 8px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }

    .advisory-body {
      font-size: 13px;
      line-height: 1.5;
      color: var(--text-main);
      word-break: break-word;
    }

    .advisory-spinner {
      display: none;
      width: 13px;
      height: 13px;
      border: 2px solid rgba(45, 212, 191, 0.25);
      border-top-color: var(--accent-teal);
      border-radius: 50%;
      flex-shrink: 0;
    }

    .advisory-spinner.active {
      display: inline-block;
      animation: advisorySpin 0.75s linear infinite;
    }

    @keyframes advisorySpin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    /* e & f) Accordions (details without heading inside summary) */
    .accordion-card {
      padding: 0;
      overflow: hidden;
    }

    .accordion-card summary {
      padding: 12px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      cursor: pointer;
      list-style: none;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
      user-select: none;
    }

    .accordion-card summary::-webkit-details-marker {
      display: none;
    }

    .accordion-card summary:hover {
      background: var(--bg-elevated);
    }

    .accordion-summary-right {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
    }

    .perf-summary-mono {
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      color: var(--text-muted);
    }

    .accordion-chevron {
      width: 16px;
      height: 16px;
      stroke: var(--text-muted);
      fill: none;
      stroke-width: 2;
      flex-shrink: 0;
    }

    .accordion-card[open] .accordion-chevron {
      transform: rotate(180deg);
    }

    .accordion-body {
      padding: 0 16px 14px 16px;
      border-top: 1px solid var(--border-subtle);
      padding-top: 12px;
    }

    /* Unified Defect Group Badges (No rainbow colors) */
    .defect-groups-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .group-badge {
      font-family: var(--font-ui);
      font-size: 12px;
      font-weight: 500;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-base);
      color: var(--text-muted);
    }

    .group-badge.active,
    .group-badge.active-teal {
      color: var(--accent-teal);
      border-color: rgba(45, 212, 191, 0.55);
      background: rgba(45, 212, 191, 0.10);
      font-weight: 600;
    }

    .group-badge.active-med {
      color: var(--status-med);
      border-color: rgba(245, 158, 11, 0.55);
      background: rgba(245, 158, 11, 0.10);
      font-weight: 600;
    }

    .group-badge.active-high {
      color: #F87171;
      border-color: rgba(239, 68, 68, 0.55);
      background: rgba(239, 68, 68, 0.10);
      font-weight: 600;
    }

    .defect-list-line {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border-subtle);
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.45;
      word-break: break-word;
    }

    /* Performance Table */
    .perf-table {
      width: 100%;
      border-collapse: collapse;
    }

    .perf-table th,
    .perf-table td {
      padding: 6px 4px;
      border-bottom: 1px solid var(--border-subtle);
      font-size: 12px;
    }

    .perf-table tr:last-child th,
    .perf-table tr:last-child td {
      border-bottom: none;
    }

    .perf-table th {
      text-align: left;
      font-weight: 500;
      color: var(--text-muted);
    }

    .perf-table td {
      text-align: right;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      font-weight: 600;
      color: var(--text-main);
    }

    /* g) Settings Card */
    .settings-card {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .settings-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }

    .register-good-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .compact-select {
      flex: 1;
      min-width: 0;
      height: 36px;
      background: var(--bg-base);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      font-family: var(--font-ui);
      font-size: 13px;
      padding: 0 10px;
      border-radius: var(--radius-control);
      cursor: pointer;
    }

    .compact-btn {
      height: 36px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      font-family: var(--font-ui);
      font-size: 13px;
      font-weight: 600;
      padding: 0 12px;
      border-radius: var(--radius-control);
      cursor: pointer;
      white-space: nowrap;
    }

    .compact-btn:hover {
      border-color: var(--accent-teal);
      color: var(--accent-teal);
    }

    /* -------------------------------------------------------------
       4. CSV LOGBOOK WORKSPACE
       ------------------------------------------------------------- */
    .logbook-workspace {
      display: none;
      flex: 1;
      flex-direction: column;
      background: var(--bg-base);
      padding: 16px;
      overflow: hidden;
      height: calc(100vh - 56px);
    }

    .logbook-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 12px;
    }

    .logbook-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-main);
    }

    .logbook-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .log-filter-select {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      font-family: var(--font-ui);
      font-size: 13px;
      font-weight: 500;
      padding: 8px 12px;
      border-radius: var(--radius-control);
      cursor: pointer;
    }

    .log-action-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      font-family: var(--font-ui);
      font-size: 13px;
      font-weight: 500;
      padding: 8px 14px;
      border-radius: var(--radius-control);
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .log-action-btn:hover {
      background: var(--bg-elevated);
      border-color: var(--accent-teal);
    }

    .log-table-wrapper {
      flex: 1;
      overflow: auto;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-card);
      background: var(--bg-card);
      box-shadow: var(--shadow-card);
    }

    .log-table {
      width: 100%;
      min-width: 1840px;
      border-collapse: collapse;
      text-align: left;
    }

    .log-col-cause,
    .log-col-action {
      min-width: 390px;
      max-width: 540px;
      line-height: 1.5;
      white-space: normal;
      word-break: break-word;
    }

    .log-col-timings {
      min-width: 120px;
    }

    .log-table th {
      position: sticky;
      top: 0;
      background: var(--bg-elevated);
      padding: 10px 12px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-subtle);
      z-index: 10;
      white-space: nowrap;
    }

    .log-table td {
      padding: 10px 12px;
      font-size: 13px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-main);
      vertical-align: top;
    }

    .log-table tr:hover {
      background: rgba(30, 38, 49, 0.55);
    }

    .mono-cell {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }

    /* Hidden Bottom Telemetry Bar (IDs preserved in DOM) */
    .app-bottom-bar {
      display: none;
    }

    /* -------------------------------------------------------------
       5. REDUCED MOTION & RESPONSIVE BREAKPOINTS
       ------------------------------------------------------------- */
    @media (prefers-reduced-motion: no-preference) {
      .result-panel {
        scroll-behavior: smooth;
      }
      .segment-btn,
      .tool-btn,
      .compact-btn,
      .log-action-btn,
      .accordion-card summary,
      .accordion-chevron,
      .defect-accent-bar,
      .group-badge,
      .anomaly-gauge-fill {
        transition: background-color 170ms ease, border-color 170ms ease, color 170ms ease, width 170ms ease, transform 170ms ease, opacity 170ms ease;
      }
      .scada-toast {
        transition: opacity 170ms ease, transform 170ms ease;
      }
      .advisory-spinner.active {
        animation: advisorySpin 0.75s linear infinite;
      }
    }

    @media (max-width: 1000px) {
      html, body {
        height: auto;
        min-height: 100vh;
        overflow-x: hidden;
        overflow-y: auto;
      }
      .app-header {
        flex-wrap: wrap;
        height: auto;
        min-height: 56px;
        padding: 8px 16px;
        gap: 8px;
      }
      .main-workspace {
        flex-direction: column;
        height: auto;
        overflow: visible;
      }
      .viewport-area {
        flex: none;
        width: 100%;
        height: 54vh;
        min-height: 360px;
      }
      .result-panel {
        width: 100%;
        max-width: none;
        min-width: 0;
        border-left: none;
        border-top: 1px solid var(--border-subtle);
        overflow-y: visible;
      }
      .verdict-card {
        position: static;
      }
    }
  </style>
</head>
<body>

  <!-- 1. TOP BAR -->
  <header class="app-header">
    <div class="brand-section">
      <span class="brand-title">Bharat AI-SoC PS5 · Unified Metal QC</span>
    </div>

    <!-- Single Segmented Navigation Menu: Demo | Image | Video | Live | Log -->
    <nav class="mode-segmented-control" role="tablist" aria-label="Inspection modes">
      <button class="segment-btn active" onclick="setMode('DEMO')" id="tab-demo" role="tab" aria-selected="true">
        <svg class="segment-icon" aria-hidden="true" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span>Demo</span>
      </button>
      <button class="segment-btn" onclick="setMode('IMAGE')" id="tab-image" role="tab" aria-selected="false">
        <svg class="segment-icon" aria-hidden="true" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
        <span>Image</span>
      </button>
      <button class="segment-btn" onclick="setMode('VIDEO')" id="tab-video" role="tab" aria-selected="false">
        <svg class="segment-icon" aria-hidden="true" viewBox="0 0 24 24"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
        <span>Video</span>
      </button>
      <button class="segment-btn" onclick="setMode('CAMERA')" id="tab-cam" role="tab" aria-selected="false">
        <svg class="segment-icon" aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
        <span>Live</span>
      </button>
      <button class="segment-btn" onclick="setMode('LOG')" id="tab-log" role="tab" aria-selected="false">
        <svg class="segment-icon" aria-hidden="true" viewBox="0 0 24 24"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
        <span>Log</span>
      </button>
    </nav>

    <!-- Compact On-Device Status Pill -->
    <div class="status-chip" id="system-status-chip">
      <span class="status-indicator-dot" aria-hidden="true"></span>
      <span id="status-chip-text">On-device · ARM64</span>
    </div>
  </header>

  <!-- 2. MAIN WORKSPACE -->
  <main class="main-workspace" id="viewport-workspace">
    <!-- Left (~65%): Viewport Area -->
    <section class="viewport-area" id="viewport-container">
      <img src="/api/frame.jpg" class="viewport-canvas-img" id="live-stream-img" alt="Unified Multi-Metal Inspection Feed">

      <!-- Top-Left Overlay Chip: Source Tag, Playlist Counter & File Selector -->
      <div class="corner-chip-left">
        <span class="counter-pill" id="playlist-counter-chip">1/32</span>
        <span id="active-source-tag">Demo · Conveyor</span>
        <select class="file-select-dropdown" id="file-selector" onchange="onFileSelected(this.value)" style="display:none;" aria-label="Select inspection media"></select>
      </div>

      <!-- Top-Right Overlay Chip: Temporal Lock Pill -->
      <div class="corner-chip-right" title="Temporal lock: verifies 4 consecutive frames before triggering AI advisory">
        <span class="lock-pill lock-pill-scan" id="lock-chip-label">Scanning…</span>
        <div class="hidden-compat" id="lock-bars">
          <div class="lock-bar" id="lock-bar-1"></div>
          <div class="lock-bar" id="lock-bar-2"></div>
          <div class="lock-bar" id="lock-bar-3"></div>
          <div class="lock-bar" id="lock-bar-4"></div>
        </div>
        <span class="hidden-compat" id="lock-ratio-text">0/4</span>
      </div>

      <!-- Non-blocking Toast Banner -->
      <div class="scada-toast" id="scada-toast" role="status" aria-live="polite"></div>

      <!-- Slim Bottom Toolbar -->
      <div class="floating-toolbar" role="toolbar" focusgroup="toolbar wrap" aria-label="Inspection controls">
        <button class="tool-btn" onclick="prevItem()" id="btn-prev" title="Previous media (P)">
          <svg aria-hidden="true" viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"/></svg>
          <span>Prev</span>
        </button>
        <button class="tool-btn" onclick="togglePause()" id="btn-pause" title="Pause or resume inspection (Space)">
          <svg aria-hidden="true" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
          <span id="btn-pause-text">Pause</span>
        </button>
        <button class="tool-btn" onclick="nextItem()" id="btn-next" title="Next media (N)">
          <svg aria-hidden="true" viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"/></svg>
          <span>Next</span>
        </button>
        <button class="tool-btn" onclick="saveFrame()" id="btn-save" title="Save current frame to disk (S)" style="display:none;">
          <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
          <span>Save</span>
        </button>
        <button class="tool-btn" onclick="document.getElementById('upload-input').click()" id="btn-upload" title="Upload inspection image or video (O)" style="display:none;">
          <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          <span>Upload</span>
        </button>
        <button class="tool-btn tool-btn-accent" onclick="registerGoodCurrent()" id="btn-register-good" title="Register active frame as good PatchCore reference (G)" style="display:none;">
          <svg aria-hidden="true" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>
          <span>Register Good</span>
        </button>
        <input type="file" id="upload-input" style="display:none;" onchange="uploadFile(this.files[0]); this.value='';" accept="image/*,video/*">
        <input type="file" id="register-good-input" style="display:none;" onchange="registerGoodUpload(this.files[0]); this.value='';" accept="image/*">
      </div>

      <!-- Empty Media / Drag & Drop Upload Zone -->
      <div class="upload-dropzone" id="upload-dropzone" onclick="document.getElementById('upload-input').click()">
        <svg aria-hidden="true" style="width:36px;height:36px;stroke:var(--text-muted);fill:none;stroke-width:1.8;" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <div class="upload-zone-title">Upload multi-metal inspection media</div>
        <div class="upload-zone-sub">Drop steel, aluminium, weld, casting, or machined part image or video here, or click to browse</div>
      </div>
    </section>

    <!-- Right (~35%): Result Panel -->
    <aside class="result-panel">
      <!-- a) Sticky Verdict Card -->
      <div class="panel-card verdict-card">
        <div class="verdict-top-row">
          <span class="verdict-badge badge-pass" id="hud-verdict-badge">PASS</span>
          <span class="action-state-pill state-pass" id="inspection-state-text">Ready</span>
        </div>

        <div class="defect-header-container">
          <div class="defect-accent-bar" id="defect-accent-bar"></div>
          <h1 class="defect-name-text" id="hud-defect-name">No defect detected</h1>
        </div>

        <div class="verdict-meta-row">
          <span class="severity-badge badge-pass" id="hud-severity-badge">None</span>
          <span class="family-label-pill" id="hud-family-badge">Steel sheet · 100%</span>
          <span id="hud-backend-badge" style="display:none;">Backend: onnxruntime_arm64</span>
        </div>
      </div>

      <!-- b) Key Metrics: 3 stat tiles in one row -->
      <div class="metrics-grid">
        <div class="stat-tile">
          <div class="metric-label">Confidence</div>
          <div class="metric-value" id="hud-conf-val">--</div>
        </div>
        <div class="stat-tile">
          <div class="metric-label">Defect area</div>
          <div class="metric-value" id="hud-area-val">0.0%</div>
        </div>
        <div class="stat-tile">
          <div class="metric-label">Defects found</div>
          <div class="metric-value" id="hud-boxes-val">0</div>
        </div>
      </div>

      <!-- c) Anomaly Check (PatchCore) -->
      <div class="panel-card" style="display:none;">
        <div class="card-header-row">
          <span class="card-title">Anomaly check</span>
          <span class="severity-badge badge-pass" id="hud-anomaly-status">Normal</span>
        </div>

        <div class="anomaly-gauge-wrap">
          <div class="anomaly-gauge-track">
            <div class="anomaly-gauge-fill fill-pass" id="hud-anomaly-bar-fill"></div>
          </div>
          <div class="anomaly-thresh-marker" id="hud-anomaly-thresh-marker" style="left: 55%;"></div>
        </div>

        <div class="anomaly-readout">
          Score <span class="mono-num" id="hud-anomaly-score">0.000</span> vs threshold <span class="mono-num" id="hud-anomaly-thresh">0.550</span>
        </div>

        <details class="sub-details">
          <summary>Details</summary>
          <div class="sub-details-body">
            <span>Bank size</span>
            <span class="mono-num" id="hud-anomaly-bank">0 / 20</span>
          </div>
        </details>
      </div>

      <!-- d) AI Advisory (Llama) -->
      <div class="panel-card advisory-card">
        <div class="advisory-header">
          <span class="card-title">AI advisory</span>
          <span class="source-chip-value" id="hud-source-tag">Llama 3.2 1B</span>
        </div>

        <div class="advisory-group" id="group-cause">
          <div class="advisory-label">
            <span>Possible cause</span>
            <span class="advisory-spinner" id="spinner-cause" aria-hidden="true"></span>
          </div>
          <div class="advisory-body" id="hud-cause-text">Waiting for Llama 3.2 1B...</div>
        </div>

        <div class="advisory-group" id="group-action">
          <div class="advisory-label">
            <span>Recommended action</span>
            <span class="advisory-spinner" id="spinner-action" aria-hidden="true"></span>
          </div>
          <div class="advisory-body" id="hud-action-text">Waiting for Llama 3.2 1B...</div>
        </div>

        <div class="advisory-group" id="group-narrative" style="display:none;">
          <div class="advisory-label">
            <span>Detailed narrative</span>
          </div>
          <div class="advisory-body" id="hud-narrative-text"></div>
        </div>
      </div>

      <!-- e) Defect Types -->
      <details class="panel-card accordion-card" style="display:none;">
        <summary>
          <span>Defect types</span>
          <span class="accordion-summary-right">
            <span id="hud-active-groups-count">0 active</span>
            <svg class="accordion-chevron" aria-hidden="true" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
          </span>
        </summary>
        <div class="accordion-body">
          <div class="defect-groups-grid" id="defect-groups-grid">
            <span class="group-badge" id="grp-badge-scratch" data-group="scratch">Scratch</span>
            <span class="group-badge" id="grp-badge-pit_void" data-group="pit_void">Pit void</span>
            <span class="group-badge" id="grp-badge-crack" data-group="crack">Crack</span>
            <span class="group-badge" id="grp-badge-stain_oxide" data-group="stain_oxide">Stain oxide</span>
            <span class="group-badge" id="grp-badge-inclusion_slag" data-group="inclusion_slag">Inclusion slag</span>
            <span class="group-badge" id="grp-badge-weld_porosity" data-group="weld_porosity">Weld porosity</span>
            <span class="group-badge" id="grp-badge-deformation" data-group="deformation">Deformation</span>
            <span class="group-badge" id="grp-badge-machining_tear" data-group="machining_tear">Machining tear</span>
          </div>
          <div class="defect-list-line" id="hud-defect-details" style="display:none;"></div>
        </div>
      </details>

      <!-- f) Performance (Collapsed by default, no heading inside summary) -->
      <details class="panel-card accordion-card">
        <summary>
          <span>Performance</span>
          <span class="accordion-summary-right">
            <span class="perf-summary-mono" id="perf-summary-pill">0 ms · 0.0 FPS · 0 tok/s</span>
            <svg class="accordion-chevron" aria-hidden="true" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
          </span>
        </summary>
        <div class="accordion-body">
          <table class="perf-table">
            <tbody>
              <tr>
                <th>Vision pre-process</th>
                <td id="hud-time-pre">0.0 ms</td>
              </tr>
              <tr>
                <th>Vision inference</th>
                <td id="hud-time-infer">0.0 ms</td>
              </tr>
              <tr>
                <th>Vision post-process</th>
                <td id="hud-time-post">0.0 ms</td>
              </tr>
              <tr>
                <th>Vision total</th>
                <td id="hud-time-total">0.0 ms</td>
              </tr>
              <tr>
                <th>LLM TTFT</th>
                <td id="hud-llm-ttft">0 ms</td>
              </tr>
              <tr>
                <th>LLM throughput</th>
                <td id="hud-llm-tps">0.0 tok/s</td>
              </tr>
              <tr>
                <th>LLM total</th>
                <td id="hud-llm-total">0 ms</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>

      <!-- g) Settings -->
      <div class="panel-card settings-card" style="display:none;">
        <label class="settings-label" for="register-family-select">Part family</label>
        <div class="register-good-controls">
          <select id="register-family-select" class="compact-select" title="Optional part family override for registering good reference">
            <option value="">Auto (current)</option>
            <option value="steel_sheet">Steel sheet</option>
            <option value="aluminium">Aluminium</option>
            <option value="weld">Weld</option>
            <option value="casting">Casting</option>
            <option value="machined_part">Machined part</option>
          </select>
          <button type="button" class="compact-btn" onclick="document.getElementById('register-good-input').click()" title="Upload a good reference image">Upload Good</button>
        </div>
      </div>
    </aside>
  </main>

  <!-- 4. CSV LOGBOOK WORKSPACE -->
  <section class="logbook-workspace" id="logbook-workspace">
    <div class="logbook-toolbar">
      <div class="logbook-title">Inspection logbook</div>
      <div class="logbook-controls">
        <select class="log-filter-select" id="log-filter-select" onchange="applyLogFilter()" aria-label="Filter logs by severity">
          <option value="ALL">All severities</option>
          <option value="CRITICAL">Major / Critical</option>
          <option value="MODERATE">Moderate</option>
          <option value="MINOR">Minor</option>
          <option value="NONE">None / Pass</option>
        </select>
        <a href="/api/download_csv" class="log-action-btn" title="Download inspection CSV">
          <svg aria-hidden="true" style="width:14px;height:14px;stroke:currentColor;fill:none;stroke-width:2;" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          <span>Download CSV</span>
        </a>
        <button class="log-action-btn" onclick="clearLogs()" title="Clear logbook records">
          <svg aria-hidden="true" style="width:14px;height:14px;stroke:currentColor;fill:none;stroke-width:2;" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
          <span>Clear logbook</span>
        </button>
      </div>
    </div>

    <div class="log-table-wrapper">
      <table class="log-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Part family</th>
            <th>Defect</th>
            <th>Group</th>
            <th>Conf / Score</th>
            <th>Anomaly</th>
            <th>Area %</th>
            <th>Boxes</th>
            <th>Severity</th>
            <th>Verdict</th>
            <th class="log-col-cause">Possible cause</th>
            <th class="log-col-action">Recommended action</th>
            <th class="log-col-timings">Timings (ms)</th>
            <th>Advisory source</th>
          </tr>
        </thead>
        <tbody id="logbook-tbody"></tbody>
      </table>
    </div>
  </section>

  <!-- 5. COMPATIBILITY TELEMETRY BAR (Hidden in DOM) -->
  <footer class="app-bottom-bar" aria-hidden="true">
    <span id="bar-cv-telemetry">0.0 ms (0.0 FPS)</span>
    <span id="bar-backend-telemetry">onnxruntime_arm64</span>
    <span id="bar-engine-telemetry">Llama 3.2 1B INT4 (ExecuTorch)</span>
    <span id="bar-latency-telemetry">Ready</span>
  </footer>

  <!-- -------------------------------------------------------------
       CLIENT LOGIC & EVENT CONTROLLERS
       ------------------------------------------------------------- -->
  <script>
    const UNIFIED_GROUPS = [
      { id: "scratch", label: "Scratch" },
      { id: "pit_void", label: "Pit void" },
      { id: "crack", label: "Crack" },
      { id: "stain_oxide", label: "Stain oxide" },
      { id: "inclusion_slag", label: "Inclusion slag" },
      { id: "weld_porosity", label: "Weld porosity" },
      { id: "deformation", label: "Deformation" },
      { id: "machining_tear", label: "Machining tear" }
    ];

    const FAMILY_LABELS = {
      steel_sheet: "Steel sheet",
      aluminium: "Aluminium",
      weld: "Weld",
      casting: "Casting",
      machined_part: "Machined part",
      not_metal: "Not metal"
    };

    let currentMode = "DEMO";
    let hasSyncedInitialMode = false;
    let modeSwitchPending = false;
    let pauseTogglePending = false;
    let statePollEpoch = 0;
    let pauseLockUntil = 0;
    let lastPlaylistSignature = "";
    let isPaused = false;
    let isFetchingFrame = false;
    let isFetchingState = false;
    let lastFrameSeq = 0;
    let currentFrameBlobUrl = null;
    let frameAbortCtrl = null;
    let pendingFrameForce = false;
    let fastPollUntil = 0;
    let currentSampleSettled = false;
    let framePollTimer = null;
    let allLogs = [];
    let toastTimer = null;
    const streamImg = document.getElementById('live-stream-img');
    let pendingSlide = null;
    function prepareSlide(direction) {
      pendingSlide = { direction, sequence: lastFrameSeq, expires: Date.now() + 4000 };
    }
    function slideToNewFrame(newSeq) {
      const slide = pendingSlide;
      if (!slide || newSeq <= slide.sequence) return;
      pendingSlide = null;
      if (Date.now() > slide.expires || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      const outgoing = streamImg.cloneNode(false);
      outgoing.removeAttribute('id');
      outgoing.setAttribute('aria-hidden', 'true');
      Object.assign(outgoing.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
      streamImg.parentElement.appendChild(outgoing);
      const options = { duration: 300, easing: 'ease-out' };
      outgoing.animate([{ transform: 'translateX(0)' }, { transform: `translateX(${-slide.direction * 100}%)` }], options);
      streamImg.animate([{ transform: `translateX(${slide.direction * 100}%)` }, { transform: 'translateX(0)' }], options);
      setTimeout(() => outgoing.remove(), 310);
    }

    function escapeHtml(str) {
      if (str === null || str === undefined) return "";
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function toSentenceCase(str) {
      if (!str) return "";
      const cleaned = String(str).replace(/_/g, " ").trim().replace(/\s+/g, " ");
      if (!cleaned) return "";
      const lower = cleaned.toLowerCase();
      return (lower.charAt(0).toUpperCase() + lower.slice(1))
        .replace(/\bpatchcore\b/g, "PatchCore")
        .replace(/\bllama\b/g, "Llama");
    }

    function formatFamilyName(fam) {
      const key = String(fam || "steel_sheet").toLowerCase();
      return FAMILY_LABELS[key] || toSentenceCase(key);
    }

    function showToast(msg, isError = false) {
      const el = document.getElementById('scada-toast');
      if (!el) return;
      el.innerText = msg;
      el.className = 'scada-toast visible' + (isError ? ' toast-error' : '');
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => {
        el.classList.remove('visible');
      }, 4200);
    }

    function normalizeSeverityBucket(sev) {
      const s = String(sev || "NONE").toUpperCase();
      if (s === "CRITICAL" || s === "HIGH" || s === "MAJOR") return "CRITICAL";
      if (s === "MODERATE" || s === "MEDIUM") return "MODERATE";
      if (s === "MINOR" || s === "LOW") return "MINOR";
      return "NONE";
    }

    function formatSeverityLabel(sev) {
      const b = normalizeSeverityBucket(sev);
      if (b === "CRITICAL") return "Major";
      if (b === "MODERATE") return "Moderate";
      if (b === "MINOR") return "Minor";
      return "None";
    }

    function severityBadgeClass(sev) {
      const b = normalizeSeverityBucket(sev);
      if (b === "CRITICAL") return "badge-high";
      if (b === "MODERATE" || b === "MINOR") return "badge-med";
      return "badge-pass";
    }

    function accentColorForSeverity(sev) {
      const b = normalizeSeverityBucket(sev);
      if (b === "CRITICAL") return "var(--status-high)";
      if (b === "MODERATE" || b === "MINOR") return "var(--status-med)";
      return "var(--status-pass)";
    }

    // 1. Smart Frame Polling Loop (?since=<frame_seq> returns 204 when unchanged -> 0 bytes transferred)
    function fetchLiveFrame(force = false) {
      if (currentMode === 'LOG') return;
      if (force) {
        lastFrameSeq = 0;
        currentSampleSettled = false;
        fastPollUntil = Date.now() + 2500;
        if (isFetchingFrame) {
          pendingFrameForce = true;
          if (frameAbortCtrl) {
            try { frameAbortCtrl.abort(); } catch (_) {}
          }
          return;
        }
      }
      if (isFetchingFrame) return;
      isFetchingFrame = true;
      frameAbortCtrl = new AbortController();
      const requestController = frameAbortCtrl;
      const requestTimeout = setTimeout(() => requestController.abort(), 3000);
      const reqSeq = force ? 0 : lastFrameSeq;
      const reqEpoch = statePollEpoch;

      fetch(`/api/frame.jpg?since=${reqSeq}&t=${Date.now()}`, {
        cache: 'no-store',
        signal: frameAbortCtrl.signal
      })
        .then(res => {
          if (res.status === 204 || !res.ok) {
            const hdr = Number(res.headers.get('X-Frame-Seq') || 0);
            if (hdr > 0 && reqSeq === hdr) {
              lastFrameSeq = hdr;
            }
            return null;
          }
          const newSeq = Number(res.headers.get('X-Frame-Seq') || 0);
          return res.blob().then(blob => ({ blob, newSeq }));
        })
        .then(payload => {
          if (!payload || !payload.blob || reqEpoch !== statePollEpoch) return;
          const nextUrl = URL.createObjectURL(payload.blob);
          const prevUrl = currentFrameBlobUrl;
          const prevSeq = lastFrameSeq;
          currentFrameBlobUrl = nextUrl;
          slideToNewFrame(payload.newSeq);
          streamImg.src = nextUrl;
          if (payload.newSeq > 0) {
            lastFrameSeq = payload.newSeq;
            if (payload.newSeq !== prevSeq && currentMode !== 'IMAGE' && currentMode !== 'LOG') {
              updateState(false);
            }
          }
          if (prevUrl) {
            setTimeout(() => URL.revokeObjectURL(prevUrl), 500);
          }
        })
        .catch(() => {})
        .finally(() => {
          clearTimeout(requestTimeout);
          isFetchingFrame = false;
          frameAbortCtrl = null;
          if (pendingFrameForce) {
            pendingFrameForce = false;
            fetchLiveFrame(true);
          }
        });
    }

    function scheduleNextFramePoll() {
      let delay = 40;
      if (currentMode === 'LOG' || (isPaused && currentMode !== 'IMAGE')) {
        delay = 350;
      } else if (currentMode === 'IMAGE') {
        const fastActive = !currentSampleSettled || Date.now() < fastPollUntil;
        delay = fastActive ? 50 : 300;
      }
      framePollTimer = setTimeout(() => {
        fetchLiveFrame(false);
        scheduleNextFramePoll();
      }, delay);
    }
    scheduleNextFramePoll();

    function clearOldAdvisoryDisplay() {
      const spinCause = document.getElementById('spinner-cause');
      const spinAction = document.getElementById('spinner-action');
      if (spinCause) spinCause.classList.add('active');
      if (spinAction) spinAction.classList.add('active');
      document.getElementById('hud-cause-text').innerText = "Generating with Llama 3.2 1B...";
      document.getElementById('hud-action-text').innerText = "Generating with Llama 3.2 1B...";
      document.getElementById('hud-source-tag').innerText = "Llama 3.2 1B";
      const stateText = document.getElementById('inspection-state-text');
      if (stateText) {
        stateText.innerText = "Scanning";
        stateText.className = "action-state-pill";
      }
    }

    // 2. Single-Menu Mode Switching (DEMO | IMAGE | VIDEO | CAMERA | LOG)
    function applyModeUI(mode) {
      currentMode = mode;
      document.querySelectorAll('.segment-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
      });
      const btnId = (mode === 'CAMERA') ? 'tab-cam' : `tab-${mode.toLowerCase()}`;
      const activeBtn = document.getElementById(btnId);
      if (activeBtn) {
        activeBtn.classList.add('active');
        activeBtn.setAttribute('aria-selected', 'true');
      }

      const viewportWorkspace = document.getElementById('viewport-workspace');
      const logbookWorkspace = document.getElementById('logbook-workspace');
      const fileSelector = document.getElementById('file-selector');
      const activeSourceTag = document.getElementById('active-source-tag');
      const btnPause = document.getElementById('btn-pause');

      if (btnPause) {
        const showPause = (mode === 'DEMO' || mode === 'VIDEO' || mode === 'CAMERA');
        btnPause.style.display = showPause ? 'inline-flex' : 'none';
      }

      if (mode === 'LOG') {
        viewportWorkspace.style.display = 'none';
        logbookWorkspace.style.display = 'flex';
        fileSelector.style.display = 'none';
        if (activeSourceTag) activeSourceTag.style.display = 'inline';
        loadLogs();
      } else {
        viewportWorkspace.style.display = 'flex';
        logbookWorkspace.style.display = 'none';
        const showDropdown = (mode === 'IMAGE' || (mode === 'VIDEO' && fileSelector.options.length > 0));
        fileSelector.style.display = showDropdown ? 'inline-block' : 'none';
        if (activeSourceTag) {
          activeSourceTag.style.display = showDropdown ? 'none' : 'inline';
        }
      }
    }

    function setMode(mode) {
      pendingSlide = null;
      hasSyncedInitialMode = true;
      modeSwitchPending = true;
      statePollEpoch++;
      pauseLockUntil = Date.now() + 800;
      isPaused = false;
      updatePauseButtonState(false);
      clearOldAdvisoryDisplay();
      applyModeUI(mode);
      fetch(`/api/switch_mode?mode=${encodeURIComponent(mode)}`, { cache: 'no-store' })
        .then(r => r.json())
        .catch(() => {})
        .finally(() => {
          modeSwitchPending = false;
          if (mode !== 'LOG') {
            fetchLiveFrame(true);
          }
          updateState();
        });
    }

    // 3. Navigation Controls
    function prevItem() {
      prepareSlide(-1);
      statePollEpoch++;
      clearOldAdvisoryDisplay();
      fastPollUntil = Date.now() + 2500;
      fetch('/api/prev', { cache: 'no-store' }).then(() => {
        fetchLiveFrame(true);
        updateState();
      });
    }

    function nextItem() {
      prepareSlide(1);
      statePollEpoch++;
      clearOldAdvisoryDisplay();
      fastPollUntil = Date.now() + 2500;
      fetch('/api/next', { cache: 'no-store' }).then(() => {
        fetchLiveFrame(true);
        updateState();
      });
    }

    function togglePause() {
      if (currentMode === 'IMAGE' || currentMode === 'LOG') return;
      statePollEpoch++;
      pauseTogglePending = true;
      pauseLockUntil = Date.now() + 1500;
      isPaused = !isPaused;
      updatePauseButtonState(isPaused);
      fetch('/api/toggle_pause', { cache: 'no-store' })
        .then(r => r.json())
        .then(data => {
          if (typeof data.is_paused === 'boolean') {
            isPaused = data.is_paused;
            updatePauseButtonState(isPaused);
          }
          pauseLockUntil = Date.now() + 800;
        })
        .catch(() => {})
        .finally(() => {
          pauseTogglePending = false;
        });
    }

    function updatePauseButtonState(paused) {
      const btn = document.getElementById('btn-pause');
      const text = document.getElementById('btn-pause-text');
      if (paused) {
        btn.classList.add('active-pause');
        text.innerText = "Resume";
      } else {
        btn.classList.remove('active-pause');
        text.innerText = "Pause";
      }
    }

    function onFileSelected(path) {
      if (!path) return;
      statePollEpoch++;
      clearOldAdvisoryDisplay();
      const targetMode = (currentMode === 'VIDEO') ? 'VIDEO' : 'IMAGE';
      if (currentMode !== targetMode) {
        applyModeUI(targetMode);
      }
      modeSwitchPending = true;
      fastPollUntil = Date.now() + 2500;
      fetch(`/api/select_item?path=${encodeURIComponent(path)}&mode=${encodeURIComponent(targetMode)}`, { cache: 'no-store' })
        .then(() => {
          fetchLiveFrame(true);
        })
        .catch(() => {})
        .finally(() => {
          modeSwitchPending = false;
          updateState();
        });
    }

    // 4. File Upload
    function uploadFile(file) {
      if (!file) return;
      const formData = new FormData();
      formData.append('file', file);
      fetch('/api/upload', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(res => {
          if (res.success) {
            setMode(res.media_type || 'IMAGE');
            updateState();
            showToast(`Uploaded ${res.name || 'media'} (${res.media_type || 'IMAGE'})`);
          } else {
            showToast(`Upload failed: ${res.error || 'unknown error'}`, true);
          }
        })
        .catch(err => showToast(`Upload error: ${err}`, true));
    }

    // 5. PatchCore Register Good (Current Active Frame OR Uploaded Good Image)
    function registerGoodCurrent() {
      const familyOverride = document.getElementById('register-family-select').value;
      const url = familyOverride
        ? `/api/register_good?family=${encodeURIComponent(familyOverride)}`
        : '/api/register_good';

      fetch(url, { method: 'POST' })
        .then(r => r.json())
        .then(res => {
          if (res.success) {
            showToast(`Registered good frame for ${formatFamilyName(res.family)} (${res.images_saved} saved)`);
            updateState();
          } else {
            showToast(`Register Good failed: ${res.error || 'unknown error'}`, true);
          }
        })
        .catch(err => showToast(`Register Good error: ${err}`, true));
    }

    function registerGoodUpload(file) {
      if (!file) return;
      const familyOverride = document.getElementById('register-family-select').value;
      const formData = new FormData();
      formData.append('file', file);
      if (familyOverride) {
        formData.append('family', familyOverride);
      }
      const url = familyOverride
        ? `/api/register_good?family=${encodeURIComponent(familyOverride)}`
        : '/api/register_good';

      fetch(url, { method: 'POST', body: formData })
        .then(r => r.json())
        .then(res => {
          if (res.success) {
            showToast(`Registered uploaded good image for ${formatFamilyName(res.family)} (${res.images_saved} saved)`);
            updateState();
          } else {
            showToast(`Register Good failed: ${res.error || 'unknown error'}`, true);
          }
        })
        .catch(err => showToast(`Register Good error: ${err}`, true));
    }

    // Drag and drop setup
    const dropzone = document.getElementById('upload-dropzone');
    window.addEventListener('dragenter', e => { e.preventDefault(); dropzone.classList.add('active-drag'); });
    dropzone.addEventListener('dragleave', e => { e.preventDefault(); dropzone.classList.remove('active-drag'); });
    window.addEventListener('dragover', e => { e.preventDefault(); });
    window.addEventListener('drop', e => {
      e.preventDefault();
      dropzone.classList.remove('active-drag');
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });

    // 6. Keyboard Shortcuts (P/Space/N/S/O/G/1-5) + Roving Tabindex Fallback
    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === '1') setMode('DEMO');
      else if (e.key === '2') setMode('IMAGE');
      else if (e.key === '3') setMode('VIDEO');
      else if (e.key === '4') setMode('CAMERA');
      else if (e.key === '5') setMode('LOG');
      else if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); togglePause(); }
      else if (e.key === 'p' || e.key === 'P') prevItem();
      else if (e.key === 'n' || e.key === 'N') nextItem();
      else if (e.key === 'o' || e.key === 'O') document.getElementById('upload-input').click();
      else if (e.key === 's' || e.key === 'S') saveFrame();
      else if (e.key === 'g' || e.key === 'G') registerGoodCurrent();
    });

    (function initToolbarFocusFallback() {
      if ('focusGroup' in HTMLElement.prototype) return;
      const toolbar = document.querySelector('.floating-toolbar[role="toolbar"]');
      if (!toolbar) return;
      const buttons = Array.from(toolbar.querySelectorAll('button')).filter(b => b.style.display !== 'none');
      if (buttons.length === 0) return;

      function setActiveIndex(activeIdx) {
        buttons.forEach((btn, idx) => btn.setAttribute('tabindex', idx === activeIdx ? '0' : '-1'));
      }

      setActiveIndex(0);
      toolbar.addEventListener('focusin', (e) => {
        const idx = buttons.indexOf(e.target);
        if (idx !== -1) setActiveIndex(idx);
      });
      toolbar.addEventListener('keydown', (e) => {
        const currentIdx = buttons.indexOf(document.activeElement);
        if (currentIdx === -1) return;
        let targetIdx = -1;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
          targetIdx = (currentIdx + 1) % buttons.length;
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
          targetIdx = (currentIdx - 1 + buttons.length) % buttons.length;
        } else if (e.key === 'Home') {
          targetIdx = 0;
        } else if (e.key === 'End') {
          targetIdx = buttons.length - 1;
        }
        if (targetIdx !== -1) {
          e.preventDefault();
          setActiveIndex(targetIdx);
          buttons[targetIdx].focus();
        }
      });
    })();

    function saveFrame() {
      fetch('/api/save_frame', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          if (data.success || data.status === 'ok') {
            showToast('Frame saved: ' + (data.filename || data.path || data.saved_path));
          } else {
            showToast('Error saving frame: ' + (data.error || 'unknown'), true);
          }
        })
        .catch(err => showToast('Failed to save frame: ' + err, true));
    }

    // 7. Periodic State Polling (Unified Shared JSON Schema)
    function updateState(force = true) {
      if (!force && isFetchingState) return;
      isFetchingState = true;
      const reqEpoch = statePollEpoch;
      const stateController = new AbortController();
      const stateTimeout = setTimeout(() => stateController.abort(), 3000);
      fetch('/api/state', { cache: 'no-store', signal: stateController.signal }).then(r => r.json()).then(data => {
        if (reqEpoch !== statePollEpoch || modeSwitchPending) {
          return;
        }
        const serverMode = String(data.mode || "DEMO").toUpperCase();
        if (!hasSyncedInitialMode) {
          hasSyncedInitialMode = true;
          if (serverMode && serverMode !== currentMode) {
            applyModeUI(serverMode);
            if (serverMode !== 'LOG') {
              fetchLiveFrame();
            }
          } else {
            applyModeUI(currentMode);
          }
        } else if (serverMode && serverMode !== currentMode) {
          applyModeUI(serverMode);
          if (serverMode !== 'LOG') {
            fetchLiveFrame();
          }
        }

        const rawTag = String(data.active_tag || currentMode);
        const cleanTag = (currentMode === 'DEMO')
          ? (data.demo_auto_paused
              ? 'Demo · Auto-paused (Llama)'
              : (data.is_paused ? 'Demo · Paused' : 'Demo · Conveyor Belt'))
          : rawTag
              .replace(/^IMAGE\s*\[\d+\/\d+\]\s*·\s*/i, 'Image · ')
              .replace(/^VIDEO\s*\[\d+\/\d+\]/i, 'Video')
              .replace(/^LIVE CAM\s*·\s*/i, 'Live · ');
        document.getElementById('active-source-tag').innerText = cleanTag || currentMode;

        // Playlist counter pill — IMAGE and VIDEO only
        const counterChip = document.getElementById('playlist-counter-chip');
        if (counterChip) {
          if ((currentMode === 'IMAGE' || currentMode === 'VIDEO') && Array.isArray(data.playlist) && data.playlist.length > 0) {
            const activeIdx = Math.max(0, Number(data.active_idx || 0));
            counterChip.innerText = `${activeIdx + 1}/${data.playlist.length}`;
            counterChip.style.display = 'inline-flex';
          } else {
            counterChip.style.display = 'none';
          }
        }

        const backend = data.vision_backend || data.vision_hardware || "onnxruntime_arm64";
        document.getElementById('status-chip-text').innerText =
          (backend === 'hailo8l_hef') ? 'On-device · Hailo-8L' : 'On-device · ARM64';
        document.getElementById('hud-backend-badge').innerText = `Backend: ${backend}`;
        document.getElementById('bar-backend-telemetry').innerText = backend;

        // Material family (clean label without redundant percentage)
        const family = data.family || data.stage0_route || "steel_sheet";
        const familyBadge = document.getElementById('hud-family-badge');
        familyBadge.innerText = formatFamilyName(family);
        familyBadge.className = 'family-label-pill' + (family === 'not_metal' ? ' meta-pill-warn' : '');

        // Defect Header (sentence case)
        const rawDefect = data.defect || data.display_title || "NO DEFECT DETECTED";
        const rawDefectUpper = String(rawDefect).toUpperCase();
        document.getElementById('hud-defect-name').innerText = toSentenceCase(rawDefect);

        // Core Metrics (show '--' for confidence when no defect is detected)
        const isCleanOrClear = rawDefectUpper === 'NO DEFECT DETECTED' ||
          rawDefectUpper.includes('CONVEYOR CLEAR') ||
          rawDefectUpper.includes('NON-METAL') ||
          (data.conf || 0) <= 0.001;
        document.getElementById('hud-conf-val').innerText = isCleanOrClear
          ? '--'
          : `${((data.conf || 0) * 100).toFixed(1)}%`;
        document.getElementById('hud-area-val').innerText = `${(data.area || 0).toFixed(1)}%`;
        document.getElementById('hud-boxes-val').innerText = data.boxes || 0;

        // Unified Defect Group Badges (semantic status / teal accent, no rainbow colors)
        const defectsArr = Array.isArray(data.defects) ? data.defects : [];
        const groupCounts = {};
        const groupSevRank = {};
        const sevRankMap = { NONE: 0, MINOR: 1, LOW: 1, MODERATE: 2, MEDIUM: 2, CRITICAL: 3, HIGH: 3, MAJOR: 3 };

        defectsArr.forEach(d => {
          const g = d.group || d.label;
          if (!g) return;
          groupCounts[g] = (groupCounts[g] || 0) + 1;
          const sKey = String(d.severity || data.severity || 'MINOR').toUpperCase();
          const rank = sevRankMap[sKey] || 1;
          if (!groupSevRank[g] || rank > groupSevRank[g]) {
            groupSevRank[g] = rank;
          }
        });

        let activeGroupsCount = 0;
        UNIFIED_GROUPS.forEach(g => {
          const badgeEl = document.getElementById(`grp-badge-${g.id}`);
          if (!badgeEl) return;
          const cnt = groupCounts[g.id] || 0;
          badgeEl.classList.remove('active', 'active-high', 'active-med', 'active-teal');
          if (cnt > 0) {
            activeGroupsCount++;
            badgeEl.classList.add('active');
            const rank = groupSevRank[g.id] || 0;
            if (rank >= 3) badgeEl.classList.add('active-high');
            else if (rank >= 1) badgeEl.classList.add('active-med');
            else badgeEl.classList.add('active-teal');
            badgeEl.innerText = cnt > 1 ? `${g.label} ×${cnt}` : g.label;
          } else {
            badgeEl.innerText = g.label;
          }
        });
        document.getElementById('hud-active-groups-count').innerText = `${activeGroupsCount} active`;

        const detailsEl = document.getElementById('hud-defect-details');
        if (defectsArr.length > 0) {
          const summaryTokens = defectsArr.slice(0, 4).map(d => {
            const lbl = toSentenceCase(d.label || d.group || 'defect');
            const c = ((d.confidence || 0) * 100).toFixed(0);
            const sv = formatSeverityLabel(d.severity || data.severity || 'MINOR');
            return `${lbl} (${c}%, ${sv})`;
          });
          if (defectsArr.length > 4) {
            summaryTokens.push(`+${defectsArr.length - 4} more`);
          }
          detailsEl.innerText = summaryTokens.join(' · ');
          detailsEl.style.display = 'block';
        } else {
          detailsEl.style.display = 'none';
        }

        // PatchCore Anomaly Telemetry + Visual Gauge
        const anom = data.anomaly || {};
        const anomScore = anom.score !== undefined ? Number(anom.score) : Number(data.anomaly_score || 0);
        const anomThresh = anom.threshold !== undefined ? Number(anom.threshold) : 0.55;
        const anomBank = anom.bank_size !== undefined ? Number(anom.bank_size) : 0;
        const anomMax = anom.max_refs !== undefined ? Number(anom.max_refs) : 20;
        const anomFlag = Boolean(anom.is_anomalous);
        const anomRawStatus = String(anom.status || (anomFlag ? "ANOMALY" : (anomBank > 0 ? "NORMAL" : "UNINDEXED"))).toUpperCase();

        document.getElementById('hud-anomaly-score').innerText = anomScore.toFixed(3);
        document.getElementById('hud-anomaly-thresh').innerText = anomThresh.toFixed(3);
        document.getElementById('hud-anomaly-bank').innerText = `${anomBank} / ${anomMax}`;

        const anomBadge = document.getElementById('hud-anomaly-status');
        if (anomRawStatus === 'GATE') {
          anomBadge.innerText = 'Skipped';
          anomBadge.className = 'severity-badge badge-med';
        } else if (anomFlag || anomRawStatus === 'ANOMALY') {
          anomBadge.innerText = 'Anomaly';
          anomBadge.className = 'severity-badge badge-high';
        } else if (anomBank > 0 || anomRawStatus === 'NORMAL') {
          anomBadge.innerText = 'Normal';
          anomBadge.className = 'severity-badge badge-pass';
        } else {
          anomBadge.innerText = 'Unindexed';
          anomBadge.className = 'severity-badge badge-low';
        }

        const gaugeFill = document.getElementById('hud-anomaly-bar-fill');
        const threshMarker = document.getElementById('hud-anomaly-thresh-marker');
        const gaugeMax = Math.max(1.0, anomThresh * 1.4, anomScore * 1.05);
        const fillPct = Math.min(100, Math.max(0, (anomScore / gaugeMax) * 100));
        const threshPct = Math.min(100, Math.max(0, (anomThresh / gaugeMax) * 100));
        if (gaugeFill) {
          gaugeFill.style.width = `${fillPct.toFixed(1)}%`;
          gaugeFill.className = 'anomaly-gauge-fill ' +
            (anomFlag ? 'fill-high' : (anomScore >= anomThresh * 0.85 ? 'fill-med' : 'fill-pass'));
        }
        if (threshMarker) {
          threshMarker.style.left = `${threshPct.toFixed(1)}%`;
        }

        // Pipeline Timings (pre, infer, post, total)
        const tm = data.timings_ms || {};
        const tPre = Number(tm.pre !== undefined ? tm.pre : 0);
        const tInfer = Number(tm.infer !== undefined ? tm.infer : (data.cv_ms || 0));
        const tPost = Number(tm.post !== undefined ? tm.post : 0);
        const tTotal = Number(tm.total !== undefined ? tm.total : (data.cv_ms || (tPre + tInfer + tPost)));

        document.getElementById('hud-time-pre').innerText = `${tPre.toFixed(1)} ms`;
        document.getElementById('hud-time-infer').innerText = `${tInfer.toFixed(1)} ms`;
        document.getElementById('hud-time-post').innerText = `${tPost.toFixed(1)} ms`;
        document.getElementById('hud-time-total').innerText = `${tTotal.toFixed(1)} ms`;

        // LLM Telemetry (ttft_ms, tokens_per_s, total_ms, narrative)
        const llm = data.llm || {};
        const ttftMs = Number(llm.ttft_ms !== undefined ? llm.ttft_ms : 0);
        const tokSpeed = Number(llm.tokens_per_s !== undefined ? llm.tokens_per_s : (data.tok_speed || 0));
        const llmTotalMs = Number(llm.total_ms !== undefined ? llm.total_ms : (data.llm_ms || 0));
        const narrative = llm.narrative || llm.summary || "";

        document.getElementById('hud-llm-ttft').innerText = `${Math.round(ttftMs)} ms`;
        document.getElementById('hud-llm-tps').innerText = `${tokSpeed.toFixed(1)} tok/s`;
        document.getElementById('hud-llm-total').innerText = `${Math.round(llmTotalMs)} ms`;

        const isImageMode = (currentMode === 'IMAGE');
        const fps = (1000 / Math.max(0.1, tTotal)).toFixed(1);
        const perfSummaryPill = document.getElementById('perf-summary-pill');
        if (perfSummaryPill) {
          perfSummaryPill.innerText = isImageMode
            ? `${Math.round(tTotal)} ms · ${Math.round(tokSpeed)} tok/s`
            : `${Math.round(tTotal)} ms · ${fps} FPS · ${Math.round(tokSpeed)} tok/s`;
        }

        // Cause, Action, Narrative & Per-Field Preloader Spinners
        const causeVal = data.cause || "";
        const actionVal = data.action || "";
        const narGroup = document.getElementById('group-narrative');
        const narText = document.getElementById('hud-narrative-text');
        const spinCause = document.getElementById('spinner-cause');
        const spinAction = document.getElementById('spinner-action');
        const isNotMetal = (family === 'not_metal') ||
          (rawDefectUpper.includes("NON-STEEL") || rawDefectUpper.includes("NON-METAL") || rawDefectUpper.includes("NOT METAL"));
        const isStandby = rawDefectUpper === "CAMERA OFFLINE" || rawDefectUpper === "NO MEDIA LOADED" || rawDefectUpper === "AWAITING VIDEO";
        const isAnalyzing = rawDefectUpper === "ANALYZING..." || rawDefectUpper === "ANALYZING" || rawDefectUpper === "SCANNING...";
        const advisoryBusy = Boolean(data.llm_busy || data.advisory_pending);
        currentSampleSettled = (currentMode === 'IMAGE') && !advisoryBusy && !isAnalyzing;
        if (typeof data.frame_seq === 'number' && data.frame_seq > 0 && data.frame_seq !== lastFrameSeq && currentMode !== 'LOG') {
          fetchLiveFrame(false);
        }

        if (advisoryBusy) {
          if (spinCause) spinCause.classList.add('active');
          if (spinAction) spinAction.classList.add('active');
          const isPlaceholder = !causeVal || causeVal.includes("Evaluating image") || causeVal.includes("Analyzing inspection") || causeVal.includes("Executing on-device") || causeVal.includes("Waiting for") || causeVal === "No defect detected";
          const isActPlaceholder = !actionVal || actionVal.includes("Generating root-cause") || actionVal.includes("Waiting for");
          document.getElementById('hud-cause-text').innerText = !isPlaceholder ? causeVal : "Generating with Llama 3.2 1B...";
          document.getElementById('hud-action-text').innerText = !isActPlaceholder ? actionVal : "Generating with Llama 3.2 1B...";
          if (narGroup) narGroup.style.display = 'none';
        } else {
          if (spinCause) spinCause.classList.remove('active');
          if (spinAction) spinAction.classList.remove('active');
          document.getElementById('hud-cause-text').innerText = causeVal || "Waiting for Llama 3.2 1B...";
          document.getElementById('hud-action-text').innerText = actionVal || "Waiting for Llama 3.2 1B...";
          if (narText) narText.innerText = "";
          if (narGroup) narGroup.style.display = 'none';
        }

        const cleanEngineSrc = String(data.engine_source || "Llama 3.2 1B")
          .replace(/\s*\(on-device[^)]*\)/gi, '')
          .trim() || "Llama 3.2 1B";
        document.getElementById('hud-source-tag').innerText = cleanEngineSrc;

        // Hidden Bottom Bar Compatibility Updates
        document.getElementById('bar-cv-telemetry').innerText = isImageMode
          ? `${tTotal.toFixed(1)} ms · ${tPre.toFixed(1)}/${tInfer.toFixed(1)}/${tPost.toFixed(1)}`
          : `${tTotal.toFixed(1)} ms (${fps} FPS) · ${tPre.toFixed(1)}/${tInfer.toFixed(1)}/${tPost.toFixed(1)}`;
        document.getElementById('bar-engine-telemetry').innerText = `Llama 3.2 1B INT4 · TTFT ${Math.round(ttftMs)}ms · ${tokSpeed.toFixed(1)} tok/s`;
        document.getElementById('bar-latency-telemetry').innerText = llmTotalMs > 0 ? `${(llmTotalMs / 1000).toFixed(2)}s` : "Ready";

        // Verdict, Severity & Action Badge Styling
        const sevBadge = document.getElementById('hud-severity-badge');
        const verdictBadge = document.getElementById('hud-verdict-badge');
        const accentBar = document.getElementById('defect-accent-bar');
        const stateText = document.getElementById('inspection-state-text');

        const rawSev = data.severity || "NONE";
        const sevBucket = normalizeSeverityBucket(rawSev);
        const rawVerdict = String(data.verdict || (sevBucket === "CRITICAL" ? "REJECT" : (sevBucket === "NONE" ? "PASS" : "REVIEW"))).toUpperCase();

        sevBadge.className = 'severity-badge';
        verdictBadge.className = 'verdict-badge';
        stateText.className = 'action-state-pill';

        if (isStandby || isAnalyzing) {
          sevBadge.classList.add('badge-low');
          verdictBadge.classList.add('badge-low');
          sevBadge.innerText = "None";
          verdictBadge.innerText = isAnalyzing ? "Scanning" : "Standby";
          accentBar.style.background = "var(--border-subtle)";
          stateText.innerText = isAnalyzing ? "Scanning" : toSentenceCase(rawDefectUpper);
        } else if (isNotMetal) {
          sevBadge.classList.add('badge-med');
          verdictBadge.classList.add('badge-med');
          stateText.classList.add('state-med');
          sevBadge.innerText = "Gate";
          verdictBadge.innerText = "Non-metal";
          accentBar.style.background = "var(--status-med)";
          stateText.innerText = "Needs review";
        } else if (rawVerdict === "PASS" && !anomFlag && sevBucket === "NONE") {
          sevBadge.classList.add('badge-pass');
          verdictBadge.classList.add('badge-pass');
          stateText.classList.add('state-pass');
          sevBadge.innerText = "None";
          verdictBadge.innerText = "PASS";
          accentBar.style.background = "var(--status-pass)";
          stateText.innerText = "Ready";
        } else if (rawVerdict === "REJECT" || sevBucket === "CRITICAL") {
          sevBadge.classList.add('badge-high');
          verdictBadge.classList.add('badge-high');
          stateText.classList.add('state-high');
          sevBadge.innerText = formatSeverityLabel(rawSev);
          verdictBadge.innerText = "DEFECT DETECTED";
          accentBar.style.background = "var(--status-high)";
          stateText.innerText = "Reject part";
        } else {
          sevBadge.classList.add('badge-med');
          verdictBadge.classList.add('badge-med');
          stateText.classList.add('state-med');
          sevBadge.innerText = formatSeverityLabel(rawSev === "NONE" && anomFlag ? "MODERATE" : rawSev);
          verdictBadge.innerText = "DEFECT DETECTED";
          accentBar.style.background = "var(--status-med)";
          stateText.innerText = "Needs review";
        }

        // Temporal Lock Pill
        const lockCnt = isNotMetal ? 0 : (data.lock_cnt || (data.is_locked ? 4 : 0));
        const isLockedNow = !isNotMetal && !isAnalyzing && !isStandby && (lockCnt >= 4 || Boolean(data.is_locked));
        const lockChipLabel = document.getElementById('lock-chip-label');
        if (lockChipLabel) {
          if (isNotMetal) {
            lockChipLabel.innerText = "Gate hold";
            lockChipLabel.className = "lock-pill lock-pill-warn";
          } else if (isStandby) {
            lockChipLabel.innerText = "Standby";
            lockChipLabel.className = "lock-pill";
          } else if (isLockedNow) {
            lockChipLabel.innerText = "Locked \u2713";
            lockChipLabel.className = "lock-pill lock-pill-locked";
          } else {
            lockChipLabel.innerText = "Scanning\u2026";
            lockChipLabel.className = "lock-pill lock-pill-scan";
          }
        }

        document.getElementById('lock-ratio-text').innerText = `${lockCnt}/4`;
        for (let i = 1; i <= 4; i++) {
          const bar = document.getElementById(`lock-bar-${i}`);
          if (!bar) continue;
          bar.className = 'lock-bar';
          if (i <= lockCnt) {
            bar.classList.add(lockCnt === 4 ? 'active-locked' : 'active');
          }
        }

        if (!pauseTogglePending && Date.now() >= pauseLockUntil && typeof data.is_paused === 'boolean') {
          isPaused = data.is_paused;
          updatePauseButtonState(isPaused);
        }

        // Update Playlist Dropdown (cache signature so open dropdown isn't clobbered every 500ms)
        const sel = document.getElementById('file-selector');
        const activeSourceTag = document.getElementById('active-source-tag');
        if (data.playlist && data.playlist.length > 0) {
          const showDropdown = (currentMode === 'IMAGE' || currentMode === 'VIDEO');
          sel.style.display = showDropdown ? 'inline-block' : 'none';
          if (activeSourceTag) {
            activeSourceTag.style.display = showDropdown ? 'none' : 'inline';
          }
          const sig = data.playlist.map(p => `${p.path}\t${p.name}`).join('\n');
          if (sig !== lastPlaylistSignature) {
            lastPlaylistSignature = sig;
            sel.innerHTML = data.playlist.map((p, idx) =>
              `<option value="${escapeHtml(p.path)}" ${idx === data.active_idx ? 'selected' : ''}>${escapeHtml(p.name)}</option>`
            ).join('');
          } else if (document.activeElement !== sel && typeof data.active_idx === 'number') {
            if (data.active_idx >= 0 && data.active_idx < sel.options.length && sel.selectedIndex !== data.active_idx) {
              sel.selectedIndex = data.active_idx;
            }
          }
        } else if (currentMode === 'VIDEO') {
          lastPlaylistSignature = "";
          sel.style.display = 'none';
          if (activeSourceTag) activeSourceTag.style.display = 'inline';
        }
      }).catch(() => {}).finally(() => {
        clearTimeout(stateTimeout);
        isFetchingState = false;
      });
    }

    // 8. Logbook Management
    function loadLogs() {
      fetch('/api/logs').then(r => r.json()).then(rows => {
        allLogs = Array.isArray(rows) ? rows : [];
        applyLogFilter();
      }).catch(() => {});
    }

    function formatTimingsCell(t) {
      if (!t) return "-";
      if (typeof t === 'object') {
        const pre = Number(t.pre || 0).toFixed(1);
        const inf = Number(t.infer || 0).toFixed(1);
        const post = Number(t.post || 0).toFixed(1);
        const tot = Number(t.total || 0).toFixed(1);
        return `${tot} ms (${pre}/${inf}/${post})`;
      }
      return `${t} ms`;
    }

    function renderLogsTable(rows) {
      const tbody = document.getElementById('logbook-tbody');
      if (!rows || rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="14" style="text-align:center; padding:32px; color:var(--text-muted); font-size:13px;">No defect inspection records logged yet</td></tr>';
        return;
      }
      tbody.innerHTML = rows.map(r => {
        const sevRaw = r.severity || r.sev || "NONE";
        const sevBucket = normalizeSeverityBucket(sevRaw);
        const badgeClass = severityBadgeClass(sevRaw);
        const family = r.family || r.part || "steel_sheet";
        const group = r.group || "-";
        const verdictRaw = String(r.verdict || (sevBucket === "CRITICAL" ? "REJECT" : (sevBucket === "NONE" ? "PASS" : "REVIEW"))).toUpperCase();
        const verdictDisp = verdictRaw === "PASS" ? "PASS" : toSentenceCase(verdictRaw);
        const scoreConf = r.score_conf !== undefined ? r.score_conf : (r.conf !== undefined ? `${r.conf}%` : "-");
        const anomalyScore = r.anomaly_score !== undefined ? Number(r.anomaly_score).toFixed(3) : "-";
        const areaPct = r.area_pct !== undefined ? `${r.area_pct}%` : (r.area !== undefined ? `${r.area}%` : "0.0%");
        const boxes = r.regions_boxes !== undefined ? r.regions_boxes : (r.boxes || 0);
        const timingsText = r.timings_ms ? formatTimingsCell(r.timings_ms) : `${r.ms || 0} ms`;

        return `
          <tr>
            <td class="mono-cell">${escapeHtml(r.ts || r.timestamp || "-")}</td>
            <td style="font-weight:600; color:var(--text-main);">${escapeHtml(formatFamilyName(family))}</td>
            <td style="font-weight:600; color:var(--text-main);">${escapeHtml(toSentenceCase(r.defect || "-"))}</td>
            <td style="color:var(--text-muted);">${escapeHtml(group === "-" ? "-" : toSentenceCase(group))}</td>
            <td class="mono-cell">${escapeHtml(scoreConf)}</td>
            <td class="mono-cell">${escapeHtml(anomalyScore)}</td>
            <td class="mono-cell">${escapeHtml(areaPct)}</td>
            <td class="mono-cell">${escapeHtml(boxes)}</td>
            <td><span class="severity-badge ${badgeClass}">${escapeHtml(formatSeverityLabel(sevRaw))}</span></td>
            <td><span class="verdict-badge ${badgeClass}">${escapeHtml(verdictDisp)}</span></td>
            <td class="log-col-cause">${escapeHtml(r.cause || "-")}</td>
            <td class="log-col-action">${escapeHtml(r.action || "-")}</td>
            <td class="mono-cell log-col-timings">${escapeHtml(timingsText)}</td>
            <td style="color:var(--text-muted);">${escapeHtml(r.src || r.advisory_source || "-")}</td>
          </tr>
        `;
      }).join('');
    }

    function applyLogFilter() {
      const filter = document.getElementById('log-filter-select').value;
      if (filter === 'ALL') {
        renderLogsTable(allLogs);
      } else {
        renderLogsTable(allLogs.filter(r => normalizeSeverityBucket(r.severity || r.sev) === filter));
      }
    }

    function clearLogs() {
      if (confirm("Clear all inspection records from the CSV logbook?")) {
        fetch('/api/clear_logs', { method: 'POST' }).then(() => loadLogs());
      }
    }

    applyModeUI(currentMode);
    setInterval(() => updateState(false), 160);
    updateState(true);
  </script>
</body>
</html>

)====rawhtml====";

std::string get_index_html() {
    std::vector<fs::path> candidates = {
        "templates/index.html",
        "../templates/index.html",
        "rpi5-native-engine/templates/index.html",
        "../rpi5-native-engine/templates/index.html"
    };
    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            std::ifstream f(p, std::ios::in | std::ios::binary);
            if (f.is_open()) {
                std::stringstream ss;
                ss << f.rdbuf();
                return ss.str();
            }
        }
    }
    return std::string(EMBEDDED_HTML);
}

} // namespace ui
} // namespace qc
