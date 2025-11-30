# Chatterbox Norwegian TTS Plugin for Pot Desktop

High-quality Norwegian text-to-speech plugin for [Pot Desktop](https://github.com/pot-app/pot-desktop), powered by the Chatterbox TTS model with formant-preserving pitch shifting algorithms.

## About

This plugin integrates the Chatterbox Norwegian TTS server with Pot Desktop, enabling natural-sounding Norwegian speech synthesis directly within the Pot translation and OCR application. The plugin supports advanced audio processing features including:

- **PSOLA-based pitch shifting** for natural voice character preservation
- **Multiple time-stretching algorithms** for speed control
- **Adjustable voice parameters** (exaggeration, temperature, cfg_weight)

## Requirements

1. **Pot Desktop**: Download from [pot-app.com](https://pot-app.com) or [GitHub releases](https://github.com/pot-app/pot-desktop/releases)
2. **Chatterbox TTS Server**: Must be running locally at `http://localhost:8000`
   - See the [main README](../README.md) for server installation and setup

## Installation

1. **Install the Plugin in Pot Desktop**:
   - Download the pre-built `plugin.com.pot-app.chatterbox_tts_no.potext` file
   - Open Pot Desktop
   - Go to: **Config → Service → TTS**
   - Click: **Add Extension → Install Plugin**
   - Select the downloaded `.potext` file
   - Update the plugin parameters; url, exaggeration, cfg_weight, etc...
   - Click **Save**.


## For Plugin Developers

### Packaging the Plugin

To create a new `.potext` file from source:

1. Compress the following files into a ZIP archive:
   - `main.js` (plugin logic)
   - `info.json` (plugin metadata)
   - `icon.png` (plugin icon)

2. Rename the ZIP file with the `.potext` extension:
   ```bash
   # Example
   mv plugin.zip plugin.com.pot-app.chatterbox_tts_no.potext
   ```

3. The resulting `.potext` file can be installed in Pot Desktop


## Troubleshooting

**Plugin shows "Connection Error"**:
- Ensure the TTS server is running: `http://localhost:8000`

**No audio output**:
- Verify the TTS server is responding: `curl http://localhost:8000/`
- Test the server directly via the web UI
- If you are using linux, you may want to check this : [https://github.com/pot-app/pot-desktop/pull/967](https://github.com/pot-app/pot-desktop/pull/967)

## License

This plugin is part of the Chatterbox Norwegian TTS project.

## Links

- [Main Project](../)
- [Pot Desktop](https://github.com/pot-app/pot-desktop)
- [Chatterbox TTS Model](https://huggingface.co/akhbar/chatterbox-tts-norwegian)



