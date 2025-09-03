# Audio Waveform Processor

🎵 **Professional Audio Analysis and Visualization Tool**

A comprehensive Flask-based web application for processing and visualizing audio files and HLS playlists with advanced features including FFT analysis, real-time waveform generation, and professional spectrograms.

## ✨ Features

### 🎬 Video & Audio Processing
- **HLS Playlist Support**: Process and analyze M3U8 playlists with video segments
- **Real-time Video Playback**: Integrated HLS.js player with synchronized audio visualization
- **Audio Extraction**: Extract and analyze audio from video segments using librosa

### 📊 Advanced Visualizations
- **Waveform Analysis**: Real-time amplitude visualization with customizable bars
- **FFT Frequency Analysis**: 12-band frequency spectrum analysis with color-coded bars
- **Professional Spectrograms**: High-quality spectrograms using librosa.display
- **Interactive Plots**: Plotly-based interactive visualizations with zoom and pan
- **Static Image Generation**: Export visualizations as high-resolution PNG images

### 🎨 Real-time Features
- **Live Audio Animation**: Web Audio API integration for real-time frequency analysis
- **Synchronized Visualization**: Audio bars synchronized with video playback
- **Simulated Animation**: Fallback animation system for enhanced user experience

### 🔧 Technical Capabilities
- **Multiple Audio Formats**: Support for MP3, WAV, and video audio extraction
- **Advanced FFT Processing**: Custom frequency band analysis with librosa
- **Professional Plotting**: Matplotlib and Plotly integration for publication-quality graphs
- **Responsive Design**: Modern, dark-themed UI with mobile-friendly interface

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/audio-waveform-processor.git
cd audio-waveform-processor

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir static video

# Add your audio/video files
# - Place audio.mp3 in the root directory
# - Place playlist.m3u8 and video segments in the video/ directory
```

### Dependencies
```
Flask==2.3.3
librosa==0.10.1
numpy==1.24.3
matplotlib==3.7.2
plotly==5.15.0
```

## 🎯 Usage

### Starting the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Main Features

#### 1. **Video Playback & Analysis**
- Click "🎬 Reproducir Video" to load and play HLS video
- Real-time audio visualization synchronized with video
- Video controls: play/pause, stop, fullscreen

#### 2. **Playlist Processing**
- Click "📋 Cargar Playlist M3U8" to analyze M3U8 playlists
- Automatic segment detection and FFT analysis
- Detailed segment information and frequency bands

#### 3. **Audio File Analysis**
- Click "🎵 Procesar Audio MP3" to analyze audio files
- Waveform generation with amplitude visualization
- Audio metadata extraction and display

#### 4. **Professional Visualizations**
Switch to the "📈 Visualización Profesional" tab for:
- **Static Visualizations**: Generate PNG images with waveform + spectrogram
- **Interactive Visualizations**: Plotly-based interactive plots
- **Multiple Sources**: Generate from MP3 files or playlist segments

#### 5. **Real-time Animation**
- Click "🎨 Animación en Vivo" for real-time audio visualization
- Web Audio API integration for actual audio analysis
- Fallback to simulated animation for enhanced experience

## 📁 Project Structure

```
audio-waveform-processor/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── audio.mp3             # Sample audio file (user-provided)
├── static/               # Generated images and static files
├── video/                # Video files and playlists
│   ├── playlist.m3u8     # HLS playlist (user-provided)
│   └── *.ts              # Video segments (user-provided)
└── templates/
    └── index.html        # Main web interface
```

## 🔧 API Endpoints

### Core Processing
- `GET /api/process-audio` - Process audio.mp3 file
- `GET /api/process-playlist` - Process M3U8 playlist

### Visualization Generation
- `GET /api/generate-audio-visualization` - Static visualization from MP3
- `GET /api/generate-playlist-visualization` - Static visualization from playlist
- `GET /api/generate-interactive-visualization` - Interactive visualization from MP3
- `GET /api/generate-interactive-playlist-visualization` - Interactive visualization from playlist

### File Serving
- `GET /video/<filename>` - Serve video files
- `GET /static/<filename>` - Serve static files

## 🎨 Visualization Types

### 1. **Simple Waveform**
- Real-time amplitude bars
- Color-coded intensity levels
- Hover effects and tooltips

### 2. **FFT Analysis**
- 12-band frequency spectrum
- Color-coded frequency ranges
- Frequency labels (20Hz - 41.6kHz)

### 3. **Professional Visualizations**
- **Waveform**: Amplitude vs. time with reference line
- **Spectrogram**: Frequency vs. time with color-coded intensity
- **Interactive**: Zoom, pan, and explore capabilities

### 4. **Real-time Animation**
- Live audio frequency analysis
- Synchronized with video playback
- Dynamic color changes based on intensity

## 🛠️ Technical Details

### Audio Processing
- **librosa**: Advanced audio analysis and feature extraction
- **FFT**: Custom frequency band analysis with 12 bands
- **Real-time**: Web Audio API for live frequency analysis

### Visualization
- **Matplotlib**: High-quality static image generation
- **Plotly**: Interactive web-based visualizations
- **librosa.display**: Professional audio visualization functions

### Web Technologies
- **Flask**: Python web framework
- **HLS.js**: HTTP Live Streaming video playback
- **Web Audio API**: Real-time audio analysis
- **Modern CSS**: Responsive design with animations

## 🎯 Use Cases

- **Audio Analysis**: Professional audio file analysis and visualization
- **Video Processing**: Extract and analyze audio from video content
- **Streaming Analysis**: Process HLS playlists and streaming content
- **Research**: Academic and research applications for audio analysis
- **Content Creation**: Generate visualizations for presentations and reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **librosa**: For advanced audio analysis capabilities
- **HLS.js**: For seamless HLS video playback
- **Plotly**: For interactive visualization features
- **Matplotlib**: For high-quality static image generation

---

**Made with ❤️ for audio analysis and visualization**

*For questions and support, please open an issue on GitHub.*
