from flask import Flask, render_template, jsonify, request, send_from_directory
import librosa
import numpy as np
import json
import os
import wave
import struct
import sys
from math import sqrt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab
import matplotlib.pyplot as plt
import io
import base64
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Sirve archivos de video desde la carpeta video/"""
    return send_from_directory('video', filename)

def getBandWidth(sample_size, sample_rate):
    return (2.0/sample_size) * (sample_rate / 2.0)

def freqToIndex(f, sample_size, sample_rate):
    # If f (frequency is lower than the bandwidth of spectrum[0]
    if f < getBandWidth(sample_size, sample_rate)/2:
        return 0
    if f > (sample_rate / 2) - (getBandWidth(sample_size, sample_rate) / 2):
        return sample_size -1
    fraction = float(f) / float(sample_rate)
    index = round(sample_size * fraction)
    return index

def average_fft_bands(fft_array, sample_size, sample_rate):
    num_bands = 12  # The number of frequency bands (12 = 1 octave)
    fft_averages = []
    
    for band in range(0, num_bands):
        avg = 0.0

        if band == 0:
            lowFreq = int(0)
        else:
            lowFreq = int(int(sample_rate / 2) / float(2 ** (num_bands - band)))
        hiFreq = int((sample_rate / 2) / float(2 ** ((num_bands-1) - band)))
        lowBound = int(freqToIndex(lowFreq, sample_size, sample_rate))
        hiBound = int(freqToIndex(hiFreq, sample_size, sample_rate))
        
        for j in range(lowBound, hiBound):
            if j < len(fft_array):
                avg += fft_array[j]

        if (hiBound - lowBound + 1) > 0:
            avg /= (hiBound - lowBound + 1)
        fft_averages.append(avg)
    
    return fft_averages

def process_audio_fft(audio_data, sample_rate):
    """Procesa audio usando FFT para obtener bandas de frecuencia"""
    try:
        logger.info(f"Procesando FFT: {len(audio_data)} muestras, {sample_rate} Hz")
        
        # Parámetros FFT
        fouriers_per_second = 24  # Frames per second
        fourier_spread = 1.0/fouriers_per_second
        fourier_width = fourier_spread
        fourier_width_index = int(fourier_width * float(sample_rate))
        
        # Calcular frecuencia
        sample_size = fourier_width_index
        freq = sample_rate / sample_size * np.arange(sample_size)
        
        # Procesar solo una muestra para obtener las bandas
        if len(audio_data) >= sample_size:
            sample_range = audio_data[:sample_size]
            # FFT the data
            fft_data = abs(np.fft.fft(sample_range))
            # Normalise the data
            fft_data *= ((2**.5)/sample_size)
            
            # Obtener promedios de bandas
            fft_averages = average_fft_bands(fft_data, sample_size, sample_rate)
            
            # Normalizar para visualización (0-100)
            max_val = max(fft_averages) if fft_averages else 1
            fft_averages = [int((a / max_val) * 100) if max_val > 0 else 0 for a in fft_averages]
            
            logger.info(f"FFT completado: {len(fft_averages)} bandas")
            return fft_averages
        else:
            logger.warning(f"Datos insuficientes para FFT: {len(audio_data)} < {sample_size}")
            return [0] * 12  # Retornar 12 bandas vacías si no hay suficientes datos
            
    except Exception as e:
        logger.error(f"Error en FFT: {e}")
        return [0] * 12

def generate_simulated_fft_data(num_segments):
    """Genera datos FFT simulados para testing"""
    logger.info(f"Generando datos FFT simulados para {num_segments} segmentos")
    all_fft_data = []
    
    for i in range(num_segments):
        # Simular 12 bandas de frecuencia con valores realistas
        bands = []
        for band in range(12):
            # Simular diferentes frecuencias con patrones realistas
            base_value = 20 + (i * 5) % 60
            freq_factor = 1 + (band * 0.3)  # Diferentes frecuencias tienen diferentes amplitudes
            noise = np.random.randint(-10, 10)
            value = int(base_value * freq_factor + noise)
            value = max(5, min(100, value))  # Limitar entre 5-100
            bands.append(value)
        all_fft_data.append(bands)
    
    return all_fft_data

@app.route('/api/process-playlist')
def process_playlist():
    try:
        logger.info("Iniciando procesamiento de playlist")
        
        # Leer el archivo playlist.m3u8
        playlist_path = "video/playlist.m3u8"
        if not os.path.exists(playlist_path):
            logger.error(f"Archivo playlist no encontrado: {playlist_path}")
            return jsonify({"error": "Archivo playlist.m3u8 no encontrado"}), 404
        
        with open(playlist_path, 'r', encoding='utf-8') as f:
            playlist_content = f.read()
        
        # Parsear el contenido del playlist
        segments = []
        lines = playlist_content.strip().split('\n')
        
        for line in lines:
            if line.endswith('.ts') and not line.startswith('#'):
                segments.append(line.strip())
        
        logger.info(f"Encontrados {len(segments)} segmentos en el playlist")
        
        # Procesar cada segmento para obtener datos de audio reales
        all_fft_data = []
        successful_processing = 0
        
        for i, segment in enumerate(segments):
            segment_path = os.path.join("video", segment)
            logger.info(f"Procesando segmento {i+1}/{len(segments)}: {segment}")
            
            if os.path.exists(segment_path):
                try:
                    # Intentar extraer audio del segmento .ts usando librosa
                    logger.info(f"Cargando audio de {segment_path}")
                    y, sr = librosa.load(segment_path, sr=None, duration=10)  # 10 segundos por segmento
                    
                    logger.info(f"Audio cargado: {len(y)} muestras, {sr} Hz")
                    
                    # Procesar con FFT
                    fft_bands = process_audio_fft(y, sr)
                    all_fft_data.append(fft_bands)
                    successful_processing += 1
                    
                except Exception as e:
                    logger.error(f"Error procesando segmento {segment}: {e}")
                    # Si falla, usar datos simulados
                    fft_bands = [20 + (i * 8) % 80 for _ in range(12)]
                    all_fft_data.append(fft_bands)
            else:
                logger.warning(f"Archivo no encontrado: {segment_path}")
                # Si el archivo no existe, usar datos simulados
                fft_bands = [20 + (i * 8) % 80 for _ in range(12)]
                all_fft_data.append(fft_bands)
        
        # Si no se pudo procesar ningún segmento real, usar datos simulados
        if successful_processing == 0:
            logger.warning("No se pudo procesar ningún segmento real, usando datos simulados")
            all_fft_data = generate_simulated_fft_data(len(segments))
        
        # Crear visualización combinada de todas las bandas
        combined_amplitudes = []
        for segment_bands in all_fft_data:
            # Promedio de todas las bandas para este segmento
            avg_amplitude = sum(segment_bands) / len(segment_bands)
            combined_amplitudes.append(int(avg_amplitude))
        
        logger.info(f"Procesamiento completado: {len(all_fft_data)} conjuntos de datos FFT")
        
        return jsonify({
            "segments": segments,
            "amplitudes": combined_amplitudes,
            "fft_bands": all_fft_data,  # Datos detallados de bandas
            "playlist_info": {
                "total_segments": len(segments),
                "duration_per_segment": 10,
                "frequency_bands": 12,
                "successful_processing": successful_processing
            }
        })
        
    except Exception as e:
        logger.error(f"Error general en process_playlist: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-audio')
def process_audio():
    try:
        # Verificar si existe el archivo de audio
        audio_file = "audio.mp3"
        if not os.path.exists(audio_file):
            return jsonify({"error": "Archivo audio.mp3 no encontrado"}), 404
        
        # Cargar audio
        y, sr = librosa.load(audio_file, sr=None)

        # Dividir en "ventanas" (ej: 200 barras)
        num_barras = 200
        hop_length = len(y) // num_barras
        amplitudes = []

        for i in range(num_barras):
            start = i * hop_length
            end = start + hop_length
            segment = y[start:end]
            rms = np.sqrt(np.mean(segment**2))  # energía RMS
            amplitudes.append(float(rms))

        # Normalizar (0-100 px para dibujar)
        max_val = max(amplitudes)
        amplitudes = [int((a / max_val) * 100) for a in amplitudes]

        return jsonify({
            "amplitudes": amplitudes,
            "sample_rate": sr,
            "duration": len(y) / sr
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_audio_visualization(audio_data, sample_rate, filename="audio_analysis"):
    """
    Genera visualizaciones de audio: waveform y espectrograma
    Usando librosa.display para gráficos profesionales
    """
    try:
        logger.info(f"Generando visualización de audio: {len(audio_data)} muestras, {sample_rate} Hz")
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Análisis de Audio: Waveform y Espectrograma', fontsize=16, fontweight='bold')
        
        # 1. GRÁFICO SUPERIOR: WAVEFORM usando librosa.display
        librosa.display.waveshow(audio_data, sr=sample_rate, color="blue", alpha=0.8, ax=ax1)
        ax1.set_title('Waveform - Amplitud vs Tiempo', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax1.set_ylabel('Amplitud', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Configurar límites del eje Y para mejor visualización
        max_amplitude = np.max(np.abs(audio_data))
        ax1.set_ylim(-max_amplitude * 1.1, max_amplitude * 1.1)
        
        # Agregar línea de referencia en y=0
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # 2. GRÁFICO INFERIOR: ESPECTROGRAMA usando librosa.display
        # Calcular espectrograma usando librosa
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        
        # Crear espectrograma con librosa.display
        img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', 
                                      ax=ax2, cmap='inferno')
        
        ax2.set_title('Espectrograma - Frecuencia vs Tiempo', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax2.set_ylabel('Frecuencia (Hz)', fontsize=12)
        
        # Agregar colorbar
        cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        cbar.set_label('Intensidad (dB)', fontsize=10)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar imagen
        output_path = f"static/{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Visualización guardada: {output_path}")
        
        return {
            'success': True,
            'image_path': f"/static/{filename}.png",
            'duration': len(audio_data) / sample_rate,
            'sample_rate': sample_rate,
            'max_amplitude': float(max_amplitude),
            'spectrogram_info': {
                'frequency_range': f"0 - {sample_rate // 2} Hz",
                'time_range': f"0 - {len(audio_data) / sample_rate:.2f} segundos"
            }
        }
        
    except Exception as e:
        logger.error(f"Error generando visualización: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def generate_interactive_visualization(audio_data, sample_rate):
    """
    Genera visualización interactiva usando Plotly
    Para mostrar en el navegador con zoom y pan
    """
    try:
        logger.info(f"Generando visualización interactiva: {len(audio_data)} muestras, {sample_rate} Hz")
        
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Waveform - Amplitud vs Tiempo', 'Espectrograma - Frecuencia vs Tiempo'),
            vertical_spacing=0.1
        )
        
        # 1. WAVEFORM
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=audio_data,
                mode='lines',
                name='Waveform',
                line=dict(color='blue', width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Agregar línea de referencia en y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
        
        # 2. ESPECTROGRAMA
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        
        # Obtener frecuencias y tiempos para el espectrograma
        freqs = librosa.fft_frequencies(sr=sample_rate)
        times = librosa.times_like(D, sr=sample_rate)
        
        fig.add_trace(
            go.Heatmap(
                z=D,
                x=times,
                y=freqs,
                colorscale='Inferno',
                name='Espectrograma',
                colorbar=dict(title="Intensidad (dB)")
            ),
            row=2, col=1
        )
        
        # Actualizar layout
        fig.update_layout(
            title_text="Análisis de Audio Interactivo: Waveform y Espectrograma",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Tiempo (segundos)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitud", row=1, col=1)
        fig.update_xaxes(title_text="Tiempo (segundos)", row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia (Hz)", row=2, col=1)
        
        # Convertir a HTML
        html_content = fig.to_html(full_html=False, include_plotlyjs=True)
        
        return {
            'success': True,
            'html_content': html_content,
            'duration': len(audio_data) / sample_rate,
            'sample_rate': sample_rate,
            'max_amplitude': float(np.max(np.abs(audio_data)))
        }
        
    except Exception as e:
        logger.error(f"Error generando visualización interactiva: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/api/generate-audio-visualization')
def generate_audio_visualization_api():
    """API endpoint para generar visualizaciones estáticas de audio"""
    try:
        # Verificar si existe el archivo de audio
        audio_file = "audio.mp3"
        if not os.path.exists(audio_file):
            return jsonify({"error": "Archivo audio.mp3 no encontrado"}), 404
        
        # Cargar audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # Generar visualización estática
        result = generate_audio_visualization(y, sr, "audio_analysis")
        
        if result['success']:
            return jsonify({
                "success": True,
                "image_url": result['image_path'],
                "audio_info": {
                    "duration": result['duration'],
                    "sample_rate": result['sample_rate'],
                    "max_amplitude": result['max_amplitude'],
                    "spectrogram_info": result['spectrogram_info']
                }
            })
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error en generate_audio_visualization_api: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-playlist-visualization')
def generate_playlist_visualization_api():
    """API endpoint para generar visualizaciones estáticas de playlist M3U8"""
    try:
        # Leer el archivo playlist.m3u8
        playlist_path = "video/playlist.m3u8"
        if not os.path.exists(playlist_path):
            return jsonify({"error": "Archivo playlist.m3u8 no encontrado"}), 404
        
        with open(playlist_path, 'r', encoding='utf-8') as f:
            playlist_content = f.read()
        
        # Parsear segmentos
        segments = []
        lines = playlist_content.strip().split('\n')
        for line in lines:
            if line.endswith('.ts') and not line.startswith('#'):
                segments.append(line.strip())
        
        if not segments:
            return jsonify({"error": "No se encontraron segmentos en el playlist"}), 404
        
        logger.info(f"Generando visualización estática para {len(segments)} segmentos")
        
        # Procesar primer segmento para visualización
        first_segment = segments[0]
        segment_path = os.path.join("video", first_segment)
        
        if os.path.exists(segment_path):
            # Cargar audio del primer segmento
            y, sr = librosa.load(segment_path, sr=None, duration=10)
            
            # Generar visualización estática
            result = generate_audio_visualization(y, sr, "playlist_analysis")
            
            if result['success']:
                return jsonify({
                    "success": True,
                    "image_url": result['image_path'],
                    "segment_info": {
                        "segment": first_segment,
                        "duration": result['duration'],
                        "sample_rate": result['sample_rate'],
                        "max_amplitude": result['max_amplitude'],
                        "spectrogram_info": result['spectrogram_info']
                    },
                    "playlist_info": {
                        "total_segments": len(segments),
                        "estimated_total_duration": len(segments) * 10
                    }
                })
            else:
                return jsonify({"error": result['error']}), 500
        else:
            return jsonify({"error": f"Segmento no encontrado: {segment_path}"}), 404
            
    except Exception as e:
        logger.error(f"Error en generate_playlist_visualization_api: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-interactive-visualization')
def generate_interactive_visualization_api():
    """API endpoint para generar visualizaciones interactivas de audio"""
    try:
        # Verificar si existe el archivo de audio
        audio_file = "audio.mp3"
        if not os.path.exists(audio_file):
            return jsonify({"error": "Archivo audio.mp3 no encontrado"}), 404
        
        # Cargar audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # Generar visualización interactiva
        result = generate_interactive_visualization(y, sr)
        
        if result['success']:
            return jsonify({
                "success": True,
                "html_content": result['html_content'],
                "audio_info": {
                    "duration": result['duration'],
                    "sample_rate": result['sample_rate'],
                    "max_amplitude": result['max_amplitude']
                }
            })
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error en generate_interactive_visualization_api: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-interactive-playlist-visualization')
def generate_interactive_playlist_visualization_api():
    """API endpoint para generar visualizaciones interactivas de playlist M3U8"""
    try:
        # Leer el archivo playlist.m3u8
        playlist_path = "video/playlist.m3u8"
        if not os.path.exists(playlist_path):
            return jsonify({"error": "Archivo playlist.m3u8 no encontrado"}), 404
        
        with open(playlist_path, 'r', encoding='utf-8') as f:
            playlist_content = f.read()
        
        # Parsear segmentos
        segments = []
        lines = playlist_content.strip().split('\n')
        for line in lines:
            if line.endswith('.ts') and not line.startswith('#'):
                segments.append(line.strip())
        
        if not segments:
            return jsonify({"error": "No se encontraron segmentos en el playlist"}), 404
        
        logger.info(f"Generando visualización interactiva para {len(segments)} segmentos")
        
        # Procesar primer segmento para visualización
        first_segment = segments[0]
        segment_path = os.path.join("video", first_segment)
        
        if os.path.exists(segment_path):
            # Cargar audio del primer segmento
            y, sr = librosa.load(segment_path, sr=None, duration=10)
            
            # Generar visualización interactiva
            result = generate_interactive_visualization(y, sr)
            
            if result['success']:
                return jsonify({
                    "success": True,
                    "html_content": result['html_content'],
                    "segment_info": {
                        "segment": first_segment,
                        "duration": result['duration'],
                        "sample_rate": result['sample_rate'],
                        "max_amplitude": result['max_amplitude']
                    },
                    "playlist_info": {
                        "total_segments": len(segments),
                        "estimated_total_duration": len(segments) * 10
                    }
                })
            else:
                return jsonify({"error": result['error']}), 500
        else:
            return jsonify({"error": f"Segmento no encontrado: {segment_path}"}), 404
            
    except Exception as e:
        logger.error(f"Error en generate_interactive_playlist_visualization_api: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
