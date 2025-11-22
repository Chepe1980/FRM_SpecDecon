import streamlit as st
import segyio
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal
import pywt
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

class CastagnaFrequencyAnalysis:
    def __init__(self):
        self.sample_rate = 4.0  # ms
        # Castagna's typical frequency ranges for seismic analysis
        self.low_freq_range = (10, 20)    # Low frequency band for tuning analysis
        self.mid_freq_range = (20, 40)    # Mid frequency band for resolution
        self.high_freq_range = (40, 80)   # High frequency band for thin beds

    def load_segy_data(self, segy_path):
        """Load SEG-Y data using segyio"""
        with segyio.open(segy_path, "r") as segyfile:
            # Read all traces
            data = []
            for trace in segyfile.trace:
                data.append(trace)
            data = np.array(data).T  # Transpose to samples x traces

            # Get sampling interval
            if segyio.BinField.Interval in segyfile.bin:
                self.sample_rate = segyfile.bin[segyio.BinField.Interval] / 1000.0

            return data

    def castagna_ricker_wavelet(self, frequency, length=0.128, dt=0.004):
        """
        Create Ricker wavelet following Castagna's spectral decomposition approach
        Castagna et al. (2003) - Instantaneous spectral analysis
        """
        t = np.arange(-length/2, length/2, dt)
        # Ricker wavelet formula as used in Castagna's work
        y = (1.0 - 2.0 * (np.pi * frequency * t) ** 2) * np.exp(-(np.pi * frequency * t) ** 2)
        return y / np.max(np.abs(y))

    def apply_spectral_decomposition(self, seismic_data, frequencies=np.arange(10, 81, 5)):
        """
        Spectral decomposition with discrete frequencies
        """
        spectral_components = {}

        for freq in frequencies:
            wavelet = self.castagna_ricker_wavelet(freq)
            filtered_traces = []

            for trace in seismic_data.T:
                # Convolve with Ricker wavelet for spectral component
                conv_result = np.convolve(trace, wavelet, mode='same')
                filtered_traces.append(conv_result)

            spectral_components[f'{freq}Hz'] = np.array(filtered_traces).T

        return spectral_components

    def create_continuous_frequency_slice(self, seismic_data, spectral_components, trace_index=0, num_interp_points=200):
        """
        Create a continuous frequency slice using interpolation
        """
        # Get original frequencies and data
        frequencies_original = sorted([float(k.replace('Hz', '')) for k in spectral_components.keys()])

        # Extract data for the specific trace
        frequency_slice_original = np.zeros((seismic_data.shape[0], len(frequencies_original)))
        for i, freq in enumerate(frequencies_original):
            frequency_slice_original[:, i] = np.abs(spectral_components[f'{freq}Hz'][:, trace_index])

        # Create finer frequency grid for interpolation
        frequencies_continuous = np.linspace(min(frequencies_original), max(frequencies_original), num_interp_points)

        # Interpolate each time sample across frequencies
        frequency_slice_continuous = np.zeros((seismic_data.shape[0], len(frequencies_continuous)))

        for time_idx in range(seismic_data.shape[0]):
            # Interpolate this time sample across frequencies
            interp_func = interp1d(frequencies_original, frequency_slice_original[time_idx, :],
                                 kind='cubic', bounds_error=False, fill_value=0)
            frequency_slice_continuous[time_idx, :] = interp_func(frequencies_continuous)

        return frequency_slice_continuous, frequencies_continuous

    def get_frequency_spectrum_at_time(self, continuous_slice, frequencies_continuous, time_ms, time_axis):
        """
        Extract frequency spectrum at specific time from continuous frequency slice
        """
        # Find closest time index
        time_idx = np.argmin(np.abs(time_axis - time_ms))
        
        # Get frequency spectrum at this time - this is the horizontal slice from the heatmap
        frequency_spectrum = continuous_slice[time_idx, :]
        
        return frequency_spectrum, time_idx

    def create_isa_frequency_spectrum(self, seismic_data, spectral_components, trace_index=0, selected_time=None, 
                                    colormap='Viridis'):
        """
        Create ISA frequency spectrum plot (heatmap only for synchronization)
        """
        # Create continuous frequency slice
        continuous_slice, frequencies_continuous = self.create_continuous_frequency_slice(
            seismic_data, spectral_components, trace_index, num_interp_points=200
        )

        time_axis = np.arange(seismic_data.shape[0]) * self.sample_rate

        # Set default selected time if not provided
        if selected_time is None:
            selected_time = time_axis[len(time_axis) // 2]

        # Create the plot - Heatmap only for synchronized zoom
        fig = go.Figure()
        
        # Continuous Frequency Volume Slice (Heatmap)
        fig.add_trace(
            go.Heatmap(
                z=continuous_slice,
                x=frequencies_continuous,
                y=time_axis,
                colorscale=colormap,
                colorbar=dict(title="Amplitude"),
                hovertemplate='<b>Frequency</b>: %{x:.1f} Hz<br><b>Time</b>: %{y:.1f} ms<br><b>Amplitude</b>: %{z:.3f}<extra></extra>',
                name='ISA Frequency Spectrum'
            )
        )

        # Add vertical line to frequency slice at selected time
        fig.add_trace(
            go.Scatter(
                x=[frequencies_continuous[0], frequencies_continuous[-1]],
                y=[selected_time, selected_time],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name='Selected Time',
                showlegend=False,
                hovertemplate='<b>Selected Time</b>: %{y:.1f} ms<extra></extra>'
            )
        )

        fig.update_layout(
            title=f'ISA Frequency Spectrum - Trace {trace_index}',
            xaxis_title="Frequency (Hz)",
            yaxis_title="Time (ms)",
            yaxis=dict(autorange='reversed'),
            height=500,
            margin=dict(l=80, r=80, t=80, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        return fig

    def create_frequency_spectrum_details(self, seismic_data, spectral_components, trace_index=0, selected_time=None, 
                                        yaxis_range=None):
        """
        Create detailed frequency spectrum plot from ISA data
        """
        # Create continuous frequency slice
        continuous_slice, frequencies_continuous = self.create_continuous_frequency_slice(
            seismic_data, spectral_components, trace_index, num_interp_points=200
        )

        time_axis = np.arange(seismic_data.shape[0]) * self.sample_rate

        # Set default selected time if not provided
        if selected_time is None:
            selected_time = time_axis[len(time_axis) // 2]

        # Get spectrum at selected time
        selected_spectrum, selected_time_idx = self.get_frequency_spectrum_at_time(
            continuous_slice, frequencies_continuous, selected_time, time_axis
        )

        fig = go.Figure()
        
        # Frequency Spectrum at Selected Time
        fig.add_trace(
            go.Scatter(
                x=frequencies_continuous,
                y=selected_spectrum,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Frequency Spectrum',
                hovertemplate='<b>Frequency</b>: %{x:.1f} Hz<br><b>Amplitude</b>: %{y:.3f}<extra></extra>'
            )
        )

        # Add dominant frequency marker
        dominant_freq_idx = np.argmax(selected_spectrum)
        dominant_freq = frequencies_continuous[dominant_freq_idx]
        dominant_amp = selected_spectrum[dominant_freq_idx]
        
        fig.add_trace(
            go.Scatter(
                x=[dominant_freq],
                y=[dominant_amp],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='star'),
                text=[f' Dominant: {dominant_freq:.1f} Hz'],
                textposition='top right',
                name='Dominant Frequency',
                hovertemplate='<b>Dominant Frequency</b>: %{x:.1f} Hz<br><b>Amplitude</b>: %{y:.3f}<extra></extra>'
            )
        )

        # Highlight Castagna frequency bands
        low_band_mask = (frequencies_continuous >= 10) & (frequencies_continuous <= 20)
        mid_band_mask = (frequencies_continuous >= 20) & (frequencies_continuous <= 40)
        high_band_mask = (frequencies_continuous >= 40) & (frequencies_continuous <= 80)
        
        if np.any(low_band_mask):
            fig.add_trace(
                go.Scatter(
                    x=frequencies_continuous[low_band_mask],
                    y=selected_spectrum[low_band_mask],
                    mode='lines',
                    line=dict(color='green', width=4),
                    name='Low Freq (10-20 Hz)',
                    showlegend=True,
                    hoverinfo='skip'
                )
            )
        
        if np.any(mid_band_mask):
            fig.add_trace(
                go.Scatter(
                    x=frequencies_continuous[mid_band_mask],
                    y=selected_spectrum[mid_band_mask],
                    mode='lines',
                    line=dict(color='orange', width=4),
                    name='Mid Freq (20-40 Hz)',
                    showlegend=True,
                    hoverinfo='skip'
                )
            )
        
        if np.any(high_band_mask):
            fig.add_trace(
                go.Scatter(
                    x=frequencies_continuous[high_band_mask],
                    y=selected_spectrum[high_band_mask],
                    mode='lines',
                    line=dict(color='purple', width=4),
                    name='High Freq (40-80 Hz)',
                    showlegend=True,
                    hoverinfo='skip'
                )
            )

        # Update layout
        fig.update_layout(
            title=f'Frequency Spectrum Details at {selected_time:.1f} ms',
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude",
            height=500,
            margin=dict(l=80, r=80, t=80, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=True
        )

        # Set Y-axis range for frequency spectrum if provided
        if yaxis_range is not None:
            fig.update_yaxes(range=yaxis_range)
        else:
            fig.update_yaxes(range=[0, np.max(selected_spectrum) * 1.1])

        return fig, selected_spectrum, frequencies_continuous

    def create_common_frequency_section(self, spectral_components, selected_frequency, colormap='Viridis'):
        """
        Create a common frequency section as described in Castagna's paper
        This shows the seismic section at a specific frequency to identify low-frequency shadows
        """
        # Get the spectral component for the selected frequency
        freq_key = f'{selected_frequency}Hz'
        if freq_key not in spectral_components:
            # Find closest available frequency
            available_freqs = [float(k.replace('Hz', '')) for k in spectral_components.keys()]
            closest_freq = min(available_freqs, key=lambda x: abs(x - selected_frequency))
            freq_key = f'{closest_freq}Hz'
            selected_frequency = closest_freq
        
        frequency_section = np.abs(spectral_components[freq_key])
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=frequency_section,
                colorscale=colormap,
                colorbar=dict(title="Amplitude"),
                hovertemplate='<b>Trace</b>: %{x}<br><b>Time</b>: %{y:.1f} ms<br><b>Amplitude</b>: %{z:.3f}<extra></extra>',
                name=f'{selected_frequency} Hz Section'
            )
        )
        
        fig.update_layout(
            title=f'Common Frequency Section - {selected_frequency} Hz',
            xaxis_title='Frequency of ISA',
            yaxis_title='Time (ms)',
            yaxis=dict(autorange='reversed'),
            height=500,
            margin=dict(l=80, r=80, t=80, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig, frequency_section

    def create_interactive_plot(self, seismic_data, spectral_components, trace_index=0, selected_time=None, 
                              colormap='Viridis', yaxis_range=None):
        """
        Create interactive plot with synchronized zoom and time selection
        """
        # Create continuous frequency slice FIRST
        continuous_slice, frequencies_continuous = self.create_continuous_frequency_slice(
            seismic_data, spectral_components, trace_index, num_interp_points=200
        )

        time_axis = np.arange(seismic_data.shape[0]) * self.sample_rate
        trace_amplitude = seismic_data[:, trace_index]

        # Set default selected time if not provided
        if selected_time is None:
            selected_time = time_axis[len(time_axis) // 2]

        # Get spectrum at selected time FROM THE CONTINUOUS SLICE
        selected_spectrum, selected_time_idx = self.get_frequency_spectrum_at_time(
            continuous_slice, frequencies_continuous, selected_time, time_axis
        )

        # Create subplots with synchronized y-axes for first two plots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'Input Seismic Trace {trace_index}',
                f'Continuous Frequency Volume Slice (Time vs Frequency)',
                f'Frequency Spectrum at {selected_time:.1f} ms (From Heatmap)'
            ),
            horizontal_spacing=0.08,
            column_widths=[0.3, 0.4, 0.3],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Input Trace: Amplitude vs Time
        fig.add_trace(
            go.Scatter(
                x=trace_amplitude,
                y=time_axis,
                mode='lines',
                line=dict(color='black', width=2.5),
                name='Input Trace',
                hovertemplate='<b>Time</b>: %{y:.1f} ms<br><b>Amplitude</b>: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Fill positive amplitudes
        fig.add_trace(
            go.Scatter(
                x=np.maximum(trace_amplitude, 0),
                y=time_axis,
                mode='lines',
                line=dict(width=0),
                fill='tozerox',
                fillcolor='rgba(0, 100, 255, 0.4)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Fill negative amplitudes
        fig.add_trace(
            go.Scatter(
                x=np.minimum(trace_amplitude, 0),
                y=time_axis,
                mode='lines',
                line=dict(width=0),
                fill='tozerox',
                fillcolor='rgba(255, 50, 50, 0.4)',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Add vertical line for time selection
        fig.add_trace(
            go.Scatter(
                x=[-np.max(np.abs(trace_amplitude)) * 0.8, np.max(np.abs(trace_amplitude)) * 0.8],
                y=[selected_time, selected_time],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name='Selected Time',
                hovertemplate='<b>Selected Time</b>: %{y:.1f} ms<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Continuous Frequency Volume Slice (Heatmap)
        fig.add_trace(
            go.Heatmap(
                z=continuous_slice,
                x=frequencies_continuous,
                y=time_axis,
                colorscale=colormap,
                colorbar=dict(
                    title="Amplitude",
                    title_side="right",
                    len=0.8,
                    y=0.5,
                    yanchor="middle"
                ),
                hovertemplate='<b>Frequency</b>: %{x:.1f} Hz<br><b>Time</b>: %{y:.1f} ms<br><b>Amplitude</b>: %{z:.3f}<extra></extra>',
                name='Frequency Slice'
            ),
            row=1, col=2
        )

        # Add vertical line to frequency slice
        fig.add_trace(
            go.Scatter(
                x=[frequencies_continuous[0], frequencies_continuous[-1]],
                y=[selected_time, selected_time],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name='Selected Time',
                showlegend=False,
                hovertemplate='<b>Selected Time</b>: %{y:.1f} ms<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. Frequency Spectrum at Selected Time (EXTRACTED FROM HEATMAP)
        fig.add_trace(
            go.Scatter(
                x=frequencies_continuous,
                y=selected_spectrum,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Frequency Spectrum',
                hovertemplate='<b>Frequency</b>: %{x:.1f} Hz<br><b>Amplitude</b>: %{y:.3f}<extra></extra>'
            ),
            row=1, col=3
        )

        # Add dominant frequency marker
        dominant_freq_idx = np.argmax(selected_spectrum)
        dominant_freq = frequencies_continuous[dominant_freq_idx]
        dominant_amp = selected_spectrum[dominant_freq_idx]
        
        fig.add_trace(
            go.Scatter(
                x=[dominant_freq],
                y=[dominant_amp],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='star'),
                text=[f' Dominant: {dominant_freq:.1f} Hz'],
                textposition='top right',
                name='Dominant Frequency',
                hovertemplate='<b>Dominant Frequency</b>: %{x:.1f} Hz<br><b>Amplitude</b>: %{y:.3f}<extra></extra>'
            ),
            row=1, col=3
        )

        # Highlight that this spectrum comes from the heatmap
        fig.add_annotation(
            x=0.5, y=1.08,
            xref="paper", yref="paper",
            text=f"← Spectrum extracted from heatmap at {selected_time:.1f} ms →",
            showarrow=False,
            font=dict(size=12, color="red"),
            row=1, col=3
        )

        # Update layout
        fig.update_layout(
            height=700,
            width=1400,
            title_text=f"Interactive Castagna Spectral Analysis - Trace {trace_index}",
            title_x=0.5,
            title_font=dict(size=16),
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Update axes - SYNCHRONIZE Y-AXES FOR FIRST TWO PLOTS
        fig.update_xaxes(title_text="Amplitude", row=1, col=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2, gridcolor='lightgray')
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=3, gridcolor='lightgray')
        
        # Synchronize y-axes for first two plots (time axes)
        fig.update_yaxes(
            title_text="Time (ms)", 
            row=1, col=1, 
            autorange="reversed", 
            gridcolor='lightgray',
            matches='y2'  # Synchronize with second plot y-axis
        )
        fig.update_yaxes(
            title_text="Time (ms)", 
            row=1, col=2, 
            autorange="reversed", 
            gridcolor='lightgray',
            matches='y'  # Synchronize with first plot y-axis
        )
        
        # Set Y-axis range for frequency spectrum if provided
        if yaxis_range is not None:
            fig.update_yaxes(
                title_text="Amplitude", 
                row=1, col=3, 
                gridcolor='lightgray', 
                range=yaxis_range
            )
        else:
            fig.update_yaxes(
                title_text="Amplitude", 
                row=1, col=3, 
                gridcolor='lightgray', 
                range=[0, np.max(selected_spectrum) * 1.1]
            )

        return fig, selected_spectrum, frequencies_continuous, selected_time

    def _get_spectral_characteristics(self, spectrum, frequencies, time):
        """Calculate spectral characteristics for display"""
        # Find dominant frequency
        dominant_freq_idx = np.argmax(spectrum)
        dominant_freq = frequencies[dominant_freq_idx]
        dominant_amp = spectrum[dominant_freq_idx]
        
        # Calculate bandwidth (FWHM)
        half_max = dominant_amp / 2
        above_half_max = spectrum >= half_max
        if np.any(above_half_max):
            bandwidth = frequencies[above_half_max][-1] - frequencies[above_half_max][0]
        else:
            bandwidth = 0

        # Calculate frequency band content (Castagna methodology)
        low_freq_mask = (frequencies >= 10) & (frequencies <= 20)
        mid_freq_mask = (frequencies >= 20) & (frequencies <= 40)
        high_freq_mask = (frequencies >= 40) & (frequencies <= 80)
        
        low_content = np.mean(spectrum[low_freq_mask]) if np.any(low_freq_mask) else 0
        mid_content = np.mean(spectrum[mid_freq_mask]) if np.any(mid_freq_mask) else 0
        high_content = np.mean(spectrum[high_freq_mask]) if np.any(high_freq_mask) else 0

        return {
            'selected_time': time,
            'dominant_frequency': dominant_freq,
            'peak_amplitude': dominant_amp,
            'bandwidth': bandwidth,
            'low_freq_content': low_content,
            'mid_freq_content': mid_content,
            'high_freq_content': high_content
        }

def main():
    st.set_page_config(
        page_title="Castagna Spectral Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
        }
        .spectrum-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .time-input-box {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 2px solid #1f77b4;
            margin: 1rem 0;
        }
        .data-flow-box {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            text-align: center;
        }
        .stNumberInput > div > div > input {
            font-size: 1.1rem;
            font-weight: bold;
        }
        .low-freq-shadow {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .isa-plot-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            text-align: center;
        }
        .details-plot-box {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #333;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Castagna Spectral Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload SEG-Y File", type=['sgy', 'segy'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_data.sgy", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize analyzer
        analyzer = CastagnaFrequencyAnalysis()
        
        # Load data
        with st.spinner("Loading SEG-Y data..."):
            seismic_data = analyzer.load_segy_data("temp_data.sgy")
        
        st.sidebar.success(f"Data loaded: {seismic_data.shape}")
        
        # Configuration options
        trace_index = st.sidebar.slider(
            "Select Trace Index", 
            min_value=0, 
            max_value=seismic_data.shape[1]-1, 
            value=0
        )
        
        # Analysis parameters
        st.sidebar.subheader("Analysis Parameters")
        # CHANGED: Minimum frequency starts at 1Hz instead of 5Hz
        min_freq = st.sidebar.slider("Minimum Frequency (Hz)", 1, 30, 10)  # Changed min from 5 to 1
        max_freq = st.sidebar.slider("Maximum Frequency (Hz)", 50, 100, 80)
        num_frequencies = st.sidebar.slider("Number of Frequencies", 10, 100, 50)
        
        # NEW: Colormap selection
        st.sidebar.subheader("Visualization Settings")
        colormap_options = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Hot', 'Cool', 'Rainbow', 'Jet']
        selected_colormap = st.sidebar.selectbox("Heatmap Colormap", colormap_options, index=0)
        
        # NEW: Y-axis range control for frequency spectrum
        st.sidebar.subheader("Frequency Spectrum Y-axis")
        auto_yaxis = st.sidebar.checkbox("Auto Y-axis range", value=True)
        if not auto_yaxis:
            y_min = st.sidebar.number_input("Y-axis Min", value=0.0, step=0.1, format="%.3f")
            y_max = st.sidebar.number_input("Y-axis Max", value=1.0, step=0.1, format="%.3f")
            yaxis_range = [y_min, y_max]
        else:
            yaxis_range = None
        
        # Time selection section in sidebar
        st.sidebar.subheader("Time Selection")
        time_axis = np.arange(seismic_data.shape[0]) * analyzer.sample_rate
        min_time = float(time_axis[0])
        max_time = float(time_axis[-1])
        default_time = float(time_axis[len(time_axis) // 2])
        
        # Initialize selected time in session state
        if 'selected_time' not in st.session_state:
            st.session_state.selected_time = default_time
        
        # Time input with number input
        selected_time_input = st.sidebar.number_input(
            "Analysis Time (ms)",
            min_value=min_time,
            max_value=max_time,
            value=st.session_state.selected_time,
            step=analyzer.sample_rate,
            format="%.1f",
            key="time_input"
        )
        
        # Apply button for time input
        if st.sidebar.button("Apply Time", use_container_width=True):
            st.session_state.selected_time = selected_time_input
            st.rerun()
        
        # MODIFIED: Fine-tune slider with 1ms step
        selected_time_slider = st.sidebar.slider(
            "Fine-tune Time",
            min_value=min_time,
            max_value=max_time,
            value=st.session_state.selected_time,
            step=1.0,  # Changed to 1ms step
            key="time_slider"
        )
        
        # Update if slider is used
        if selected_time_slider != st.session_state.selected_time:
            st.session_state.selected_time = selected_time_slider
            st.rerun()
        
        # NEW: Common Frequency Section Parameters
        st.sidebar.subheader("Common Frequency Section")
        common_freq = st.sidebar.slider(
            "Select Frequency for Common Section (Hz)",
            min_value=min_freq,
            max_value=max_freq,
            value=15,
            step=1,
            help="Display seismic section at this specific frequency to identify low-frequency shadows"
        )
        
        # Process button
        if st.sidebar.button("Run Spectral Analysis", type="primary"):
            with st.spinner("Performing spectral decomposition..."):
                # Apply spectral decomposition
                frequencies_continuous = np.linspace(min_freq, max_freq, num_frequencies)
                spectral_components = analyzer.apply_spectral_decomposition(
                    seismic_data, frequencies_continuous
                )
                
                # Store in session state
                st.session_state.spectral_components = spectral_components
                st.session_state.frequencies_continuous = frequencies_continuous
                st.session_state.analyzer = analyzer
                st.session_state.seismic_data = seismic_data
                st.session_state.trace_index = trace_index
        
        # Display results if analysis is done
        if 'spectral_components' in st.session_state:
            # Data flow explanation
            st.markdown(f"""
            <div class="data-flow-box">
                <h3>🔄 Data Flow: How the Frequency Spectrum is Calculated</h3>
                <p><b>Seismic Trace</b> → <b>Spectral Decomposition</b> → <b>Continuous Frequency Volume</b> → <b>Horizontal Slice at {st.session_state.selected_time:.1f} ms</b> → <b>Frequency Spectrum</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Time input box in main area
            st.markdown(f"""
            <div class="time-input-box">
                <h3>⏰ Time Selection Control</h3>
                <p>Current analysis time: <b>{st.session_state.selected_time:.1f} ms</b></p>
                <p>Time range: {min_time:.1f} ms to {max_time:.1f} ms | Sample rate: {analyzer.sample_rate} ms</p>
                <p><b>NEW:</b> Fine-tune slider now has 1ms precision for precise time selection</p>
                <p><b>NEW:</b> Input trace, continuous frequency volume, and ISA frequency spectrum plots are synchronized for zooming</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Information box
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Interactive Instructions</h3>
            <ul>
            <li><b>Plot Order:</b> 1. Input Trace → 2. Continuous Frequency Volume → 3. Frequency Spectrum → 4. ISA Frequency Spectrum → 5. Frequency Spectrum Details</li>
            <li><b>Synchronized Zoom:</b> Plots 1, 2, and 4 are synchronized (Input Trace, Continuous Frequency Volume, ISA Frequency Spectrum)</li>
            <li>Use the <b>time input box</b> in the sidebar to select analysis time</li>
            <li>Click <b>Apply Time</b> to extract a new frequency spectrum from the heatmap</li>
            <li>Use the <b>fine-tune slider</b> for precise time selection (now with 1ms precision)</li>
            <li><b>NEW:</b> Choose different colormaps for the heatmap visualization</li>
            <li><b>NEW:</b> Manually control Y-axis range for frequency spectrum plot</li>
            <li><b>NEW:</b> ISA Frequency Plot with synchronized zoom and detailed spectrum view</li>
            <li><b>NEW:</b> Common Frequency Section display for low-frequency shadow detection</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 1. First Plot: Input Trace, Continuous Frequency Volume, Frequency Spectrum
            st.subheader("1. Main Analysis Plots")
            fig, spectrum, frequencies, current_time = st.session_state.analyzer.create_interactive_plot(
                st.session_state.seismic_data,
                st.session_state.spectral_components,
                st.session_state.trace_index,
                st.session_state.selected_time,
                colormap=selected_colormap,
                yaxis_range=yaxis_range
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Second Plot: ISA Frequency Spectrum (synchronized with above plots)
            st.subheader("2. ISA Frequency Spectrum")
            st.markdown("""
            <div class="isa-plot-box">
                <h3>🔬 ISA Frequency Spectrum - Synchronized View</h3>
                <p>This plot is synchronized with the Input Trace and Continuous Frequency Volume plots above</p>
            </div>
            """, unsafe_allow_html=True)
            
            isa_fig = st.session_state.analyzer.create_isa_frequency_spectrum(
                st.session_state.seismic_data,
                st.session_state.spectral_components,
                st.session_state.trace_index,
                st.session_state.selected_time,
                colormap=selected_colormap
            )
            
            st.plotly_chart(isa_fig, use_container_width=True)
            
            # 3. Third Plot: Frequency Spectrum Details
            st.subheader("3. Frequency Spectrum Details")
            st.markdown("""
            <div class="details-plot-box">
                <h3>📊 Detailed Frequency Analysis</h3>
                <p>Detailed view of the frequency spectrum with Castagna frequency bands highlighted</p>
            </div>
            """, unsafe_allow_html=True)
            
            details_fig, details_spectrum, details_frequencies = st.session_state.analyzer.create_frequency_spectrum_details(
                st.session_state.seismic_data,
                st.session_state.spectral_components,
                st.session_state.trace_index,
                st.session_state.selected_time,
                yaxis_range
            )
            
            st.plotly_chart(details_fig, use_container_width=True)
            
            # 4. Fourth Plot: Common Frequency Section
            st.subheader("4. Common Frequency Section - Low-Frequency Shadow Detection")
            st.markdown("""
            <div class="low-freq-shadow">
                <h3>🔍 Castagna Low-Frequency Shadow Detection</h3>
                <p>Based on Castagna et al. (2003) - Compare different frequency sections to identify hydrocarbon-related low-frequency shadows</p>
                <p><b>Look for:</b> Strong low-frequency energy beneath potential reservoirs that disappears at higher frequencies</p>
            </div>
            """, unsafe_allow_html=True)
            
            common_freq_fig, common_freq_data = st.session_state.analyzer.create_common_frequency_section(
                st.session_state.spectral_components,
                common_freq,
                colormap=selected_colormap
            )
            
            st.plotly_chart(common_freq_fig, use_container_width=True)
            
            # Interpretation guidance for low-frequency shadows
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **📊 Low-Frequency Shadow Indicators:**
                - Strong amplitude at low frequencies (8-15 Hz)
                - Located beneath amplitude anomalies/bright spots
                - Disappears at higher frequencies (25+ Hz)
                - Better defines reservoir boundaries
                """)
            
            with col2:
                st.info("""
                **🎯 Castagna Methodology:**
                - Compare 10Hz vs 30Hz sections
                - Look for persistent low-frequency energy
                - Use multiple frequencies for confirmation
                - Combine with amplitude analysis
                """)
            
            # Verify the data flow - create continuous slice separately for verification
            continuous_slice, _ = st.session_state.analyzer.create_continuous_frequency_slice(
                st.session_state.seismic_data,
                st.session_state.spectral_components,
                st.session_state.trace_index
            )
            
            st.markdown(f"""
            <div class="info-box">
            <h3>🔍 Data Verification</h3>
            <p>The frequency spectrum is directly extracted from the continuous frequency volume (heatmap):</p>
            <ul>
            <li><b>Heatmap Shape:</b> {continuous_slice.shape[0]} time samples × {continuous_slice.shape[1]} frequency points</li>
            <li><b>Extraction:</b> Taking row at time index corresponding to {st.session_state.selected_time:.1f} ms</li>
            <li><b>Result:</b> 1D array of {len(spectrum)} amplitude values vs {len(frequencies)} frequencies</li>
            <li><b>Time Index:</b> {np.argmin(np.abs(time_axis - st.session_state.selected_time))} (closest to {st.session_state.selected_time:.1f} ms)</li>
            <li><b>Colormap:</b> {selected_colormap}</li>
            <li><b>Y-axis Range:</b> {'Auto' if auto_yaxis else f'Manual [{yaxis_range[0]}, {yaxis_range[1]}]'}</li>
            <li><b>Frequency Range:</b> {min_freq} Hz to {max_freq} Hz</li>
            <li><b>Common Frequency Section:</b> {common_freq} Hz</li>
            <li><b>Synchronized Plots:</b> Input Trace, Continuous Frequency Volume, ISA Frequency Spectrum</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Spectral characteristics
            characteristics = st.session_state.analyzer._get_spectral_characteristics(
                spectrum, frequencies, st.session_state.selected_time
            )
            
            # Display spectrum information
            st.markdown(f"""
            <div class="spectrum-info">
                <h3>📊 Spectrum Analysis at {characteristics['selected_time']:.1f} ms</h3>
                <p><b>Frequency Range:</b> {frequencies[0]:.1f} Hz to {frequencies[-1]:.1f} Hz | 
                <b>Spectrum Amplitude Range:</b> {np.min(spectrum):.3f} to {np.max(spectrum):.3f}</p>
                <p><b>Data Source:</b> Horizontal slice from continuous frequency volume at {characteristics['selected_time']:.1f} ms</p>
                <p><b>Visualization Settings:</b> Colormap: {selected_colormap} | Y-axis: {'Auto' if auto_yaxis else 'Manual'} | Synchronized Zoom: Enabled</p>
                <p><b>Common Frequency Analysis:</b> {common_freq} Hz section displayed for shadow detection</p>
                <p><b>Plot Order:</b> 1. Input Trace → 2. Continuous Frequency Volume → 3. Frequency Spectrum → 4. ISA Frequency Spectrum → 5. Frequency Spectrum Details</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display characteristics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Selected Time", f"{characteristics['selected_time']:.1f} ms")
                st.metric("Dominant Frequency", f"{characteristics['dominant_frequency']:.1f} Hz")
                st.metric("Peak Amplitude", f"{characteristics['peak_amplitude']:.3f}")
            
            with col2:
                st.metric("Bandwidth (FWHM)", f"{characteristics['bandwidth']:.1f} Hz")
                st.metric("Low Freq (10-20Hz)", f"{characteristics['low_freq_content']:.3f}")
                st.metric("Mid Freq (20-40Hz)", f"{characteristics['mid_freq_content']:.3f}")
            
            with col3:
                st.metric("High Freq (40-80Hz)", f"{characteristics['high_freq_content']:.3f}")
                
                # Castagna interpretation
                st.subheader("Castagna Interpretation")
                if characteristics['dominant_frequency'] < 20:
                    st.info("🔵 **Low Frequency**: Potential tuning effects, good for thick reservoir characterization")
                elif characteristics['dominant_frequency'] < 40:
                    st.info("🟢 **Mid Frequency**: Good resolution for medium-thick beds")
                else:
                    st.info("🟡 **High Frequency**: Excellent thin bed resolution")
            
            # Frequency band analysis
            st.subheader("Frequency Band Analysis")
            band_data = {
                'Frequency Band': ['Low (10-20 Hz)', 'Mid (20-40 Hz)', 'High (40-80 Hz)'],
                'Average Amplitude': [
                    characteristics['low_freq_content'],
                    characteristics['mid_freq_content'], 
                    characteristics['high_freq_content']
                ]
            }
            
            band_fig = px.bar(
                band_data, 
                x='Frequency Band', 
                y='Average Amplitude',
                title='Average Amplitude by Frequency Band',
                color='Frequency Band',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            band_fig.update_layout(showlegend=False)
            st.plotly_chart(band_fig, use_container_width=True)
            
            # Additional spectrum details
            with st.expander("📈 Detailed Spectrum Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Spectrum Statistics:**")
                    st.write(f"- Minimum amplitude: {np.min(spectrum):.4f}")
                    st.write(f"- Maximum amplitude: {np.max(spectrum):.4f}")
                    st.write(f"- Mean amplitude: {np.mean(spectrum):.4f}")
                    st.write(f"- Standard deviation: {np.std(spectrum):.4f}")
                    st.write(f"- Number of frequency points: {len(frequencies)}")
                    
                with col2:
                    st.write("**Frequency Content:**")
                    total_energy = np.sum(spectrum)
                    low_percent = (characteristics['low_freq_content'] / total_energy * 100) if total_energy > 0 else 0
                    mid_percent = (characteristics['mid_freq_content'] / total_energy * 100) if total_energy > 0 else 0
                    high_percent = (characteristics['high_freq_content'] / total_energy * 100) if total_energy > 0 else 0
                    
                    st.write(f"- Low frequency content: {low_percent:.1f}%")
                    st.write(f"- Mid frequency content: {mid_percent:.1f}%")
                    st.write(f"- High frequency content: {high_percent:.1f}%")
                    st.write(f"- Dominant frequency position: {np.argmax(spectrum)}/{len(spectrum)}")
            
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div style='text-align: center; padding: 4rem;'>
            <h2>Welcome to Castagna Spectral Analysis</h2>
            <p style='font-size: 1.2rem; color: #666;'>
                Upload a SEG-Y file to perform interactive spectral analysis based on Castagna's methodology.
            </p>
            <div class="info-box" style='max-width: 600px; margin: 2rem auto;'>
                <h3>📁 Supported Files</h3>
                <ul style='text-align: left;'>
                    <li>SEG-Y format (.sgy, .segy)</li>
                    <li>2D or 3D seismic data</li>
                    <li>Standard SEG-Y headers</li>
                </ul>
            </div>
            <div class="info-box" style='max-width: 600px; margin: 2rem auto;'>
                <h3>🔬 Analysis Features</h3>
                <ul style='text-align: left;'>
                    <li>Spectral decomposition using Ricker wavelets</li>
                    <li>Interactive time selection via text input</li>
                    <li>Continuous frequency volume visualization</li>
                    <li>Frequency spectrum extracted directly from frequency volume</li>
                    <li>Castagna frequency band analysis</li>
                    <li>Dominant frequency detection</li>
                    <li>Bandwidth calculation (FWHM)</li>
                    <li><b>NEW:</b> 1ms precision in fine-tune time slider</li>
                    <li><b>NEW:</b> Multiple colormap options for heatmap</li>
                    <li><b>NEW:</b> Manual Y-axis range control for frequency spectrum</li>
                    <li><b>NEW:</b> Synchronized zoom between input trace, continuous frequency volume, and ISA frequency spectrum</li>
                    <li><b>NEW:</b> Plot order: 1. Input Trace → 2. Continuous Frequency Volume → 3. Frequency Spectrum → 4. ISA Frequency Spectrum → 5. Frequency Spectrum Details</li>
                    <li><b>NEW:</b> Common Frequency Section for low-frequency shadow detection</li>
                </ul>
            </div>
            <div class="info-box" style='max-width: 600px; margin: 2rem auto;'>
                <h3>🎯 Castagna Methodology</h3>
                <p style='text-align: left;'>
                    Based on Castagna et al. (2003) "Instantaneous spectral analysis" - 
                    This tool provides frequency-dependent seismic attribute analysis 
                    for reservoir characterization and thin-bed detection.
                </p>
                <p style='text-align: left;'>
                    <b>NEW:</b> Low-frequency shadow detection using common frequency sections
                    as described in the paper for direct hydrocarbon indication.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
