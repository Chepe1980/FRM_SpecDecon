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
        """Load SEG-Y data using segyio with proper trace handling"""
        try:
            with segyio.open(segy_path, "r", strict=False) as segyfile:
                # Get basic file information
                num_traces = segyfile.tracecount
                num_samples = segyfile.samples.size
                
                st.sidebar.info(f"SEG-Y File Info: {num_traces} traces, {num_samples} samples")
                
                # Read all traces manually - most reliable method
                data = []
                for i, trace in enumerate(segyfile.trace):
                    data.append(trace)
                
                data = np.array(data).T  # Transpose to samples x traces
                
                # Get sampling interval - convert to milliseconds
                try:
                    if hasattr(segyfile, 'bin'):
                        # Try different possible field names for sample interval
                        if segyio.BinField.Interval in segyfile.bin:
                            sample_interval_microsec = segyfile.bin[segyio.BinField.Interval]
                            self.sample_rate = sample_interval_microsec / 1000.0  # Convert μs to ms
                        else:
                            # If we can't read from header, use default
                            self.sample_rate = 4.0
                            st.sidebar.warning("Using default sample rate: 4.0 ms")
                except Exception as e:
                    # If we can't read from header, use default
                    self.sample_rate = 4.0
                    st.sidebar.warning(f"Using default sample rate: 4.0 ms. Error: {e}")
                
                st.sidebar.success(f"Loaded data shape: {data.shape}")
                st.sidebar.info(f"Sample rate: {self.sample_rate} ms")
                
                return data
                
        except Exception as e:
            st.error(f"Error loading SEG-Y file: {e}")
            # Create dummy data for testing
            st.warning("Using dummy data for demonstration")
            dummy_data = np.random.randn(1251, 24)
            # Add some realistic seismic signal
            t = np.arange(1251) * 4.0  # Time in ms
            for i in range(24):
                freq = 10 + i * 2  # Varying frequency across traces
                dummy_data[:, i] += 0.5 * np.sin(2 * np.pi * freq * t / 1000) * np.exp(-t / 1000)
            return dummy_data

    def castagna_ricker_wavelet(self, frequency, length=0.128, dt=0.004):
        """
        Create Ricker wavelet following Castagna's spectral decomposition approach
        Based on Castagna et al. (2003) - Instantaneous spectral analysis
        
        RICKER WAVELET EQUATION:
        ψ(t) = (1 - 2π²f²t²) * exp(-π²f²t²)
        
        Where:
        - f = center frequency (Hz)
        - t = time (seconds)
        - ψ(t) = wavelet amplitude at time t
        
        Reference: Castagna, J.P., Sun, S., and Siegfried, R.W., 2003, 
        Instantaneous spectral analysis: Detection of low-frequency shadows 
        associated with hydrocarbons: The Leading Edge, 22, 120-127.
        """
        t = np.arange(-length/2, length/2, dt)
        # Ricker wavelet formula as used in Castagna's work
        y = (1.0 - 2.0 * (np.pi * frequency * t) ** 2) * np.exp(-(np.pi * frequency * t) ** 2)
        return y / np.max(np.abs(y))

    def morlet_wavelet(self, frequency, length=0.128, dt=0.004, omega0=6.0):
        """
        Create Morlet wavelet for alternative spectral decomposition
        
        MORLET WAVELET EQUATION:
        ψ(t) = π^(-1/4) * exp(iω₀t) * exp(-t²/2)
        
        Complex form:
        ψ(t) = π^(-1/4) * [exp(iω₀t) - exp(-ω₀²/2)] * exp(-t²/2)
        
        Where:
        - ω₀ = central frequency parameter (typically 5-6 for seismic)
        - f = center frequency (Hz)
        - t = time (seconds)
        
        Advantages:
        - Better time-frequency localization than Ricker
        - Complex-valued output enables phase analysis
        - Better for analyzing non-stationary signals
        
        Reference: Morlet, J., 1983, Sampling theory and wave propagation: 
        NATO ASI Series, 1, 233-261.
        """
        t = np.arange(-length/2, length/2, dt)
        
        # Normalization constant
        normalization = np.pi ** (-0.25)
        
        # Angular frequency
        omega = 2.0 * np.pi * frequency
        
        # Morlet wavelet (complex) - corrected implementation
        # Scale the time axis by frequency for proper wavelet scaling
        scaled_t = omega * t
        wavelet_complex = normalization * np.exp(1j * omega0 * scaled_t) * np.exp(-scaled_t**2 / 2)
        
        # For seismic applications, we often use the real part
        wavelet_real = np.real(wavelet_complex)
        
        return wavelet_real / np.max(np.abs(wavelet_real))

    def apply_spectral_decomposition(self, seismic_data, frequencies=np.arange(10, 81, 5), wavelet_type='ricker'):
        """
        Spectral decomposition with discrete frequencies
        Based on Castagna's Instantaneous Spectral Analysis (ISA) methodology
        
        MATHEMATICAL BASIS:
        The seismic trace s(t) is decomposed into frequency components using 
        convolution with wavelets:
        
        S(f, t) = s(t) ∗ ψ_f(t)
        
        Where:
        - S(f, t) = spectral component at frequency f and time t
        - s(t) = input seismic trace
        - ψ_f(t) = wavelet at frequency f (Ricker or Morlet)
        - ∗ denotes convolution
        
        This produces a time-frequency representation that avoids windowing
        artifacts of traditional Fourier methods.
        """
        spectral_components = {}

        for freq in frequencies:
            if wavelet_type == 'ricker':
                wavelet = self.castagna_ricker_wavelet(freq)
            elif wavelet_type == 'morlet':
                wavelet = self.morlet_wavelet(freq)
            else:
                raise ValueError("Wavelet type must be 'ricker' or 'morlet'")
                
            filtered_traces = []

            for trace in seismic_data.T:
                # Convolve with wavelet for spectral component
                conv_result = np.convolve(trace, wavelet, mode='same')
                # Take absolute value for amplitude spectrum
                filtered_traces.append(np.abs(conv_result))

            spectral_components[f'{freq}Hz'] = np.array(filtered_traces).T

        return spectral_components

    def apply_morlet_spectral_decomposition(self, seismic_data, frequencies=np.arange(10, 81, 2)):
        """
        Spectral decomposition using Morlet wavelets with higher frequency resolution
        
        MORLET DECOMPOSITION THEORY:
        The Morlet wavelet provides superior time-frequency localization
        compared to Ricker wavelets due to its Gaussian envelope and
        complex-valued nature.
        
        Key advantages:
        1. Better time-frequency resolution trade-off
        2. Enables phase and amplitude analysis
        3. More suitable for non-stationary seismic signals
        4. Better frequency localization
        
        Reference: Goupillaud, P., Grossmann, A., and Morlet, J., 1984, 
        Cycle-octave and related transforms in seismic signal analysis: 
        Geoexploration, 23, 85-102.
        """
        return self.apply_spectral_decomposition(seismic_data, frequencies, wavelet_type='morlet')

    def create_continuous_frequency_slice(self, seismic_data, spectral_components, trace_index=0, num_interp_points=200):
        """
        Create a continuous frequency slice using interpolation
        
        METHODOLOGY:
        Cubic spline interpolation is used to create a continuous frequency
        volume from discrete frequency components, enabling smooth frequency
        visualization and analysis.
        """
        # Get original frequencies and data
        frequencies_original = sorted([float(k.replace('Hz', '')) for k in spectral_components.keys()])

        # Extract data for the specific trace
        frequency_slice_original = np.zeros((seismic_data.shape[0], len(frequencies_original)))
        for i, freq in enumerate(frequencies_original):
            frequency_slice_original[:, i] = spectral_components[f'{freq}Hz'][:, trace_index]

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
        
        This represents a horizontal slice through the time-frequency volume
        at a specific time, showing amplitude vs frequency.
        """
        # Find closest time index
        time_idx = np.argmin(np.abs(time_axis - time_ms))
        
        # Get frequency spectrum at this time - this is the horizontal slice from the heatmap
        frequency_spectrum = continuous_slice[time_idx, :]
        
        return frequency_spectrum, time_idx

    def create_isa_frequency_spectrum(self, seismic_data, spectral_components, trace_index=0, selected_time=None, 
                                    colormap='Viridis', wavelet_type='ricker'):
        """
        Create ISA frequency spectrum plot (heatmap only for synchronization)
        
        THEORETICAL BACKGROUND:
        This visualization shows the Instantaneous Spectral Analysis (ISA) 
        results as a time-frequency heatmap. Each horizontal slice represents
        the frequency spectrum at that particular time, while vertical slices
        show amplitude variation with time at specific frequencies.
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

        wavelet_name = "Ricker" if wavelet_type == 'ricker' else "Morlet"
        fig.update_layout(
            title=f'ISA Frequency Spectrum ({wavelet_name} Wavelet) - Trace {trace_index}',
            xaxis_title="Frequency (Hz)",
            yaxis_title="Time (ms)",
            yaxis=dict(autorange='reversed'),
            height=500,
            margin=dict(l=80, r=80, t=80, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        return fig, selected_spectrum, frequencies_continuous, selected_time

    def create_frequency_spectrum_details(self, spectrum, frequencies, selected_time, yaxis_range=None, wavelet_type='ricker'):
        """
        Create detailed frequency spectrum plot from ISA data
        
        FREQUENCY BAND ANALYSIS:
        Based on Castagna's methodology, seismic frequencies are categorized into:
        - Low frequency (10-20 Hz): Tuning effects, thick reservoir characterization
        - Mid frequency (20-40 Hz): Medium-thick bed resolution  
        - High frequency (40-80 Hz): Thin bed resolution, stratigraphic details
        
        HYDROCARBON INDICATORS:
        Low-frequency shadows beneath amplitude anomalies may indicate:
        - Hydrocarbon-related attenuation
        - Tuning effects
        - Fluid-related dispersion
        """
        fig = go.Figure()
        
        # Frequency Spectrum at Selected Time
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=spectrum,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Frequency Spectrum',
                hovertemplate='<b>Frequency</b>: %{x:.1f} Hz<br><b>Amplitude</b>: %{y:.3f}<extra></extra>'
            )
        )

        # Add dominant frequency marker
        dominant_freq_idx = np.argmax(spectrum)
        dominant_freq = frequencies[dominant_freq_idx]
        dominant_amp = spectrum[dominant_freq_idx]
        
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
        low_band_mask = (frequencies >= 10) & (frequencies <= 20)
        mid_band_mask = (frequencies >= 20) & (frequencies <= 40)
        high_band_mask = (frequencies >= 40) & (frequencies <= 80)
        
        if np.any(low_band_mask):
            fig.add_trace(
                go.Scatter(
                    x=frequencies[low_band_mask],
                    y=spectrum[low_band_mask],
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
                    x=frequencies[mid_band_mask],
                    y=spectrum[mid_band_mask],
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
                    x=frequencies[high_band_mask],
                    y=spectrum[high_band_mask],
                    mode='lines',
                    line=dict(color='purple', width=4),
                    name='High Freq (40-80 Hz)',
                    showlegend=True,
                    hoverinfo='skip'
                )
            )

        wavelet_name = "Ricker" if wavelet_type == 'ricker' else "Morlet"
        # Update layout
        fig.update_layout(
            title=f'Frequency Spectrum Details ({wavelet_name} Wavelet) at {selected_time:.1f} ms',
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
            fig.update_yaxes(range=[0, np.max(spectrum) * 1.1])

        return fig

    def create_common_frequency_section(self, spectral_components, selected_frequency, colormap='Viridis', wavelet_type='ricker'):
        """
        Create a common frequency section as described in Castagna's paper
        This shows the seismic section at a specific frequency to identify low-frequency shadows
        
        LOW-FREQUENCY SHADOW THEORY:
        Low-frequency shadows beneath hydrocarbon reservoirs can be caused by:
        1. Intrinsic attenuation in gas-filled reservoirs
        2. Tuning effects from thin layers
        3. Wave interference patterns
        4. Fluid-related velocity dispersion
        
        Reference: Castagna et al. (2003) identified multiple mechanisms for
        low-frequency shadows beyond simple attenuation.
        """
        # Get the spectral component for the selected frequency
        freq_key = f'{selected_frequency}Hz'
        if freq_key not in spectral_components:
            # Find closest available frequency
            available_freqs = [float(k.replace('Hz', '')) for k in spectral_components.keys()]
            closest_freq = min(available_freqs, key=lambda x: abs(x - selected_frequency))
            freq_key = f'{closest_freq}Hz'
            selected_frequency = closest_freq
        
        frequency_section = spectral_components[freq_key]
        
        # Create the plot
        fig = go.Figure()
        
        wavelet_name = "Ricker" if wavelet_type == 'ricker' else "Morlet"
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
            title=f'Common Frequency Section ({wavelet_name} Wavelet) - {selected_frequency} Hz',
            xaxis_title='Trace Number',
            yaxis_title='Time (ms)',
            yaxis=dict(autorange='reversed'),
            height=500,
            margin=dict(l=80, r=80, t=80, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig, frequency_section

    def create_interactive_plot(self, seismic_data, spectral_components, trace_index=0, selected_time=None, 
                              colormap='Viridis', yaxis_range=None, wavelet_type='ricker'):
        """
        Create interactive plot with synchronized zoom and time selection
        
        METHODOLOGY OVERVIEW:
        This implements Castagna's Instantaneous Spectral Analysis (ISA) which
        provides superior time-frequency resolution compared to traditional
        Fourier-based methods by avoiding windowing artifacts.
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

        wavelet_name = "Ricker" if wavelet_type == 'ricker' else "Morlet"
        # Create subplots with synchronized y-axes for first two plots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'Input Seismic Trace {trace_index}',
                f'Continuous Frequency Volume Slice ({wavelet_name} Wavelet)',
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
            title_text=f"Interactive Castagna Spectral Analysis ({wavelet_name} Wavelet) - Trace {trace_index}",
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
        """Calculate spectral characteristics for display
        
        SPECTRAL METRICS:
        - Dominant frequency: Frequency with maximum amplitude
        - Bandwidth (FWHM): Full width at half maximum
        - Frequency band content: Average amplitude in Castagna's frequency bands
        
        These metrics help characterize reservoir properties and identify
        hydrocarbon-related frequency anomalies.
        """
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


def display_analysis_results(analyzer, spectrum, frequencies, selected_time, time_axis, continuous_slice, 
                           wavelet_type, selected_colormap, auto_yaxis, yaxis_range, min_freq, max_freq, common_freq):
    """Helper function to display analysis results for both wavelet types"""
    
    wavelet_name = "Ricker" if wavelet_type == 'ricker' else "Morlet"
    
    # Verify the data flow - create continuous slice separately for verification
    st.markdown(f"""
    <div class="info-box">
    <h3>🔍 Data Verification ({wavelet_name} Wavelet)</h3>
    <p>The frequency spectrum is directly extracted from the continuous frequency volume (heatmap):</p>
    <ul>
    <li><b>Heatmap Shape:</b> {continuous_slice.shape[0]} time samples × {continuous_slice.shape[1]} frequency points</li>
    <li><b>Extraction:</b> Taking row at time index corresponding to {selected_time:.1f} ms</li>
    <li><b>Result:</b> 1D array of {len(spectrum)} amplitude values vs {len(frequencies)} frequencies</li>
    <li><b>Time Index:</b> {np.argmin(np.abs(time_axis - selected_time))} (closest to {selected_time:.1f} ms)</li>
    <li><b>Wavelet Type:</b> {wavelet_name}</li>
    <li><b>Colormap:</b> {selected_colormap}</li>
    <li><b>Y-axis Range:</b> {'Auto' if auto_yaxis else f'Manual [{yaxis_range[0]}, {yaxis_range[1]}]'}</li>
    <li><b>Frequency Range:</b> {min_freq} Hz to {max_freq} Hz</li>
    <li><b>Common Frequency Section:</b> {common_freq} Hz</li>
    <li><b>Synchronized Plots:</b> Input Trace, Continuous Frequency Volume, ISA Frequency Spectrum</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Spectral characteristics
    characteristics = analyzer._get_spectral_characteristics(
        spectrum, frequencies, selected_time
    )
    
    # Display spectrum information
    st.markdown(f"""
    <div class="spectrum-info">
        <h3>📊 Spectrum Analysis at {characteristics['selected_time']:.1f} ms ({wavelet_name} Wavelet)</h3>
        <p><b>Frequency Range:</b> {frequencies[0]:.1f} Hz to {frequencies[-1]:.1f} Hz | 
        <b>Spectrum Amplitude Range:</b> {np.min(spectrum):.3f} to {np.max(spectrum):.3f}</p>
        <p><b>Data Source:</b> Horizontal slice from continuous frequency volume at {characteristics['selected_time']:.1f} ms</p>
        <p><b>Visualization Settings:</b> Wavelet: {wavelet_name} | Colormap: {selected_colormap} | Y-axis: {'Auto' if auto_yaxis else 'Manual'} | Synchronized Zoom: Enabled</p>
        <p><b>Common Frequency Analysis:</b> {common_freq} Hz section displayed for shadow detection</p>
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
        title=f'Average Amplitude by Frequency Band ({wavelet_name} Wavelet)',
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
        .theory-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .equation-box {
            background-color: #2c3e50;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            font-family: monospace;
        }
        .wavelet-comparison {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: #333;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Castagna Spectral Analysis</h1>', unsafe_allow_html=True)
    
    # Theory and Methodology Section
    st.markdown("""
    <div class="theory-box">
        <h2>📚 Theoretical Background & Methodology</h2>
        <h3>Instantaneous Spectral Analysis (ISA) - Castagna et al. (2003)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🔬 Core Methodology</h4>
        <p><b>Instantaneous Spectral Analysis (ISA)</b> provides continuous time-frequency analysis 
        using wavelet transforms to avoid windowing problems of conventional Fourier analysis.</p>
        
        <div class="equation-box">
        <b>Ricker Wavelet Equation:</b><br>
        ψ(t) = (1 - 2π²f²t²) · exp(-π²f²t²)
        </div>
        
        <div class="equation-box">
        <b>Morlet Wavelet Equation:</b><br>
        ψ(t) = π^(-1/4) · exp(iω₀t) · exp(-t²/2)
        </div>
        
        <div class="equation-box">
        <b>Spectral Decomposition:</b><br>
        S(f, t) = s(t) ∗ ψ_f(t)
        </div>
        
        <p><b>Advantages over FFT:</b></p>
        <ul>
        <li>No windowing artifacts</li>
        <li>Superior time-frequency resolution</li>
        <li>Better vertical resolution</li>
        <li>No spectral notching</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Hydrocarbon Detection</h4>
        <p><b>Low-Frequency Shadows:</b> Anomalous low-frequency energy beneath potential reservoirs</p>
        
        <p><b>Detection Mechanisms:</b></p>
        <ul>
        <li>Anomalous attenuation in gas reservoirs</li>
        <li>Tuning frequency effects</li>
        <li>Frequency-dependent AVO</li>
        <li>Wave interference patterns</li>
        </ul>
        
        <p><b>Frequency Bands (Castagna):</b></p>
        <ul>
        <li><b>Low (10-20 Hz):</b> Thick reservoir characterization</li>
        <li><b>Mid (20-40 Hz):</b> Medium bed resolution</li>
        <li><b>High (40-80 Hz):</b> Thin bed detection</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Wavelet Comparison Section
    st.markdown("""
    <div class="wavelet-comparison">
        <h3>🔄 Wavelet Comparison: Ricker vs Morlet</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <h4>Ricker Wavelet (Tab 1)</h4>
                <ul>
                <li><b>Type:</b> Real-valued, symmetric</li>
                <li><b>Advantages:</b> Simple, zero-phase, good for seismic</li>
                <li><b>Limitations:</b> Limited frequency localization</li>
                <li><b>Best for:</b> General seismic analysis, amplitude studies</li>
                </ul>
            </div>
            <div>
                <h4>Morlet Wavelet (Tab 2)</h4>
                <ul>
                <li><b>Type:</b> Complex-valued, modulated Gaussian</li>
                <li><b>Advantages:</b> Better time-frequency localization, enables phase analysis</li>
                <li><b>Limitations:</b> More computationally intensive</li>
                <li><b>Best for:</b> Detailed frequency analysis, non-stationary signals</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # References
    with st.expander("📖 References & Further Reading"):
        st.markdown("""
        **Primary Reference:**
        - Castagna, J.P., Sun, S., and Siegfried, R.W., 2003, Instantaneous spectral analysis: 
          Detection of low-frequency shadows associated with hydrocarbons: The Leading Edge, 22, 120-127.
        
        **Wavelet Theory:**
        - Morlet, J., 1983, Sampling theory and wave propagation: NATO ASI Series, 1, 233-261.
        - Goupillaud, P., Grossmann, A., and Morlet, J., 1984, Cycle-octave and related transforms 
          in seismic signal analysis: Geoexploration, 23, 85-102.
        - Daubechies, I., 1992, Ten lectures on wavelets: Society for Industrial and Applied Mathematics.
        
        **Related Works:**
        - Partyka, G., Gridley, J., and Lopez, J., 1999, Interpretational applications of spectral 
          decomposition in reservoir characterization: The Leading Edge, 18, 353-360.
        - Marfurt, K.J., and Kirlin, R.L., 2001, Narrow-band spectral analysis and thin-bed tuning: 
          Geophysics, 66, 1274-1283.
        - Chakraborty, A., and Okaya, D., 1995, Frequency-time decomposition of seismic data using 
          wavelet-based methods: Geophysics, 60, 1906-1916.
        - Ebrom, D., 1996, The low-frequency gas shadow on seismic sections: SEG/EAGE Summer Research 
          Workshop on Wave Propagation in Rocks.
        
        **Key Equations:**
        - Ricker Wavelet: ψ(t) = (1 - 2π²f²t²) · exp(-π²f²t²)
        - Morlet Wavelet: ψ(t) = π^(-1/4) · exp(iω₀t) · exp(-t²/2)
        - Spectral Decomposition: S(f, t) = s(t) ∗ ψ_f(t)
        - Bandwidth: FWHM = f_max - f_min where amplitude ≥ 0.5 × max_amplitude
        """)
    
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
        
        # Show data information
        st.sidebar.info(f"Data dimensions: {seismic_data.shape[0]} samples × {seismic_data.shape[1]} traces")
        st.sidebar.info(f"Sample rate: {analyzer.sample_rate} ms")
        
        # Configuration options - Handle single trace case
        if seismic_data.shape[1] > 1:
            trace_index = st.sidebar.slider(
                "Select Trace Index", 
                min_value=0, 
                max_value=seismic_data.shape[1]-1, 
                value=0
            )
        else:
            trace_index = 0
            st.sidebar.info("Single trace detected - using trace 0")
        
        # Analysis parameters
        st.sidebar.subheader("Analysis Parameters")
        min_freq = st.sidebar.slider("Minimum Frequency (Hz)", 1, 30, 10)
        max_freq = st.sidebar.slider("Maximum Frequency (Hz)", 50, 100, 80)
        num_frequencies = st.sidebar.slider("Number of Frequencies", 10, 100, 50)
        
        # Visualization settings
        st.sidebar.subheader("Visualization Settings")
        colormap_options = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Hot', 'Cool', 'Rainbow', 'Jet']
        selected_colormap = st.sidebar.selectbox("Heatmap Colormap", colormap_options, index=0)
        
        st.sidebar.subheader("Frequency Spectrum Y-axis")
        auto_yaxis = st.sidebar.checkbox("Auto Y-axis range", value=True)
        if not auto_yaxis:
            y_min = st.sidebar.number_input("Y-axis Min", value=0.0, step=0.1, format="%.3f")
            y_max = st.sidebar.number_input("Y-axis Max", value=1.0, step=0.1, format="%.3f")
            yaxis_range = [y_min, y_max]
        else:
            yaxis_range = None
        
        # Time selection section in sidebar - ALL IN MILLISECONDS
        st.sidebar.subheader("Time Selection (ms)")
        time_axis = np.arange(seismic_data.shape[0]) * analyzer.sample_rate
        min_time = float(time_axis[0])
        max_time = float(time_axis[-1])
        default_time = float(time_axis[len(time_axis) // 2])
        
        # Initialize selected time in session state
        if 'selected_time' not in st.session_state:
            st.session_state.selected_time = default_time
        
        # Time input with number input - IN MILLISECONDS
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
        
        # Fine-tune slider with 1ms step - IN MILLISECONDS
        selected_time_slider = st.sidebar.slider(
            "Fine-tune Time (ms)",
            min_value=min_time,
            max_value=max_time,
            value=st.session_state.selected_time,
            step=1.0,
            key="time_slider"
        )
        
        # Update if slider is used
        if selected_time_slider != st.session_state.selected_time:
            st.session_state.selected_time = selected_time_slider
            st.rerun()
        
        # Common Frequency Section Parameters
        st.sidebar.subheader("Common Frequency Section")
        common_freq = st.sidebar.slider(
            "Select Frequency for Common Section (Hz)",
            min_value=min_freq,
            max_value=max_freq,
            value=15,
            step=1,
            help="Display seismic section at this specific frequency to identify low-frequency shadows"
        )
        
        # Create tabs
        tab1, tab2 = st.tabs(["🎯 Ricker Wavelet (Original)", "🌀 Morlet Wavelet"])
        
        # Tab 1: Ricker Wavelet (Original - Unchanged)
        with tab1:
            # Process button for Ricker
            if st.sidebar.button("Run Ricker Analysis", type="primary", key="ricker_btn"):
                with st.spinner("Performing Ricker spectral decomposition..."):
                    # Apply spectral decomposition with Ricker wavelet
                    frequencies_continuous = np.linspace(min_freq, max_freq, num_frequencies)
                    spectral_components = analyzer.apply_spectral_decomposition(
                        seismic_data, frequencies_continuous, wavelet_type='ricker'
                    )
                    
                    # Store in session state
                    st.session_state.ricker_spectral_components = spectral_components
                    st.session_state.ricker_frequencies_continuous = frequencies_continuous
                    st.session_state.ricker_analyzer = analyzer
                    st.session_state.ricker_seismic_data = seismic_data
                    st.session_state.ricker_trace_index = trace_index
        
            # Display results if Ricker analysis is done
            if 'ricker_spectral_components' in st.session_state:
                # Data flow explanation
                st.markdown(f"""
                <div class="data-flow-box">
                    <h3>🔄 Data Flow: How the Frequency Spectrum is Calculated (Ricker Wavelet)</h3>
                    <p><b>Seismic Trace</b> → <b>Spectral Decomposition</b> → <b>Continuous Frequency Volume</b> → <b>Horizontal Slice at {st.session_state.selected_time:.1f} ms</b> → <b>Frequency Spectrum</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Time input box in main area
                st.markdown(f"""
                <div class="time-input-box">
                    <h3>⏰ Time Selection Control - Ricker Wavelet</h3>
                    <p>Current analysis time: <b>{st.session_state.selected_time:.1f} ms</b></p>
                    <p>Wavelet type: <b>Ricker</b></p>
                    <p>Time range: {min_time:.1f} ms to {max_time:.1f} ms | Sample rate: {analyzer.sample_rate} ms</p>
                    <p><b>Features:</b> Fine-tune slider with 1ms precision, synchronized zoom between plots</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Information box
                st.markdown("""
                <div class="info-box">
                <h3>🎯 Interactive Instructions (Ricker Wavelet)</h3>
                <ul>
                <li><b>Plot Order:</b> 1. Input Trace → 2. Continuous Frequency Volume → 3. Frequency Spectrum → 4. ISA Frequency Spectrum → 5. Frequency Spectrum Details</li>
                <li><b>Synchronized Zoom:</b> Plots 1, 2, and 4 are synchronized (Input Trace, Continuous Frequency Volume, ISA Frequency Spectrum)</li>
                <li>Use the <b>time input box</b> in the sidebar to select analysis time</li>
                <li>Click <b>Apply Time</b> to extract a new frequency spectrum from the heatmap</li>
                <li>Use the <b>fine-tune slider</b> for precise time selection (1ms precision)</li>
                <li><b>Original Castagna Method:</b> Using Ricker wavelet as per the 2003 paper</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # 1. First Plot: Input Trace, Continuous Frequency Volume, Frequency Spectrum
                st.subheader("1. Main Analysis Plots (Ricker Wavelet)")
                fig, spectrum, frequencies, current_time = st.session_state.ricker_analyzer.create_interactive_plot(
                    st.session_state.ricker_seismic_data,
                    st.session_state.ricker_spectral_components,
                    st.session_state.ricker_trace_index,
                    st.session_state.selected_time,
                    colormap=selected_colormap,
                    yaxis_range=yaxis_range,
                    wavelet_type='ricker'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Second Plot: ISA Frequency Spectrum (synchronized with above plots)
                st.subheader("2. ISA Frequency Spectrum (Ricker Wavelet)")
                st.markdown("""
                <div class="isa-plot-box">
                    <h3>🔬 ISA Frequency Spectrum (Ricker Wavelet) - Synchronized View</h3>
                    <p>This plot is synchronized with the Input Trace and Continuous Frequency Volume plots above</p>
                </div>
                """, unsafe_allow_html=True)
                
                isa_fig, isa_spectrum, isa_frequencies, isa_time = st.session_state.ricker_analyzer.create_isa_frequency_spectrum(
                    st.session_state.ricker_seismic_data,
                    st.session_state.ricker_spectral_components,
                    st.session_state.ricker_trace_index,
                    st.session_state.selected_time,
                    colormap=selected_colormap,
                    wavelet_type='ricker'
                )
                
                st.plotly_chart(isa_fig, use_container_width=True)
                
                # 3. Third Plot: Frequency Spectrum Details
                st.subheader("3. Frequency Spectrum Details (Ricker Wavelet)")
                st.markdown("""
                <div class="details-plot-box">
                    <h3>📊 Detailed Frequency Analysis (Ricker Wavelet)</h3>
                    <p>Detailed view of the frequency spectrum with Castagna frequency bands highlighted</p>
                </div>
                """, unsafe_allow_html=True)
                
                details_fig = st.session_state.ricker_analyzer.create_frequency_spectrum_details(
                    isa_spectrum, isa_frequencies, st.session_state.selected_time, yaxis_range, 'ricker'
                )
                
                st.plotly_chart(details_fig, use_container_width=True)
                
                # 4. Fourth Plot: Common Frequency Section
                st.subheader("4. Common Frequency Section (Ricker Wavelet) - Low-Frequency Shadow Detection")
                st.markdown("""
                <div class="low-freq-shadow">
                    <h3>🔍 Castagna Low-Frequency Shadow Detection (Ricker Wavelet)</h3>
                    <p>Based on Castagna et al. (2003) - Compare different frequency sections to identify hydrocarbon-related low-frequency shadows</p>
                    <p><b>Look for:</b> Strong low-frequency energy beneath potential reservoirs that disappears at higher frequencies</p>
                </div>
                """, unsafe_allow_html=True)
                
                common_freq_fig, common_freq_data = st.session_state.ricker_analyzer.create_common_frequency_section(
                    st.session_state.ricker_spectral_components,
                    common_freq,
                    colormap=selected_colormap,
                    wavelet_type='ricker'
                )
                
                st.plotly_chart(common_freq_fig, use_container_width=True)
                
                # Get continuous slice for verification
                continuous_slice, _ = st.session_state.ricker_analyzer.create_continuous_frequency_slice(
                    st.session_state.ricker_seismic_data,
                    st.session_state.ricker_spectral_components,
                    st.session_state.ricker_trace_index
                )
                
                # Display Ricker-specific results
                display_analysis_results(st.session_state.ricker_analyzer, spectrum, frequencies, 
                                      st.session_state.selected_time, time_axis, continuous_slice, 
                                      'ricker', selected_colormap, auto_yaxis, yaxis_range, 
                                      min_freq, max_freq, common_freq)
        
        # Tab 2: Morlet Wavelet
        with tab2:
            # Process button for Morlet
            if st.sidebar.button("Run Morlet Analysis", type="primary", key="morlet_btn"):
                with st.spinner("Performing Morlet spectral decomposition with higher frequency resolution..."):
                    # Apply spectral decomposition with Morlet wavelet - use higher frequency resolution
                    morlet_num_frequencies = min(100, num_frequencies * 2)  # Double the frequency resolution for Morlet
                    frequencies_continuous = np.linspace(min_freq, max_freq, morlet_num_frequencies)
                    spectral_components = analyzer.apply_morlet_spectral_decomposition(
                        seismic_data, frequencies_continuous
                    )
                    
                    # Store in session state
                    st.session_state.morlet_spectral_components = spectral_components
                    st.session_state.morlet_frequencies_continuous = frequencies_continuous
                    st.session_state.morlet_analyzer = analyzer
                    st.session_state.morlet_seismic_data = seismic_data
                    st.session_state.morlet_trace_index = trace_index
        
            # Display results if Morlet analysis is done
            if 'morlet_spectral_components' in st.session_state:
                # Data flow explanation
                st.markdown(f"""
                <div class="data-flow-box">
                    <h3>🔄 Data Flow: How the Frequency Spectrum is Calculated (Morlet Wavelet)</h3>
                    <p><b>Seismic Trace</b> → <b>Spectral Decomposition</b> → <b>Continuous Frequency Volume</b> → <b>Horizontal Slice at {st.session_state.selected_time:.1f} ms</b> → <b>Frequency Spectrum</b></p>
                    <p><b>Note:</b> Morlet analysis uses higher frequency resolution ({len(st.session_state.morlet_frequencies_continuous)} frequencies) for better time-frequency localization</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Time input box in main area
                st.markdown(f"""
                <div class="time-input-box">
                    <h3>⏰ Time Selection Control - Morlet Wavelet</h3>
                    <p>Current analysis time: <b>{st.session_state.selected_time:.1f} ms</b></p>
                    <p>Wavelet type: <b>Morlet</b></p>
                    <p>Time range: {min_time:.1f} ms to {max_time:.1f} ms | Sample rate: {analyzer.sample_rate} ms</p>
                    <p><b>Features:</b> Higher frequency resolution ({len(st.session_state.morlet_frequencies_continuous)} frequencies), better time-frequency localization</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Information box
                st.markdown("""
                <div class="info-box">
                <h3>🎯 Interactive Instructions (Morlet Wavelet)</h3>
                <ul>
                <li><b>Plot Order:</b> 1. Input Trace → 2. Continuous Frequency Volume → 3. Frequency Spectrum → 4. ISA Frequency Spectrum → 5. Frequency Spectrum Details</li>
                <li><b>Synchronized Zoom:</b> Plots 1, 2, and 4 are synchronized (Input Trace, Continuous Frequency Volume, ISA Frequency Spectrum)</li>
                <li>Use the <b>time input box</b> in the sidebar to select analysis time</li>
                <li>Click <b>Apply Time</b> to extract a new frequency spectrum from the heatmap</li>
                <li>Use the <b>fine-tune slider</b> for precise time selection (1ms precision)</li>
                <li><b>Morlet Advantages:</b> Higher frequency resolution ({len(st.session_state.morlet_frequencies_continuous)} frequencies), better time-frequency localization, suitable for non-stationary signals</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # 1. First Plot: Input Trace, Continuous Frequency Volume, Frequency Spectrum
                st.subheader("1. Main Analysis Plots (Morlet Wavelet)")
                fig, spectrum, frequencies, current_time = st.session_state.morlet_analyzer.create_interactive_plot(
                    st.session_state.morlet_seismic_data,
                    st.session_state.morlet_spectral_components,
                    st.session_state.morlet_trace_index,
                    st.session_state.selected_time,
                    colormap=selected_colormap,
                    yaxis_range=yaxis_range,
                    wavelet_type='morlet'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Second Plot: ISA Frequency Spectrum (synchronized with above plots)
                st.subheader("2. ISA Frequency Spectrum (Morlet Wavelet)")
                st.markdown("""
                <div class="isa-plot-box">
                    <h3>🔬 ISA Frequency Spectrum (Morlet Wavelet) - Synchronized View</h3>
                    <p>This plot is synchronized with the Input Trace and Continuous Frequency Volume plots above</p>
                </div>
                """, unsafe_allow_html=True)
                
                isa_fig, isa_spectrum, isa_frequencies, isa_time = st.session_state.morlet_analyzer.create_isa_frequency_spectrum(
                    st.session_state.morlet_seismic_data,
                    st.session_state.morlet_spectral_components,
                    st.session_state.morlet_trace_index,
                    st.session_state.selected_time,
                    colormap=selected_colormap,
                    wavelet_type='morlet'
                )
                
                st.plotly_chart(isa_fig, use_container_width=True)
                
                # 3. Third Plot: Frequency Spectrum Details
                st.subheader("3. Frequency Spectrum Details (Morlet Wavelet)")
                st.markdown("""
                <div class="details-plot-box">
                    <h3>📊 Detailed Frequency Analysis (Morlet Wavelet)</h3>
                    <p>Detailed view of the frequency spectrum with Castagna frequency bands highlighted</p>
                </div>
                """, unsafe_allow_html=True)
                
                details_fig = st.session_state.morlet_analyzer.create_frequency_spectrum_details(
                    isa_spectrum, isa_frequencies, st.session_state.selected_time, yaxis_range, 'morlet'
                )
                
                st.plotly_chart(details_fig, use_container_width=True)
                
                # 4. Fourth Plot: Common Frequency Section
                st.subheader("4. Common Frequency Section (Morlet Wavelet) - Low-Frequency Shadow Detection")
                st.markdown("""
                <div class="low-freq-shadow">
                    <h3>🔍 Castagna Low-Frequency Shadow Detection (Morlet Wavelet)</h3>
                    <p>Based on Castagna et al. (2003) - Compare different frequency sections to identify hydrocarbon-related low-frequency shadows</p>
                    <p><b>Look for:</b> Strong low-frequency energy beneath potential reservoirs that disappears at higher frequencies</p>
                </div>
                """, unsafe_allow_html=True)
                
                common_freq_fig, common_freq_data = st.session_state.morlet_analyzer.create_common_frequency_section(
                    st.session_state.morlet_spectral_components,
                    common_freq,
                    colormap=selected_colormap,
                    wavelet_type='morlet'
                )
                
                st.plotly_chart(common_freq_fig, use_container_width=True)
                
                # Get continuous slice for verification
                continuous_slice, _ = st.session_state.morlet_analyzer.create_continuous_frequency_slice(
                    st.session_state.morlet_seismic_data,
                    st.session_state.morlet_spectral_components,
                    st.session_state.morlet_trace_index
                )
                
                # Display Morlet-specific results
                display_analysis_results(st.session_state.morlet_analyzer, spectrum, frequencies, 
                                      st.session_state.selected_time, time_axis, continuous_slice, 
                                      'morlet', selected_colormap, auto_yaxis, yaxis_range, 
                                      min_freq, max_freq, common_freq)
    
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
                    <li><b>Tab 1:</b> Spectral decomposition using Ricker wavelets (Original Castagna method)</li>
                    <li><b>Tab 2:</b> Spectral decomposition using Morlet wavelets (Enhanced time-frequency localization with higher frequency resolution)</li>
                    <li>Interactive time selection in milliseconds</li>
                    <li>Continuous frequency volume visualization</li>
                    <li>Frequency spectrum extracted directly from frequency volume</li>
                    <li>Castagna frequency band analysis</li>
                    <li>Dominant frequency detection</li>
                    <li>Bandwidth calculation (FWHM)</li>
                    <li>1ms precision in fine-tune time slider</li>
                    <li>Multiple colormap options for heatmap</li>
                    <li>Manual Y-axis range control for frequency spectrum</li>
                    <li>Synchronized zoom between input trace, continuous frequency volume, and ISA frequency spectrum</li>
                    <li>Common Frequency Section for low-frequency shadow detection</li>
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
                    <b>Dual Wavelet Approach:</b> Compare Ricker (original method) vs Morlet (enhanced localization with higher frequency resolution)
                    for comprehensive spectral analysis.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
