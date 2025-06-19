"""
🔋 电池数据分析工具 v2.2 - 完整专业版
=========================================

v2.2 完整功能：
- 📈 基本曲线分析：多循环、多工步类型对比分析
- 🔬 增强dQ/dV分析：分离充放电过程，改进寻峰算法
- 📊 循环性能分析：容量衰减和库伦效率演变
- 📄 专业性能报告：自动生成并支持导出
- 🎨 学术风格界面：黑白配色，Times New Roman字体
- ⚡ 全面缓存优化：智能数据处理，快速响应
- 🔍 错误诊断功能：详细调试信息，问题定位

v2.2 核心改进：
- ✅ 修正寻峰算法：直接在用户平滑曲线上寻峰
- ✅ 充放电分离：基于电流方向自动识别和分别分析
- ✅ 优化数据处理：改进放电数据预处理逻辑
- ✅ 增强错误报告：提供详细调试信息
- ✅ 参数自适应：智能调整检测参数
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
import hashlib

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="🔋 Battery Data Analysis Tool v2.2",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 学术风格CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Times New Roman', serif;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333333;
        text-align: center;
        margin-bottom: 3rem;
        font-family: 'Times New Roman', serif;
    }
    .metric-card {
        background: #f8f9fa;
        border: 2px solid #000000;
        padding: 1rem;
        border-radius: 5px;
        color: #000000;
        text-align: center;
        font-family: 'Times New Roman', serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        border: 1px solid #000000;
        font-family: 'Times New Roman', serif;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000;
        color: white;
    }
    .stMarkdown {
        font-family: 'Times New Roman', serif;
    }
    .css-1d391kg {
        font-family: 'Times New Roman', serif;
    }
</style>
""", unsafe_allow_html=True)


# 学术风格图表配置
def get_academic_layout(title_text=None, x_title=None, y_title=None, font_size=18, title_font_size=20):
    """获取学术风格的图表布局配置"""
    layout = {
        'font': {
            'family': 'Times New Roman',
            'size': font_size,
            'color': 'black'
        },
        'xaxis': {
            'title': {
                'text': x_title,
                'font': {
                    'family': 'Times New Roman',
                    'size': font_size,
                    'color': 'black'
                }
            },
            'tickfont': {
                'family': 'Times New Roman',
                'size': font_size - 2,
                'color': 'black'
            },
            'linecolor': 'black',
            'linewidth': 2,
            'mirror': True,
            'ticks': 'outside',
            'tickwidth': 2,
            'ticklen': 8
        },
        'yaxis': {
            'title': {
                'text': y_title,
                'font': {
                    'family': 'Times New Roman',
                    'size': font_size,
                    'color': 'black'
                }
            },
            'tickfont': {
                'family': 'Times New Roman',
                'size': font_size - 2,
                'color': 'black'
            },
            'linecolor': 'black',
            'linewidth': 2,
            'mirror': True,
            'ticks': 'outside',
            'tickwidth': 2,
            'ticklen': 8
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'showlegend': True,
        'legend': {
            'font': {
                'family': 'Times New Roman',
                'size': font_size - 2,
                'color': 'black'
            },
            'bgcolor': 'white',
            'bordercolor': 'black',
            'borderwidth': 1
        }
    }

    if title_text:
        layout['title'] = {
            'text': title_text,
            'font': {
                'family': 'Times New Roman',
                'size': title_font_size,
                'color': 'black'
            },
            'x': 0.5,
            'xanchor': 'center'
        }

    return layout


# 英文标签映射
COLUMN_MAPPING = {
    '循环号': 'Cycle Number',
    '工步类型': 'Step Type',
    '时间': 'Time',
    '总时间': 'Total Time',
    '电流(A)': 'Current (A)',
    '电压(V)': 'Voltage (V)',
    '容量(mAh)': 'Capacity (mAh)',
    '能量(Wh)': 'Energy (Wh)',
    '功率(W)': 'Power (W)',
    '放电容量(mAh)': 'Discharge Capacity (mAh)',
    '充放电效率(%)': 'Coulombic Efficiency (%)',
    '充电容量(mAh)': 'Charge Capacity (mAh)',
    '充电能量(Wh)': 'Charge Energy (Wh)',
    '放电能量(Wh)': 'Discharge Energy (Wh)'
}


def get_english_label(chinese_label):
    """获取英文标签"""
    return COLUMN_MAPPING.get(chinese_label, chinese_label)


# 缓存数据读取函数
@st.cache_data
def load_excel_data(file_content, record_sheet_name, cycle_sheet_name):
    """缓存的Excel数据读取函数"""
    try:
        file_buffer = io.BytesIO(file_content)
        excel_file = pd.ExcelFile(file_buffer)
        available_sheets = excel_file.sheet_names

        record_data = pd.DataFrame()
        cycle_data = pd.DataFrame()

        if record_sheet_name in available_sheets:
            record_data = pd.read_excel(file_buffer, sheet_name=record_sheet_name)

        if cycle_sheet_name in available_sheets:
            file_buffer.seek(0)
            cycle_data = pd.read_excel(file_buffer, sheet_name=cycle_sheet_name)

        return record_data, cycle_data, available_sheets

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), [f"Reading Error: {str(e)}"]


# 缓存颜色生成函数
@st.cache_data
def generate_colors(n_colors, color_theme='academic_black'):
    """生成不同主题的颜色 - 缓存版"""
    colors = []

    if color_theme == 'academic_black':
        if n_colors == 1:
            colors = ['black']
        else:
            for i in range(n_colors):
                gray_value = int(120 * i / max(1, n_colors - 1))
                colors.append(f'rgb({gray_value}, {gray_value}, {gray_value})')

    elif color_theme == 'custom_blue':
        for i in range(n_colors):
            intensity = 0.3 + 0.7 * i / max(1, n_colors - 1)
            colors.append(f'rgba({int(52 * intensity)}, {int(152 * intensity)}, {int(219 * intensity)}, 0.8)')

    elif color_theme == 'custom_purple':
        for i in range(n_colors):
            intensity = 0.3 + 0.7 * i / max(1, n_colors - 1)
            colors.append(f'rgba({int(155 * intensity)}, {int(89 * intensity)}, {int(182 * intensity)}, 0.8)')

    else:
        import matplotlib.pyplot as plt
        try:
            cm = plt.cm.get_cmap(color_theme)
            colors = [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.8)'
                      for r, g, b, a in [cm(i / max(1, n_colors - 1)) for i in range(n_colors)]]
        except:
            if n_colors == 1:
                colors = ['black']
            else:
                for i in range(n_colors):
                    gray_value = int(120 * i / max(1, n_colors - 1))
                    colors.append(f'rgb({gray_value}, {gray_value}, {gray_value})')

    return colors


# 缓存数据过滤函数
@st.cache_data
def filter_record_data_cached(record_data_hash, cycles_tuple, step_types_tuple, record_data_pickle):
    """缓存的数据过滤函数"""
    import pickle
    record_data = pickle.loads(record_data_pickle)

    filtered_data = record_data[
        (record_data['循环号'].isin(cycles_tuple)) &
        (record_data['工步类型'].isin(step_types_tuple))
        ]
    return filtered_data


# 修正的数据预处理函数
@st.cache_data
def preprocess_charge_discharge_data(voltage_tuple, capacity_tuple, process_type='charge'):
    """预处理充放电数据 - 根据过程类型优化处理"""
    voltage = np.array(voltage_tuple)
    capacity = np.array(capacity_tuple)

    # 去除NaN值
    valid_mask = ~(np.isnan(voltage) | np.isnan(capacity))
    voltage = voltage[valid_mask]
    capacity = capacity[valid_mask]

    if len(voltage) < 3:
        return np.array([]), np.array([])

    # 创建数据框便于处理
    df = pd.DataFrame({'Voltage': voltage, 'Capacity': capacity})

    if process_type == 'charge':
        # 充电：电压递增，容量递增
        df = df.sort_values('Voltage').reset_index(drop=True)
        # 去除电压重复值
        df = df.drop_duplicates(subset=['Voltage'], keep='first')
        # 确保容量单调递增
        df = df[df['Capacity'].diff().fillna(1) >= 0]

    elif process_type == 'discharge':
        # 放电：电压递减，容量可能递增（累积放电容量）或递减
        df = df.sort_values('Voltage', ascending=False).reset_index(drop=True)
        # 去除电压重复值
        df = df.drop_duplicates(subset=['Voltage'], keep='first')
        # 对于放电，按电压降序排列后，容量应该是单调的

    else:
        # 默认按电压排序
        df = df.sort_values('Voltage').reset_index(drop=True)

    if len(df) < 3:
        return np.array([]), np.array([])

    return df['Voltage'].values, df['Capacity'].values


# 改进的移动平均函数
@st.cache_data
def enhanced_smoothing(data_tuple, method='savgol', window_size=7, polyorder=3):
    """增强的数据平滑函数"""
    data = np.array(data_tuple)

    if len(data) < window_size:
        return data

    if method == 'savgol':
        # Savitzky-Golay滤波，保持峰值特征
        # 确保窗口大小为奇数
        window = window_size if window_size % 2 == 1 else window_size + 1
        # 确保多项式阶数小于窗口大小
        poly_order = min(polyorder, window - 1)

        if len(data) >= window and window > poly_order:
            try:
                return savgol_filter(data, window, poly_order)
            except:
                # 如果Savitzky-Golay失败，回退到移动平均
                return np.convolve(data, np.ones(min(window_size, len(data))) / min(window_size, len(data)),
                                   mode='valid')
        else:
            return data
    elif method == 'moving_average':
        window = min(window_size, len(data))
        return np.convolve(data, np.ones(window) / window, mode='valid')
    else:
        return data


# 修正的dQ/dV计算函数 - 分充放电
@st.cache_data
def calculate_charge_discharge_dqdv(voltage_tuple, capacity_tuple, current_tuple, cycle,
                                    process_type='both', smoothing_method='savgol',
                                    smoothing_window=7, interpolation_points=1000, epsilon=0.000004):
    """修正的充电和放电dQ/dV曲线计算"""
    voltage = np.array(voltage_tuple)
    capacity = np.array(capacity_tuple)
    current = np.array(current_tuple)

    if len(voltage) < 5:
        return pd.DataFrame()

    # 去除NaN值
    valid_mask = ~(np.isnan(voltage) | np.isnan(capacity) | np.isnan(current))
    voltage = voltage[valid_mask]
    capacity = capacity[valid_mask]
    current = current[valid_mask]

    if len(voltage) < 5:
        return pd.DataFrame()

    results = []

    # 降低电流阈值，避免数据太少
    current_threshold = 0.005  # 降低到5mA

    if process_type in ['charge', 'both']:
        # 充电过程 (电流>阈值)
        charge_mask = current > current_threshold
        if np.sum(charge_mask) > 5:
            v_charge = voltage[charge_mask]
            c_charge = capacity[charge_mask]

            v_clean, c_clean = preprocess_charge_discharge_data(
                tuple(v_charge), tuple(c_charge), 'charge'
            )

            if len(v_clean) > 5:
                dqdv_charge = _calculate_single_dqdv(v_clean, c_clean, smoothing_method,
                                                     smoothing_window, interpolation_points, epsilon)
                if len(dqdv_charge['voltage']) > 0:
                    df_charge = pd.DataFrame({
                        'Voltage (V)': dqdv_charge['voltage'],
                        'dQ/dV (mAh/V)': dqdv_charge['dqdv'],
                        'Process Type': 'Charge',
                        'Cycle Number': cycle
                    })
                    results.append(df_charge)

    if process_type in ['discharge', 'both']:
        # 放电过程 (电流<-阈值)
        discharge_mask = current < -current_threshold
        if np.sum(discharge_mask) > 5:
            v_discharge = voltage[discharge_mask]
            c_discharge = capacity[discharge_mask]

            v_clean, c_clean = preprocess_charge_discharge_data(
                tuple(v_discharge), tuple(c_discharge), 'discharge'
            )

            if len(v_clean) > 5:
                # 直接计算放电dQ/dV，不做容量反转
                dqdv_discharge = _calculate_single_dqdv(v_clean, c_clean, smoothing_method,
                                                        smoothing_window, interpolation_points, epsilon)
                if len(dqdv_discharge['voltage']) > 0:
                    df_discharge = pd.DataFrame({
                        'Voltage (V)': dqdv_discharge['voltage'],
                        'dQ/dV (mAh/V)': dqdv_discharge['dqdv'],
                        'Process Type': 'Discharge',
                        'Cycle Number': cycle
                    })
                    results.append(df_discharge)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def _calculate_single_dqdv(voltage, capacity, smoothing_method, smoothing_window, interpolation_points, epsilon):
    """计算单个过程的dQ/dV"""
    try:
        # 确保数据长度足够
        if len(voltage) < 5 or len(capacity) < 5:
            return {'voltage': np.array([]), 'dqdv': np.array([])}

        # 插值
        interp_func = interpolate.interp1d(voltage, capacity, kind='linear', fill_value="extrapolate")
        voltage_interp = np.linspace(voltage.min(), voltage.max(),
                                     num=min(len(voltage) + interpolation_points, 2000))
        capacity_interp = interp_func(voltage_interp)

        # 计算差分
        dq = np.diff(capacity_interp)
        dv = np.diff(voltage_interp)

        # 过滤小的dV值
        dv_filtered = np.where(np.abs(dv) < epsilon, np.sign(dv) * epsilon, dv)
        dqdv_raw = np.abs(dq / dv_filtered)

        # 应用平滑
        if smoothing_method == 'savgol' and len(dqdv_raw) >= smoothing_window:
            # 确保窗口大小为奇数且多项式阶数合适
            window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
            polyorder = min(3, window - 1)
            dqdv_smoothed = enhanced_smoothing(tuple(dqdv_raw), method='savgol',
                                               window_size=window, polyorder=polyorder)
        else:
            dqdv_smoothed = enhanced_smoothing(tuple(dqdv_raw), method='moving_average',
                                               window_size=min(smoothing_window, len(dqdv_raw)))

        # 调整电压数组长度以匹配平滑后的数据
        voltage_final = voltage_interp[1:len(dqdv_smoothed) + 1]

        # 过滤异常值
        if len(dqdv_smoothed) > 10:
            q25, q75 = np.percentile(dqdv_smoothed, [25, 75])
            iqr = q75 - q25
            lower_bound = max(0, q25 - 1.5 * iqr)
            upper_bound = q75 + 2.0 * iqr  # 允许更高的上界保留峰值

            valid_mask = (dqdv_smoothed >= lower_bound) & (dqdv_smoothed <= upper_bound)
            voltage_final = voltage_final[valid_mask]
            dqdv_smoothed = dqdv_smoothed[valid_mask]

        return {'voltage': voltage_final, 'dqdv': dqdv_smoothed}

    except Exception as e:
        return {'voltage': np.array([]), 'dqdv': np.array([])}


# 修正的寻峰函数
@st.cache_data
def find_dqdv_peaks_improved(voltage_tuple, dqdv_tuple, cycle, process_type,
                             prominence=30.0, distance=20, height=50.0, rel_height=0.5):
    """修正的dQ/dV峰值检测 - 直接在用户平滑后的曲线上寻峰"""
    from scipy.signal import find_peaks

    voltage = np.array(voltage_tuple)
    dqdv = np.array(dqdv_tuple)

    if len(voltage) < 10 or len(dqdv) < 10:
        return pd.DataFrame()

    try:
        # 直接在传入的平滑数据上寻峰，不再额外平滑
        max_dqdv = np.max(dqdv)

        # 根据数据特征调整参数
        adapted_prominence = min(prominence, max_dqdv * 0.15)
        adapted_height = min(height, max_dqdv * 0.25)

        # 直接在用户设置的平滑结果上寻找峰值
        peaks, properties = find_peaks(
            dqdv,  # 直接使用传入的已平滑数据
            prominence=adapted_prominence,
            distance=distance,
            height=adapted_height,
            rel_height=rel_height
        )

        if len(peaks) == 0:
            return pd.DataFrame()

        # 创建峰值数据框
        peak_data = pd.DataFrame({
            'Peak Index': peaks,
            'Voltage (V)': voltage[peaks],
            'dQ/dV (mAh/V)': dqdv[peaks],
            'Prominence': properties['prominences'],
            'Peak Height': properties['peak_heights'],
            'Relative Height (%)': (dqdv[peaks] / max_dqdv * 100),
            'Process Type': process_type,
            'Cycle Number': cycle
        })

        # 按峰值高度排序
        peak_data = peak_data.sort_values('dQ/dV (mAh/V)', ascending=False).reset_index(drop=True)

        # 限制峰值数量
        if len(peak_data) > 8:
            peak_data = peak_data.head(8)

        return peak_data

    except Exception as e:
        return pd.DataFrame()


class BatteryAnalyzerApp:
    def __init__(self):
        self.record_data = pd.DataFrame()
        self.cycle_data = pd.DataFrame()

    def generate_colors(self, n_colors, color_theme='academic_black'):
        """生成不同主题的颜色"""
        return generate_colors(n_colors, color_theme)

    def find_valid_last_cycle(self, column_name='放电容量(mAh)', min_threshold=0.1):
        """找到最后一个有效的循环数据"""
        sorted_cycles = self.cycle_data.sort_values('循环号', ascending=False)
        for idx, row in sorted_cycles.iterrows():
            value = row[column_name]
            if pd.notna(value) and value > min_threshold:
                return row
        return self.cycle_data.sort_values('循环号').iloc[-1]

    def calculate_charge_discharge_dqdv(self, cycle, process_type='both', smoothing_method='savgol',
                                        smoothing_window=7, interpolation_points=1000, epsilon=0.000004):
        """计算充放电dQ/dV曲线"""
        cycle_data = self.record_data[self.record_data['循环号'] == cycle].copy()

        if len(cycle_data) < 5:
            return pd.DataFrame()

        # 检查必要的列
        required_cols = ['电压(V)', '容量(mAh)', '电流(A)']
        if not all(col in cycle_data.columns for col in required_cols):
            return pd.DataFrame()

        voltage = cycle_data['电压(V)'].values
        capacity = cycle_data['容量(mAh)'].values
        current = cycle_data['电流(A)'].values

        return calculate_charge_discharge_dqdv(
            tuple(voltage), tuple(capacity), tuple(current), cycle,
            process_type, smoothing_method, smoothing_window, interpolation_points, epsilon
        )

    def find_dqdv_peaks_for_process(self, dqdv_data, prominence=30.0, distance=20,
                                    height=50.0, rel_height=0.5):
        """为不同过程类型寻找峰值"""
        all_peaks = []

        for process_type in dqdv_data['Process Type'].unique():
            process_data = dqdv_data[dqdv_data['Process Type'] == process_type]
            if len(process_data) > 10:
                cycle = process_data['Cycle Number'].iloc[0]
                peaks = find_dqdv_peaks_improved(
                    tuple(process_data['Voltage (V)'].values),
                    tuple(process_data['dQ/dV (mAh/V)'].values),
                    cycle, process_type, prominence, distance, height, rel_height
                )
                if not peaks.empty:
                    all_peaks.append(peaks)

        if all_peaks:
            return pd.concat(all_peaks, ignore_index=True)
        else:
            return pd.DataFrame()

    def generate_performance_report(self):
        """生成性能报表"""
        if self.cycle_data.empty:
            return {'error': 'No cycle data available for report generation'}

        sorted_cycles = self.cycle_data.sort_values('循环号')
        first_cycle = sorted_cycles.iloc[0]
        last_valid_cycle = self.find_valid_last_cycle('放电容量(mAh)')

        if first_cycle['放电容量(mAh)'] <= 0:
            return {'error': 'Abnormal first cycle discharge capacity data, cannot calculate retention'}

        capacity_retention = (last_valid_cycle['放电容量(mAh)'] / first_cycle['放电容量(mAh)']) * 100
        first_cycle_efficiency = first_cycle['充放电效率(%)']

        valid_efficiency = self.cycle_data['充放电效率(%)']
        valid_efficiency = valid_efficiency[(valid_efficiency > 0) & (valid_efficiency <= 100)]
        avg_efficiency = valid_efficiency.mean() if not valid_efficiency.empty else 0

        actual_last_cycle_num = last_valid_cycle['循环号']

        return {
            'Total Cycles': len(self.cycle_data),
            'Last Analyzed Cycle': int(actual_last_cycle_num),
            'Capacity Retention (%)': round(capacity_retention, 2),
            'First Cycle CE (%)': round(first_cycle_efficiency, 2),
            'Average CE (%)': round(avg_efficiency, 2),
            'First Cycle Capacity (mAh)': round(first_cycle['放电容量(mAh)'], 2),
            'Last Cycle Capacity (mAh)': round(last_valid_cycle['放电容量(mAh)'], 2)
        }


# 创建应用实例
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = BatteryAnalyzerApp()

analyzer = st.session_state.analyzer

# 主界面
st.markdown('<h1 class="main-header">🔋 Battery Data Analysis Tool v2.2</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Complete Professional Suite: Enhanced dQ/dV • Cycle Analysis • Performance Reports</p>',
    unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("📊 Data Upload")

    # 文件上传
    data_file = st.file_uploader(
        "Upload Battery Data File (Excel)",
        type=['xlsx', 'xls'],
        help="Excel file should contain 'record' and 'cycle' worksheets"
    )

    # 工作表名称设置
    st.subheader("📋 Worksheet Settings")
    record_sheet_name = st.text_input("Record Sheet Name", value="record")
    cycle_sheet_name = st.text_input("Cycle Sheet Name", value="cycle")

    # 数据加载
    if data_file is not None:
        try:
            st.info(f"📁 File: {data_file.name}")
            file_content = data_file.getvalue()

            with st.spinner("🔄 Loading data..."):
                record_data, cycle_data, available_sheets = load_excel_data(
                    file_content, record_sheet_name, cycle_sheet_name
                )

            if available_sheets and available_sheets[0].startswith("Reading Error"):
                st.error(f"❌ {available_sheets[0]}")
            else:
                analyzer.record_data = record_data
                analyzer.cycle_data = cycle_data
                st.success(f"📄 Worksheets: {', '.join(available_sheets)}")

                if not record_data.empty:
                    st.success(f"✅ Record data: {len(record_data)} rows")
                    with st.expander("📋 Record Data Preview"):
                        st.dataframe(record_data.head())

                if not cycle_data.empty:
                    st.success(f"✅ Cycle data: {len(cycle_data)} rows")

        except Exception as e:
            st.error(f"❌ File reading failed: {str(e)}")

    # 参数设置
    if not analyzer.record_data.empty:
        st.header("⚙️ Parameter Settings")

        required_columns = ['循环号', '工步类型', '电流(A)']
        missing_columns = [col for col in required_columns if col not in analyzer.record_data.columns]

        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            available_cycles = []
            available_step_types = []
        else:
            available_cycles = sorted(analyzer.record_data['循环号'].unique())
            available_step_types = list(analyzer.record_data['工步类型'].unique())

        axis_options = ['时间', '总时间', '电流(A)', '电压(V)', '容量(mAh)', '能量(Wh)', '功率(W)']

        if available_cycles and available_step_types:
            # 基本参数
            st.subheader("🎯 Cycle Selection")
            selected_cycles = st.multiselect(
                "Select Cycle Numbers",
                available_cycles,
                default=available_cycles[:3] if len(available_cycles) >= 3 else available_cycles
            )

            selected_step_types = st.multiselect(
                "Select Step Types",
                available_step_types,
                default=[available_step_types[0]] if available_step_types else []
            )

            # 轴参数
            st.subheader("📈 Chart Axis Settings")
            x_axis = st.selectbox("X-axis Parameter", axis_options, index=0)
            y_axis = st.selectbox("Y-axis Parameter", axis_options, index=3)

            # 改进的dQ/dV参数
            st.subheader("🔬 Enhanced dQ/dV Analysis")
            dqdv_cycles = st.multiselect(
                "dQ/dV Analysis Cycles",
                available_cycles,
                default=available_cycles[:3] if len(available_cycles) >= 3 else available_cycles
            )

            # 过程类型选择
            process_type = st.selectbox(
                "Analysis Process Type",
                ["both", "charge", "discharge"],
                index=0,
                help="Choose which process to analyze: both, charge only, or discharge only"
            )

            # 平滑参数
            col1, col2 = st.columns(2)
            with col1:
                smoothing_method = st.selectbox(
                    "Smoothing Method",
                    ["savgol", "moving_average"],
                    help="Savitzky-Golay preserves peak features better"
                )
                smoothing_window = st.slider("Smoothing Window", 3, 15, 7, step=2,
                                             help="Larger window = smoother curves, better peak detection")
            with col2:
                interpolation_points = st.slider("Interpolation Points", 500, 2000, 1000, step=100)
                epsilon = st.number_input("Voltage Threshold", value=0.000004, format="%.6f")

            # 寻峰参数
            st.subheader("🏔️ Enhanced Peak Detection")
            enable_peak_finding = st.checkbox("Enable Peak Detection", value=True)

            if enable_peak_finding:
                col1, col2 = st.columns(2)
                with col1:
                    peak_prominence = st.slider("Peak Prominence", 10.0, 100.0, 30.0, step=5.0,
                                                help="Lower values find more peaks")
                    peak_distance = st.slider("Minimum Peak Distance", 10, 50, 20)
                with col2:
                    min_peak_height = st.slider("Minimum Peak Height", 20.0, 200.0, 50.0, step=10.0)
                    max_peaks_display = st.slider("Max Peaks Display", 3, 15, 10)

                # 高级参数
                with st.expander("🔧 Advanced Peak Parameters"):
                    rel_height = st.slider("Relative Height", 0.1, 1.0, 0.5, step=0.1)
                    st.info("**New Feature**: Peaks are now detected on heavily smoothed curves for better accuracy")
            else:
                peak_prominence = 30.0
                peak_distance = 20
                min_peak_height = 50.0
                max_peaks_display = 10
                rel_height = 0.5

            # 图表样式
            st.subheader("🎨 Chart Style")
            color_theme = st.selectbox(
                "Color Scheme",
                ["academic_black", "custom_blue", "custom_purple", "viridis", "plasma"],
                index=0
            )

            col1, col2 = st.columns(2)
            with col1:
                line_width = st.slider("Line Width", 1, 5, 2)
                marker_size = st.slider("Marker Size", 2, 10, 4)
            with col2:
                font_size = st.slider("Font Size", 12, 24, 18)
                title_font_size = st.slider("Title Font Size", 16, 28, 20)

            # 高级图表选项
            with st.expander("🔧 Advanced Chart Options"):
                connect_gaps = st.checkbox("Connect Data Gaps", value=False,
                                           help="Connect lines across missing data points")

            # 颜色主题说明
            if color_theme == "academic_black":
                st.info("🎓 **Academic Style**: Black to gray gradient, suitable for publications")
            elif color_theme in ["custom_blue", "custom_purple"]:
                st.info("🎨 **Custom Colors**: Professional gradient colors")
            elif color_theme in ["viridis", "plasma"]:
                st.info("🌈 **Scientific Colormaps**: Perceptually uniform gradients")
            else:
                st.info("🎯 **Categorical Colors**: Distinct colors for clear differentiation")

            # 性能提示
            st.subheader("⚡ Performance Info")
            if len(dqdv_cycles) > 5:
                st.warning(f"⚠️ {len(dqdv_cycles)} cycles selected, calculation may take longer")

            st.info("""
            🚀 **v2.2 Improvements**
            - Separate charge/discharge dQ/dV analysis
            - Direct peak detection on user-smoothed curves
            - Lowered current threshold to 5mA for better data capture
            - Enhanced error reporting with debug information
            - Improved discharge data processing (no artificial capacity reversal)
            """)

            # 额外的参数说明
            with st.expander("ℹ️ Parameter Guidelines"):
                st.markdown("""
                **Current Detection:**
                - Charge: Current > 5mA
                - Discharge: Current < -5mA

                **Smoothing Recommendations:**
                - Savitzky-Golay: Better for preserving peak shapes
                - Window size 7-11: Good balance of smoothing and detail

                **Peak Detection:**
                - Prominence 20-50: Good for typical battery dQ/dV
                - Distance 15-25: Prevents false peak merging

                **Common Issues:**
                - If no data: Check current column for valid values
                - If no peaks: Try lower prominence/height values
                - If too noisy: Increase smoothing window
                """)

        else:
            # 默认值
            selected_cycles = []
            selected_step_types = []
            x_axis = "时间"
            y_axis = "电压(V)"
            dqdv_cycles = []
            process_type = "both"
            smoothing_method = "savgol"
            smoothing_window = 7
            interpolation_points = 1000
            epsilon = 0.000004
            color_theme = "academic_black"
            enable_peak_finding = False
            peak_prominence = 30.0
            peak_distance = 20
            min_peak_height = 50.0
            max_peaks_display = 10
            rel_height = 0.5
            line_width = 2
            font_size = 18
            marker_size = 4
            title_font_size = 20
            connect_gaps = False
    else:
        # 数据为空时的默认值
        selected_cycles = []
        selected_step_types = []
        x_axis = "时间"
        y_axis = "电压(V)"
        dqdv_cycles = []
        process_type = "both"
        smoothing_method = "savgol"
        smoothing_window = 7
        interpolation_points = 1000
        epsilon = 0.000004
        color_theme = "academic_black"
        enable_peak_finding = False
        peak_prominence = 30.0
        peak_distance = 20
        min_peak_height = 50.0
        max_peaks_display = 10
        rel_height = 0.5
        line_width = 2
        font_size = 18
        marker_size = 4
        title_font_size = 20
        connect_gaps = False

# 主内容区域
if analyzer.record_data.empty:
    st.info("👆 Please upload an Excel file in the sidebar")
    st.markdown("""
    ### 📋 Excel File Format Requirements

    **Required columns in record worksheet:**
    - `循环号` - Cycle number (integer)
    - `工步类型` - Step type (string)
    - `电压(V)` - Voltage (numeric)
    - `容量(mAh)` - Capacity (numeric)
    - `电流(A)` - Current (numeric) **[New requirement for charge/discharge analysis]**

    ### 🆕 Version 2.2 New Features
    - **Charge/Discharge Separation**: Automatically identifies charge (I>0) and discharge (I<0) processes
    - **Enhanced Peak Detection**: Uses Savitzky-Golay filtering for better peak preservation
    - **Improved Smoothing**: Multiple smoothing methods available
    - **Better Data Processing**: Advanced preprocessing for each process type
    """)
else:
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Basic Curves", "🔬 Enhanced dQ/dV Analysis", "📊 Cycle Performance", "📄 Performance Report"])

    with tab1:
        st.header("📈 Basic Curve Analysis")

        if selected_cycles and selected_step_types and not analyzer.record_data.empty:
            # 检查必要的列是否存在
            required_cols = ['循环号', '工步类型', x_axis, y_axis]
            missing_cols = [col for col in required_cols if col not in analyzer.record_data.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns in data: {missing_cols}")
                st.info("Please check data file format and ensure required columns are present")
            else:
                # 获取数据 - 使用缓存优化
                try:
                    import pickle
                    import hashlib

                    # 创建数据哈希用于缓存
                    data_hash = hashlib.md5(
                        str(analyzer.record_data.shape).encode() +
                        str(analyzer.record_data.columns.tolist()).encode()
                    ).hexdigest()

                    # 使用缓存过滤数据
                    record_data_pickle = pickle.dumps(analyzer.record_data)
                    filtered_data = filter_record_data_cached(
                        data_hash,
                        tuple(selected_cycles),
                        tuple(selected_step_types),
                        record_data_pickle
                    )

                    if not filtered_data.empty:
                        # 生成颜色 - 支持多种主题
                        colors = analyzer.generate_colors(len(selected_cycles), color_theme)

                        # 创建Plotly图表
                        fig = go.Figure()

                        for i, cycle in enumerate(selected_cycles):
                            cycle_data = filtered_data[filtered_data['循环号'] == cycle]
                            if not cycle_data.empty:
                                # 如果选择了多个工步类型且不连接间隙，需要分别处理每个工步类型
                                if len(selected_step_types) > 1 and not connect_gaps:
                                    for j, step_type in enumerate(selected_step_types):
                                        step_data = cycle_data[cycle_data['工步类型'] == step_type]
                                        if not step_data.empty:
                                            # 按时间或指定X轴排序以确保正确连接
                                            step_data = step_data.sort_values(x_axis)

                                            fig.add_trace(go.Scatter(
                                                x=step_data[x_axis],
                                                y=step_data[y_axis],
                                                mode='lines+markers',
                                                name=f'Cycle {cycle} - {step_type}',
                                                line=dict(color=colors[i], width=line_width),
                                                marker=dict(size=marker_size, color=colors[i]),
                                                hovertemplate=f'Cycle {cycle} - {step_type}<br>{get_english_label(x_axis)}: %{{x}}<br>{get_english_label(y_axis)}: %{{y}}<extra></extra>',
                                                showlegend=(j == 0),  # 只为第一个工步类型显示图例
                                                connectgaps=False
                                            ))
                                else:
                                    # 单一工步类型或选择连接间隙
                                    cycle_data = cycle_data.sort_values(x_axis)

                                    fig.add_trace(go.Scatter(
                                        x=cycle_data[x_axis],
                                        y=cycle_data[y_axis],
                                        mode='lines+markers',
                                        name=f'Cycle {cycle}',
                                        line=dict(color=colors[i], width=line_width),
                                        marker=dict(size=marker_size, color=colors[i]),
                                        hovertemplate=f'Cycle {cycle}<br>{get_english_label(x_axis)}: %{{x}}<br>{get_english_label(y_axis)}: %{{y}}<extra></extra>',
                                        connectgaps=connect_gaps
                                    ))

                        # 应用学术风格
                        academic_layout = get_academic_layout(
                            title_text=f'{get_english_label(y_axis)} vs {get_english_label(x_axis)}',
                            x_title=get_english_label(x_axis),
                            y_title=get_english_label(y_axis),
                            font_size=font_size,
                            title_font_size=title_font_size
                        )
                        academic_layout.update({
                            'height': 600,
                            'hovermode': 'x unified'
                        })
                        fig.update_layout(**academic_layout)

                        st.plotly_chart(fig, use_container_width=True)

                        # 数据统计
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Data Points", len(filtered_data))
                        with col2:
                            st.metric("Analyzed Cycles", len(selected_cycles))
                        with col3:
                            st.metric("Step Types", len(selected_step_types))
                        with col4:
                            try:
                                if x_axis in filtered_data.columns and len(filtered_data) > 0:
                                    # 尝试计算数值范围
                                    x_values = filtered_data[x_axis]

                                    # 检查数据类型并尝试转换为数值
                                    if pd.api.types.is_numeric_dtype(x_values):
                                        time_span = x_values.max() - x_values.min()
                                        st.metric("Data Range", f"{time_span:.2f}")
                                    elif pd.api.types.is_datetime64_any_dtype(x_values):
                                        time_span = (x_values.max() - x_values.min()).total_seconds()
                                        st.metric("Time Span", f"{time_span:.0f}s")
                                    else:
                                        # 尝试转换为数值类型
                                        try:
                                            numeric_values = pd.to_numeric(x_values, errors='coerce')
                                            if not numeric_values.isna().all():
                                                time_span = numeric_values.max() - numeric_values.min()
                                                st.metric("Data Range", f"{time_span:.2f}")
                                            else:
                                                st.metric("Data Range", f"{len(x_values.unique())} types")
                                        except:
                                            st.metric("Data Range", f"{len(x_values.unique())} types")
                                else:
                                    st.metric("Data Range", "N/A")
                            except Exception as e:
                                st.metric("Data Range", "Calc. Failed")

                        # 缓存提示
                        st.success("💡 Data filtering cached, faster response when switching parameters")
                    else:
                        st.warning("⚠️ No data matches the criteria")
                except Exception as e:
                    st.error(f"❌ Data processing error: {e}")
        else:
            if analyzer.record_data.empty:
                st.info("👆 Please upload data file first")
            else:
                st.info("👆 Please select cycle numbers and step types in the sidebar")

    with tab2:
        st.header("🔬 Enhanced dQ/dV Analysis with Charge/Discharge Separation")

        if dqdv_cycles and not analyzer.record_data.empty:
            required_cols = ['循环号', '电压(V)', '容量(mAh)', '电流(A)']
            missing_cols = [col for col in required_cols if col not in analyzer.record_data.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Enhanced dQ/dV analysis requires Current (A) column for charge/discharge separation")
            else:
                # 计算充放电dQ/dV数据
                all_dqdv_data = []
                successful_cycles = []
                debug_info = []  # 添加调试信息

                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner('Calculating enhanced dQ/dV curves...'):
                    for i, cycle in enumerate(dqdv_cycles):
                        try:
                            status_text.text(
                                f'Processing Cycle {cycle} ({process_type})... ({i + 1}/{len(dqdv_cycles)})')
                            progress_bar.progress((i + 1) / len(dqdv_cycles))

                            # 获取循环数据进行调试
                            cycle_data = analyzer.record_data[analyzer.record_data['循环号'] == cycle]
                            if len(cycle_data) > 0:
                                current = cycle_data['电流(A)'].values
                                charge_points = np.sum(current > 0.005)
                                discharge_points = np.sum(current < -0.005)

                                debug_info.append({
                                    'Cycle': cycle,
                                    'Total Points': len(cycle_data),
                                    'Charge Points': charge_points,
                                    'Discharge Points': discharge_points,
                                    'Current Range': f"{current.min():.3f} to {current.max():.3f} A"
                                })

                            dqdv_data = analyzer.calculate_charge_discharge_dqdv(
                                cycle, process_type, smoothing_method, smoothing_window,
                                interpolation_points, epsilon
                            )

                            if not dqdv_data.empty:
                                all_dqdv_data.append(dqdv_data)
                                successful_cycles.append(cycle)
                        except Exception as e:
                            st.warning(f"⚠️ Cycle {cycle} calculation failed: {e}")
                            debug_info.append({
                                'Cycle': cycle,
                                'Error': str(e),
                                'Total Points': 'N/A',
                                'Charge Points': 'N/A',
                                'Discharge Points': 'N/A',
                                'Current Range': 'N/A'
                            })

                progress_bar.empty()
                status_text.empty()

                # 显示调试信息
                if debug_info:
                    with st.expander("🔍 Debug Information - Data Analysis Details", expanded=not successful_cycles):
                        debug_df = pd.DataFrame(debug_info)
                        st.dataframe(debug_df, use_container_width=True, hide_index=True)

                        if not successful_cycles:
                            st.error("❌ No successful calculations. Check the debug info above:")
                            st.info("🔧 **Troubleshooting Tips:**")
                            st.info("• Ensure Current (A) column contains valid numeric data")
                            st.info("• Check if current values are reasonable (not all zero)")
                            st.info("• Try lowering current threshold or adjusting smoothing parameters")
                            st.info("• Verify that voltage and capacity data are not constant")

                if successful_cycles:
                    # 合并所有数据
                    combined_dqdv = pd.concat(all_dqdv_data, ignore_index=True)

                    # 创建图表
                    fig = go.Figure()

                    # 为每个循环和过程类型生成颜色
                    unique_processes = combined_dqdv['Process Type'].unique()
                    unique_cycles = combined_dqdv['Cycle Number'].unique()

                    colors = analyzer.generate_colors(len(unique_cycles), color_theme)

                    # 绘制曲线
                    for i, cycle in enumerate(unique_cycles):
                        cycle_data = combined_dqdv[combined_dqdv['Cycle Number'] == cycle]

                        for j, process in enumerate(unique_processes):
                            process_data = cycle_data[cycle_data['Process Type'] == process]

                            if not process_data.empty:
                                # 线型区分充放电
                                line_dash = 'solid' if process == 'Charge' else 'dash'
                                line_color = colors[i]

                                fig.add_trace(go.Scatter(
                                    x=process_data['Voltage (V)'],
                                    y=process_data['dQ/dV (mAh/V)'],
                                    mode='lines',
                                    name=f'Cycle {cycle} - {process}',
                                    line=dict(color=line_color, width=line_width, dash=line_dash),
                                    hovertemplate=f'Cycle {cycle} - {process}<br>Voltage: %{{x:.3f}} V<br>dQ/dV: %{{y:.2f}}<extra></extra>'
                                ))

                    # 寻峰并添加峰值标记
                    if enable_peak_finding:
                        all_peaks_data = []

                        for i, cycle in enumerate(successful_cycles):
                            cycle_dqdv = combined_dqdv[combined_dqdv['Cycle Number'] == cycle]
                            peaks_data = analyzer.find_dqdv_peaks_for_process(
                                cycle_dqdv, peak_prominence, peak_distance, min_peak_height, rel_height
                            )

                            if not peaks_data.empty:
                                all_peaks_data.append(peaks_data)

                                # 在图上标记峰值
                                for process in peaks_data['Process Type'].unique():
                                    process_peaks = peaks_data[peaks_data['Process Type'] == process]
                                    marker_symbol = 'triangle-up' if process == 'Charge' else 'triangle-down'

                                    fig.add_trace(go.Scatter(
                                        x=process_peaks['Voltage (V)'],
                                        y=process_peaks['dQ/dV (mAh/V)'],
                                        mode='markers',
                                        name=f'Peaks C{cycle} {process}',
                                        marker=dict(
                                            symbol=marker_symbol,
                                            size=marker_size + 8,
                                            color=colors[i],
                                            line=dict(width=2, color='white')
                                        ),
                                        hovertemplate=f'Peak - C{cycle} {process}<br>V: %{{x:.3f}}<br>dQ/dV: %{{y:.1f}}<extra></extra>'
                                    ))

                    # 应用学术风格
                    academic_layout = get_academic_layout(
                        title_text='Enhanced dQ/dV Analysis - Charge/Discharge Separation',
                        x_title='Voltage (V)',
                        y_title='dQ/dV (mAh/V)',
                        font_size=font_size,
                        title_font_size=title_font_size
                    )
                    academic_layout.update({'height': 600, 'hovermode': 'x unified'})
                    fig.update_layout(**academic_layout)

                    st.plotly_chart(fig, use_container_width=True)

                    # 显示参数和统计
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Successful Cycles", len(successful_cycles))
                    with col2:
                        st.metric("Process Types", len(unique_processes))
                    with col3:
                        st.metric("Smoothing Method", smoothing_method.title())
                    with col4:
                        st.metric("Smoothing Window", smoothing_window)

                    # 显示计算详情
                    st.subheader("📋 Calculation Details")
                    details_data = []
                    for cycle in successful_cycles:
                        cycle_data = combined_dqdv[combined_dqdv['Cycle Number'] == cycle]
                        for process in cycle_data['Process Type'].unique():
                            process_data = cycle_data[cycle_data['Process Type'] == process]
                            details_data.append({
                                'Cycle': cycle,
                                'Process': process,
                                'Data Points': len(process_data),
                                'Voltage Range (V)': f"{process_data['Voltage (V)'].min():.3f} - {process_data['Voltage (V)'].max():.3f}",
                                'Max dQ/dV': f"{process_data['dQ/dV (mAh/V)'].max():.2f}"
                            })

                    details_df = pd.DataFrame(details_data)
                    st.dataframe(details_df, use_container_width=True, hide_index=True)

                    # 峰值结果
                    if enable_peak_finding and 'all_peaks_data' in locals() and all_peaks_data:
                        st.subheader("🏔️ Peak Detection Results")
                        combined_peaks = pd.concat(all_peaks_data, ignore_index=True)

                        if len(combined_peaks) > max_peaks_display:
                            combined_peaks = combined_peaks.head(max_peaks_display)

                        # 格式化显示
                        peaks_display = combined_peaks[['Cycle Number', 'Process Type', 'Voltage (V)',
                                                        'dQ/dV (mAh/V)', 'Prominence', 'Relative Height (%)']].copy()
                        peaks_display['Voltage (V)'] = peaks_display['Voltage (V)'].round(3)
                        peaks_display['dQ/dV (mAh/V)'] = peaks_display['dQ/dV (mAh/V)'].round(1)
                        peaks_display['Prominence'] = peaks_display['Prominence'].round(1)
                        peaks_display['Relative Height (%)'] = peaks_display['Relative Height (%)'].round(1)

                        st.dataframe(peaks_display, use_container_width=True, hide_index=True)

                        # 峰值统计
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Peaks", len(combined_peaks))
                        with col2:
                            charge_peaks = len(combined_peaks[combined_peaks['Process Type'] == 'Charge'])
                            st.metric("Charge Peaks", charge_peaks)
                        with col3:
                            discharge_peaks = len(combined_peaks[combined_peaks['Process Type'] == 'Discharge'])
                            st.metric("Discharge Peaks", discharge_peaks)
                        with col4:
                            avg_prominence = combined_peaks['Prominence'].mean()
                            st.metric("Avg Prominence", f"{avg_prominence:.1f}")

                    # 改进提示
                    st.success(
                        "🎉 Enhanced dQ/dV analysis complete! Charge and discharge processes analyzed separately.")
                    st.info(
                        "💡 **Key Improvements**: Peak detection now uses additional smoothing for better accuracy, and charge/discharge processes are separated based on current direction.")

                else:
                    st.error("❌ No successful dQ/dV calculations. Please check data quality or adjust parameters.")
        else:
            if analyzer.record_data.empty:
                st.info("👆 Please upload data file first")
            else:
                st.info("👆 Please select cycles for analysis")

    with tab3:
        st.header("📊 Cycle Performance Analysis")

        if not analyzer.cycle_data.empty:
            # 检查必要的列是否存在
            required_cols = ['循环号', '放电容量(mAh)', '充放电效率(%)']
            missing_cols = [col for col in required_cols if col not in analyzer.cycle_data.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns in cycle data: {missing_cols}")
                st.info(
                    "Cycle performance analysis requires: Cycle Number, Discharge Capacity (mAh), Coulombic Efficiency (%) columns")
            else:
                try:
                    # 过滤有效数据
                    valid_data = analyzer.cycle_data[
                        (analyzer.cycle_data['放电容量(mAh)'] > 0) &
                        (analyzer.cycle_data['充放电效率(%)'] > 0) &
                        (analyzer.cycle_data['充放电效率(%)'] <= 100) &
                        (pd.notna(analyzer.cycle_data['放电容量(mAh)'])) &
                        (pd.notna(analyzer.cycle_data['充放电效率(%)']))
                        ].copy()

                    if not valid_data.empty:
                        # 生成颜色 - 支持多种主题
                        colors = analyzer.generate_colors(2, color_theme)  # 两条线：容量和效率

                        # 创建子图
                        fig = make_subplots(
                            rows=1, cols=2,
                            shared_xaxes=False,
                            horizontal_spacing=0.15
                        )

                        # 放电容量图
                        fig.add_trace(
                            go.Scatter(
                                x=valid_data['循环号'],
                                y=valid_data['放电容量(mAh)'],
                                mode='lines+markers',
                                name='Discharge Capacity',
                                line=dict(color=colors[0], width=line_width),
                                marker=dict(size=marker_size, color=colors[0]),
                                hovertemplate='Cycle: %{x}<br>Discharge Capacity: %{y:.2f} mAh<extra></extra>'
                            ),
                            row=1, col=1
                        )

                        # 充放电效率图
                        fig.add_trace(
                            go.Scatter(
                                x=valid_data['循环号'],
                                y=valid_data['充放电效率(%)'],
                                mode='lines+markers',
                                name='Coulombic Efficiency',
                                line=dict(color=colors[1], width=line_width),
                                marker=dict(size=marker_size, color=colors[1]),
                                hovertemplate='Cycle: %{x}<br>Coulombic Efficiency: %{y:.2f}%<extra></extra>'
                            ),
                            row=1, col=2
                        )

                        # 应用学术风格
                        academic_layout = get_academic_layout()
                        academic_layout.update({
                            'height': 600,
                            'showlegend': False,
                            'margin': dict(l=80, r=80, t=100, b=80)
                        })
                        fig.update_layout(**academic_layout)

                        # 手动添加外部标题
                        fig.add_annotation(
                            text="Discharge Capacity Fade",
                            xref="paper", yref="paper",
                            x=0.225, y=1.12,
                            showarrow=False,
                            font=dict(family='Times New Roman', size=font_size, color='black'),
                            xanchor='center'
                        )

                        fig.add_annotation(
                            text="Coulombic Efficiency Evolution",
                            xref="paper", yref="paper",
                            x=0.775, y=1.12,
                            showarrow=False,
                            font=dict(family='Times New Roman', size=font_size, color='black'),
                            xanchor='center'
                        )

                        # 更新每个轴的字体和标签设置
                        fig.update_xaxes(
                            title_text="Cycle Number",
                            tickfont=dict(family='Times New Roman', size=font_size - 4, color='black'),
                            title_font=dict(family='Times New Roman', size=font_size - 2, color='black'),
                            linecolor='black',
                            linewidth=2,
                            mirror=True,
                            ticks='outside',
                            tickwidth=2,
                            ticklen=6,
                            showline=True,
                            zeroline=False,
                            rangemode='tozero'
                        )
                        fig.update_yaxes(
                            tickfont=dict(family='Times New Roman', size=font_size - 4, color='black'),
                            title_font=dict(family='Times New Roman', size=font_size - 2, color='black'),
                            linecolor='black',
                            linewidth=2,
                            mirror=True,
                            ticks='outside',
                            tickwidth=2,
                            ticklen=6,
                            showline=True,
                            zeroline=False
                        )

                        # 单独设置Y轴标题
                        fig.update_yaxes(title_text="Discharge Capacity (mAh)", row=1, col=1)
                        fig.update_yaxes(title_text="Coulombic Efficiency (%)", row=1, col=2)

                        st.plotly_chart(fig, use_container_width=True)

                        # 关键指标
                        col1, col2, col3, col4 = st.columns(4)

                        first_capacity = valid_data.iloc[0]['放电容量(mAh)']
                        last_capacity = valid_data.iloc[-1]['放电容量(mAh)']
                        capacity_retention = (last_capacity / first_capacity) * 100
                        avg_efficiency = valid_data['充放电效率(%)'].mean()

                        with col1:
                            st.metric("Valid Cycles", len(valid_data))
                        with col2:
                            st.metric("Capacity Retention", f"{capacity_retention:.1f}%")
                        with col3:
                            st.metric("Average CE", f"{avg_efficiency:.1f}%")
                        with col4:
                            capacity_fade = (first_capacity - last_capacity) / len(valid_data)
                            st.metric("Avg. Capacity Fade", f"{capacity_fade:.2f} mAh/cycle")

                        # 过滤信息
                        if len(valid_data) < len(analyzer.cycle_data):
                            filtered_count = len(analyzer.cycle_data) - len(valid_data)
                            st.info(f"ℹ️ Automatically filtered {filtered_count} abnormal data points")

                    else:
                        st.error("❌ No valid cycle performance data")
                except Exception as e:
                    st.error(f"❌ Cycle performance analysis error: {e}")
        else:
            st.warning("⚠️ Please upload cycle data file to view cycle performance analysis")

    with tab4:
        st.header("📄 Battery Performance Report")

        if not analyzer.cycle_data.empty:
            # 检查必要的列是否存在
            required_cols = ['循环号', '放电容量(mAh)', '充放电效率(%)']
            missing_cols = [col for col in required_cols if col not in analyzer.cycle_data.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns in cycle data: {missing_cols}")
                st.info(
                    "Performance report generation requires: Cycle Number, Discharge Capacity (mAh), Coulombic Efficiency (%) columns")
            else:
                try:
                    report = analyzer.generate_performance_report()

                    if 'error' not in report:
                        # 关键指标卡片
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Capacity Retention</h3>
                                <h2>{report['Capacity Retention (%)']}%</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>First Cycle CE</h3>
                                <h2>{report['First Cycle CE (%)']}%</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Average CE</h3>
                                <h2>{report['Average CE (%)']}%</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # 详细报表
                        st.subheader("📊 Detailed Performance Metrics")

                        # 创建报表DataFrame
                        report_data = []
                        for key, value in report.items():
                            report_data.append({'Metric': key, 'Value': value})

                        report_df = pd.DataFrame(report_data)

                        # 美化表格显示
                        st.dataframe(
                            report_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # 导出功能
                        st.subheader("💾 Export Report")

                        col1, col2 = st.columns(2)

                        with col1:
                            # 导出为CSV
                            csv_buffer = io.StringIO()
                            report_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                            csv_data = csv_buffer.getvalue()

                            st.download_button(
                                label="📥 Download CSV Report",
                                data=csv_data,
                                file_name=f"Battery_Performance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # 导出为Excel
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                report_df.to_excel(writer, sheet_name='Performance Report', index=False)
                                if not analyzer.cycle_data.empty:
                                    analyzer.cycle_data.to_excel(writer, sheet_name='Cycle Data', index=False)

                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_buffer.getvalue(),
                                file_name=f"Battery_Performance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetsheet"
                            )

                        # 数据质量信息
                        if 'Data Description' in report:
                            st.info(f"ℹ️ {report['Data Description']}")

                    else:
                        st.error(f"❌ {report['error']}")
                except Exception as e:
                    st.error(f"❌ Performance report generation error: {e}")
        else:
            st.warning("⚠️ Please upload cycle data file to generate performance report")

# 页脚
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style='text-align: center; color: #333; padding: 20px; font-family: Times New Roman;'>
        <p>🔋 Battery Data Analysis Tool v2.2 | Complete Professional Suite</p>
        <p>✅ Basic Curves • 🆕 Enhanced dQ/dV (Charge/Discharge) • 📊 Cycle Performance • 📄 Reports</p>
        <p>🎓 Academic Style • ⚡ Cache Acceleration • 🔍 Error Diagnosis • 📈 Publication Ready</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("🗑️ Clear Cache", help="Clear calculation cache"):
        st.cache_data.clear()
        st.success("✅ Cache cleared")
        st.rerun()
