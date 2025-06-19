"""
🔋 电池数据分析工具 v2.1 - 学术版
=======================================

主要特性：
- 📊 智能数据清洗和异常值处理
- 🎨 学术风格的黑白配色和专业界面
- ⚙️ 全参数可调的分析算法
- 📈 学术标准的图表格式
- 📄 专业的性能报表生成
- ⚡ 多层缓存加速系统

学术风格优化：
- 18号标准字体
- 英文标签和标题
- 黑色线条配色
- 居中标题布局
- 专业图表格式
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
    page_title="🔋 Battery Data Analysis Tool",
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

    /* 自定义字体样式 */
    .stMarkdown {
        font-family: 'Times New Roman', serif;
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        font-family: 'Times New Roman', serif;
    }
</style>
""", unsafe_allow_html=True)


# 学术风格图表配置 - 修复函数参数
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

    # 添加标题配置（如果提供）
    if title_text:
        layout['title'] = {
            'text': title_text,
            'font': {
                'family': 'Times New Roman',
                'size': title_font_size,
                'color': 'black'
            },
            'x': 0.5,  # 居中
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
        # 使用BytesIO直接读取，避免临时文件问题
        import io
        file_buffer = io.BytesIO(file_content)

        # 读取Excel文件
        excel_file = pd.ExcelFile(file_buffer)
        available_sheets = excel_file.sheet_names

        record_data = pd.DataFrame()
        cycle_data = pd.DataFrame()

        if record_sheet_name in available_sheets:
            record_data = pd.read_excel(file_buffer, sheet_name=record_sheet_name)

        if cycle_sheet_name in available_sheets:
            file_buffer.seek(0)  # 重置缓冲区指针
            cycle_data = pd.read_excel(file_buffer, sheet_name=cycle_sheet_name)

        return record_data, cycle_data, available_sheets

    except Exception as e:
        # 返回错误信息用于调试
        return pd.DataFrame(), pd.DataFrame(), [f"Reading Error: {str(e)}"]


# 缓存颜色生成函数
@st.cache_data
def generate_colors(n_colors, color_theme='academic_black'):
    """生成不同主题的颜色 - 缓存版"""
    colors = []

    if color_theme == 'academic_black':
        # 学术风格黑色系
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
        # 使用matplotlib颜色映射
        import matplotlib.pyplot as plt
        try:
            cm = plt.cm.get_cmap(color_theme)
            colors = [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.8)'
                      for r, g, b, a in [cm(i / max(1, n_colors - 1)) for i in range(n_colors)]]
        except:
            # 如果颜色映射不存在，回退到学术黑色
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


# 缓存dQ/dV计算的辅助函数
@st.cache_data
def remove_duplicates_and_reverse(voltage_tuple, capacity_tuple):
    """去除重复值和反向变化值 - 缓存版"""
    voltage = np.array(voltage_tuple)
    capacity = np.array(capacity_tuple)

    # 1. 去除电压的重复值和反向变化值
    v_points_noRepeatValue = []
    for i in range(len(voltage)):
        k = 1
        for j in range(len(v_points_noRepeatValue)):
            if voltage[i] == v_points_noRepeatValue[j][0]:
                k = 0
                break
            if voltage[i] < v_points_noRepeatValue[j][0]:
                k = 0
                break
        if k == 1:
            v_points_noRepeatValue.append((voltage[i], capacity[i]))

    # 2. 去除容量的重复值和反向变化值
    qv_points_noRepeatValue = []
    for i in range(len(v_points_noRepeatValue)):
        k = 1
        for j in range(len(qv_points_noRepeatValue)):
            if v_points_noRepeatValue[i][1] == qv_points_noRepeatValue[j][1]:
                k = 0
                break
            if v_points_noRepeatValue[i][1] < qv_points_noRepeatValue[j][1]:
                k = 0
                break
        if k == 1:
            qv_points_noRepeatValue.append(v_points_noRepeatValue[i])

    if len(qv_points_noRepeatValue) == 0:
        return np.array([]), np.array([])

    df_v_ori = pd.DataFrame(qv_points_noRepeatValue, columns=['Voltage', 'Capacity'])
    df_v_ori_rvs = df_v_ori.sort_index(ascending=False)

    return df_v_ori_rvs['Voltage'].values, df_v_ori_rvs['Capacity'].values


# 缓存寻峰算法函数
@st.cache_data
def find_dqdv_peaks(voltage_tuple, dqdv_tuple, cycle, prominence=50.0, distance=30, height=100.0, rel_height=0.8):
    """寻找dQ/dV曲线中的峰值 - 缓存版，针对电池dQ/dV优化"""
    from scipy.signal import find_peaks

    voltage = np.array(voltage_tuple)
    dqdv = np.array(dqdv_tuple)

    if len(voltage) < 10 or len(dqdv) < 10:
        return pd.DataFrame()

    try:
        # 对于dQ/dV曲线，通常不需要过度的自适应调整
        # 直接使用用户设定的参数，因为dQ/dV曲线的尺度相对固定

        # 寻找峰值
        peaks, properties = find_peaks(
            dqdv,
            prominence=prominence,
            distance=distance,
            height=height,
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
            'Relative Height (%)': (dqdv[peaks] / np.max(dqdv) * 100),
            'Cycle Number': cycle
        })

        # 按峰值高度排序
        peak_data = peak_data.sort_values('dQ/dV (mAh/V)', ascending=False).reset_index(drop=True)

        # 对于dQ/dV曲线，通常主要峰值不会太多
        if len(peak_data) > 6:  # 最多保留6个最显著的峰值
            peak_data = peak_data.head(6)

        return peak_data

    except Exception as e:
        return pd.DataFrame()


@st.cache_data
def moving_average(data_tuple, window_size):
    """移动平均平滑 - 缓存版"""
    data = np.array(data_tuple)
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# 缓存dQ/dV计算函数
@st.cache_data
def calculate_dqdv_cached(voltage_tuple, capacity_tuple, cycle, smoothing_window=2, interpolation_points=1000,
                          epsilon=0.000004):
    """计算dQ/dV曲线 - 缓存版"""
    voltage = np.array(voltage_tuple)
    capacity = np.array(capacity_tuple)

    if len(voltage) < 3:
        return pd.DataFrame()

    # 去除NaN值
    valid_mask = ~(np.isnan(voltage) | np.isnan(capacity))
    voltage = voltage[valid_mask]
    capacity = capacity[valid_mask]

    if len(voltage) < 3:
        return pd.DataFrame()

    # 去除重复值和反向变化值
    voltage_clean, capacity_clean = remove_duplicates_and_reverse(tuple(voltage), tuple(capacity))

    if len(voltage_clean) < 3:
        return pd.DataFrame()

    try:
        # 插值处理
        interp_func = interpolate.interp1d(voltage_clean, capacity_clean,
                                           kind='linear', fill_value="extrapolate")

        voltage_interpolated = np.linspace(voltage.min(), voltage.max(),
                                           num=len(voltage) + interpolation_points)
        capacity_interpolated = interp_func(voltage_interpolated)

        # 计算dq/dv
        dq_interpolated = np.diff(capacity_interpolated)
        dv_interpolated = np.diff(voltage_interpolated)

        # 对dV进行阈值过滤
        dv_interpolated_filtered = np.where(
            np.abs(dv_interpolated) < epsilon,
            np.sign(dv_interpolated) * epsilon,
            dv_interpolated
        )

        dq_dv_interpolated = dq_interpolated / dv_interpolated_filtered
        dq_dv_smoothed = moving_average(tuple(np.abs(dq_dv_interpolated)), smoothing_window)
        voltage_smoothed = voltage_interpolated[1:len(dq_dv_smoothed) + 1]

        dqdv_data = pd.DataFrame({
            'Voltage (V)': voltage_smoothed,
            'dQ/dV (mAh/V)': dq_dv_smoothed,
            'Cycle Number': cycle
        })

        # 过滤异常值
        if len(dqdv_data) > 10:
            q25, q75 = dqdv_data['dQ/dV (mAh/V)'].quantile([0.25, 0.75])
            iqr = q75 - q25
            lower_bound = max(0, q25 - 1.5 * iqr)
            upper_bound = q75 + 1.5 * iqr

            dqdv_data = dqdv_data[
                (dqdv_data['dQ/dV (mAh/V)'] >= lower_bound) &
                (dqdv_data['dQ/dV (mAh/V)'] <= upper_bound)
                ]

        return dqdv_data

    except Exception as e:
        return pd.DataFrame()


class BatteryAnalyzerApp:
    def __init__(self):
        self.record_data = pd.DataFrame()
        self.cycle_data = pd.DataFrame()

    def generate_colors(self, n_colors, color_theme='academic_black'):
        """生成不同主题的颜色 - 使用缓存版本"""
        return generate_colors(n_colors, color_theme)

    def find_valid_last_cycle(self, column_name='放电容量(mAh)', min_threshold=0.1):
        """找到最后一个有效的循环数据"""
        sorted_cycles = self.cycle_data.sort_values('循环号', ascending=False)
        for idx, row in sorted_cycles.iterrows():
            value = row[column_name]
            if pd.notna(value) and value > min_threshold:
                return row
        return self.cycle_data.sort_values('循环号').iloc[-1]

    def find_dqdv_peaks(self, cycle, prominence=50.0, distance=30, height=100.0, rel_height=0.8):
        """寻找dQ/dV峰值 - 使用缓存版本，针对电池dQ/dV优化"""
        cycle_data = self.record_data[self.record_data['循环号'] == cycle].copy()

        if len(cycle_data) < 10:
            return pd.DataFrame()

        voltage = cycle_data['电压(V)'].values
        capacity = cycle_data['容量(mAh)'].values

        # 首先计算dQ/dV
        dqdv_data = calculate_dqdv_cached(
            tuple(voltage), tuple(capacity), cycle,
            smoothing_window=3, interpolation_points=1000, epsilon=0.000004
        )

        if dqdv_data.empty:
            return pd.DataFrame()

        # 寻找峰值
        return find_dqdv_peaks(
            tuple(dqdv_data['Voltage (V)'].values),
            tuple(dqdv_data['dQ/dV (mAh/V)'].values),
            cycle, prominence, distance, height, rel_height
        )

    def calculate_dqdv(self, cycle, smoothing_window=2, interpolation_points=1000, epsilon=0.000004):
        """计算dQ/dV曲线 - 使用缓存版本"""
        cycle_data = self.record_data[self.record_data['循环号'] == cycle].copy()

        if len(cycle_data) < 3:
            return pd.DataFrame()

        voltage = cycle_data['电压(V)'].values
        capacity = cycle_data['容量(mAh)'].values

        # 使用缓存版本的计算函数
        return calculate_dqdv_cached(tuple(voltage), tuple(capacity), cycle, smoothing_window, interpolation_points,
                                     epsilon)

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


# 创建应用实例 - 使用会话状态管理
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = BatteryAnalyzerApp()

analyzer = st.session_state.analyzer

# 主界面
st.markdown('<h1 class="main-header">🔋 Battery Data Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Battery Testing Data Analysis and Visualization Platform</p>',
            unsafe_allow_html=True)

# 侧边栏 - 数据上传和参数设置
with st.sidebar:
    st.header("📊 Data Upload")

    # 文件上传 - 单个Excel文件包含record和cycle层
    data_file = st.file_uploader(
        "Upload Battery Data File (Excel)",
        type=['xlsx', 'xls'],
        help="Excel file should contain 'record' and 'cycle' worksheets"
    )

    # 工作表名称设置
    st.subheader("📋 Worksheet Settings")
    record_sheet_name = st.text_input("Record Sheet Name", value="record",
                                      help="Worksheet containing detailed test data")
    cycle_sheet_name = st.text_input("Cycle Sheet Name", value="cycle", help="Worksheet containing cycle summary data")

    # 数据加载和预览
    if data_file is not None:
        try:
            # 显示文件基本信息
            st.info(f"📁 File Name: {data_file.name}")
            st.info(f"📏 File Size: {len(data_file.getvalue())} bytes")

            # 获取文件内容用于缓存
            file_content = data_file.getvalue()

            # 使用缓存函数读取数据
            with st.spinner("🔄 Loading data..."):
                record_data, cycle_data, available_sheets = load_excel_data(
                    file_content, record_sheet_name, cycle_sheet_name
                )

            # 检查是否有错误信息
            if available_sheets and available_sheets[0].startswith("Reading Error"):
                st.error(f"❌ {available_sheets[0]}")
                st.info("💡 Please ensure the file format is correct:")
                st.info("• Upload standard Excel files (.xlsx or .xls)")
                st.info("• Check if the file is corrupted")
                st.info("• Try re-saving the Excel file")
            else:
                # 更新analyzer中的数据
                analyzer.record_data = record_data
                analyzer.cycle_data = cycle_data

                st.success(f"📄 Found worksheets: {', '.join(available_sheets)}")

                # 显示加载结果
                if not record_data.empty:
                    st.success(
                        f"✅ Record data loaded successfully! ({len(record_data)} rows × {len(record_data.columns)} columns)")

                    # 显示record数据预览
                    with st.expander("📋 Record Data Preview", expanded=False):
                        st.write("**Data Shape:**", record_data.shape)
                        st.write("**Columns:**", list(record_data.columns))
                        st.dataframe(record_data.head(), use_container_width=True)
                else:
                    if record_sheet_name in available_sheets:
                        st.warning(f"⚠️ '{record_sheet_name}' worksheet exists but is empty")
                    else:
                        st.warning(f"⚠️ Worksheet '{record_sheet_name}' not found")
                        st.info(f"📋 Available worksheets: {available_sheets}")

                if not cycle_data.empty:
                    st.success(
                        f"✅ Cycle data loaded successfully! ({len(cycle_data)} rows × {len(cycle_data.columns)} columns)")

                    # 显示cycle数据预览
                    with st.expander("📋 Cycle Data Preview", expanded=False):
                        st.write("**Data Shape:**", cycle_data.shape)
                        st.write("**Columns:**", list(cycle_data.columns))
                        st.dataframe(cycle_data.head(), use_container_width=True)
                else:
                    if cycle_sheet_name in available_sheets:
                        st.warning(f"⚠️ '{cycle_sheet_name}' worksheet exists but is empty")
                    else:
                        st.warning(f"⚠️ Worksheet '{cycle_sheet_name}' not found")
                        st.info(f"📋 Available worksheets: {available_sheets}")

        except Exception as e:
            st.error(f"❌ File reading failed: {str(e)}")
            st.info("🔧 Troubleshooting suggestions:")
            st.info("1. Ensure file is valid Excel format (.xlsx or .xls)")
            st.info("2. Check if file contains data")
            st.info("3. Try opening file with Excel to verify format")
            st.info("4. If file is large, please wait a moment")

            # 显示详细错误信息用于调试
            with st.expander("🔍 Detailed Error Information", expanded=False):
                st.code(str(e))

    if not analyzer.record_data.empty:
        st.header("⚙️ Parameter Settings")

        # 检查必要的列是否存在
        required_columns = ['循环号', '工步类型']
        missing_columns = [col for col in required_columns if col not in analyzer.record_data.columns]

        if missing_columns:
            st.error(f"❌ Missing required columns in data: {missing_columns}")
            st.info("Please ensure data contains: Cycle Number, Step Type, Voltage (V), Capacity (mAh), etc.")
            available_cycles = []
            available_step_types = []
        else:
            # 获取可用选项
            try:
                available_cycles = sorted(analyzer.record_data['循环号'].unique())
                available_step_types = list(analyzer.record_data['工步类型'].unique())
            except Exception as e:
                st.error(f"❌ Data reading error: {e}")
                available_cycles = []
                available_step_types = []

        axis_options = ['时间', '总时间', '电流(A)', '电压(V)', '容量(mAh)', '能量(Wh)', '功率(W)']
        if available_cycles and available_step_types:
            # 基本参数
            st.subheader("🎯 Cycle Selection")
            selected_cycles = st.multiselect(
                "Select Cycle Numbers",
                available_cycles,
                default=available_cycles[:3] if len(available_cycles) >= 3 else available_cycles,
                help="Select cycle numbers for analysis"
            )

            selected_step_types = st.multiselect(
                "Select Step Types",
                available_step_types,
                default=[available_step_types[0]] if available_step_types else [],
                help="Select step types for analysis"
            )

            # 轴参数
            st.subheader("📈 Chart Axis Settings")
            x_axis = st.selectbox("X-axis Parameter", axis_options, index=0)
            y_axis = st.selectbox("Y-axis Parameter", axis_options, index=3)

            # dQ/dV参数
            st.subheader("🔬 dQ/dV Analysis Parameters")
            dqdv_cycles = st.multiselect(
                "dQ/dV Analysis Cycles",
                available_cycles,
                default=available_cycles[:3] if len(available_cycles) >= 3 else available_cycles
            )

            smoothing_window = st.slider("Smoothing Window Size", 1, 10, 2,
                                         help="Moving average window size, larger for smoother curves")
            interpolation_points = st.slider("Interpolation Points", 100, 2000, 1000, step=100,
                                             help="Number of interpolation points, more for smoother curves")
            epsilon = st.number_input("Voltage Differential Threshold", value=0.000004, format="%.6f",
                                      help="Prevent noise from small dV values")

            # 寻峰参数
            st.subheader("🏔️ Peak Finding Parameters")
            enable_peak_finding = st.checkbox("Enable Peak Detection", value=True, help="Find peaks in dQ/dV curves")

            if enable_peak_finding:
                st.info("💡 **Tip**: For dQ/dV curves, prominence should typically be 30-100 to find major peaks only.")
                col1, col2 = st.columns(2)
                with col1:
                    peak_prominence = st.slider("Peak Prominence", 10.0, 150.0, 50.0, step=5.0,
                                                help="Minimum prominence required (higher = fewer, more significant peaks)")
                    peak_distance = st.slider("Minimum Peak Distance", 10, 100, 30,
                                              help="Minimum distance between peaks in data points")
                with col2:
                    min_peak_height = st.slider("Minimum Peak Height", 50.0, 300.0, 100.0, step=10.0,
                                                help="Minimum height for peak detection")
                    max_peaks_display = st.slider("Max Peaks to Display", 1, 10, 5,
                                                  help="Maximum number of peaks to show in table")

                # 高级参数
                with st.expander("🔧 Advanced Peak Parameters"):
                    rel_height = st.slider("Relative Height Threshold", 0.1, 1.0, 0.8, step=0.1,
                                           help="Relative height for peak width calculation")
                    st.info("**Recommended**: Prominence 40-80 for typical battery dQ/dV curves")
            else:
                # 默认值
                peak_prominence = 50.0
                peak_distance = 30
                min_peak_height = 100.0
                max_peaks_display = 5
                rel_height = 0.8

            # 图表样式参数
            st.subheader("🎨 Chart Style Settings")
            color_theme = st.selectbox(
                "Select Color Scheme",
                [
                    "academic_black",  # 学术黑色风格
                    "custom_blue",  # 自定义蓝色
                    "custom_purple",  # 自定义紫色
                    "viridis",  # 经典渐变
                    "plasma",
                    "inferno",
                    "magma",
                    "cividis",  # 色盲友好
                    "tab10",  # 分类颜色
                    "Set1",
                    "Set2"
                ],
                index=0,  # 默认选择学术黑色
                help="Select chart color gradient theme"
            )

            # 样式参数
            col1, col2 = st.columns(2)
            with col1:
                line_width = st.slider("Line Width", 1, 5, 2, help="Chart line thickness")
                marker_size = st.slider("Marker Size", 2, 10, 4, help="Data point marker size")
            with col2:
                font_size = st.slider("Font Size", 12, 24, 18, help="Chart text font size")
                title_font_size = st.slider("Title Font Size", 16, 28, 20, help="Chart title font size")

            # 高级图表选项
            with st.expander("🔧 Advanced Chart Options"):
                connect_gaps = st.checkbox("Connect Data Gaps", value=False,
                                           help="Connect lines across missing data points")

            # 颜色主题说明
            if color_theme == "academic_black":
                st.info("🎓 **Academic Style**: Black to gray gradient, suitable for publications")
            elif color_theme in ["custom_blue", "custom_purple"]:
                st.info("🎨 **Custom Colors**: Professional gradient colors")
            elif color_theme in ["viridis", "plasma", "inferno", "magma"]:
                st.info("🌈 **Scientific Colormaps**: Perceptually uniform gradients")
            elif color_theme == "cividis":
                st.info("👁️ **Colorblind-Friendly**: Accessible for all users")
            else:
                st.info("🎯 **Categorical Colors**: Distinct colors for clear differentiation")

            # 性能优化提示
            st.subheader("⚡ Performance Optimization")
            if len(dqdv_cycles) > 5:
                st.warning(f"⚠️ Selected {len(dqdv_cycles)} cycles for dQ/dV analysis, calculation may take longer")

            st.info("""
            🚀 **Cache Acceleration Enabled**
            - dQ/dV calculation results auto-cached
            - Data reading intelligently cached
            - Faster execution with same parameters
            """)

        else:
            # 设置默认值，避免变量未定义错误
            selected_cycles = []
            selected_step_types = []
            x_axis = "时间"
            y_axis = "电压(V)"
            dqdv_cycles = []
            smoothing_window = 2
            interpolation_points = 1000
            epsilon = 0.000004
    else:
        # 数据为空时的默认值
        selected_cycles = []
        selected_step_types = []
        x_axis = "时间"
        y_axis = "电压(V)"
        dqdv_cycles = []
        smoothing_window = 2
        interpolation_points = 1000
        epsilon = 0.000004
        color_theme = "academic_black"
        enable_peak_finding = False
        peak_prominence = 50.0
        peak_distance = 30
        min_peak_height = 100.0
        max_peaks_display = 5
        rel_height = 0.8
        line_width = 2
        font_size = 18
        marker_size = 4
        title_font_size = 20
        connect_gaps = False

# 主内容区域
if analyzer.record_data.empty:
    st.info("👆 Please upload an Excel file containing record and cycle sheets in the sidebar")
    st.markdown("""
    ### 📋 Excel File Format Requirements

    **File Structure:**
    - Upload one Excel file containing two worksheets:
      - `record` worksheet: Detailed test data
      - `cycle` worksheet: Cycle summary data

    **Record worksheet must contain the following columns:**
    - `循环号` - Integer, cycle number
    - `工步类型` - String, e.g., "恒流放电", "恒流充电", etc.
    - `电压(V)` - Numeric, battery voltage
    - `容量(mAh)` - Numeric, cumulative capacity
    - `时间`, `总时间`, `电流(A)`, `能量(Wh)`, `功率(W)`, etc.

    **Cycle worksheet must contain the following columns:**
    - `循环号` - Integer, corresponding to record data cycle number
    - `放电容量(mAh)` - Numeric, discharge capacity per cycle
    - `充放电效率(%)` - Numeric, coulombic efficiency percentage
    - `充电容量(mAh)`, `充电能量(Wh)`, `放电能量(Wh)`, etc.

    ### 🚨 Common Issues and Solutions
    - **Worksheet Names**: Default searches for "record" and "cycle" worksheets, customizable in sidebar
    - **Column Name Errors**: Ensure column names match requirements exactly (including brackets and units)
    - **Data Types**: Cycle numbers should be integers, capacity/voltage/efficiency should be numeric
    - **Missing Worksheets**: If only record data exists, cycle analysis features will be unavailable

    ### 💡 Usage Tips
    - Check "Available Worksheets" list after file upload
    - If worksheet names differ, modify in "Worksheet Settings" in sidebar
    - Use "Data Preview" function to check data format
    - Record data is required, Cycle data is optional

    ### 🚀 Feature Highlights
    - 📊 Intelligent data cleaning and outlier processing
    - 🎨 Academic-style black and white theme with professional interface
    - ⚙️ Fully adjustable analysis algorithms
    - 📈 Academic standard chart formats
    - 📄 Professional performance report generation
    - 🔍 Data preview function for format checking
    - 📋 Automatic Excel worksheet structure recognition
    """)
else:
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Basic Curves", "🔬 dQ/dV Analysis", "📊 Cycle Performance", "📄 Performance Report"])

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
                                                showlegend=(j == 0)  # 只为第一个工步类型显示图例
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

                        # 应用学术风格 - 使用修复后的参数
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
        st.header("🔬 dQ/dV Differential Capacity Analysis")

        if dqdv_cycles and not analyzer.record_data.empty:
            # 检查必要的列是否存在
            required_cols = ['循环号', '电压(V)', '容量(mAh)']
            missing_cols = [col for col in required_cols if col not in analyzer.record_data.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns in data: {missing_cols}")
                st.info("dQ/dV analysis requires: Cycle Number, Voltage (V), Capacity (mAh) columns")
            else:
                # 计算dQ/dV数据 - 使用缓存优化
                all_dqdv_data = []
                successful_cycles = []

                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner('Calculating dQ/dV curves...'):
                    for i, cycle in enumerate(dqdv_cycles):
                        try:
                            status_text.text(f'Calculating Cycle {cycle}... ({i + 1}/{len(dqdv_cycles)})')
                            progress_bar.progress((i + 1) / len(dqdv_cycles))

                            dqdv_data = analyzer.calculate_dqdv(cycle, smoothing_window, interpolation_points, epsilon)
                            if not dqdv_data.empty:
                                all_dqdv_data.append(dqdv_data)
                                successful_cycles.append(cycle)
                        except Exception as e:
                            st.warning(f"⚠️ Cycle {cycle} calculation failed: {e}")

                # 清除进度指示器
                progress_bar.empty()
                status_text.empty()

                if successful_cycles:
                    # 生成颜色 - 支持多种主题
                    colors = analyzer.generate_colors(len(successful_cycles), color_theme)

                    # 创建Plotly图表
                    fig = go.Figure()

                    for i, cycle in enumerate(successful_cycles):
                        cycle_dqdv = all_dqdv_data[i]
                        fig.add_trace(go.Scatter(
                            x=cycle_dqdv['Voltage (V)'],
                            y=cycle_dqdv['dQ/dV (mAh/V)'],
                            mode='lines',
                            name=f'Cycle {cycle}',
                            line=dict(color=colors[i], width=line_width),
                            hovertemplate=f'Cycle {cycle}<br>Voltage: %{{x:.3f}} V<br>dQ/dV: %{{y:.2f}}<extra></extra>'
                        ))

                    # 如果启用了寻峰功能，添加峰值标记
                    if enable_peak_finding:
                        all_peaks_data = []
                        for i, cycle in enumerate(successful_cycles):
                            try:
                                peaks_data = analyzer.find_dqdv_peaks(
                                    cycle, peak_prominence, peak_distance, min_peak_height, rel_height
                                )
                                if not peaks_data.empty:
                                    all_peaks_data.append(peaks_data)

                                    # 在图上标记峰值
                                    fig.add_trace(go.Scatter(
                                        x=peaks_data['Voltage (V)'],
                                        y=peaks_data['dQ/dV (mAh/V)'],
                                        mode='markers',
                                        name=f'Peaks Cycle {cycle}',
                                        marker=dict(
                                            symbol='triangle-up',
                                            size=marker_size + 8,  # 峰值标记比普通标记大
                                            color=colors[i],
                                            line=dict(width=2, color='white')
                                        ),
                                        hovertemplate=f'Peak - Cycle {cycle}<br>Voltage: %{{x:.3f}} V<br>dQ/dV: %{{y:.1f}}<br>Prominence: %{{customdata:.1f}}<extra></extra>',
                                        customdata=peaks_data['Prominence']
                                    ))
                            except Exception as e:
                                st.warning(f"⚠️ Peak finding failed for Cycle {cycle}: {e}")

                    # 应用学术风格 - 使用修复后的参数
                    academic_layout = get_academic_layout(
                        title_text='dQ/dV Differential Capacity Curves',
                        x_title='Voltage (V)',
                        y_title='dQ/dV (mAh/V)',
                        font_size=font_size,
                        title_font_size=title_font_size
                    )
                    academic_layout.update({
                        'height': 600,
                        'hovermode': 'x unified'
                    })
                    fig.update_layout(**academic_layout)

                    st.plotly_chart(fig, use_container_width=True)

                    # 参数显示
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Successful Cycles", len(successful_cycles))
                    with col2:
                        st.metric("Smoothing Window", smoothing_window)
                    with col3:
                        st.metric("Interpolation Points", interpolation_points)
                    with col4:
                        st.metric("Voltage Threshold", f"{epsilon:.6f}")

                    # 显示计算详情
                    st.subheader("📋 Calculation Details")
                    details_df = pd.DataFrame({
                        'Cycle Number': successful_cycles,
                        'Data Points': [len(data) for data in all_dqdv_data],
                        'Voltage Range (V)': [f"{data['Voltage (V)'].min():.3f} - {data['Voltage (V)'].max():.3f}" for
                                              data in
                                              all_dqdv_data],
                        'dQ/dV Range': [f"{data['dQ/dV (mAh/V)'].min():.2f} - {data['dQ/dV (mAh/V)'].max():.2f}" for
                                        data in
                                        all_dqdv_data]
                    })
                    st.dataframe(details_df, use_container_width=True)

                    # 显示寻峰结果
                    if enable_peak_finding and 'all_peaks_data' in locals() and all_peaks_data:
                        st.subheader("🏔️ Peak Detection Results")

                        # 合并所有峰值数据
                        combined_peaks = pd.concat(all_peaks_data, ignore_index=True)

                        # 限制显示的峰值数量
                        if len(combined_peaks) > max_peaks_display:
                            combined_peaks = combined_peaks.head(max_peaks_display)
                            st.info(f"ℹ️ Showing top {max_peaks_display} peaks (sorted by dQ/dV value)")

                        # 格式化峰值表格
                        peaks_display = combined_peaks.copy()
                        peaks_display['Voltage (V)'] = peaks_display['Voltage (V)'].round(3)
                        peaks_display['dQ/dV (mAh/V)'] = peaks_display['dQ/dV (mAh/V)'].round(1)
                        peaks_display['Prominence'] = peaks_display['Prominence'].round(1)
                        peaks_display['Relative Height (%)'] = peaks_display['Relative Height (%)'].round(1)

                        st.dataframe(
                            peaks_display.drop('Peak Index', axis=1),
                            use_container_width=True,
                            hide_index=True
                        )

                        # 峰值统计
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Peaks Found", len(combined_peaks))
                        with col2:
                            avg_prominence = combined_peaks['Prominence'].mean()
                            st.metric("Avg Prominence", f"{avg_prominence:.2f}")
                        with col3:
                            voltage_span = combined_peaks['Voltage (V)'].max() - combined_peaks['Voltage (V)'].min()
                            st.metric("Voltage Span", f"{voltage_span:.3f} V")
                        with col4:
                            max_peak = combined_peaks['dQ/dV (mAh/V)'].max()
                            st.metric("Max Peak Value", f"{max_peak:.2f}")

                    elif enable_peak_finding:
                        st.info(
                            "ℹ️ No peaks found with current parameters. Try adjusting prominence or distance thresholds.")

                    # 缓存状态提示
                    st.info("💡 Tip: dQ/dV calculation results cached, faster recalculation with same parameters")

                else:
                    st.error("❌ No successful dQ/dV curve calculations, please check data quality or adjust parameters")
        else:
            if analyzer.record_data.empty:
                st.info("👆 Please upload data file first")
            else:
                st.info("👆 Please select cycle numbers for analysis in the sidebar")

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
                        # 创建子图 - 不设置子图标题
                        fig = make_subplots(
                            rows=1, cols=2,
                            shared_xaxes=False,  # 不共享X轴
                            horizontal_spacing=0.15  # 增加子图间距
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

                        fig.update_yaxes(title_text="Discharge Capacity (mAh)", row=1, col=1)
                        fig.update_yaxes(title_text="Coulombic Efficiency (%)", row=1, col=2)

                        # 为每个子图单独设置X轴标题
                        fig.update_xaxes(title_text="Cycle Number", row=1, col=1)
                        fig.update_xaxes(title_text="Cycle Number", row=1, col=2)

                        # 应用学术风格 - 使用修复后的参数（不传递额外参数）
                        academic_layout = get_academic_layout()
                        academic_layout.update({
                            'height': 600,  # 调整高度
                            'showlegend': False,
                            'margin': dict(l=80, r=80, t=100, b=80)  # 增加顶部边距为标题留出空间
                        })
                        fig.update_layout(**academic_layout)

                        # 手动添加外部标题
                        fig.add_annotation(
                            text="Discharge Capacity Fade",
                            xref="paper", yref="paper",
                            x=0.225, y=1.12,  # 提高Y位置，确保在图表外部
                            showarrow=False,
                            font=dict(family='Times New Roman', size=font_size, color='black'),
                            xanchor='center'
                        )

                        fig.add_annotation(
                            text="Coulombic Efficiency Evolution",
                            xref="paper", yref="paper",
                            x=0.775, y=1.12,  # 提高Y位置，确保在图表外部
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
                            rangemode='tozero'  # X轴从0开始
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

# 缓存管理
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style='text-align: center; color: #333; padding: 20px; font-family: Times New Roman;'>
        <p>🔋 Battery Data Analysis Tool v2.1 | Professional • Intelligent • Academic • High Performance</p>
        <p>Excel Data Import • Full Parameter Control • Intelligent Data Cleaning • Academic Styling • Cache Acceleration</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='padding: 20px;'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Cache", help="Clear all calculation cache, free memory"):
        st.cache_data.clear()
        st.success("✅ Cache cleared")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
