import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from scipy import integrate

# 页面配置
st.set_page_config(
    page_title="气体分析工具",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 生成不重复的高对比度颜色
def generate_distinct_colors(n):
    """生成n个不重复的高对比度颜色"""
    base_colors = [
        '#DC143C', '#228B22', '#4169E1', '#FF4500', '#9932CC', '#008B8B',
        '#B22222', '#32CD32', '#1E90FF', '#FF6347', '#8A2BE2', '#00CED1',
        '#CD5C5C', '#7CFC00', '#6495ED', '#FF8C00', '#9370DB', '#20B2AA',
        '#F08080', '#98FB98', '#87CEEB', '#DDA0DD', '#90EE90', '#ADD8E6',
        '#800000', '#008000', '#000080', '#800080', '#008080', '#804000'
    ]

    # 循环使用颜色
    colors = []
    for i in range(n):
        colors.append(base_colors[i % len(base_colors)])
    return colors


def detect_encoding(file_bytes):
    """检测文件编码"""
    try:
        result = chardet.detect(file_bytes)
        return result['encoding']
    except:
        return 'utf-8'


def process_gas_data(file_content, filename, ar_column_choice="自动选择"):
    """处理气体强度数据"""
    try:
        # 检测编码
        detected_encoding = detect_encoding(file_content)

        # 尝试不同编码读取文件
        encodings = [detected_encoding, 'utf-8', 'gbk', 'gb2312', 'ascii', 'latin-1']
        lines = None

        for encoding in encodings:
            try:
                if encoding:
                    content = file_content.decode(encoding)
                    lines = content.split('\n')
                    st.success(f"✅ 成功使用 {encoding} 编码读取文件")
                    break
            except:
                continue

        if lines is None:
            st.error("❌ 无法读取文件，请检查文件格式")
            return None

        st.info(f"📄 文件总行数: {len(lines)}")

        # 寻找"Data"行和表头行
        data_start_line = None
        header_line = None

        for i, line in enumerate(lines):
            line_clean = line.strip().replace('"', '')

            # 寻找"Data"标识行
            if 'Data' in line_clean and (',' in line_clean or line_clean == 'Data'):
                st.success(f"✅ 找到Data标识行在第{i + 1}行")

                # 检查下一行是否是表头
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip().replace('"', '')
                    if 'Time' in next_line and 'ms' in next_line:
                        header_line = i + 1
                        data_start_line = i + 2
                        st.success(f"✅ 找到表头行在第{header_line + 1}行")
                        break

        if data_start_line is None:
            st.error("❌ 未找到Data标识行或表头行")
            return None

        # 解析表头
        header = lines[header_line].strip().replace('"', '').split(',')
        cleaned_header = [col.strip() for col in header if col.strip()]

        st.info(f"📊 表头列数: {len(cleaned_header)}")

        # 读取数据行
        data_lines = lines[data_start_line:]
        parsed_data = []

        progress_bar = st.progress(0)

        for i, line in enumerate(data_lines):
            if i % 100 == 0:
                progress = min(i / len(data_lines), 1.0)
                progress_bar.progress(progress)

            line_clean = line.strip().replace('"', '')
            if not line_clean:
                continue

            parts = line_clean.split(',')

            if len(parts) >= len(cleaned_header) - 2:
                mixed_row = []
                for j in range(len(cleaned_header)):
                    if j < len(parts):
                        part = parts[j].strip()
                    else:
                        part = '0'

                    # 处理第一列的时间格式
                    if j == 0 and ':' in part:
                        mixed_row.append(part)
                    else:
                        try:
                            val = float(part)
                            mixed_row.append(val)
                        except:
                            mixed_row.append(0)

                parsed_data.append(mixed_row)

            # 限制读取行数
            if len(parsed_data) >= 50000:
                st.warning(f"⚠️ 达到最大行数限制，停止读取")
                break

        progress_bar.progress(1.0)

        if not parsed_data:
            st.error("❌ 未找到有效的数值数据")
            return None

        # 创建DataFrame
        df = pd.DataFrame(parsed_data, columns=cleaned_header)
        st.success(f"✅ 成功解析 {len(parsed_data)} 行数据")

        # 数据处理
        processed_df = df.copy()

        # 检查是否存在Time列并安全删除
        try:
            if 'Time' in processed_df.columns:
                numeric_df = processed_df.drop(['Time'], axis=1)
            else:
                numeric_df = processed_df.copy()
        except Exception as e:
            st.warning(f"删除Time列时出错: {e}")
            numeric_df = processed_df.copy()

        # 确保所有列都是数值类型
        for col in numeric_df.columns:
            try:
                # 确保列数据是Series类型
                col_data = numeric_df[col]
                if not isinstance(col_data, pd.Series):
                    col_data = pd.Series(col_data)
                numeric_df.loc[:, col] = pd.to_numeric(col_data, errors='coerce')
            except Exception as e:
                st.warning(f"转换列 {col} 时出错: {e}")
                # 如果转换失败，尝试强制转换为float
                try:
                    numeric_df.loc[:, col] = numeric_df[col].astype(float, errors='ignore')
                except:
                    pass

        # 处理时间列
        if 'ms' in numeric_df.columns:
            try:
                numeric_df.loc[:, 'time_minutes'] = numeric_df['ms'] / 60000
                numeric_df.loc[:, 'time_minutes_relative'] = numeric_df['time_minutes'] - numeric_df[
                    'time_minutes'].min()
                st.success(f"✅ 时间范围: 0 - {numeric_df['time_minutes_relative'].max():.2f} 分钟")
            except Exception as e:
                st.error(f"处理时间列时出错: {e}")
                return None
        else:
            st.error("❌ 未找到ms列")
            return None

        # 寻找Ar列进行归一化
        ar_columns = [col for col in numeric_df.columns if 'Ar' in str(col) and 'time' not in str(col)]

        if len(ar_columns) >= 1:
            # 显示所有找到的Ar列及其统计信息
            st.info(f"📊 找到 {len(ar_columns)} 个Ar列: {ar_columns}")

            # 获取第一个Ar列（因为看起来你只有一个名为"Ar"的列，但包含2列数据）
            ar_column_raw = numeric_df[ar_columns[0]]

            # 检查是否是多维数据
            if hasattr(ar_column_raw, 'shape') and len(ar_column_raw.shape) > 1:
                st.info(f"Ar列包含 {ar_column_raw.shape[1]} 列数据")

                # 显示每一列的统计
                for j in range(ar_column_raw.shape[1]):
                    col_data = ar_column_raw.iloc[:, j]
                    st.write(
                        f"  Ar列{j}: 均值={col_data.mean():.2e}, 范围=[{col_data.min():.2e}, {col_data.max():.2e}]")

                # 根据用户选择确定使用哪一列
                if ar_column_choice == "第一个Ar列":
                    ar_column = ar_column_raw.iloc[:, 0]
                    st.success(f"✅ 使用Ar的第1列进行归一化（均值: {ar_column.mean():.2e}）")
                elif ar_column_choice == "第二个Ar列":
                    if ar_column_raw.shape[1] > 1:
                        ar_column = ar_column_raw.iloc[:, 1]
                        st.success(f"✅ 使用Ar的第2列进行归一化（均值: {ar_column.mean():.2e}）")
                    else:
                        ar_column = ar_column_raw.iloc[:, 0]
                        st.warning("只有一列数据，使用第1列")
                else:
                    # 自动选择：使用第二列（数值较大的那列，通常是信号强度）
                    if ar_column_raw.shape[1] > 1:
                        # 比较两列的均值，选择较大的那列
                        col0_mean = ar_column_raw.iloc[:, 0].mean()
                        col1_mean = ar_column_raw.iloc[:, 1].mean()

                        if col1_mean > col0_mean:
                            ar_column = ar_column_raw.iloc[:, 1]
                            st.success(f"✅ 自动选择Ar的第2列（较大均值: {col1_mean:.2e}）")
                        else:
                            ar_column = ar_column_raw.iloc[:, 0]
                            st.success(f"✅ 自动选择Ar的第1列（较大均值: {col0_mean:.2e}）")
                    else:
                        ar_column = ar_column_raw.iloc[:, 0]
                        st.info("只有一列数据，使用第1列")
            else:
                # 如果不是多维数据，直接使用
                ar_column = ar_column_raw
                st.success(f"✅ 使用Ar列进行归一化（均值: {ar_column.mean():.2e}）")

        else:
            st.error("❌ 未找到Ar列用于归一化")
            return None

        # Ar列归一化处理
        try:
            # 确保ar_column是一维Series
            if not isinstance(ar_column, pd.Series):
                if hasattr(ar_column, 'values'):
                    # 如果是DataFrame或多维，转换为Series
                    ar_values = ar_column.values
                    if ar_values.ndim > 1:
                        ar_values = ar_values.flatten()
                    ar_column = pd.Series(ar_values, index=numeric_df.index[:len(ar_values)])
                else:
                    ar_column = pd.Series(ar_column)

            # 确保长度匹配
            if len(ar_column) != len(numeric_df):
                min_len = min(len(ar_column), len(numeric_df))
                ar_column = ar_column.iloc[:min_len]
                st.warning(f"调整Ar列长度为 {min_len}")

            ar_mean = float(ar_column.mean())
            ar_safe = ar_column.copy()

            # 安全地替换值
            ar_safe = ar_safe.replace(0, ar_mean)
            ar_safe = ar_safe.fillna(ar_mean)

            # 处理极小值
            mask = ar_safe.abs() < 1e-15
            ar_safe.loc[mask] = ar_mean

            st.success(f"✅ Ar列处理完成，长度: {len(ar_safe)}, 均值: {ar_mean:.2e}")

        except Exception as e:
            st.error(f"Ar列处理时出错: {e}")
            # 如果处理失败，创建一个默认的ar_safe
            ar_safe = pd.Series([1.0] * len(numeric_df), index=numeric_df.index)
            st.warning("使用默认Ar值进行归一化")

        # 获取需要归一化的气体列
        exclude_cols = ['ms', 'time_minutes', 'time_minutes_relative'] + ar_columns
        gas_columns = [col for col in numeric_df.columns if col not in exclude_cols]

        st.info(f"📈 需要归一化的气体: {len(gas_columns)} 种")

        # 逐列进行归一化
        for gas in gas_columns:
            try:
                # 确保数据是Series类型
                gas_data = numeric_df[gas]
                if not isinstance(gas_data, pd.Series):
                    gas_data = pd.Series(gas_data)

                # 确保ar_safe和gas_data长度一致
                if len(gas_data) != len(ar_safe):
                    min_len = min(len(gas_data), len(ar_safe))
                    gas_data = gas_data.iloc[:min_len]
                    ar_safe_temp = ar_safe.iloc[:min_len]
                else:
                    ar_safe_temp = ar_safe

                intensity_data = gas_data / ar_safe_temp
                numeric_df.loc[:len(intensity_data) - 1, f'{gas}_intensity'] = intensity_data.values

            except Exception as e:
                st.warning(f"处理气体 {gas} 时出错: {e}")
                continue

        # 将原始的Time列添加回去（如果存在的话）
        try:
            if 'Time' in processed_df.columns:
                numeric_df.loc[:, 'Time'] = processed_df['Time']
        except Exception as e:
            st.warning(f"添加Time列时出错: {e}")

        return numeric_df

    except Exception as e:
        st.error(f"处理数据时出错: {str(e)}")
        return None


def apply_baseline_correction(df, gas_columns, baseline_start_time=60, integration_start_time=3):
    """应用基线校正"""
    try:
        # 移除前integration_start_time分钟的数据点
        time_mask = df['time_minutes_relative'] >= integration_start_time
        df_cleaned = df[time_mask].copy().reset_index(drop=True)

        st.info(f"📊 移除前{integration_start_time}分钟的数据")

        # 寻找基线数据点
        time_baseline_mask = df_cleaned['time_minutes_relative'] >= baseline_start_time
        baseline_data = df_cleaned[time_baseline_mask].head(5)

        if len(baseline_data) == 0:
            baseline_data = df_cleaned.tail(5)

        st.success(f"✅ 基线数据点数: {len(baseline_data)}")

        # 基线校正
        corrected_df = df_cleaned.copy()
        baseline_info = {}

        intensity_cols = [f'{gas}_intensity' for gas in gas_columns if f'{gas}_intensity' in df_cleaned.columns]

        for col in intensity_cols:
            gas_name = col.replace('_intensity', '')

            try:
                baseline_time = baseline_data['time_minutes_relative'].values.reshape(-1, 1)
                baseline_intensity = baseline_data[col].values

                if len(baseline_time) >= 2:
                    reg = LinearRegression()
                    reg.fit(baseline_time, baseline_intensity)

                    all_time = df_cleaned['time_minutes_relative'].values.reshape(-1, 1)
                    baseline_values = reg.predict(all_time)

                    corrected_df[f'{gas_name}_corrected'] = df_cleaned[col] - baseline_values
                    corrected_df[f'{gas_name}_flattened'] = df_cleaned[col] - baseline_values

                    baseline_info[gas_name] = {
                        'slope': reg.coef_[0],
                        'intercept': reg.intercept_,
                        'baseline_mean': baseline_intensity.mean(),
                        'baseline_std': baseline_intensity.std()
                    }
                else:
                    baseline_mean = baseline_intensity.mean()
                    corrected_df[f'{gas_name}_corrected'] = df_cleaned[col] - baseline_mean
                    corrected_df[f'{gas_name}_flattened'] = df_cleaned[col] - baseline_mean

                    baseline_info[gas_name] = {
                        'slope': 0,
                        'intercept': baseline_mean,
                        'baseline_mean': baseline_mean,
                        'baseline_std': 0
                    }
            except:
                corrected_df[f'{gas_name}_corrected'] = df_cleaned[col]
                corrected_df[f'{gas_name}_flattened'] = df_cleaned[col]
                baseline_info[gas_name] = {'slope': 0, 'intercept': 0, 'baseline_mean': 0, 'baseline_std': 0}

        return corrected_df, baseline_info

    except Exception as e:
        st.error(f"基线校正时出错: {str(e)}")
        return df, {}


def calculate_gas_areas(df, gas_columns, integration_start_time=3):
    """计算气体曲线面积积分"""
    try:
        results = {}

        integration_mask = df['time_minutes_relative'] >= integration_start_time
        integration_df = df[integration_mask]
        time_data = integration_df['time_minutes_relative'].values

        for gas in gas_columns:
            corrected_col = f'{gas}_corrected'
            if corrected_col in integration_df.columns:
                try:
                    intensity_data = integration_df[corrected_col].values

                    total_area = integrate.trapz(np.abs(intensity_data), time_data)
                    net_area = integrate.trapz(intensity_data, time_data)

                    results[gas] = {
                        'total_area': total_area,
                        'net_area': net_area,
                        'max_intensity': intensity_data.max(),
                        'min_intensity': intensity_data.min(),
                        'mean_intensity': intensity_data.mean(),
                        'std_intensity': intensity_data.std()
                    }
                except:
                    results[gas] = {
                        'total_area': 0, 'net_area': 0,
                        'max_intensity': 0, 'min_intensity': 0,
                        'mean_intensity': 0, 'std_intensity': 0
                    }

        return results

    except Exception as e:
        st.error(f"计算面积时出错: {str(e)}")
        return {}


def main():
    try:
        # 标题和说明
        st.title("🧪 气体分析工具")
        st.markdown("**专业的气体强度数据分析与可视化平台**")

        # 初始化session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'corrected_data' not in st.session_state:
            st.session_state.corrected_data = None
        if 'baseline_info' not in st.session_state:
            st.session_state.baseline_info = None
        if 'integration_results' not in st.session_state:
            st.session_state.integration_results = None
        if 'deleted_points' not in st.session_state:
            st.session_state.deleted_points = set()

        # 侧边栏设置
        st.sidebar.header("📊 图表设置")
        font_size = st.sidebar.slider("字体大小", min_value=8, max_value=20, value=12)
        line_width = st.sidebar.slider("线条粗细", min_value=1, max_value=5, value=2)

        st.sidebar.header("🔧 Ar列选择")
        ar_column_choice = st.sidebar.selectbox(
            "选择用于归一化的Ar列",
            ["自动选择", "第一个Ar列", "第二个Ar列"],
            index=0,
            help="Ar列包含两列数据，选择用于归一化的列"
        )

        st.sidebar.header("⚙️ 分析参数")
        baseline_start_time = st.sidebar.number_input("基线开始时间 (分钟)", min_value=10, max_value=200, value=60)
        integration_start_time = st.sidebar.number_input("积分开始时间 (分钟)", min_value=0, max_value=30, value=3)

        st.sidebar.header("🧮 积分计算")
        calculate_integration = st.sidebar.checkbox("计算面积积分", value=True)

        # 文件上传
        st.header("📁 文件上传")
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="支持包含气体强度数据的CSV文件"
        )

        if uploaded_file is not None:
            file_content = uploaded_file.read()

            with st.spinner("正在处理文件..."):
                processed_data = process_gas_data(file_content, uploaded_file.name, ar_column_choice)

            if processed_data is not None:
                st.session_state.data = processed_data

                # 显示数据基本信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总数据点", len(processed_data))
                with col2:
                    intensity_cols = [col for col in processed_data.columns if col.endswith('_intensity')]
                    st.metric("气体种类", len(intensity_cols))
                with col3:
                    time_range = processed_data['time_minutes_relative'].max()
                    st.metric("时间跨度", f"{time_range:.1f} 分钟")
                with col4:
                    st.metric("数据完整性", f"{processed_data.notna().mean().mean():.1%}")

        # 数据处理和分析
        if st.session_state.data is not None:
            df = st.session_state.data

            # 数据预览
            st.header("📊 数据预览")
            display_cols = ['time_minutes_relative'] + [col for col in df.columns if col.endswith('_intensity')][:5]
            st.dataframe(df[display_cols].head(10), use_container_width=True)

            # 交互式图表
            st.header("📈 数据可视化")

            # 图表选项卡
            tab1, tab2, tab3, tab4 = st.tabs(["原始数据", "基线校正", "拉平基线图", "数据管理"])

            with tab1:
                st.subheader("气体强度随时间变化")

                # 气体选择
                intensity_cols = [col for col in df.columns if col.endswith('_intensity')]
                gas_names = [col.replace('_intensity', '') for col in intensity_cols]

                selected_gases = st.multiselect(
                    "选择要显示的气体",
                    gas_names,
                    default=gas_names[:5] if len(gas_names) > 5 else gas_names
                )

                if selected_gases:
                    # 过滤数据
                    filtered_df = df.copy()
                    if st.session_state.deleted_points:
                        mask = ~df.index.isin(st.session_state.deleted_points)
                        filtered_df = df[mask]

                    # 生成颜色
                    colors = generate_distinct_colors(len(selected_gases))

                    # 创建图表
                    fig = go.Figure()

                    for i, gas in enumerate(selected_gases):
                        col_name = f'{gas}_intensity'
                        if col_name in filtered_df.columns:
                            fig.add_trace(go.Scatter(
                                x=filtered_df['time_minutes_relative'],
                                y=filtered_df[col_name],
                                mode='lines',
                                name=gas,
                                line=dict(color=colors[i], width=line_width),
                                hovertemplate=f'<b>{gas}</b><br>时间: %{{x:.2f}} 分钟<br>强度: %{{y:.2e}}<extra></extra>'
                            ))

                    fig.update_layout(
                        title=dict(text='气体强度随时间变化图', font=dict(size=font_size + 4)),
                        xaxis=dict(
                            title=dict(text='时间 (分钟)', font=dict(size=font_size)),
                            tickfont=dict(size=font_size - 2)
                        ),
                        yaxis=dict(
                            title=dict(text='气体信号强度', font=dict(size=font_size)),
                            tickfont=dict(size=font_size - 2)
                        ),
                        height=600,
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("基线校正")

                if st.button("执行基线校正", type="primary"):
                    with st.spinner("正在进行基线校正..."):
                        gas_columns = [col.replace('_intensity', '') for col in df.columns if
                                       col.endswith('_intensity')]
                        corrected_data, baseline_info = apply_baseline_correction(df, gas_columns, baseline_start_time,
                                                                                  integration_start_time)

                        st.session_state.corrected_data = corrected_data
                        st.session_state.baseline_info = baseline_info

                        if calculate_integration:
                            integration_results = calculate_gas_areas(corrected_data, gas_columns,
                                                                      integration_start_time)
                            st.session_state.integration_results = integration_results

                        st.success("✅ 基线校正完成！")

                # 显示校正后的数据
                if st.session_state.corrected_data is not None:
                    corrected_df = st.session_state.corrected_data

                    corrected_cols = [col for col in corrected_df.columns if col.endswith('_corrected')]
                    corrected_gas_names = [col.replace('_corrected', '') for col in corrected_cols]

                    selected_corrected_gases = st.multiselect(
                        "选择要显示的校正后气体",
                        corrected_gas_names,
                        default=corrected_gas_names[:5] if len(corrected_gas_names) > 5 else corrected_gas_names,
                        key="corrected_gases"
                    )

                    if selected_corrected_gases:
                        colors = generate_distinct_colors(len(selected_corrected_gases))
                        fig_corrected = go.Figure()

                        for i, gas in enumerate(selected_corrected_gases):
                            col_name = f'{gas}_corrected'
                            if col_name in corrected_df.columns:
                                fig_corrected.add_trace(go.Scatter(
                                    x=corrected_df['time_minutes_relative'],
                                    y=corrected_df[col_name],
                                    mode='lines',
                                    name=f'{gas} (校正)',
                                    line=dict(color=colors[i], width=line_width),
                                    hovertemplate=f'<b>{gas} (校正)</b><br>时间: %{{x:.2f}} 分钟<br>校正强度: %{{y:.2e}}<extra></extra>'
                                ))

                        fig_corrected.update_layout(
                            title=dict(text='基线校正后气体强度图', font=dict(size=font_size + 4)),
                            xaxis=dict(
                                title=dict(text='时间 (分钟)', font=dict(size=font_size)),
                                tickfont=dict(size=font_size - 2)
                            ),
                            yaxis=dict(
                                title=dict(text='校正后气体信号强度', font=dict(size=font_size)),
                                tickfont=dict(size=font_size - 2)
                            ),
                            height=600,
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )

                        st.plotly_chart(fig_corrected, use_container_width=True)

            with tab3:
                st.subheader("拉平基线后的图表")

                if st.session_state.corrected_data is not None:
                    corrected_df = st.session_state.corrected_data

                    flattened_cols = [col for col in corrected_df.columns if col.endswith('_flattened')]
                    flattened_gas_names = [col.replace('_flattened', '') for col in flattened_cols]

                    if flattened_gas_names:
                        selected_flattened_gases = st.multiselect(
                            "选择要显示的拉平基线后气体",
                            flattened_gas_names,
                            default=flattened_gas_names[:5] if len(flattened_gas_names) > 5 else flattened_gas_names,
                            key="flattened_gases"
                        )

                        if selected_flattened_gases:
                            colors = generate_distinct_colors(len(selected_flattened_gases))
                            fig_flattened = go.Figure()

                            for i, gas in enumerate(selected_flattened_gases):
                                col_name = f'{gas}_flattened'
                                if col_name in corrected_df.columns:
                                    fig_flattened.add_trace(go.Scatter(
                                        x=corrected_df['time_minutes_relative'],
                                        y=corrected_df[col_name],
                                        mode='lines',
                                        name=f'{gas} (拉平)',
                                        line=dict(color=colors[i], width=line_width),
                                        hovertemplate=f'<b>{gas} (拉平)</b><br>时间: %{{x:.2f}} 分钟<br>拉平强度: %{{y:.2e}}<extra></extra>'
                                    ))

                            fig_flattened.update_layout(
                                title=dict(text='拉平基线后气体强度图', font=dict(size=font_size + 4)),
                                xaxis=dict(
                                    title=dict(text='时间 (分钟)', font=dict(size=font_size)),
                                    tickfont=dict(size=font_size - 2)
                                ),
                                yaxis=dict(
                                    title=dict(text='拉平后气体信号强度', font=dict(size=font_size)),
                                    tickfont=dict(size=font_size - 2)
                                ),
                                height=600,
                                showlegend=True,
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )

                            st.plotly_chart(fig_flattened, use_container_width=True)
                    else:
                        st.info("没有找到拉平基线数据，请先执行基线校正")
                else:
                    st.info("请先执行基线校正")

            with tab4:
                st.subheader("数据点管理")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**选择要删除的数据点范围:**")
                    time_min = float(df['time_minutes_relative'].min())
                    time_max = float(df['time_minutes_relative'].max())

                    delete_range = st.slider(
                        "时间范围 (分钟)",
                        min_value=time_min,
                        max_value=time_max,
                        value=(time_min, time_max),
                        step=0.1
                    )

                    if st.button("删除选定范围的数据点"):
                        mask = (df['time_minutes_relative'] >= delete_range[0]) & (
                                    df['time_minutes_relative'] <= delete_range[1])
                        points_to_delete = df[mask].index.tolist()
                        st.session_state.deleted_points.update(points_to_delete)
                        st.success(f"✅ 已删除 {len(points_to_delete)} 个数据点")

                with col2:
                    st.write("**已删除的数据点:**")
                    st.info(f"共删除了 {len(st.session_state.deleted_points)} 个数据点")

                    if st.button("恢复所有数据点"):
                        st.session_state.deleted_points.clear()
                        st.success("✅ 已恢复所有数据点")

            # 分析报表
            if st.session_state.integration_results is not None and calculate_integration:
                st.header("📋 分析报表")

                integration_results = st.session_state.integration_results
                baseline_info = st.session_state.baseline_info

                # 创建报表数据
                report_data = []
                for gas in integration_results.keys():
                    if gas in baseline_info:
                        baseline = baseline_info[gas]
                        integration = integration_results[gas]

                        report_data.append({
                            '气体': gas,
                            '基线斜率': f"{baseline['slope']:.6e}",
                            '基线截距': f"{baseline['intercept']:.6e}",
                            '总面积': f"{integration['total_area']:.6e}",
                            '净面积': f"{integration['net_area']:.6e}",
                            '最大强度': f"{integration['max_intensity']:.6e}",
                            '最小强度': f"{integration['min_intensity']:.6e}",
                            '平均强度': f"{integration['mean_intensity']:.6e}"
                        })

                if report_data:
                    report_df = pd.DataFrame(report_data)
                    st.dataframe(report_df, use_container_width=True)

                    # 下载报表
                    csv = report_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载分析报表 (CSV)",
                        data=csv,
                        file_name="gas_analysis_report.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"应用运行时出错: {str(e)}")
        st.info("请刷新页面重试")


if __name__ == "__main__":
    main()
