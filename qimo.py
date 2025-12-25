import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder  

# ---------------------- 1. 基础配置（保持不变） ----------------------
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="💯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 中文字体配置（解决乱码，微调字体大小更美观）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['figure.figsize'] = (12, 7)      # 小幅放大图表，显示更清晰
plt.rcParams['font.size'] = 11                # 微调基础字体大小
plt.rcParams['axes.titlesize'] = 14           # 放大标题字体，更醒目
plt.rcParams['axes.labelsize'] = 12           # 放大坐标轴标签字体

# ---------------------- 2. 数据加载（保持不变） ----------------------
@st.cache_data(show_spinner="正在加载学生数据...")
def load_student_data():
    encoding_list = ['utf-8-sig', 'gbk']
    df = None
    for enc in encoding_list:
        try:
            df = pd.read_csv('student_data_adjusted_rounded.csv', encoding=enc)
            break
        except:
            continue
    
    if df is None:
        st.error("❌ 无法读取CSV文件，请确认文件存在且格式正确")
        st.stop()
    
    required_cols = ['专业', '性别', '每周学习时长（小时）', '期中考试分数', '期末考试分数', '上课出勤率', '作业完成率']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ 缺少必要列：{', '.join(missing_cols)}")
        st.stop()
    return df

df = load_student_data()

# 定义特征信息
TARGET_COL = '期末考试分数'
FEATURE_COLS = [
    col for col in df.columns 
    if col not in ['学号', TARGET_COL] and df[col].dtype in [object, int, float]
]
REAL_GENDERS = df['性别'].dropna().unique().tolist() if '性别' in FEATURE_COLS else []
REAL_MAJORS = df['专业'].dropna().unique().tolist() if '专业' in FEATURE_COLS else []

# ---------------------- 3. 模型加载（保持不变） ----------------------
@st.cache_resource(show_spinner="正在加载预测模型...")
def load_model():
    model_path = 'student_final_score_model.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"❌ 未找到模型文件：{model_path}，请确认文件存在于当前目录")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.stop()

# ---------------------- 4. 特征编码（保持不变） ----------------------
def encode_features(gender, major):
    try:
        gender_le = LabelEncoder()
        gender_le.fit(REAL_GENDERS)
        gender_code = gender_le.transform([gender])[0]
    except:
        st.warning("⚠️ 性别编码失败，使用默认值0")
        gender_code = 0
    
    try:
        major_le = LabelEncoder()
        major_le.fit(REAL_MAJORS)
        major_code = major_le.transform([major])[0]
    except:
        st.warning("⚠️ 专业编码失败，使用默认值0")
        major_code = 0
    
    return gender_code, major_code

# ---------------------- 5. 侧边栏（保持不变） ----------------------
with st.sidebar:
    st.title("📌 导航菜单")
    st.divider()
    nav_option = st.radio(
        label="请选择功能页面",
        options=["项目介绍", "专业成绩分析", "成绩预测"],
        index=0,
        format_func=lambda x: f"📑 {x}"
    )
    st.divider()
    st.subheader("📊 专业成绩分析预览")
    preview_items = [
        "各专业学习时长对比图",
        "期中/期末成绩趋势图",
        "专业出勤率排名",
        "大数据管理专业专项分析"
    ]
    for item in preview_items:
        st.write(f"• {item}")

# ---------------------- 6. 页面1：项目介绍页（已修改：适配images文件夹图片） ----------------------
def show_project_intro():
    # 核心修改：分栏实现标题左、图片右（右上角）布局
    title_column, image_column = st.columns([3, 1])  # 3:1 列宽比例，标题占主导，图片在右侧
    with title_column:
        st.title("🎓 学生成绩分析与预测系统")
    with image_column:
        # 关键修改：路径改为 images/捕获.PNG，适配子文件夹存放
        # width=180 可按需调整图片大小
        st.image("images/捕获.PNG", width=800)  # 已适配images文件夹，无需其他修改

    st.divider()

    # 项目概述
    st.header("📂 项目概述")
    st.write(
        "本项目基于Streamlit搭建轻量化学生成绩分析平台，"
        "助力教育工作者洞察学生学习表现，同时通过机器学习模型实现期末考试成绩智能预测。"
    )
    
    st.subheader("✨ 主要特点")
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.markdown("""
        - 📊 多维度数据可视化展示
        - 🔍 精细化专业学业分析
        """)
    with feat_col2:
        st.markdown("""
        - 🤖 基于机器学习的智能预测
        - 💡 个性化学习建议推送
        """)

    st.divider()

    # 项目目标
    st.header("🎯 项目目标")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("目标一：分析优化")
        st.markdown("""
        - 识别关键学习指标
        - 探索成绩相关性
        - 提供教学改进参考
        """)
    with col2:
        st.subheader("目标二：可视化跟踪")
        st.markdown("""
        - 跨专业对比分析
        - 学习趋势动态追踪
        - 异常学习情况识别
        """)
    with col3:
        st.subheader("目标三：智能预测")
        st.markdown("""
        - 预训练机器学习模型
        - 个性化成绩预测
        - 提前干预风险预警
        """)

    st.divider()

    # 技术架构
    st.header("🔧 技术架构")
    arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)
    with arch_col1:
        st.subheader("前端框架")
        st.write("Streamlit")
    with arch_col2:
        st.subheader("数据处理")
        st.write("Pandas\nNumpy")
    with arch_col3:
        st.subheader("可视化")
        st.write("Matplotlib")
    with arch_col4:
        st.subheader("机器学习")
        st.write("Scikit-Learn")

# ---------------------- 7. 页面2：专业成绩分析页（保持不变） ----------------------
def show_major_analysis():
    st.title("📊 专业成绩分析")
    st.divider()

    # 第1行：性别比例图表 + 数据表
    st.subheader("一、各专业男女性别比例分析")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        gender_count = df.groupby(['专业', '性别']).size().unstack(fill_value=0)
        for g in ['男', '女']:
            if g not in gender_count.columns:
                gender_count[g] = 0
        gender_count = gender_count[['男', '女']]
        
        fig, ax = plt.subplots()
        bar_width = 0.35
        index = np.arange(len(gender_count.index))
        bar1 = ax.bar(index - bar_width/2, gender_count['男'], bar_width, label='男生', color='#4287f5', alpha=0.8)
        bar2 = ax.bar(index + bar_width/2, gender_count['女'], bar_width, label='女生', color='#f54242', alpha=0.8)
        
        ax.set_xlabel('专业')
        ax.set_ylabel('学生人数')
        ax.set_title('各专业男女性别分布')
        ax.set_xticks(index)
        ax.set_xticklabels(gender_count.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bar1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom')
        for bar in bar2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.subheader("性别比例详细数据表")
        gender_detail = df.groupby(['专业', '性别']).size().unstack(fill_value=0)
        for g in ['男', '女']:
            if g not in gender_detail.columns:
                gender_detail[g] = 0
        
        gender_detail['总人数'] = gender_detail['男'] + gender_detail['女']
        gender_detail['男生占比(%)'] = (gender_detail['男'] / gender_detail['总人数'] * 100).round(2)
        gender_detail['女生占比(%)'] = (gender_detail['女'] / gender_detail['总人数'] * 100).round(2)
        
        gender_detail_df = gender_detail.reset_index()
        gender_detail_df.insert(0, '序号', range(1, len(gender_detail_df)+1))
        st.dataframe(gender_detail_df, use_container_width=True, hide_index=True)

    st.divider()

    # 第2行：学习指标图表 + 数据表
    st.subheader("二、各专业学习指标分析")
    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.subheader("期中期末成绩趋势图")
        score_data = df.groupby('专业').agg({
            '期中考试分数': 'mean',
            '期末考试分数': 'mean'
        }).round(2)
        
        fig, ax = plt.subplots()
        ax.plot(score_data.index, score_data['期中考试分数'], marker='o', linewidth=2, label='期中平均分', color='#28c76f')
        ax.plot(score_data.index, score_data['期末考试分数'], marker='s', linewidth=2, label='期末平均分', color='#f9a825')
        
        ax.set_xlabel('专业')
        ax.set_ylabel('平均分数')
        ax.set_title('各专业期中期末成绩趋势')
        ax.set_xticklabels(score_data.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i, (mid, final) in enumerate(zip(score_data['期中考试分数'], score_data['期末考试分数'])):
            ax.text(i, mid + 1, f'{mid}', ha='center', va='bottom', fontsize=10)
            ax.text(i, final + 1, f'{final}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
    with col4:
        st.subheader("学习指标详细数据表")
        study_detail = df.groupby('专业').agg({
            '每周学习时长（小时）': lambda x: round(x.mean(), 2),
            '期中考试分数': lambda x: round(x.mean(), 2),
            '期末考试分数': lambda x: round(x.mean(), 2)
        }).round(2)
        
        study_detail.columns = ['每周学习时长(小时)', '期中平均分', '期末平均分']
        study_detail_df = study_detail.reset_index()
        study_detail_df.insert(0, '序号', range(1, len(study_detail_df)+1))
        st.dataframe(study_detail_df, use_container_width=True, hide_index=True)

    st.divider()

    # 第3行：出勤率分析图表 + 数据表
    st.subheader("三、各专业出勤率分析")
    col5, col6 = st.columns(2, gap="medium")
    with col5:
        st.subheader("各专业平均出勤率柱状图")
        attendance_data = (df.groupby('专业')['上课出勤率'].mean() * 100).round(2).sort_values(ascending=False)
        
        fig, ax = plt.subplots()
        bars = ax.bar(attendance_data.index, attendance_data.values, color='#8e44ad', alpha=0.8)
        
        ax.set_xlabel('专业')
        ax.set_ylabel('出勤率(%)')
        ax.set_title('各专业平均出勤率排名')
        ax.set_xticklabels(attendance_data.index, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    with col6:
        st.subheader("出勤率排行数据表")
        attendance_detail = (df.groupby('专业')['上课出勤率'].mean() * 100).round(2).reset_index()
        attendance_detail.columns = ['专业', '平均出勤率(%)']
        attendance_detail = attendance_detail.sort_values(by='平均出勤率(%)', ascending=False)
        attendance_detail.insert(0, '排名', range(1, len(attendance_detail)+1))
        
        st.dataframe(attendance_detail, use_container_width=True, hide_index=True)

    st.divider()

    # 第4行：大数据管理专业专项分析
    st.subheader("四、大数据管理专业专项分析")
    col7 = st.columns(1)[0]
    with col7:
        bigdata_df = df[df['专业'] == '大数据管理'].copy()
        if len(bigdata_df) == 0:
            st.warning("⚠️ 未找到'大数据管理'专业，展示所有专业数据替代")
            bigdata_df = df.copy()
            target_major = "所有专业"
        else:
            target_major = "大数据管理专业"
        
        col7_1, col7_2, col7_3 = st.columns(3)
        with col7_1:
            st.info(f"**分析对象**\n{target_major}")
            st.info(f"**学生总数**\n{len(bigdata_df)} 人")
        with col7_2:
            avg_study_hour = bigdata_df['每周学习时长（小时）'].mean().round(2)
            avg_attendance = (bigdata_df['上课出勤率'].mean() * 100).round(2)
            st.success(f"**每周平均学习时长**\n{avg_study_hour} 小时")
            st.success(f"**平均出勤率**\n{avg_attendance}%")
        with col7_3:
            avg_final_score = bigdata_df['期末考试分数'].mean().round(2)
            min_final_score = bigdata_df['期末考试分数'].min()
            max_final_score = bigdata_df['期末考试分数'].max()
            st.warning(f"**期末平均分**\n{avg_final_score} 分")
            st.warning(f"**期末分数范围**\n{min_final_score} - {max_final_score} 分")
        
        st.divider()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(
            bigdata_df['每周学习时长（小时）'],
            bigdata_df['期末考试分数'],
            c=bigdata_df['期末考试分数'],
            cmap='viridis',
            alpha=0.7,
            s=60
        )
        
        z = np.polyfit(bigdata_df['每周学习时长（小时）'], bigdata_df['期末考试分数'], 1)
        p = np.poly1d(z)
        ax.plot(bigdata_df['每周学习时长（小时）'], p(bigdata_df['每周学习时长（小时）']), 
                "r--", linewidth=2, label=f'趋势线（y={z[0]:.2f}x+{z[1]:.2f}）')
        
        ax.set_xlabel('每周学习时长（小时）', fontsize=11)
        ax.set_ylabel('期末考试分数', fontsize=11)
        ax.set_title(f'{target_major}：每周学习时长与期末考试成绩关系', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, label='期末考试分数')
        
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------- 8. 页面3：成绩预测页（保持不变） ----------------------
def show_score_prediction():
    st.title("🤖 成绩预测")
    st.divider()
    st.markdown(f"基于真实学生数据（共{len(df)}条记录），特征自动匹配数据分布")
    st.divider()
    
    model = load_model()
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📝 学生信息输入")
        student_id = st.text_input("学号", value="2024001", help="仅用于标识，不参与预测")
        
        gender = st.selectbox("性别", options=REAL_GENDERS) if '性别' in FEATURE_COLS else st.text_input("性别", value="男")
        major = st.selectbox("专业", options=REAL_MAJORS) if '专业' in FEATURE_COLS else st.text_input("专业", value="信息系统")
        
        st.markdown("#### 📚 学习特征")
        study_hours_min, study_hours_max = int(df['每周学习时长（小时）'].min()), int(df['每周学习时长（小时）'].max())
        study_hours = st.slider(
            "每周学习时长（小时）", 
            min_value=study_hours_min, 
            max_value=study_hours_max, 
            value=(study_hours_min + study_hours_max) // 2,
            help=f"范围：{study_hours_min} - {study_hours_max} 小时"
        )
        
        attendance_min, attendance_max = round(df['上课出勤率'].min(), 2), round(df['上课出勤率'].max(), 2)
        attendance = st.slider(
            "上课出勤率", 
            min_value=attendance_min, 
            max_value=attendance_max, 
            value=round((attendance_min + attendance_max) / 2, 2), 
            step=0.01,
            help=f"范围：{attendance_min} - {attendance_max}"
        )
        
        midterm_min, midterm_max = int(df['期中考试分数'].min()), int(df['期中考试分数'].max())
        midterm_score = st.slider(
            "期中考试分数", 
            min_value=midterm_min, 
            max_value=midterm_max, 
            value=(midterm_min + midterm_max) // 2,
            help=f"范围：{midterm_min} - {midterm_max} 分"
        )
        
        homework_min, homework_max = round(df['作业完成率'].min(), 2), round(df['作业完成率'].max(), 2)
        homework_rate = st.slider(
            "作业完成率", 
            min_value=homework_min, 
            max_value=homework_max, 
            value=round((homework_min + homework_max) / 2, 2), 
            step=0.01,
            help=f"范围：{homework_min} - {homework_max}"
        )
    
        predict_btn = st.button("🚀 预测期末成绩", type="primary")
    
    with col2:
        st.subheader("📊 预测结果")
        if predict_btn:
            gender_code, major_code = encode_features(gender, major)
            
            features = pd.DataFrame({
                "性别": [gender_code],
                "专业": [major_code],
                "每周学习时长（小时）": [study_hours],
                "上课出勤率": [attendance],
                "期中考试分数": [midterm_score],
                "作业完成率": [homework_rate]
            })
            
            try:
                final_score = model.predict(features)[0]
                final_score = round(final_score, 1)
            except Exception as e:
                st.error(f"❌ 预测失败：{str(e)}")
                return
            
            st.success(f"🎯 预测期末成绩：{final_score} 分")
            if final_score >= 80:
                st.image("images/优秀.jpg", width=300, caption="优秀！继续保持")
            elif 60 <= final_score < 80:
                st.image("images/恭喜.jpg", width=300, caption="良好！有待提升")
            else:
                st.image("images/鼓励.jpg", width=300, caption="加油！需要努力")
            
            st.divider()
            st.subheader("💡 学习建议")
            if final_score >= 80:
                st.markdown("🎉 成绩优秀！建议保持当前的学习节奏，可适当拓展学科相关的实践项目，进一步提升专业能力。")
            elif 60 <= final_score < 80:
                st.markdown(f"📖 成绩良好！建议适当增加每周学习时长（当前：{study_hours}小时），同时提高作业完成质量，争取更优异的成绩。")
            else:
                st.markdown(f"⚠️ 成绩待提升！建议优先提高上课出勤率（当前：{attendance}），并针对性加强期中考试相关知识点的复习，尽快提升成绩。")
        else:
            st.info("ℹ️ 请填写左侧信息后，点击「🚀 预测期末成绩」按钮获取结果")

# ---------------------- 9. 导航逻辑（保持不变） ----------------------
if nav_option == "项目介绍":
    show_project_intro()
elif nav_option == "专业成绩分析":
    show_major_analysis()
elif nav_option == "成绩预测":
    show_score_prediction()
