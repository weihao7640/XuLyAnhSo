import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Báo cáo cuối kỳ môn Xử lý ảnh👋")

st.sidebar.success("Chọn chương trình chạy ở trên ")

st.markdown(
    """
        <h3>Họ tên sinh viên thực hiện:</h2> <br>
        <div>\t 1. Nguyễn Đức Toàn - MSSV: 20110220 </div>
        <div>\t 2. Nguyễn Duy Hào - MSSV: 20110220 </div> 
    """,
    unsafe_allow_html=True
)

st.image('vango.jpg')
