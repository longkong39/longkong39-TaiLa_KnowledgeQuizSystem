import streamlit as st

# 页面配置，设置页面宽布局，使侧边栏和主体内容更适合
st.set_page_config(page_title="侧边菜单示例", layout="wide")

# 初始化会话状态，如果第一次运行会话状态不会自动创建
if 'page' not in st.session_state:
    st.session_state.page = '首页'

# 定义侧边菜单
def sidebar_menu():
    selection = st.sidebar.selectbox("导航", ["首页", "关于", "联系我们"])
    st.session_state.page = selection

# 主体内容根据侧边菜单的选择显示不同页面
def main_content():
    if st.session_state.page == "首页":
        st.title("欢迎来到首页")
        st.write("这是应用的首页内容...")
    elif st.session_state.page == "关于":
        st.title("关于我们")
        st.write("这里是关于我们的介绍...")
    elif st.session_state.page == "联系我们":
        st.title("联系我们")
        st.write("联系方式和地址等信息...")

# 应用主体
def main():
    sidebar_menu()
    main_content()

if __name__ == "__main__":
    main()