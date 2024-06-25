import sys 
# sys.path.append('/home/aistudio/PRTS/packages')
sys.path.insert(0, '/home/aistudio/PRTS/packages')
import streamlit as st

def main():
    st.markdown(st.__version__)
    input_txt = st.chat_input("input")

if __name__ == "__main__":
    main()