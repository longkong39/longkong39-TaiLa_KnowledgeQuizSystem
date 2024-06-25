# -*- encoding: utf-8 -*-
import sys 
# sys.path.append('PRTS/packages')
# sys.path.insert(0, 'packages')
import os
import shutil
import time
from pathlib import Path

import numpy as np
import streamlit as st
from utils import make_prompt, read_yaml, get_timestamp, mkdir
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer

config = read_yaml("config/config.yaml")

st.set_page_config(
    page_title=config.get("title"),
    layout="wide", 
    page_icon=":robot:",
)

if 'page' not in st.session_state:
    st.session_state.page = '首页'

def sidebar_menu():
    selection = st.sidebar.selectbox("导航", ["首页", "对话", "知识库"])
    st.session_state.page = selection


def main_content():
    if st.session_state.page == "首页":
        css = """
            <style>
            .reportview-container .main .block-container {
                max-width: 900px;
                padding-top: 5rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 3rem;
            }
            .stTitle {
                text-align: center;
            }
            </style>
        """
        # 使用st.markdown插入CSS样式
        st.markdown(css, unsafe_allow_html=True)
        # 设置居中的标题
        st.markdown("<h1 class='stTitle'>泰拉知识问答系统</h1>", unsafe_allow_html=True)
        st.image('source/image_0.jpg')
        st.sidebar.image('source/image_2.jpg')
    elif st.session_state.page == "对话":
        talk()
    elif st.session_state.page == "知识库":
        knowledge_base()


def init_ui_parameters():
    st.session_state["params"] = {}

    st.sidebar.markdown("### 🛶 参数设置")
    llm_path = config.get('LLMPath')
    llm = os.listdir(llm_path)
    llm = st.sidebar.selectbox('LLM', llm)
    st.session_state["params"]["llm"] = os.path.join(llm_path, llm)

    options = ['cuda', 'cuda:1', 'mps', 'tpu', 'cpu']
    device = st.sidebar.selectbox('tokenizer device', options)
    st.session_state["params"]["device"] = device

    param = config.get("Parameter")
    param_max_length = param.get("max_length")
    max_length = st.sidebar.slider(
        "max_length",
        min_value=param_max_length.get("min_value"),
        max_value=param_max_length.get("max_value"),
        value=param_max_length.get("default"),
        step=param_max_length.get("step"),
        help=param_max_length.get("tip"),
    )
    st.session_state["params"]["max_length"] = max_length

    param_top = param.get("top_p")
    top_p = st.sidebar.slider(
        "top_p",
        min_value=param_top.get("min_value"),
        max_value=param_top.get("max_value"),
        value=param_top.get("default"),
        step=param_top.get("step"),
        help=param_top.get("tip"),
    )
    st.session_state["params"]["top_p"] = top_p

    param_temp = param.get("temperature")
    temperature = st.sidebar.slider(
        "temperature",
        min_value=param_temp.get("min_value"),
        max_value=param_temp.get("max_value"),
        value=param_temp.get("default"),
        step=param_temp.get("stemp"),
        help=param_temp.get("tip"),
    )
    st.session_state["params"]["temperature"] = temperature
    st.sidebar.markdown("### 📖 知识库选择")
    if os.path.isdir('db_base'):
        mkdir('db_base')
    kb_options = ['None'] + os.listdir('db_base')
    st.session_state["params"]["select_knowledge_base"] = st.sidebar.selectbox('Knowledge base', kb_options)


def init_ui_db():
    st.markdown("### 🧻 知识库")
    menu_col1, menu_col2, menu_col3 = st.columns([1, 1, 1])
    select_name = menu_col1.text_input("🏷️ 知识库名称:", placeholder="请输入名称...")
    select_embedding = menu_col2.selectbox("🔍 Embedding:", ['text2vec-base-chinese'])
    select_chunk_size = menu_col2.selectbox("🗂️ chunk_size:", [100, 200, 300, 500])
    uploaded_files = st.file_uploader(
        "default",
        accept_multiple_files=True,
        label_visibility="hidden",
        help="支持多个文件的选取",
    )
    upload_dir = config.get("upload_dir")
    btn_upload = st.button("上传文档并加载")
    if btn_upload:
        time_stamp = get_timestamp()
        if 'embedding_extract' not in st.session_state or not st.session_state["embedding_extract"]:
            st.session_state["embedding_extract"] = init_encoder('models/Embeddings/text2vec-base-chinese')
        doc_dir = Path(upload_dir) / time_stamp

        tips("正在上传文件到平台中...", icon="⏳")
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
        mkdir(doc_dir)
        for file_data in uploaded_files:
            bytes_data = file_data.getvalue()
            save_path = doc_dir / file_data.name
            with open(save_path, "wb") as f:
                f.write(bytes_data)
        tips("上传完毕！")

        split_docs = []
        with st.spinner(f"正在从{doc_dir}提取内容...."):
            file_paths = [os.path.join(doc_dir, path) for path in os.listdir(doc_dir)]
            for path in file_paths:
                loader = UnstructuredFileLoader(path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=select_chunk_size, chunk_overlap=0)
                split_doc = text_splitter.split_documents(data)
                split_docs += split_doc
        with st.spinner(f"正在构建知识库...."):
            db_save_path = os.path.join(config.get('db_save_path'), select_name)
            db = Chroma.from_documents(split_docs, st.session_state["embedding_extract"], persist_directory=db_save_path)
            db.persist()
            tips("知识库构建成功！")

        shutil.rmtree(doc_dir.resolve())
        tips("现在可以提问问题了哈！")


@st.cache_resource
def init_encoder(path):
    embeddings = HuggingFaceEmbeddings(model_name=path)
    return embeddings


def init_llm(path):
    model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
    device_map="auto"
    )
    return model


def init_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer


def init_db(path,embedding):
    db = Chroma(persist_directory=path, embedding_function=embedding)
    return db


def predict(text, similarDocs):
    if similarDocs:
        info = ""
        for similardoc in similarDocs:
            info = info + similardoc.page_content + '\n'
    else:
        info = None
    response, elapse = get_model_response(text, info)

    responce = response + f"\n**推理耗时:{elapse:.5f}s**"
    bot_print(response)


def get_model_response(text, context):
    s_model = time.perf_counter()
    prompt_msg = make_prompt(text, context)
    messages = [
    {"role": "system", "content": "你是阿米娅，你了解泰拉大地上的所有事情，并且了解每一位罗德岛干员。"},
    {"role": "user", "content": prompt_msg}
    ]
    text = st.session_state["tokenizer"].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = st.session_state["tokenizer"]([text], return_tensors="pt").to(st.session_state["params"]["device"])
    generated_ids = st.session_state["llm"].generate(
        model_inputs.input_ids,
        # max_new_tokens=st.session_state["params"]["max_length"],
        max_length=st.session_state["params"]["max_length"],
        top_p = st.session_state["params"]["max_length"],
        temperature = st.session_state["params"]["temperature"]
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = st.session_state["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
    elapse = time.perf_counter() - s_model

    if not response:
        response = "抱歉，我并不能正确回答该问题。"
    return response, elapse


def bot_print(content, avatar: str = "🤖"):
    with st.chat_message("assistant", avatar=avatar):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in content.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)


def tips(txt: str, wait_time: int = 2, icon: str = "🎉"):
    st.toast(txt, icon=icon)
    time.sleep(wait_time)


# 对话
def talk():
    title = config.get("title")
    version = config.get("version", "1.0.0")
    version = "Ver: " + version
    st.markdown(
        f"<h3 style='text-align: center;'>{title} {version}</h3><br/>",
        unsafe_allow_html=True,
    )

    # 初始化侧边参数
    init_ui_parameters()

    if 'llm' not in st.session_state or not st.session_state["llm"]:
        st.session_state["llm"] = init_llm(st.session_state["params"]["llm"])
    if 'tokenizer' not in st.session_state or not st.session_state["tokenizer"]:
        st.session_state["tokenizer"] = init_tokenizer(st.session_state["params"]["llm"])
    if st.session_state["params"]["select_knowledge_base"] != 'None':
        if 'db' not in st.session_state or not st.session_state['db']:
            db_path = os.path.join('PRTS/db_base', st.session_state["params"]["select_knowledge_base"])
            if 'embedding_extract' not in st.session_state or not st.session_state["embedding_extract"]:
                st.session_state["embedding_extract"] = init_encoder('models/Embeddings/text2vec-base-chinese')
            st.session_state['db'] = init_db(db_path, st.session_state["embedding_extract"])

    input_txt = st.chat_input("问点啥吧！")
    if input_txt:
        with st.chat_message("user", avatar="😀"):
            st.markdown(input_txt)

        if st.session_state["params"]["select_knowledge_base"] == 'None':
            predict(
                input_txt,
                None
            )
        else:
            # Embedding
            # query_embedding = st.session_state["embedding_extract"](input_txt)
            with st.spinner("正在搜索相关文档..."):
                similarDocs = st.session_state['db'].similarity_search(input_txt, k=1)
                # uid = st.session_state.get("connect_id", None)
                # search_res, search_elapse = db_tools.search_local(
                #     query_embedding, top_k=search_top, uid=uid
                # )

            predict(
                input_txt,
                similarDocs
            )


# 知识库
def knowledge_base():
    init_ui_db()


def main():
    sidebar_menu()
    main_content()


if __name__ == "__main__":
    main()