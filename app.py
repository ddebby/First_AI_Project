from utils import *
import streamlit as st

IMAGE_TYPES = ["bmp", "jpg","png"]

inf_learner = load_learner('export.pkl')
class_map = inf_learner.dls.vocab 


def main():
    """Run to execute main application"""
    st.title("🐕狗狗分类模型🐕")
    st.write("> 这个人工智能模型能够分辨: 【" + ', '.join(list(class_map))+'】')
    image = st.file_uploader("🔎请上传一张用于检测的图像:", IMAGE_TYPES)

    if image:
        st.image(image, use_column_width=True,width=450)
        img = PILImage.create(image)

        pred, _, prod = inf_learner.predict(img)
        prediction_string = f"**预测结果为**： {pred}， **置信度**： {prod.max()*100:.04f}%."

        st.markdown(prediction_string)
        st.balloons()

main()