import numpy as np
import pandas as pd
from time import time
import streamlit as st
import plotly.express as px
from graphviz import Digraph
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from variables import *
from utils import rescale_data, save_result, score, reset_cache, clear_cache
from models import PolynomialClassifierModel, DeepLearningModel, ARIMAModel, CombiningModel, RBFNNModel

######################################### Cấu hình streamlit #########################################
st.set_page_config(
    page_title="Dự báo",
    page_icon=":sparkles:",
)


######################################### Chọn tập dữ liệu #########################################
try:
    st.title('Dự Báo Dữ Liệu Bằng Các Mô Hình Học Máy')

    uploaded_file = st.file_uploader(
        "**Chọn tệp dữ liệu CSV:**", type=["csv"], on_change=reset_cache)
    if uploaded_file is not None:
        file_name = uploaded_file.name
        data.load_data(uploaded_file=uploaded_file)
        # Thông tin tập dữ liệu
        with st.expander(f"**Thông tin tập dữ liệu {file_name}**"):
            st.write(data.data)

    ######################################### Tiền xử lý dữ liệu #########################################
    if data.data is not None:
        with st.spinner('Đang tiền xử lý tập dữ liệu, xin vui lòng đợi.'):
            data.preprocess_data()

    ######################################### Vẽ biểu đồ tập dữ liệu #########################################
        st.write("---")
        st.subheader('Biểu đồ đường tập dữ liệu ' + file_name)

        column_names = data.data.columns.tolist()
        selected_predict_column_name = st.selectbox(
            "**Chọn cột dữ liệu để dự đoán:**", column_names)
        st.line_chart(data=data.data, y=selected_predict_column_name)
        data.target_data = data.data[selected_predict_column_name]

    ######################################### Chọn mô hình #########################################
        st.write("---")
        if st.session_state.model1_name is None:
            flag = False
            model1_name = None
        else:
            flag = True
        model1_name = st.selectbox(
            "**Chọn mô hình để dự đoán:**",
            ["Polynomial Classifier", "ARIMA", "RNN", "LSTM", "GRU", "RBFNN"],
            on_change=reset_cache, placeholder="Chọn mô hình", index=None)

        st.session_state.model1_name = model1_name
        if st.session_state.model1_name is not None:
            flag = True

    ######################################### Chọn mô hình để kết hợp #########################################
        if model1_name is not None and flag:
            combine = st.toggle("{}".format("**Kết Hợp Mô Hình**"))

        if combine:
            model2_name = st.selectbox(
                "**Chọn mô hình để kết hợp:**",
                ["Polynomial Classifier", "RNN", "LSTM", "GRU", "RBFNN"],
                placeholder="Chọn mô hình kết hợp",
                index=None
            )

            st.session_state.model2_name = model2_name

            # Chọn phương thức kết hợp
            if st.session_state.model2_name is not None:
                combine_strategy = st.radio(
                    "**Chọn phương thức kết hợp:**",
                    ["**Tuần tự Cộng**", "**Tuần tự Nhân**", "**Song song**"],
                    horizontal=True,
                    on_change=clear_cache).lstrip('*').rstrip('*')

    ######################################### Vẽ lưu đồ kết hợp #########################################
            if st.session_state.model2_name is not None:
                model1_name = st.session_state.model1_name
                model2_name = st.session_state.model2_name
                model1_label = model1_name + \
                    '_1' if model1_name == model2_name else model1_name
                model2_label = model2_name + \
                    '_2' if model1_name == model2_name else model2_name

                st.write('**Mô hình kết hợp:**')
                graph = Digraph(f'{model1_name} + {model2_name}')

                with graph.subgraph() as s1:
                    s1.attr(rank='same')
                    s1.node(file_name, label=file_name)

                with graph.subgraph() as s2:
                    s2.attr(rank='same')
                    s2.node(model1_label, label=model1_label)
                    s2.node(model2_label, label=model2_label)

                with graph.subgraph() as s3:
                    s3.attr(rank='same')
                    s3.node('Output', label='Output')

                if combine_strategy != 'Song song':
                    graph.node('i', shape='point')
                    graph.edge(file_name, model1_label, label='Y')
                    graph.edge(
                        model1_label, model2_label, label='Y - L' if combine_strategy == 'Tuần tự Cộng' else 'Y/L')
                    graph.edge(model1_label, 'i', dir='none', label='L')
                    graph.edge(model2_label, 'i', dir='none', label='N')
                    graph.edge('i', 'Output', label='L + N' if combine_strategy ==
                            'Tuần tự Cộng' else 'L*N')
                else:
                    graph.node('i', shape='point')
                    graph.edge(file_name, model1_label, label="y")
                    graph.edge(file_name, model2_label, label="y")
                    graph.edge(model1_label, 'i', label="y'")
                    graph.edge(model2_label, 'i', label="y''")
                    graph.edge('i', 'Output', label="w*y'' + (1-w)*y'")

                graph.attr(rankdir='LR')
                st.graphviz_chart(graph)

        st.write("---")
        if model1_name is None or not flag:
            st.subheader("Vui lòng chọn mô hình để dự đoán")
        else:
            st.subheader("Chọn tham số cho mô hình " + model1_name +
                        (f" và {model2_name}" if model2_name is not None and model2_name != model1_name else ""))

    ######################################### Chọn tham số #########################################
            # Thông tin cho mô hình ARIMA
            if model1_name == "ARIMA":
                with st.expander("**Thông tin chi tiết về tập dữ liệu**"):
                    st.subheader("Các tính chất của dữ liệu")
                    st.pyplot(seasonal_decompose(
                        data.target_data[-300:], model='multiplicative', period=12).plot())
                    st.subheader("ACF và PACF của dữ liệu")
                    col_acf, col_pacf = st.columns(2)
                    with col_acf:
                        st.pyplot(plot_acf(data.target_data))
                    with col_pacf:
                        st.pyplot(plot_pacf(data.target_data))
                    st.subheader("Tính dừng của dữ liệu")
                    p_value = adfuller(data.target_data)
                    st.write(f"**P-value:  {p_value[1]}**")
                    st.write(
                        "**Dữ liệu có tính dừng**" if p_value[1] < 0.05 else "**Dữ liệu không có tính dừng**")
                    if p_value[1] > 0.05:
                        d_order = ndiffs(data.target_data, test='adf')
                        st.write(
                            f"**Số bậc sai phân khuyên dùng để dữ liệu có tính dừng: {d_order}**")
                        st.subheader(
                            f"ACF và PACF của dữ liệu sau khi sai phân bậc {d_order}")
                        col_acf_diff, col_pacf_diff = st.columns(2)
                        diff_data = data.target_data
                        for i in range(d_order):
                            diff_data = diff_data.diff().dropna()
                        with col_acf_diff:
                            st.pyplot(plot_acf(diff_data))
                        with col_pacf_diff:
                            st.pyplot(plot_pacf(diff_data))

            col11, col12 = st.columns(2)
            col111, col112, col113 = st.columns(3)
            if model1_name == "ARIMA":
                is_auto = st.checkbox("Tự Động Chọn Tham Số Tốt Nhất")
            col21, col22 = st.columns(2)
            col211, col212, col213 = st.columns(3)
            col31, col32 = st.columns(2)

            # Chọn tỉ lệ chia tập train/test/validation
            with col21:
                if (model1_name == "ARIMA" and model2_name == "Polynomial Classifier") \
                    or (model1_name == "Polynomial Classifier" and model2_name == "Polynomial Classifier") \
                        or (model2_name is None and (model1_name == "ARIMA" or model1_name == "Polynomial Classifier")):
                    train_size = st.number_input(
                        "**Chia tỉ lệ training %:**", min_value=1, max_value=99, value=80, step=1, on_change=clear_cache)
                    test_size = 100 - train_size
                    st.write("**Tỷ lệ testing: **" + str(test_size) + "**%**")
                else:
                    col_train, col_validate = st.columns(2)
                    with col_train:
                        train_size = st.number_input(
                            "**Chia tỉ lệ training %:**", min_value=1, max_value=99, value=80, step=1, on_change=clear_cache)
                        test_size = 100 - train_size
                        st.write("**Tỷ lệ testing: **" + str(test_size) + "**%**")
                    with col_validate:
                        validate_size = st.number_input(
                            "**Chia tỉ lệ validating %:**", min_value=1, max_value=train_size-1, value=20, step=1, on_change=clear_cache)
            if model2_name is None:
                # Chọn ngày để dự đoán
                if model1_name != "ARIMA":
                    with col11:
                        d = st.number_input("**Số ngày dùng để dự đoán:**",
                                            value=2, step=1, min_value=1, on_change=clear_cache)

                    with col12:
                        t = st.number_input("**Số ngày muốn dự đoán:**", value=1,
                                            step=1, min_value=1, on_change=clear_cache)

                    # Chọn bậc của đa thức và các tham số khác
                    with col22:
                        if model1_name == "Polynomial Classifier":
                            degree = st.number_input(
                                "**Số bậc của đa thức:**", value=2, step=1, min_value=1, on_change=clear_cache)
                        else:
                            unit = st.number_input(
                                "**Số node cho hidden layer:**", value=60, step=1, min_value=1, on_change=clear_cache)

                    with col31:
                        epoch = st.number_input(
                            "**Số lượng epoch cho mỗi lần train:**", value=100, step=1, min_value=1, on_change=clear_cache)
                    with col32:
                        batch_size = st.number_input(
                            "**Số lượng batch_size cho mỗi lần epoch:**", value=64, step=1, min_value=1, on_change=clear_cache)
                else:
                    with col111:
                        p_arima = st.number_input("**Số bậc tự hồi qui (AR):**",
                                                value=0, step=1, min_value=0, on_change=clear_cache, disabled=is_auto)
                    with col112:
                        d_arima = st.number_input("**Số bậc sai phân (I):**",
                                                value=0, step=1, min_value=0, on_change=clear_cache, disabled=is_auto)
                    with col113:
                        q_arima = st.number_input("**Số bậc trung bình trượt (MA):**",
                                                value=0, step=1, min_value=0, on_change=clear_cache, disabled=is_auto)
            else:
                if model1_name == "ARIMA":
                    with col111:
                        p_arima = st.number_input("**Số bậc tự hồi qui (AR):**",
                                                value=0, step=1, min_value=0, on_change=clear_cache, disabled=is_auto)
                    with col112:
                        d_arima = st.number_input("**Số bậc sai phân (I):**",
                                                value=0, step=1, min_value=0, on_change=clear_cache, disabled=is_auto)
                    with col113:
                        q_arima = st.number_input("**Số bậc trung bình trượt (MA):**",
                                                value=0, step=1, min_value=0, on_change=clear_cache, disabled=is_auto)

                    with col31:
                        epoch = st.number_input(
                            "**Số lượng epoch cho mỗi lần train:**", value=100, step=1, min_value=1, on_change=clear_cache)
                    with col32:
                        batch_size = st.number_input(
                            "**Số lượng batch_size cho mỗi lần epoch:**", value=64, step=1, min_value=1, on_change=clear_cache)

                    if model2_name != "Polynomial Classifier":
                        with col22:
                            unit = st.number_input(
                                "**Số node cho hidden layer:**", value=60, step=1, min_value=1, on_change=clear_cache)
                    else:
                        with col22:
                            degree = st.number_input(
                                "**Số bậc của đa thức:**", value=2, step=1, min_value=1, on_change=clear_cache)
                else:
                    with col11:
                        d = st.number_input("**Số ngày dùng để dự đoán:**",
                                            value=2, step=1, min_value=1, on_change=clear_cache)

                    with col12:
                        t = st.number_input("**Số ngày muốn dự đoán:**", value=1,
                                            step=1, min_value=1, on_change=clear_cache)

                    if model1_name == "Polynomial Classifier":
                        with col22:
                            degree = st.number_input(
                                "**Số bậc của đa thức:**", value=2, step=1, min_value=1, on_change=clear_cache)

                        if model2_name == "Polynomial Classifier":
                            with col31:
                                epoch = st.number_input(
                                    "**Số lượng epoch cho mỗi lần train:**", value=100, step=1, min_value=1, on_change=clear_cache)
                            with col32:
                                batch_size = st.number_input(
                                    "**Số lượng batch_size cho mỗi lần epoch:**", value=64, step=1, min_value=1, on_change=clear_cache)

                        else:
                            with col211:
                                unit = st.number_input(
                                    "**Số node cho hidden layer:**", value=60, step=1, min_value=1, on_change=clear_cache)
                            with col212:
                                epoch = st.number_input(
                                    "**Số lượng epoch cho mỗi lần train:**", value=100, step=1, min_value=1, on_change=clear_cache)
                            with col213:
                                batch_size = st.number_input(
                                    "**Số lượng batch_size cho mỗi epcho:**", value=64, step=1, min_value=1, on_change=clear_cache)

                    else:
                        if model2_name == "Polynomial Classifier":
                            with col22:
                                degree = st.number_input(
                                    "**Số bậc của đa thức:**", value=2, step=1, min_value=1, on_change=clear_cache)
                            with col211:
                                unit = st.number_input(
                                    "**Số node cho hidden layer:**", value=60, step=1, min_value=1, on_change=clear_cache)
                            with col212:
                                epoch = st.number_input(
                                    "**Số lượng epoch cho mỗi lần train:**", value=100, step=1, min_value=1, on_change=clear_cache)
                            with col213:
                                batch_size = st.number_input(
                                    "**Số lượng batch_size cho mỗi epcho:**", value=64, step=1, min_value=1, on_change=clear_cache)
                        else:
                            with col22:
                                unit = st.number_input(
                                    "**Số node cho hidden layer:**", value=60, step=1, min_value=1, on_change=clear_cache)
                            with col31:
                                epoch = st.number_input(
                                    "**Số lượng epoch cho mỗi lần train:**", value=100, step=1, min_value=1, on_change=clear_cache)
                            with col32:
                                batch_size = st.number_input(
                                    "**Số lượng batch_size cho mỗi lần epoch:**", value=64, step=1, min_value=1, on_change=clear_cache)

            if st.session_state.data is None:
                st.session_state.data = data
            if st.session_state.model1_name is None:
                st.session_state.model1_name = model1_name

    ######################################### Chuẩn hóa dữ liệu #########################################
            st.write("---")
            st.subheader("Chuẩn Hóa Dữ Liệu")
            standardize_strategy = st.radio(
                "**Chọn phương thức chuẩn hóa:**",
                ["**Không chuẩn hóa**", "**Z-score**", "**Min-max**"], horizontal=True, on_change=clear_cache).lstrip("*").rstrip("*")

            col_min, col_max = st.columns(2)

            if standardize_strategy == "Min-max":
                with col_min:
                    min_scale = st.number_input(
                        "**Min:**", value=0)
                with col_max:
                    max_scale = st.number_input(
                        "**Max:**", value=1)

            if st.button("Chuẩn Hóa Dữ Liệu", type="primary", on_click=clear_cache, disabled=standardize_strategy == "Không chuẩn hóa"):
                try:
                    data.standardize_data(standardize_strategy,
                                        min_scale, max_scale)
                except ValueError:
                    st.error(
                        "**Không thể chuẩn hóa, giá trị Min phải nhỏ hơn giá trị Max. Giá trị nhận được:**", (min_scale, max_scale))
                st.session_state.data = data

    ######################################### In dữ liệu trước và sau chuẩn hóa #########################################
            col41, col42 = st.columns(2)
            if st.session_state.data.scaled_data is not None:
                data = st.session_state.data
            if standardize_strategy != "Không chuẩn hóa":
                with col41:
                    st.write("**Dữ liệu trước khi chuẩn hóa:**")
                    st.write(data.target_data[:5])
                if data.scaled_data is not None:
                    with col42:
                        st.write("**Dữ liệu sau khi chuẩn hóa:**")
                        st.write(data.scaled_data[:5])
            else:
                data.scaled_data = None

    ######################################### Chia tập training/testing/validating #########################################
            data.split_data(
                test_size, validate_size if validate_size is not None else None)

            if validate_size is not None:
                st.write("**Kích thước tập train:**", data.train_data.size,
                        "**Kích thước tập test:**", data.test_data.size,
                        "**Kích thước tập validate:**", data.valid_data.size)
            else:
                data.valid_data = None
                st.write("**Kích thước tập train:**", data.train_data.size,
                        "**Kích thước tập test:**", data.test_data.size)

    ######################################### Khởi tạo mô hình #########################################
            st.session_state.data = data
            data = st.session_state.data
            if model2_name is None:
                if model1_name == "Polynomial Classifier":
                    model = PolynomialClassifierModel(
                        data.train_data, data.test_data, d, t, degree, epoch, batch_size)
                elif model1_name == "ARIMA":
                    model = ARIMAModel(
                        data.train_data, data.test_data, p_arima, d_arima, q_arima, is_auto)
                elif model1_name == "RBFNN":
                    model = RBFNNModel(data.train_data, data.test_data,
                                    data.valid_data, d, t, unit, epoch, batch_size)
                else:
                    model = DeepLearningModel(
                        model1_name, data.train_data, data.test_data, data.valid_data, d, t, unit, epoch, batch_size)
            else:
                model = CombiningModel(model1_name, model2_name, combine_strategy,
                                    data.train_data, data.test_data, data.valid_data, validate_size,
                                    d if model1_name != "ARIMA" else p_arima,
                                    t if model1_name != "ARIMA" else 1, degree, unit, epoch, batch_size,
                                    p_arima, d_arima, q_arima, is_auto)

    ######################################### Tìm siêu tham số  #########################################
            if model2_name is None:
                if model1_name != "ARIMA" and model1_name != "Polynomial Classifier":
                    st.write("---")
                    if st.button("Tìm Siêu Tham Số Cho Mô Hình {} {}"
                                .format(model1_name, "Chưa Chuẩn Hóa" if data.scaled_data is None else "Đã Chuẩn Hóa"), type="primary"):
                        with st.spinner("**Đang tìm tham số tối ưu, xin vui lòng đợi.**"):
                            result = model.hyperparameter()
                            trials_df = []
                            for res in result:
                                params = res['params']
                                target = res['target']
                                row = {
                                    "validation_loss": -target,
                                    "batch_size": int(params["batch_size"]),
                                    "epoch": int(params["epoch"]),
                                    "learning_rate": params["learning_rate"],
                                    "unit": int(params["unit"])
                                }
                                trials_df.append(row)

                            trials_df = pd.DataFrame(trials_df)
                            # Find row with lowest validation loss
                            best_trial_index = trials_df['validation_loss'].idxmin(
                            )

                            # Highlight row with lowest validation loss in green
                            highlighted_df = trials_df.style.apply(lambda x: [
                                'background: lightgreen' if x.name == best_trial_index else '' for _ in x], axis=1)

                            with st.expander(f"**Tham số tốt nhất của mô hình {model1_name}**"):
                                st.table(highlighted_df)
            else:
                pass

    ######################################### Train mô hình #########################################
            st.write("---")
            if st.button("Train Mô Hình {} {}".format("Kết Hợp" if model2_name is not None else '',
                                                    "Chưa Chuẩn Hóa" if data.scaled_data is None else "Đã Chuẩn Hóa"), type="primary"):
                with st.spinner("**Mô hình đang được train, xin vui lòng đợi.**"):
                    start_time = time()
                    model.train()
                    train_time = float("{:.2f}".format(time() - start_time))

                    st.success("**Train hoàn thành**")
                    st.session_state.model = model
                    st.session_state.train_time = train_time

    ######################################### In thông số của mô hình #########################################
            st.write("---")
            model = st.session_state.model

            if model is not None:
                try:
                    auto = "auto " if is_auto else ""
                    if model2_name is None:
                        if model1_name == "Polynomial Classifier" and model.model is not None:
                            st.write(
                                "**Ma trận trọng số của mô hình Polynomial Classifier:**")
                            vmatrix_latex = "\omega = \\begin{vmatrix} "
                            for i in range(len(model.model)):
                                vmatrix_latex += " & ".join(["{:.4f}".format(x)
                                                            for x in model.model[i]]) + " \\\\ "
                            vmatrix_latex += " \\end{vmatrix}"
                            st.latex(vmatrix_latex)
                        elif model1_name == "ARIMA":
                            with st.expander("**Thông số của mô hình ARIMA**"):
                                if model.trace_df is not None:
                                    min_aic_index = model.trace_df['AIC'].idxmin()
                                    min_aic_style = {
                                        'selector': f'tr:nth-child({min_aic_index + 1})',
                                        'props': [('background-color', '#66ff66')]
                                    }
                                    st.table(model.trace_df.style.set_table_styles(
                                        [min_aic_style]))
                                st.text(model.model.fit(
                                    model.train_set).summary().as_text())
                        else:
                            with st.expander(f"**Thông số của mô hình {model1_name}:**"):
                                model.model.summary(print_fn=st.text)

                    else:
                        if model1_name == "ARIMA":
                            with st.expander("**Thông số của mô hình ARIMA**"):
                                if model.model1.trace_df is not None:
                                    min_aic_index = model.model1.trace_df['AIC'].idxmin(
                                    )
                                    min_aic_style = {
                                        'selector': f'tr:nth-child({min_aic_index + 1})',
                                        'props': [('background-color', '#66ff66')]
                                    }
                                    st.table(model.model1.trace_df.style.set_table_styles(
                                        [min_aic_style]))
                                st.text(model.model1.model.fit(
                                    model.train_set).summary().as_text())

                            if model2_name == "Polynomial Classifier":
                                st.write(
                                    "**Ma trận trọng số của mô hình Polynomial Classifier:**")
                                vmatrix_latex = "\omega = \\begin{vmatrix} "
                                for i in range(len(model.model2.model)):
                                    vmatrix_latex += " & ".join(["{:.4f}".format(x)
                                                                for x in model.model2.model[i]]) + " \\\\ "
                                vmatrix_latex += " \\end{vmatrix}"
                                st.latex(vmatrix_latex)
                            else:
                                with st.expander(f"**Thông số của mô hình {model2_name}:**"):
                                    model.model2.model.summary(
                                        print_fn=st.text)

                        elif model1_name == "Polynomial Classifier":
                            pc1 = "Polynomial Classifier_1" if model2_name == "Polynomial Classifier" else "Polynomial Classifier"

                            st.write(
                                f"**Ma trận trọng số của mô hình {pc1}:**")
                            vmatrix_latex = "\omega = \\begin{vmatrix} "
                            for i in range(len(model.model1.model)):
                                vmatrix_latex += " & ".join(["{:.4f}".format(x)
                                                            for x in model.model1.model[i]]) + " \\\\ "
                            vmatrix_latex += " \\end{vmatrix}"
                            st.latex(vmatrix_latex)

                            if model2_name == "Polynomial Classifier":
                                st.write(
                                    "**Ma trận trọng số của mô hình Polynomial Classifier_2:**")
                                vmatrix_latex = "\omega = \\begin{vmatrix} "
                                for i in range(len(model.model2.model)):
                                    vmatrix_latex += " & ".join(["{:.4f}".format(x)
                                                                for x in model.model2.model[i]]) + " \\\\ "
                                vmatrix_latex += " \\end{vmatrix}"
                                st.latex(vmatrix_latex)
                            else:
                                with st.expander(f"**Thông số của mô hình {model2_name}:**"):
                                    model.model2.model.summary(
                                        print_fn=st.text)

                        else:
                            model1_label = model1_name + \
                                '_1' if model1_name == model2_name else model1_name
                            model2_label = model2_name + \
                                '_2' if model1_name == model2_name else model2_name

                            with st.expander(f"**Thông số của mô hình {model1_label}:**"):
                                model.model1.model.summary(print_fn=st.text)

                            if model2_name == "Polynomial Classifier":
                                st.write(
                                    "**Ma trận trọng số của mô hình Polynomial Classifier:**")
                                vmatrix_latex = "\omega = \\begin{vmatrix} "
                                for i in range(len(model.model2.model)):
                                    vmatrix_latex += " & ".join(["{:.4f}".format(x)
                                                                for x in model.model2.model[i]]) + " \\\\ "
                                vmatrix_latex += " \\end{vmatrix}"
                                st.latex(vmatrix_latex)

                            else:
                                with st.expander(f"**Thông số của mô hình {model2_label}:**"):
                                    model.model2.model.summary(
                                        print_fn=st.text)

                        if model.w_parallel is not None:
                            st.write("**Trọng số cho mô hình song song:**")
                            st.latex(f"\omega = {model.w_parallel}")
                except:
                    pass

    ######################################### Test mô hình #########################################
                test_flag = False
                if hasattr(model, "model"):
                    if model.model is not None:
                        test_flag = True
                if hasattr(model, "model2"):
                    if model.model2 is not None:
                        test_flag = True
                if st.button("Test Mô Hình", type="primary", disabled=not test_flag):
                    with st.spinner("**Đang test mô hình, xin vui lòng đợi.**"):
                        try:
                            start_time = time()

                            predict, actual = model.test()

                            test_time = float(
                                "{:.2f}".format(time() - start_time))

                            st.session_state.test_time = test_time
                            st.session_state.predict = predict
                            st.session_state.actual = actual
                        except (ValueError, AttributeError, TypeError) as error:
                            st.write(error)

    ######################################### In kết quả dự báo #########################################
            st.write("---")
            predict = st.session_state.predict
            actual = st.session_state.actual
            if predict is not None:
                result_table = pd.DataFrame(
                    {"Dự đoán": predict.flatten(), "Thực tế": actual.flatten()})

                st.session_state.result_table = result_table
                if data.scaled_data is not None:
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)

                    result_table = pd.DataFrame(
                        {"Dự đoán": predict_inv.flatten(), "Thực tế": actual_inv.flatten()})

                    st.session_state.result_table = result_table

                st.table(result_table[:5])

    ####################################### Tính lỗi và thời gian thực thi #########################################
                st.subheader("Các thông số đánh giá")

                if data.scaled_data is not None:
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                else:
                    metrics = score(predict, actual)
                metrics.update({"Thời gian train": st.session_state.train_time,
                                "Thời gian test": st.session_state.test_time})
                metric_table = pd.DataFrame(metrics, index=[
                                            f"{model1_name} {combine_strategy} {model2_name}" if model2_name is not None else f"{model1_name}"])
                st.session_state.metric_table = metric_table

                st.table(metric_table)

    ######################################### Vẽ biểu đồ dự báo và thực tế #########################################
                if t == None:
                    t = 1
                    d = 0
                if t == 1:
                    with st.spinner("**Đang vẽ biểu đồ so sánh, xin vui lòng đợi.**"):
                        st.subheader("Biểu đồ so sánh")

                        index_column = np.array(
                            data.target_data[data.train_data.size +
                                            (data.valid_data.size if data.valid_data is not None else 0) +
                                            d:].index).flatten()

                        if hasattr(model, "model2"):
                            # if model.model1_name != "ARIMA":
                            index_column = index_column[len(
                                model.test_set) - len(predict) - d:]

                        predict = pd.DataFrame({"Dự đoán": predict.flatten()})
                        predict = predict.set_index(index_column)
                        actual = pd.DataFrame({"Thực tế": actual.flatten()})
                        actual = actual.set_index(index_column)

                        df_result = pd.concat([predict, actual], axis=1)

                        fig1 = px.line(df_result, x=df_result.index, y=df_result.columns,
                                    color_discrete_sequence=["red", "blue"])
                        fig1.update_traces(
                            patch={"line": {"width": 1, "dash": "dot"}})
                        fig1.update_layout(xaxis_title="Date")

                        st.plotly_chart(fig1)

                        # Rescale data và vẽ lại biểu đồ
                        if predict is not None and data.scaled_data is not None:
                            predict_inv = rescale_data(data.target_data,
                                                    predict, standardize_strategy)
                            actual_inv = rescale_data(
                                data.target_data, actual, standardize_strategy)

                            predict_inv = pd.DataFrame(
                                {"Dự đoán": predict_inv.flatten()})
                            predict_inv = predict_inv.set_index(index_column)
                            actual_inv = pd.DataFrame(
                                {"Thực tế": actual_inv.flatten()})
                            actual_inv = actual_inv.set_index(index_column)

                            df_rescale = pd.concat(
                                [predict_inv, actual_inv], axis=1)

                            if st.button("Rescale Mô Hình", type="primary"):
                                fig2 = px.line(df_rescale, x=df_rescale.index,
                                            y=df_rescale.columns, color_discrete_sequence=["red", "blue"])
                                fig2.update_traces(
                                    patch={"line": {"width": 1, "dash": "dot"}})
                                fig2.update_layout(xaxis_title="Date")

                                st.plotly_chart(fig2)

        ######################################### Lưu kết quả #########################################
                    st.write("---")
                    st.subheader("Lưu kết quả")
                    common_params = ["dataset_name", "column_name", "model", "d", "t",
                                    "split", "train_size", "test_size", "standardize"]
                    common_values = [file_name, selected_predict_column_name, model1_name, d, t,
                                    f"{100-test_size}/{test_size}", data.train_data.size, data.test_data.size,
                                    "Chưa chuẩn hóa" if data.scaled_data is None
                                    else (standardize_strategy if standardize_strategy != "Min-max"
                                        else f"Min-max[{min_scale}, {max_scale}]")]
                    common_meanings = ["Tên tập dữ liệu", "Tên cột dự đoán", "Tên mô hình", "Số ngày dùng để dự đoán", "Số ngày để dự đoán",
                                    "Tỉ lệ tập train/test", "Kích thước tập train", "Kích thước tập test", "Phương thức chuẩn hóa"]

                    if model1_name == "Polynomial Classifier":
                        extra_params = ["degree", "epoch", "batch_size"]
                        extra_values = [degree, epoch, batch_size]
                        extra_meanings = [
                            "Số bậc đa thức", "Số lượng epoch cho mỗi lần train", "Số lượng batch_size cho mỗi lần epoch"]
                    elif model1_name != "ARIMA":
                        extra_params = ["unit", "epoch", "batch_size"]
                        extra_values = [unit, epoch, batch_size]
                        extra_meanings = ["Số node cho hidden layer",
                                        "Số lượng epoch cho mỗi lần train", "Số lượng batch_size cho mỗi lần epoch"]
                    else:
                        common_values[3] = model.p_arima
                        common_values[4] = 1
                        extra_params = ["p_arima", "d_arima", "q_arima"]
                        extra_values = [model.p_arima,
                                        model.d_arima, model.q_arima]
                        extra_meanings = ["Số bậc tự hồi quy",
                                        "Số sai phân", "Số bậc trung bình trượt"]

                    if model2_name is not None:
                        common_values[2] = f"{model1_name} {combine_strategy} {model2_name}"
                        if model1_name != model2_name:
                            extra_params2 = []
                            extra_values2 = []
                            extra_meanings2 = []
                            if model1_name == "Polynomial Classifier":
                                extra_params2 = ["unit"]
                                extra_values2 = [unit]
                                extra_meanings2 = ["Số node cho hidden layer"]

                            if model1_name == "ARIMA":
                                if model2_name == "Polynomial Classifier":
                                    extra_params2 = [
                                        "degree", "epoch", "batch_size"]
                                    extra_values2 = [degree, epoch, batch_size]
                                    extra_meanings2 = ["Số bậc đa thức",
                                                    "Số lượng epoch cho mỗi lần train", "Số lượng batch_size cho mỗi lần epoch"]
                                else:
                                    extra_params2 = ["unit", "epoch", "batch_size"]
                                    extra_values2 = [unit, epoch, batch_size]
                                    extra_meanings2 = ["Số node cho hidden layer",
                                                    "Số lượng epoch cho mỗii lần train", "Số lượng batch_size cho mỗi lần epoch"]

                            else:
                                if model2_name == "Polynomial Classifier":
                                    extra_params2 = ["degree"]
                                    extra_values2 = [degree]
                                    extra_meanings2 = ["Số bậc đa thức"]

                            extra_params = extra_params + extra_params2
                            extra_values = extra_values + extra_values2
                            extra_meanings = extra_meanings + extra_meanings2

                    params = common_params + extra_params
                    values = common_values + extra_values
                    meanings = common_meanings + extra_meanings

                    info_table = pd.DataFrame({
                        "Tham số": params,
                        "Giá trị": values,
                        "Ý nghĩa": meanings
                    })

                    st.session_state.info_table = info_table
                    if result_table is not None:
                        sheet_name = st.text_input("**Nhập tên sheet**", "Sheet1")
                        st.write(
                            "**Để tránh xung đột, vui lòng đóng các cửa sổ Excel trước khi lưu.**")
                        if st.button("Lưu Kết Quả Vào report/output", type="primary"):
                            with st.spinner("**Kết quả đang được lưu, xin vui lòng đợi.**"):
                                save_result(st.session_state.result_table, st.session_state.metric_table,
                                            st.session_state.info_table, sheet_name)

except:
    pass
