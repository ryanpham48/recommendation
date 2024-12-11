import streamlit as st
import pandas as pd
import pickle
st.markdown("""
    <style>
    body {
        background-color: #F7F8FA;
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 10px;
    }
    .css-1aumxhk {  /* Sidebar */
        background-color: #E8F5E9;
        font-family: 'Georgia', serif;
    }
    h1, h2, h3 {
        color: #4CAF50;
        font-weight: bold;
    }
    .stDownloadButton {
        font-size: 18px;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
    }
    .st-bar-chart {
        background-color: #E3F2FD;
    }
    .stButton > button {
        font-size: 16px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)
# GUI
st.markdown(
    "<h1 style='text-align: center;'>DATA SCIENCE PROJECT<br>Hệ Thống Gợi Ý Sản Phẩm</h1>",
    unsafe_allow_html=True,
)
menu = ["Tổng Quan", "Thực Hiện & Đánh Giá Model", "Gợi ý theo thông tin khách hàng", "Gợi ý sản theo thông tin sản phẩm"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Lý Quốc Hồng Phúc & Phạm Anh Vũ""")
image_width = 400
st.sidebar.image('phucly.png')
st.sidebar.image('vupham.jpg', width=image_width)
st.sidebar.write("#### Giảng viên hướng dẫn: Cô Khuất Thùy Phương")
st.sidebar.image('khuat_thuy_phuong.jpg')
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")

if choice == 'Tổng Quan':
    # Giao diện Streamlit
    st.title("Yêu Cầu Thực Hiện")
    st.image('hasaki_banner.jpg', use_container_width=True)
    # Nội dung phát biểu bài toán
    st.write("""
    **Công ty Hasaki mong muốn xây dựng một hệ thống đề xuất sản phẩm nhằm cá nhân hóa trải nghiệm người dùng, giúp khách hàng dễ dàng tìm kiếm và lựa chọn sản phẩm phù hợp với sở thích và nhu cầu của họ. Hệ thống này sẽ phân tích dữ liệu về sản phẩm và hành vi của người dùng để đưa ra các gợi ý hiệu quả, tăng cường sự hài lòng của khách hàng và thúc đẩy doanh số bán hàng.**

    **Cụ thể, bài toán đặt ra là:**
    1. Với khách hàng đã có lịch sử mua sắm hoặc tương tác: hệ thống cần dựa trên thông tin mua sắm và nội dung đánh giá của những người dùng khác có sở thích tương tự để đưa ra gợi ý chính xác hơn.
    2. Với khách hàng mới (chưa có nhiều tương tác với hệ thống), hệ thống cần sử dụng thông tin về sản phẩm để đề xuất các sản phẩm tương tự.
    """)

elif choice == 'Thực Hiện & Đánh Giá Model':
    # Giao diện Streamlit
    st.title("Model Evaluation")
    st.image('Hasaki.logo.wide.jpg', use_container_width=True)
    # Nội dung phương pháp giải quyết bài toán
    st.markdown("""
    ##### Để giải quyết nhu cầu của Hasaki, hệ thống sẽ sử dụng kết hợp hai phương pháp: **Content-based Filtering** và **Collaborative Filtering** 
    I. Collaborative Filtering:  
    > * Nguyên lý: Dựa vào hành vi người dùng (lịch sử mua sắm và nội dung đánh giá), tìm kiếm các khách hàng có hành vi mua sắm hoặc đánh giá tương tự để tìm ra những mối liên hệ tiềm ẩn giữa khách hàng và sản phẩm mà Content-Based Filtering không thể, để đề xuất sản phẩm phù hợp cho người dùng.  
    > * Thuật toán: Sử dụng các mô hình từ thư viện Surprise để dự đoán điểm đánh giá sản phẩm cho người dùng. Đánh giá các thuật toán:""")
    st.image('RMSE&MAE.png', use_container_width=True)
    st.image('ALS.png', use_container_width=True)
    st.write(""" Lựa chọn sử dụng thuật toán ALS. Số lượng sản phẩm đề xuất (chọn 5 sản phẩm). Dataframe của 5 sản phẩm được đề xuất có điểm đánh giá cao nhất ứng với mã khách hàng được nhập.""")
    st.markdown("""
    II.  Content-based Filtering:  
    > * Nguyên lý: Phân tích thông tin về sản phẩm (như thành phần, công dụng, loại da phù hợp, giá cả, v.v.) để tìm các sản phẩm tương tự dựa trên đặc trưng của chúng.  
    > * Thuật toán: Sử dụng thuật toán Cosine Similarity """)
    st.image('gensimresults.png', use_container_width=True)
    st.image('gensim_vs_consine.png', use_container_width=True)


elif choice == 'Gợi ý theo thông tin khách hàng':    
    # Hàm để kiểm tra khách hàng và đề xuất sản phẩm
    def recommend_products_for_customer(ma_khach_hang, data_sub_pandas, products_sub_pandas, best_algorithm):
        # Kiểm tra nếu khách hàng đã đánh giá sản phẩm
        df_select = data_sub_pandas[(data_sub_pandas['ma_khach_hang'] == ma_khach_hang) & (data_sub_pandas['so_sao'] >= 3)]

        if df_select.empty:
            return pd.DataFrame(), "Khách hàng không có sản phẩm đã đánh giá >= 3."

        # Dự đoán điểm cho các sản phẩm chưa đánh giá
        df_score = pd.DataFrame(data_sub_pandas['ma_san_pham'].unique(), columns=['ma_san_pham'])
        df_score['EstimateScore'] = df_score['ma_san_pham'].apply(
            lambda x: best_algorithm.predict(ma_khach_hang, x).est
        )

        # Lấy top 5 sản phẩm dựa trên EstimateScore
        top_5_df = df_score.sort_values(by=['EstimateScore'], ascending=False).head(5)
        top_5_df['ma_khach_hang'] = ma_khach_hang

        # Kết hợp với thông tin sản phẩm từ products_sub_pandas
        enriched_top_5_df = pd.merge(
            top_5_df,
            products_sub_pandas,
            on='ma_san_pham',
            how='left'
        )
        return enriched_top_5_df, None
    # Hiển thị đề xuất ra bảng
    def display_recommended_products_1(recommend_products_for_customer, cols=5):
        for i in range(0, len(recommend_products_for_customer), cols):
            cols = st.columns(cols)
            for j, col in enumerate(cols):
                if i + j < len(recommend_products_for_customer):
                    product = recommend_products_for_customer.iloc[i + j]
                    with col:   
                        st.write(product['ten_san_pham'])                    
                        expander = st.expander(f"Mô tả")
                        product_description = product['mo_ta']
                        truncated_description = ' '.join(product_description.split()[:100]) + '...'
                        expander.write(truncated_description)
                        expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")
                        
    # Đọc dữ liệu khách hàng, sản phẩm, và đánh giá
    customers = pd.read_csv('Khach_hang.csv')
    products = pd.read_csv('San_pham.csv')
    reviews = pd.read_csv('Danh_gia_new.csv')

    # Giao diện Streamlit
    st.title("Hệ thống gợi ý sản phẩm theo thông tin khách hàng")

    st.image('hasaki_banner.jpg', use_container_width=True)

    # Nhập thông tin khách hàng
    #ho_ten_input = st.text_input("Nhập họ và tên khách hàng:")
    #ma_khach_hang_input = st.text_input("Nhập mã khách hàng:")
    # Tăng kích thước chữ cho nhãn "Nhập họ và tên khách hàng"
    st.markdown('<p style="font-size:30px; font-weight:bold;">Nhập họ và tên khách hàng:</p>', unsafe_allow_html=True)
    ho_ten_input = st.text_input("ho_ten_input", key="ho_ten_input", label_visibility="hidden")

    # Tăng kích thước chữ cho nhãn "Nhập mã khách hàng"
    st.markdown('<p style="font-size:30px; font-weight:bold;">Nhập mã khách hàng:</p>', unsafe_allow_html=True)
    ma_khach_hang_input = st.text_input("ma_khach_hang_input", key="ma_khach_hang_input", label_visibility="hidden")

    if ho_ten_input and ma_khach_hang_input:
        try:
            ma_khach_hang_input = int(ma_khach_hang_input)  # Chuyển mã khách hàng thành số nguyên
        except ValueError:
            st.error("Mã khách hàng phải là một số nguyên.")
        else:
            # Kiểm tra thông tin khách hàng
            customer_match = customers[
                (customers['ho_ten'].str.contains(ho_ten_input, case=False, na=False)) &
                (customers['ma_khach_hang'] == ma_khach_hang_input)
            ]

            if not customer_match.empty:
                st.success(f"Thông tin khách hàng hợp lệ: {ho_ten_input} (Mã: {ma_khach_hang_input})")

                # Đọc model được lưu trữ trong file best_algorithm.pkl
                with open('best_algorithm.pkl', 'rb') as f:
                    best_algorithm_new = pickle.load(f)

                # Gợi ý sản phẩm
                recommendations, error = recommend_products_for_customer(
                    ma_khach_hang=ma_khach_hang_input,
                    data_sub_pandas=reviews,
                    products_sub_pandas=products,
                    best_algorithm=best_algorithm_new
                )

                if error:
                    st.warning(error)
                elif not recommendations.empty:
                    st.write("Các sản phẩm gợi ý cho khách hàng:")
                    display_recommended_products_1(recommendations, cols=5)
                else:
                    st.write("Không có sản phẩm nào được đề xuất.")
            else:
                st.error("Không tìm thấy thông tin khách hàng.")
    
elif choice == 'Gợi ý theo thông tin sản phẩm':
    # Nhập tên sản phẩm, tìm kiếm mã sản phẩm, và đề xuất các sản phẩm liên quan
    def get_products_recommendations(products, product_id, cosine_sim, nums=5):
        # Tìm chỉ mục sản phẩm dựa trên mã sản phẩm
        matching_indices = products.index[products['ma_san_pham'] == product_id].tolist()

        if not matching_indices:
            return pd.DataFrame()
        idx = matching_indices[0]

        # Tính toán độ tương đồng của sản phẩm được chọn với các sản phẩm khác
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sắp xếp sản phẩm theo độ tương đồng
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Lấy các sản phẩm tương tự (bỏ qua sản phẩm chính)
        sim_scores = sim_scores[1:nums + 1]

        # Lấy chỉ số sản phẩm
        product_indices = [i[0] for i in sim_scores]

        # Trả về danh sách sản phẩm được đề xuất
        return products.iloc[product_indices]
    
    # Hiển thị đề xuất ra bảng
    def display_recommended_products_2(get_products_recommendations, cols=4):
        for i in range(0, len(get_products_recommendations), cols):
            cols = st.columns(cols)
            for j, col in enumerate(cols):
                if i + j < len(get_products_recommendations):
                    product = get_products_recommendations.iloc[i + j]
                    with col:   
                        st.write(product['ten_san_pham'])                    
                        expander = st.expander(f"Mô tả")
                        product_description = product['mo_ta']
                        truncated_description = ' '.join(product_description.split()[:100]) + '...'
                        expander.write(truncated_description)
                        expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")
                        
    # Đọc dữ liệu sản phẩm
    products = pd.read_csv('San_pham.csv')

    # Open and read file to cosine_sim_new
    with open('products_cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)

    # Giao diện Streamlit
    st.title("Hệ thống gợi ý sản phẩm theo thông tin sản phẩm")

    st.image('hasaki_banner.jpg', use_container_width=True)

    # Người dùng nhập tên sản phẩm
    st.markdown('<p style="font-size:30px; font-weight:bold;">Nhập tên sản phẩm để tìm kiếm:</p>', unsafe_allow_html=True)
    product_name_input = st.text_input("product_name_input", key="product_name_input", label_visibility="hidden")

    # Kiểm tra tên sản phẩm
    if product_name_input:
        matching_products = products[products['ten_san_pham'].str.contains(product_name_input, case=False, na=False)]
        
        if not matching_products.empty:
            
            # Hiển thị các sản phẩm tìm được
            st.write("Danh mục sản phẩm tìm được:")
            
            # Display the matching products as a DataFrame
            st.dataframe(matching_products[['ma_san_pham', 'ten_san_pham']])  # Streamlit DataFrame display
            
            # Người dùng chọn sản phẩm từ danh sách
            selected_product = st.selectbox(
                "Chọn sản phẩm để xem gợi ý:",
                options=matching_products.itertuples(),
                format_func=lambda x: x.ten_san_pham)

            if selected_product:
                st.write("### Bạn đã chọn:")
                st.write(f"- **Tên:** {selected_product.ten_san_pham}")
                st.write(f"- **Mã:** {selected_product.ma_san_pham}")
                st.write(f"- **Mô tả:** {selected_product.mo_ta[:5000]}...")

                # Lấy danh sách sản phẩm gợi ý
                recommendations = get_products_recommendations(
                    products, selected_product.ma_san_pham, cosine_sim_new, nums=4)

                if not recommendations.empty:
                    st.write("Các sản phẩm gợi ý liên quan:")
                    display_recommended_products_2(recommendations, cols=4)
                else:
                    st.write("Không tìm thấy sản phẩm liên quan.")
        else:
            st.write("Không tìm thấy sản phẩm phù hợp.")
