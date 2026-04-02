import pandas as pd
import streamlit as st

st.title("Sales Summary Dashboard")
st.subheader("Interactive sales summary with category-based filtering")

sales_data = pd.DataFrame(
    {
        "Product": ["Laptop", "Mouse", "Keyboard", "Desk Chair", "Monitor", "Webcam"],
        "Category": ["Electronics", "Electronics", "Electronics", "Furniture", "Electronics", "Electronics"],
        "Sales": [1200, 45, 80, 300, 250, 95],
    }
)

categories = sorted(sales_data["Category"].unique())
selected_category = st.sidebar.selectbox("Select Category", categories)

filtered_data = sales_data[sales_data["Category"] == selected_category]

st.dataframe(filtered_data, use_container_width=True)
st.line_chart(filtered_data.set_index("Product")["Sales"])
