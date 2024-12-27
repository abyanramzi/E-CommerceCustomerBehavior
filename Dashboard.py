import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import altair as alt

# Set the base directory relative to this script's location
base_dir = os.path.dirname(__file__)
customers_file_path = os.path.join(base_dir, 'Data', 'customers_dataset.csv')
products_file_path = os.path.join(base_dir, 'Data', 'products_dataset.csv')
order_items_file_path = os.path.join(base_dir, 'Data', 'order_items_dataset.csv')
orders_file_path = os.path.join(base_dir, 'Data', 'orders_dataset.csv')
product_translation_file_path = os.path.join(base_dir, 'Data', 'product_category_name_translation.csv')

# Load the datasets
customers_df = pd.read_csv(customers_file_path)
products_df = pd.read_csv(products_file_path)
order_items_df = pd.read_csv(order_items_file_path)
orders_df = pd.read_csv(orders_file_path)
product_translation_df = pd.read_csv(product_translation_file_path)

# Change string format
customers_df['customer_city'] = customers_df['customer_city'].str.title()
product_translation_df['product_category_name_english'] = product_translation_df['product_category_name_english'].str.replace('_', ' ').str.title()

# Data Assessing
## Customer data
print(customers_df.info())
print(customers_df.isnull().sum())
print(customers_df.duplicated().sum())
## Product data
print(products_df.info())
print(products_df.isnull().sum())
print(products_df.duplicated().sum())
## Order items data
print(order_items_df.info())
print(order_items_df.isnull().sum())
print(order_items_df.duplicated().sum())
## Orders data
print(orders_df.info())
print(orders_df.isnull().sum())
print(orders_df.duplicated().sum())
## Product translation data
print(product_translation_df.info())
print(product_translation_df.isnull().sum())
print(product_translation_df.duplicated().sum()) 

# Cleaning Data
customers_df = customers_df.drop_duplicates(subset='customer_id')
products_df = products_df.drop_duplicates(subset='product_id')
orders_df = orders_df.drop_duplicates(subset='order_id')


# Exploratory Data Analysis
# Merged Data
## Merged product and product translation data
products_merged = products_df.merge(product_translation_df, on='product_category_name')
print(products_merged.head())
## Merged all data
merged_data = (
    orders_df
    .merge(order_items_df, on='order_id')
    .merge(products_merged, on='product_id')
    .merge(customers_df, on='customer_id')
)
print("Merged Data:")
print(merged_data.shape)
print(merged_data.head())


# 1. Count total sales and customers for region with more than 900 customers
## Group by city to get total sales by region (city)
sales_by_region = merged_data.groupby('customer_city').agg({'price': 'sum'}).reset_index()
print(sales_by_region)
## Count number of customers in each city
customers_distribution = customers_df['customer_city'].value_counts().reset_index()
customers_distribution.columns = ['customer_city', 'num_customers']
print(customers_distribution)
## Merge sales and customer distribution on customer_city
merged_region_data = pd.merge(customers_distribution, sales_by_region, on='customer_city')
# Top 50 region by sales
all_region_sales_data = merged_region_data.groupby('customer_city').agg(
    total_sales=('price', 'sum'),
    num_customers=('num_customers', 'sum')
).reset_index()
top_n_region_by_sales = all_region_sales_data.nlargest(50,"total_sales")
average_customers_in_top_n = top_n_region_by_sales['num_customers'].mean()
average_sales_in_top_n = top_n_region_by_sales["total_sales"].mean()
print("Average number of customers for top 50 regions by total sales: ", average_customers_in_top_n)

## Filter for regions with more than 1000 customers
filtered_region_data = merged_region_data[merged_region_data['num_customers'] > average_customers_in_top_n]
# Filter for region with sales more than average
average_sales_in_filtered_region = filtered_region_data['price'].mean()
print("Average number of customers for top 50 regions by total sales: ", average_sales_in_filtered_region)
average_customers_in_filtered_region = filtered_region_data['num_customers'].mean()
print("Average number of customers for top 50 regions by total sales: ", average_customers_in_filtered_region)

## Sorting for city color pallete
city_order_by_customers = filtered_region_data.sort_values(by='num_customers', ascending=False)['customer_city'].tolist()
## Indexing top region
top_regions_index = filtered_region_data['customer_city']
## Calculate mean sales per customer for each region
region_sales_data = filtered_region_data.groupby('customer_city').agg(
    total_sales=('price', 'sum'),
    num_customers=('num_customers', 'sum')
).reset_index()
region_sales_data['mean_sales_per_customers'] = region_sales_data['total_sales'] / region_sales_data['num_customers']
## Sorting for visualization based on mean
city_order_by_mean_sales = region_sales_data.sort_values(by='mean_sales_per_customers', ascending=False)['customer_city'].tolist()

# 2. RFM
## Filtered data for top regions
filtered_data = merged_data[(merged_data['customer_city'].isin(top_regions_index ))]
## Check the filtered data
print("Filtered Data:")
print(filtered_data.shape)
print(filtered_data.head())
## Convert purchase timestamp to datetime
filtered_data.loc[:, 'order_purchase_timestamp'] = pd.to_datetime(filtered_data['order_purchase_timestamp'], errors='coerce')
## Calculate Recency, Frequency, and Monetary only for customers with orders
latest_purchase = filtered_data['order_purchase_timestamp'].max()
rfm_df = filtered_data.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (latest_purchase - x.max()).days,
    'order_id': 'count',  # Frequency
    'price': 'sum'        # Ensure 'price' column exists
}).reset_index()
## Rename columns
rfm_df.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']
## Check frequency of orders and the customers
print(rfm_df['Frequency'].value_counts())
total_rfm_customers = rfm_df['customer_id'].nunique()
print(total_rfm_customers)
## Create bins for Recency, Frequency, and Monetary
rfm_df['R_Score'] = pd.cut(rfm_df['Recency'], bins=5, labels=range(5, 0, -1))
rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'], bins=5, labels=range(1, 6))
rfm_df['M_Score'] = pd.cut(rfm_df['Monetary'], bins=5, labels=range(1, 6))
## Check for NaN values and drop if necessary
rfm_df.dropna(inplace=True)
## Combine RFM Scores
rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
## Summary of RFM scores
rfm_summary = rfm_df.groupby('RFM_Score').agg({'customer_id': 'count'}).reset_index()
print(rfm_summary)
## Merge rfm_df with filtered_data (cleaned duplicates customer id) to get region information
rfm_with_region = pd.merge(rfm_df, filtered_data[['customer_id', 'customer_city']].drop_duplicates(subset="customer_id"), on='customer_id', how='left')
## Count customers by region
region_counts_rfm = rfm_with_region['customer_city'].value_counts().reset_index()
region_counts_rfm.columns = ['customer_city', 'num_customers']
## Count customers by region and RFM Score
region_rfm_counts = rfm_with_region.groupby(['customer_city', 'RFM_Score']).size().unstack(fill_value=0)

def segment_customers(row):
    if row['RFM_Score'] == '511':
        return 'Champions'
    elif row['RFM_Score'].startswith('4'):
        return 'Loyal Customers'
    elif row['RFM_Score'].startswith('2') and row['Recency'] < 30:
        return 'Potential Loyalists'
    elif row['RFM_Score'].startswith('1'):
        return 'Lost Customers'
    else:
        return 'At Risk'

# Apply the segmentation function
rfm_df['Customer_Segment'] = rfm_df.apply(segment_customers, axis=1)

# Step 3: Aggregate data for visualization
segment_counts = rfm_df['Customer_Segment'].value_counts().reset_index()
segment_counts.columns = ['Customer_Segment', 'Number_of_Customers']


# 3. Find top product on top regions
filtered_sales_data = merged_data[merged_data['customer_city'].isin(filtered_region_data['customer_city'].unique())]
category_sales_in_filtered_cities = (filtered_sales_data.groupby('product_category_name_english')['price'].sum().reset_index())
## Filter top N product
top_n_categories = category_sales_in_filtered_cities.nlargest(10,'price')
## Sorting for visualization based on total sales of prodcut category
product_order_by_total_sales = top_n_categories.sort_values(by='price', ascending=False)['product_category_name_english'].tolist()


# 4 Analysis top product category sales 
def process_category_data(merged_data, customers_distribution, top_region_index, category_name):
    category_data = merged_data[merged_data['product_category_name_english'] == category_name]

    sales_by_region = category_data.groupby('customer_city').agg({'price': 'sum'}).reset_index()
    sales_by_region.columns = ['customer_city', f'{category_name.lower().replace(" ", "_")}_sales']

    merged_category_data = pd.merge(customers_distribution, sales_by_region, on='customer_city', how='inner')
    filtered_category_data = merged_category_data[merged_category_data['customer_city'].isin(top_region_index)]
    
    total_sales = filtered_category_data[f'{category_name.lower().replace(" ", "_")}_sales'].sum()
    total_customers = filtered_category_data['num_customers'].sum()

    sales_data = filtered_category_data.groupby('customer_city').agg(
        total_sales = (f'{category_name.lower().replace(" ", "_")}_sales', 'sum'),
        num_customers = ('num_customers', 'sum')
    ).reset_index()
    sales_data['mean_sales_per_customers'] = sales_data['total_sales']/sales_data['num_customers']

    return{
        'total_sales': total_sales,
        'total_customers': total_customers,
        'sales_data': sales_data,
        'filtered_data': filtered_category_data
    }

categories = product_order_by_total_sales
category_result = {}

for category in categories:
    category_result[category] = process_category_data(
        merged_data,
        customers_distribution,
        top_regions_index,
        category
    )

health_beauty_results = category_result["Health Beauty"]
total_sales_health_beauty = health_beauty_results['total_sales']
total_customers_health_beauty = health_beauty_results['total_customers']
health_beauty_sales_data = health_beauty_results['sales_data']
filtered_health_beauty_data = health_beauty_results['filtered_data']

health_beauty_order_by_mean_sales = health_beauty_sales_data.sort_values(by='mean_sales_per_customers', ascending=False)['customer_city'].tolist()

watches_gifts_results = category_result["Watches Gifts"]
total_sales_watches_gifts = watches_gifts_results['total_sales']
total_customers_watches_gifts = watches_gifts_results['total_customers']
watches_gifts_sales_data = watches_gifts_results['sales_data']
filtered_watches_gifts_data = watches_gifts_results['filtered_data']
watches_gifts_order_by_mean_sales = watches_gifts_sales_data.sort_values(by='mean_sales_per_customers', ascending=False)['customer_city'].tolist()


### Next to streamlit
# Display
## Color Palette
custom_palette = ['#002855', '#023E7D', '#0466C8', '#005F73', '#0A9396', '#57B5B8', '#EE9B00', '#EDAE49', '#6A040F', '#9B2226']
color_mapping = {city: color for city, color in zip(city_order_by_customers, custom_palette)}
## Page config
st.set_page_config(
    page_title="RFM Analysis Dashboard",
    layout="wide"
)
## Inject custom CSS to style the app
st.markdown(
    """
    <style>
        .streamlit-container {
            max-width: 1200px;  /* Change this value to set the max width */
            margin: auto;
        }
        h1 {
            text-align: center;  /* Center the title */
        }
    </style>
    """,
    unsafe_allow_html=True
)
## Streamlit Code
st.title("Analyzing Customer Behavior In Brazilian E-Commerce (Olist Store)")

tab1, tab2 = st.tabs(["RFM Analysis", "Product Analysis"])

with tab1:
    with st.container(height=750):
        chart1, chart2 = st.columns([1.5,1])
        ## First chart
        with chart1:
            scatter_plot = alt.Chart(filtered_region_data).mark_circle(size=100).encode(
                x=alt.X('price', title='Total Sales (BRL)'),
                y=alt.Y('num_customers', title='Number of Customers'),
                color=alt.condition(
                    alt.datum.price > average_sales_in_filtered_region,
                    alt.value("#bde0fe"),
                    alt.value("#023E7D")
                ),
                tooltip=['customer_city', 'price', 'num_customers']
            ).properties(
                title=alt.TitleParams(
                    text='Number of Customers Vs Total Sales in Region with More Than 900 Customers',
                    anchor='middle'  # Center-align the title
                ),
                height=350
            )
            st.altair_chart(scatter_plot, use_container_width=True)
        ## Second Chart
        with chart2:
                bars = alt.Chart(region_counts_rfm).mark_bar().encode(
                    x=alt.X('num_customers', title='Number of Customers'),
                    y=alt.Y('customer_city:N', sort=city_order_by_customers, title=None),
                    color=alt.condition(
                        alt.datum.num_customers > average_customers_in_filtered_region, 
                        alt.value("#bde0fe"),
                        alt.value("#023E7D")
                    ),
                    tooltip=['customer_city', 'num_customers']
                ).properties(
                    title=alt.TitleParams(
                        text='Region with Most RFM Customers',
                        anchor='middle'
                    ),
                    height=350
                )
                st.altair_chart(bars, use_container_width=True)


        chart6, chart7 = st.columns([1.5,1])

        with chart6:
            heatmap_data = region_rfm_counts.reset_index().melt(id_vars='customer_city', var_name='RFM Score', value_name='Count')
            heatmap_data.columns = ['Region', 'RFM Score', 'Count']

            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('RFM Score:N', title='RFM Score', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Region:N', title='Region'),
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Customer Count'),  # Adjusted color scheme
                tooltip=['Region', 'RFM Score', 'Count']
            ).properties(
                title=alt.TitleParams(
                    text='Regions with More than 1000 Customers',
                    anchor='middle'  # Center-align the title
                ), height=350)

            st.altair_chart(heatmap, use_container_width=True)

        with chart7:
            dominant_segment = segment_counts.loc[segment_counts['Number_of_Customers'].idxmax(), 'Customer_Segment']
            bars = alt.Chart(segment_counts).mark_bar().encode(
                y=alt.Y('Customer_Segment:N', sort=['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Lost Customers'], title='Customer Segment'),
                x=alt.X('Number_of_Customers:Q', title='Number of Customers'),
                color=alt.condition(
                alt.datum.Customer_Segment == dominant_segment,  # Condition to highlight the highest segment
                alt.value("#bde0fe"),
                alt.value("#023E7D")
            ),
                tooltip=[alt.Tooltip('Customer_Segment:N', title='Customer Segment'),
                alt.Tooltip('Number_of_Customers:Q', title='Number of Customers')] 
            ).properties(
                # width=800,
                height=350,
                title="Customer Segments Distribution"
            )
            st.altair_chart(bars, use_container_width=True)
        


def process_category_data(merged_data, customers_distribution, top_region_index, category_name):
    category_data = merged_data[merged_data['product_category_name_english'] == category_name]

    sales_by_region = category_data.groupby('customer_city').agg({'price': 'sum'}).reset_index()
    sales_by_region.columns = ['customer_city', f'{category_name.lower().replace(" ", "_")}_sales']

    merged_category_data = pd.merge(customers_distribution, sales_by_region, on='customer_city', how='inner')
    filtered_category_data = merged_category_data[merged_category_data['customer_city'].isin(top_region_index)]
    
    total_sales = filtered_category_data[f'{category_name.lower().replace(" ", "_")}_sales'].sum()
    total_customers = filtered_category_data['num_customers'].sum()

    sales_data = filtered_category_data.groupby('customer_city').agg(
        total_sales = (f'{category_name.lower().replace(" ", "_")}_sales', 'sum'),
        num_customers = ('num_customers', 'sum')
    ).reset_index()
    sales_data['mean_sales_per_customers'] = sales_data['total_sales']/sales_data['num_customers']

    return{
        'total_sales': total_sales,
        'total_customers': total_customers,
        'sales_data': sales_data,
        'filtered_data': filtered_category_data
    }

categories = product_order_by_total_sales
category_result = {}

for category in categories:
    category_result[category] = process_category_data(
        merged_data,
        customers_distribution,
        top_regions_index,
        category
    )

health_beauty_results = category_result["Health Beauty"]
total_sales_health_beauty = health_beauty_results['total_sales']
total_customers_health_beauty = health_beauty_results['total_customers']
health_beauty_sales_data = health_beauty_results['sales_data']
filtered_health_beauty_data = health_beauty_results['filtered_data']

health_beauty_order_by_mean_sales = health_beauty_sales_data.sort_values(by='mean_sales_per_customers', ascending=False)['customer_city'].tolist()

watches_gifts_results = category_result["Watches Gifts"]
total_sales_watches_gifts = watches_gifts_results['total_sales']
total_customers_watches_gifts = watches_gifts_results['total_customers']
watches_gifts_sales_data = watches_gifts_results['sales_data']
filtered_watches_gifts_data = watches_gifts_results['filtered_data']
watches_gifts_order_by_mean_sales = watches_gifts_sales_data.sort_values(by='mean_sales_per_customers', ascending=False)['customer_city'].tolist()

with tab2:
# with st.expander("See Product Charts"):
    ## Columns for the Third and Fourth Charts
    tabs = st.tabs(product_order_by_total_sales)
    for i, tab in enumerate(tabs):
        filtered_data_map = {
            category: category_result[category]['filtered_data']
            for category in product_order_by_total_sales
        }
        sales_data_map = {
            category: category_result[category]['sales_data']
            for category in product_order_by_total_sales
        }
        category = product_order_by_total_sales[i]
        filtered_data = filtered_data_map[category]
        sales_data = sales_data_map[category]
        sorted_cities = sales_data.sort_values(by='mean_sales_per_customers', ascending=False)['customer_city'].tolist()

        with tab:
            chart3, chart4, chart5 = st.columns([0.75, 0.75, 1])
            ### Second chart
            with chart3:
                bars = alt.Chart(top_n_categories).mark_bar().encode(
                    x=alt.X('price', title='Total Sales (BRL)', axis=alt.Axis(labels=True, tickCount=10, labelAngle=0)),
                    y=alt.Y('product_category_name_english', sort=product_order_by_total_sales, title='Product Category'),
                    color=alt.condition(
                        alt.datum.product_category_name_english == f'{category}',
                        alt.value("#bde0fe"),
                        alt.value("#023E7D")
                    ),
                    tooltip=['price', 'product_category_name_english']
                ).properties(
                    title=alt.TitleParams(
                        text='Top Sales of Product Categories in Top Regions',
                        anchor='middle'  # Center-align the title
                    ), height=350)
                st.altair_chart(bars, use_container_width=True)

            with chart4:
                scatter_plot = alt.Chart(filtered_data).mark_circle(size=100).encode(
                    x=alt.X(f'{category.lower().replace(" ", "_")}_sales', title='Total Sales (BRL)'),
                    y=alt.Y('num_customers', title='Number of Customers'),
                    color=alt.Color(
                        'customer_city',
                        scale=alt.Scale(domain=list(color_mapping.keys()), range=list(color_mapping.values())),
                        legend=None
                    ),
                    tooltip=['customer_city', f'{category.lower().replace(" ", "_")}_sales', 'num_customers']
                ).properties(
                    title=alt.TitleParams(
                        text=f'{category} Customers vs Sales',
                        anchor='middle'
                    ),
                    height=350
                )
                st.altair_chart(scatter_plot, use_container_width=True)

            with chart5:
                bars = alt.Chart(sales_data).mark_bar().encode(
                    x=alt.X('mean_sales_per_customers:Q', title='Mean Sales per Customer (BRL)', axis=alt.Axis(labels=True, tickCount=10, labelAngle=0)),
                    y=alt.Y('customer_city:N', sort=sorted_cities, title=None, axis=alt.Axis(labels=True, tickCount=10, labelAngle=0)),
                    color=alt.Color(
                        'customer_city:N',
                        scale=alt.Scale(domain=list(color_mapping.keys()), range=list(color_mapping.values())),
                        legend=alt.Legend(title='Region (City)')
                    ),
                    tooltip=['customer_city', 'mean_sales_per_customers']
                ).properties(
                    title=alt.TitleParams(
                        text=f'{category} Product Mean Sales per Customer by Region',
                        anchor='middle'  # Center-align the title
                    ),
                    height=350
                )
                st.altair_chart(bars, use_container_width=True)


