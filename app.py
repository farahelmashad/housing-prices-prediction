import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px

df = pd.read_csv('assets/Housing_cleaned.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Housing Prices Dashboard"

sidebar = dbc.Col([
    html.H5("Filters", className="text-white mt-3 mb-4", style={"fontFamily": "Segoe UI, sans-serif"}),
    html.Hr(style={"borderColor": "#777"}),

    html.Label('Price Range', className="text-light"),
    dcc.RangeSlider(
        id='price-slider',
        min=df['price'].min(),
        max=df['price'].max(),
        value=[df['price'].min(), df['price'].max()],
        tooltip={"placement": "bottom", "always_visible": True},
        className="mb-4"
    ),

    html.Label('Area Range', className="text-light"),
    dcc.RangeSlider(
        id='area-slider',
        min=df['area'].min(),
        max=df['area'].max(),
        value=[df['area'].min(), df['area'].max()],
        tooltip={"placement": "bottom", "always_visible": True},
        className="mb-4"
    ),

    html.Label('Furnishing Status', className="text-light"),
    dcc.Dropdown(
        id='furnishing-dropdown',
        options=[{'label': 'All', 'value': 'All'}] +
                [{'label': status, 'value': status} for status in df['furnishingstatus'].unique()],
        value='All',
        clearable=False,
        className="mb-4"
    ),

    html.Label('Number of Rooms', className="text-light"),
    dcc.RangeSlider(
        id='rooms-slider',
        min=df['total_rooms'].min(),
        max=df['total_rooms'].max(),
        step=1,
        value=[df['total_rooms'].min(), df['total_rooms'].max()],
        tooltip={"placement": "bottom", "always_visible": True},
        className="mb-4"
    ),

    html.Label('Stories', className="text-light"),
    dcc.RangeSlider(
        id='stories-slider',
        min=df['stories'].min(),
        max=df['stories'].max(),
        step=1,
        value=[df['stories'].min(), df['stories'].max()],
        tooltip={"placement": "bottom", "always_visible": True},
        className="mb-4"
    ),

    html.Label('Luxury Features', className='text-light'),
    dcc.RangeSlider(
        id='luxury-slider',
        min=df['luxury_features'].min(),
        max=df['luxury_features'].max(),
        step=1,
        value=[df['luxury_features'].min(), df['luxury_features'].max()],
        tooltip={"placement": "bottom", "always_visible": True},
        className="mb-4"
    ),

    html.Label("Main Road Access", className="text-light"),
    dbc.RadioItems(
        id='mainroad-radio',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Yes', 'value': True},
            {'label': 'No', 'value': False}
        ],
        value='All',
        inline=True,
        className='text-light mb-4'
    ),

    html.Label("Guestroom", className="text-light"),
    dbc.RadioItems(
        id='guestroom-radio',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Yes', 'value': True},
            {'label': 'No', 'value': False}
        ],
        value='All',
        inline=True,
        className='text-light mb-4'
    ),

    html.Label("Basement", className="text-light"),
    dbc.RadioItems(
        id='basement-radio',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Yes', 'value': True},
            {'label': 'No', 'value': False}
        ],
        value='All',
        inline=True,
        className='text-light mb-4'
    ),

    html.Label("Air Conditioning", className="text-light"),
    dbc.RadioItems(
        id='ac-radio',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Yes', 'value': True},
            {'label': 'No', 'value': False}
        ],
        value='All',
        inline=True,
        className='text-light mb-4'
    ),

], width=3, style={
    'backgroundColor': "#1C1C1C",  
    'height': '100vh',
    'position': 'sticky',
    'top': 0,
    'padding': '20px',
    'overflowY': 'auto',  
    'fontFamily': 'Segoe UI, sans-serif'
})

main_content = dbc.Col([
    html.H3("🏡 House Prices Dashboard",
            className="text-center mt-3 mb-3",
            style={"fontFamily": "Segoe UI, Roboto, Helvetica, sans-serif",
                   "fontWeight": "bold",
                   "fontSize": "28px"}),
    # Cards' Row
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Average Price", className="fw-bold"),
                          dbc.CardBody(html.H5(id="avg-price", className="fw-bold"))]), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("Total Properties", className="fw-bold"),
                          dbc.CardBody(html.H5(id="total-properties", className="fw-bold"))]), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("Luxury Houses %", className="fw-bold"),
                          dbc.CardBody(html.H5(id="luxury-percent", className="fw-bold"))]), width=3)            
    ], className="mb-4", justify="center"),

    # Row 1
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="graph-price-area", figure={}, style={"height": "100%", "width": "100%"}), style={"height": "400px"}), width=6, className="mb-4"),
        dbc.Col(
            dbc.Card([
                    dbc.CardHeader(
                        dbc.Tabs([
                                dbc.Tab(label="Avg Price", tab_id="tab-bar"),
                                dbc.Tab(label="Distribution", tab_id="tab-pie"),
                            ], id="furnishing-tabs", active_tab="tab-bar")
                    ),
                    dbc.CardBody(dcc.Graph(id="graph-furnishing", style={"height": "100%", "width": "100%"}))
                ],
                style={"height": "400px"}
            ),
            width=6, className="mb-4"
        ),
    ]),

    # Row 2
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="graph-mainroad-bar", figure={}, style={"height": "100%", "width": "100%"}), style={"height": "350px"}), width=6, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id="graph-bedrooms-box", figure={}, style={"height": "100%", "width": "100%"}), style={"height": "350px"}), width=6, className="mb-4"),
    ]),

    #Row 3
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="graph-luxury-count", figure={}, style={"height": "100%", "width": "100%"}), style={"height": "400px"}), width=6, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id="graph-price-per-area", figure={}, style={"height": "100%", "width": "100%"}), style={"height": "400px"}), width=6, className="mb-4"),
    ]),

    # Row 4
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id="graph-comfort", style={"height": "100%", "width": "100%"}), 
                        style={"height": "400px"}), width=6, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id="graph-parking", style={"height": "100%", "width": "100%"}), 
                        style={"height": "400px"}), width=6, className="mb-4"),
    ]),

], width=9)

@app.callback(
    [
        Output("avg-price", "children"),
        Output("total-properties", "children"),
        Output("luxury-percent", "children"),
        Output("graph-price-area", "figure"),
        Output("graph-furnishing", "figure"),
        Output("graph-mainroad-bar", "figure"),
        Output("graph-bedrooms-box", "figure"),
        Output("graph-luxury-count", "figure"),
        Output("graph-price-per-area", "figure"),
        Output("graph-comfort", "figure"),
        Output("graph-parking", "figure"),

    ],
    [
        Input("price-slider", "value"),
        Input("area-slider", "value"),
        Input("furnishing-dropdown", "value"),
        Input("rooms-slider", "value"),
        Input("stories-slider", "value"),
        Input("luxury-slider", "value"),
        Input("mainroad-radio", "value"),
        Input("guestroom-radio", "value"),
        Input("basement-radio", "value"),
        Input("ac-radio", "value"),
        Input("furnishing-tabs", "active_tab")
    ]
)
def update_dashboard(price_range, area_range, furnishing, rooms_range, stories_range,
                     luxury_range, mainroad, guestroom, basement, ac, active_tab):

    dff = df.copy()
    dff = dff[(dff['price'] >= price_range[0]) & (dff['price'] <= price_range[1])]
    dff = dff[(dff['area'] >= area_range[0]) & (dff['area'] <= area_range[1])]
    dff = dff[(dff['total_rooms'] >= rooms_range[0]) & (dff['total_rooms'] <= rooms_range[1])]
    dff = dff[(dff['stories'] >= stories_range[0]) & (dff['stories'] <= stories_range[1])]
    dff = dff[(dff['luxury_features'] >= luxury_range[0]) & (dff['luxury_features'] <= luxury_range[1])]

    if furnishing != 'All':
        dff = dff[dff['furnishingstatus'] == furnishing]

    for col, val in {'mainroad': mainroad,'guestroom': guestroom,'basement': basement,'airconditioning': ac}.items():
        if val != 'All':
            dff = dff[dff[col] == val]

    # Cards' Row
    avg_price = f"${dff['price'].mean():,.0f}" if not dff.empty else "N/A"
    total_properties = f"{len(dff)}" if not dff.empty else "0"
    luxuryPerc = f"{round((len(dff[dff['luxury_features'] > 2]) / len(dff)) * 100, 2)}" if not dff.empty else "0%"

    # Row 1
    # -- Price vs Area Scatter plot
    fig_price_area = px.scatter(
        dff, x='area', y='price',  size='luxury_features',
        hover_data=['bedrooms','bathrooms','total_rooms'],
        title="Price vs Area"
    )
    
    # -- Avg price
    avg_price_furnishing = dff.groupby('furnishingstatus')['price'].mean().reset_index()
    fig_furnishing_bar = px.bar(
        avg_price_furnishing, x='furnishingstatus', y='price', color='furnishingstatus',
        title="Average Price by Furnishing Status"
    )

    fig_furnishing_pie = px.pie(
        dff, names='furnishingstatus', title="Furnishing Status Distribution"
    )

    if active_tab == "tab-bar":
        fig_furnishing = fig_furnishing_bar
    else:
        fig_furnishing = fig_furnishing_pie

    # Row 2
    # -- Avg price for Main Road access 
    avg_price_mainroad = dff.groupby('mainroad')['price'].mean().reset_index()
    fig_mainroad_price = px.bar(
        avg_price_mainroad,
        x='mainroad',
        y='price',
        color='mainroad',
        title="Average Price: Main Road Access vs No Main Road"
    )

    # -- No of Bedrooms price distribution
    fig_bedrooms_box = px.box(
        dff, x='bedrooms', y='price', title="Price Distribution by Number of Bedrooms", points=False
    )

    # Row 3
    # -- Avg Price for each Luxury features count
    luxury_avg_price = dff.groupby('luxury_features')['price'].mean().reset_index()
    luxury_avg_price.columns = ['Luxury Features', 'Average Price']

    fig_luxury_count = px.bar(
        luxury_avg_price,
        x='Luxury Features',
        y='Average Price',
        title="Average Price by Number of Luxury Features",
        color='Average Price',
        color_continuous_scale='Blues'
    )

    # -- Avg price for each story bar plot
    avg_price_stories = dff.groupby('stories')['price'].mean().reset_index()
    avg_price_stories.columns = ['Stories', 'Average Price']

    fig_price_stories = px.bar(
        avg_price_stories,
        x='Stories',
        y='Average Price',
        title="Average Price by Number of Stories",
    )

    # Row 4
    # -- Percentage of Availability of each luxury feature found in houses
    luxuries = {}
    for col in ['guestroom', 'basement', 'mainroad', 'hotwaterheating', 'airconditioning', 'prefarea']:
        if not dff.empty:
            luxuries[col] = round((dff[col].sum() / len(dff)) * 100, 2)
        else:
            luxuries[col] = 0

    luxuries_df = pd.DataFrame({
        "Luxuries": luxuries.keys(),
        "Percentage": luxuries.values()
    })

    fig_luxuries = px.bar(
        luxuries_df, x="Luxuries", y="Percentage", color="Luxuries",
        title="Percentage of Houses with Each Luxury Feature",
        text="Percentage"
    )
    fig_luxuries.update_traces(texttemplate='%{text:.2f}%', textposition="outside")

    # --- Parking Pie
    if not dff.empty:
        fig_parking = px.pie(
            dff, names="parking", title="Parking Slots Distribution"
        )
    else:
        fig_parking = px.pie(
            names=["No Data"], values=[1], title="Parking Slots Distribution"
        )


    return avg_price, total_properties, luxuryPerc, fig_price_area, fig_furnishing, \
           fig_mainroad_price, fig_bedrooms_box, fig_luxury_count, fig_price_stories, fig_luxuries, fig_parking
           
app.layout = dbc.Container([dbc.Row([sidebar, main_content], justify="start")], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
