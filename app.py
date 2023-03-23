import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pickle
import xgboost
from sklearn.model_selection import train_test_split
import numpy as np
import time

st.set_page_config(page_title="RaccooL", page_icon=":raccoon:",layout="wide")

@st.cache_data
def load_game_details():
    return pd.read_excel('data.xlsx')

game_details = load_game_details()


progress_text = "Our **raccoons** are **foraging**. They are **the** **best** at what they **do** :raccoon:"
my_bar = st.progress(0)

for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

my_bar.empty()


# Define Home page

def home():
    st.sidebar.title('Navigation Center:')
    menu = ['Home', 'Game Searching', 'Game Recommendation', 'Game Price Calculator', 'About us']
    page = st.sidebar.radio('Select a page', menu)
    
    st.sidebar.write('')

    st.sidebar.write('### Follow me on Social Media')
    st.sidebar.write('[LinkedIn Profile](https://www.linkedin.com/in/-ricardo-guedes/)')
    st.sidebar.write('[GitHub Profile](https://github.com/BillyGrey)')

    st.sidebar.write('---')
    image2 = Image.open('raccoon.png')
    st.sidebar.image(image2, use_column_width=True)

    if page == 'Home':
        front()
    elif page == 'Game Searching':
        game_searching()
    elif page == 'Game Recommendation':
        game_recommendation()
    elif page == 'Game Price Calculator':
        model()
    else:
        about()

#Define Front page
@st.cache_resource
def front():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    cola, colb = st.columns([3,2])
    with cola:
        st.markdown("<h1 style='text-align: right; font-size: 110px; color: #53290B;'>RaccooL</h1>", unsafe_allow_html=True)

    with colb:
        image1 = Image.open('icon.png')
        st.image(image1, width=170)  

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'> A tool made to help, optimize and organize your game searching experience. </h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>All the details of +6700 games in your hands. Happy Foraging! </h4>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    

    st.header("Games Statistics:")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Top Trending Games :video_game:")

    # Filter the games based on specific appid values
    tgame_appids = [990080,1326470,1058830,668580,1693980]
    top_games = game_details[game_details['appid'].isin(tgame_appids)]

    # Display the image and name of each game in a horizontal layout
    game_columns = st.columns(len(top_games))
    for i, game in enumerate(top_games.itertuples()):
        game_columns[i].image(game.header_image, width=200)
        game_columns[i].write(game.name)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Top Rated Games :chart_with_upwards_trend: ")

    # Filter the games based on specific appid values
    rgame_appids = [620,1794680,413150,400,105600]
    rated_games = game_details[game_details['appid'].isin(rgame_appids)]

    # Display the image and name of each game in a horizontal layout
    game_columns = st.columns(len(rated_games))
    for i, game in enumerate(rated_games.itertuples()):
        game_columns[i].image(game.header_image, width=200)
        game_columns[i].write(game.name)
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Top Most Played Games :busts_in_silhouette:")

    # Filter the games based on specific appid values
    pgame_appids = [570,578080,440,304930,252490]
    played_games = game_details[game_details['appid'].isin(pgame_appids)]

    # Display the image and name of each game in a horizontal layout
    game_columns = st.columns(len(played_games))
    for i, game in enumerate(played_games.itertuples()):
        game_columns[i].image(game.header_image, width=200)
        game_columns[i].write(game.name)
    


# Define Game Searching page

def game_searching():
    
    st.title('Welcome to our Game Searching Tool')
    options = game_details['name'].tolist()
    selected_game = st.selectbox('Select a game and our racoons will gather all available information about it for you', options)
    game = game_details[game_details['name'] == selected_game]
    game_id = game['appid'].iloc[0]

    def game_video(game_id):
        url = f"https://store.steampowered.com/api/appdetails?appids={game_id}"
        response = requests.get(url)
        if response.status_code == 200:
            # extract the relevant data from the response
            json_data = response.json()
            data = json_data[str(game_id)]['data']
            if 'movies' in data:
                new_df = pd.json_normalize(data)
                new_df1 = pd.json_normalize(new_df['movies'])
                new_df2 = pd.json_normalize(new_df1[1])
                result = new_df2.head(1)
                video_url = result["webm.480"]
                if not video_url.empty:
                    return video_url.iloc[0]
            # If "movies" field is not present, return None or a default video URL
            return None
        else:
            return None

    # Create a two-column layout
    
    col1, col2 = st.columns(2)
    # Display the game image in the first column
    with col1:
        # Display the game details (title, genre, developer, publisher, price) next to the image
        
        st.markdown(f"<p style='font-size:40px'><b>{game['name'].iloc[0]}</b></p>", unsafe_allow_html=True)
        st.subheader('**Genre:**')
        st.markdown(f"<p style='font-size:16px'><b>{game['genre'].iloc[0]}</b></p>", unsafe_allow_html=True)
        st.subheader('**Developer:**')
        st.markdown(f"<p style='font-size:16px'><b>{game['developer'].iloc[0]}</b></p>", unsafe_allow_html=True)
        st.subheader('**Publisher:**')
        st.markdown(f"<p style='font-size:16px'><b>{game['publisher'].iloc[0]}</b></p>", unsafe_allow_html=True)
        st.subheader('**Price:**')
        st.markdown(f"<p style='font-size:16px'><b>{game['price'].iloc[0]}</b>€</p>", unsafe_allow_html=True)
        st.subheader('**Description:**')
        st.markdown(f"<p style='font-size:16px'><b>{game['short_description'].iloc[0]}</b></p>", unsafe_allow_html=True)
        
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(game['header_image'].iloc[0], use_column_width=True)
        image3 = Image.open('sorry.png')
        video_url = game_video(game_id)
        if video_url is None:
            st.image(image3, use_column_width=True) 
        else:
            st.video(video_url)

    # Create a center column layout
    col3 = st.columns(1)  
    with col3[0]:
        st.write('---')

    # Create a two-column layout
    col4, col5 = st.columns(2)

    # Display additional game details (year of release, categories, languages, review ratio, rentability)
    with col4:
        st.subheader('**Game Properties:**')
        st.write('**Year of Release:**', game['release_date.date'].iloc[0])
        st.write('**Categories:**', game['categories'].iloc[0])
        st.write('**Languages:**', game['languages'].iloc[0])
        st.write('**Review Ratio:**', game['review_ration'].iloc[0], 'positive/negative reviews')
        st.write('**Rentability:**', game['rentability'].iloc[0], 'minutes/cent')

    with col5:
        st.subheader('**System Requirements:**')
        st.write('**Recommended for PC:**', game['pc_requirements.recommended'].iloc[0])
        st.write('**Recommended for Mac:**', game['mac_requirements.recommended'].iloc[0])
        st.write('**Recommended for Linux:**', game['linux_requirements.recommended'].iloc[0])


    # Define the price_data function here
    def price_data(game_id):
        url = f'https://steampricehistory.com/app/{game_id}'
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", class_="breakdown-table")
            
            if table is None:
                return "Information Unavailable"
            
            rows = table.find_all("tr")
            data = []
            
            for row in rows:
                columns = row.find_all("td")
                row_data = []
                
                for column in columns:
                    row_data.append(column.text.strip())
                
                data.append(row_data)

            columns = [col.text.strip() for col in rows[0].find_all('th')]
            df_ph = pd.DataFrame(data[1:], columns=columns)
            df_ph['Price'] = df_ph['Price'].str.replace('$', '').astype(float)

            fig, ax = plt.subplots()
            df_ph.plot(x="Date", y="Price", ax=ax)
            ax.plot(df_ph['Price']) 
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title('Price History')
            tick_positions = range(0, len(df_ph), 3)
            tick_labels = df_ph.iloc[tick_positions]["Date"]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=70)
            ax.invert_xaxis()

            fig.set_size_inches(12, 6)
            plt.subplots_adjust(bottom=0.2)

            return st.pyplot(fig)
        
        else:
            return "No data available"
    
    col6 = st.columns(1)  
    with col6[0]:
        st.subheader('**Price History:**')
        price_data(game['appid'].iloc[0])


    def players_data(game_id):
        url = f"https://steamcharts.com/app/{game_id}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", class_="common-table")
            if table is None:
                return "Information Unavailable"
            rows = table.find_all("tr")

            data = []
            for row in rows:
                columns = row.find_all("td")
                row_data = []
                for column in columns:
                    row_data.append(column.text.strip())
                data.append(row_data)

            columns = [col.text.strip() for col in rows[0].find_all('th')]
            df_plh = pd.DataFrame(data[1:], columns=columns)
            df_plh["Avg. Players"] = df_plh["Avg. Players"].astype(float)

            fig, ax = plt.subplots()
            df_plh.plot(x="Month", y="Avg. Players", ax=ax)
            ax.set_title("Average Players per Month")
            ax.set_xlabel("Month")
            ax.set_ylabel("Average Players")
            tick_positions = range(0, len(df_plh), 3)
            tick_labels = df_plh.iloc[tick_positions]["Month"]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=70)

            ax.invert_xaxis()  # invert x-axis
            fig.set_size_inches(12, 6)
            fig.subplots_adjust(bottom=0.2)

            # Compute and return summary statistics
            return fig, df_plh["Avg. Players"].min(), df_plh["Avg. Players"].max(), df_plh["Avg. Players"].mean()

        else:
            return "No data available"

    # Get game id and call players_data function
    
    fig, min_players, max_players, mean_players = players_data(game_id)

    # Display plot
    st.subheader('**Number of Players History:**')
    st.pyplot(fig)

    # Display summary statistics
    
       
    col7, col8 = st.columns(2)
    with col7:
        st.subheader('**Player Statistics:**')
        st.write("**Estimated number of owners:**", game['owners'].iloc[0])
        st.write(f"**Lowest number of players:** {int(min_players)}")
        st.write(f"**Highest number of players:** {int(max_players)}")
        st.write(f"**Mean number of players**: {int(mean_players)}")
        st.write("**Current number of online players:**",game['ccu'].iloc[0])
        st.write("**Average playthrough time:**",game['average_forever'].iloc[0])

    with col8:
        st.subheader('**Additional Information:**')
        st.write('**Age Restrictions:**', game['required_age'].iloc[0])
        st.write('**Warnings:**', game['content_descriptors.notes'].iloc[0])
        st.write('**Achievements:**', game['Achievements'].iloc[0])
        st.write('**Controller Support:**', game['controller_support'].iloc[0])
        st.write('**Subtitles:**', game['Subtitles'].iloc[0])
        st.write('**Virtual Reality:**', game['VR'].iloc[0])
        
  

    

# Define Game Recommendation page

def game_recommendation():
    st.title('Game Recommendation Tool')
    st.header('Filter your game search with your favourite game properties')
    # Create sliders and checkboxes for filtering
    cola1, cola2,cola3 = st.columns([2,1,1])
    with cola1:
        price_range = st.slider('**Price Range**', 0, 200, (0, 200), key='price_range')
        age_limit = st.slider('**Required Age**', 0, 18, 18, key='age_limit')
    with cola2:
        st.write('')
    with cola3:
        st.subheader("**System**")
        windows_platform = st.checkbox('Windows',value=True)
        mac_platform = st.checkbox('Mac')
        linux_platform = st.checkbox('Linux')
    cola6, cola7, cola8 = st.columns(3)
    with cola6:
        st.subheader("**Play Mode**")
        single_player = st.checkbox('Single-Player',value=True)
        multiplayer = st.checkbox('Multiplayer')
    with cola7:
        st.subheader("**Gadgets**")
        controller_support = st.checkbox('Controller Support')
        vr_support = st.checkbox('VR Support')
    with cola8:
        st.subheader("**App State**")
        in_app_purchases = st.checkbox('In-App Purchases')
        early_access = st.checkbox('Early Access')

    # Filter the games based on selected options
    filtered_games = game_details[(game_details['price'].between(price_range[0], price_range[1])) &
                                (game_details['required_age'] <= age_limit) &
                                (game_details['controller_support'] == controller_support) &
                                (game_details['platforms.windows'] == windows_platform) &
                                (game_details['platforms.mac'] == mac_platform) &
                                (game_details['platforms.linux'] == linux_platform) &
                                (game_details['VR'] == vr_support) &
                                (game_details['Single-player'] == single_player) &
                                (game_details['Multi-player'] == multiplayer) &
                                (game_details['In-AppPurchases'] == in_app_purchases) &
                                (game_details['Early Access'] == early_access )]
    
    st.write('---')
    st.markdown("<br>", unsafe_allow_html=True)
    # Display filtered games
    if filtered_games.empty:
        st.write('No games found that match your criteria.')
    else:
        st.write(f'{len(filtered_games)} games found that match your criteria.')
        colz, coly, colx, colw = st.columns(4)
        for i, game in enumerate(filtered_games.head(12).itertuples()):
            if i % 4 == 0:
                current_col = colz
            elif i % 4 == 1:
                current_col = coly
            elif i % 4 == 2:
                current_col = colx
            else:
                current_col = colw
            
            with current_col:
                st.image(game.header_image, width=200, use_column_width=True)
                st.write(game.name)
                




# Define About page

def about():
    st.title('About us')
    st.write('Raccoons are medium-sized mammals native to North America. They have a distinctive appearance, with a black "mask" around their eyes and a bushy tail with black and white stripes. Their fur is grayish-brown, and they have long, sharp claws that enable them to climb trees and dig for food. Raccoons are opportunistic feeders and will eat just about anything they can find, including insects, fruits, nuts, small animals, and even garbage. They are known for their cleverness and resourcefulness, and they have been observed using tools to obtain food. Raccoons are also highly adaptable and can thrive in a variety of habitats, including urban areas. However, they can be considered pests in some situations, as they may raid gardens or garbage cans in search of food. Despite their reputation as pests, **raccoons are fascinating animals** with unique behaviors and adaptations. They are often featured in folklore and popular culture, and many people find them charming and endearing')
    st.image("https://www.rd.com/wp-content/uploads/2021/04/GettyImages-1165332891-scaled.jpg")
#Define Model page

def model():
    st.title('Game Price Calculator')
    st.subheader('Want to know how much a game should be priced? Find it here!')
    st.write("Our raccoons worked hard and fabricated this model themselves. The test F-score is 0.97 and the mean squared error is only 1.75€")
    df_usar = pd.read_excel('model.xlsx')
    
    xgb_model = pickle.load(open('model.p','rb'))

    def price_prediction(input_data):
        input_data_as_numpy_array = np.asarray(input_data, dtype=object)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = xgb_model.predict(input_data_reshaped)
        return prediction[0] # return the prediction value only

    def xgb():

        col1, col2,col3,col4,col5 = st.columns(5)
        with col1:
            rage = st.text_input('Required Age',value=18)
            windows = st.text_input('Windows',value=1)
            mac = st.text_input('Mac',value=0)
            linux = st.text_input('Linux',value=0)
            controller = st.text_input('Controller Support',value=1)
            positive = st.text_input('Positive Reviews',value=50)
        with col2:
            negative = st.text_input('Negative Reviews',value=2)
            owners = st.text_input('Owners',value=20000)
            averagetime = st.text_input('Average Play Time',value=300)
            rentability = st.text_input('Rentability',value=12)
            review = st.text_input('Reviews',value=25)
            dlc = st.text_input('DLC Count',value=0)
            multiplayer = st.text_input('Massive Multiplayer',value=0)
        with col3:
            adventure = st.text_input('Adventure',value=0)
            casual = st.text_input('Casual',value=0)
            indie = st.text_input('Indie',value=0)
            action = st.text_input('Action',value=0)
            rpg = st.text_input('RPG',value=0)
            simulation = st.text_input('Simulation',value=0)
            sports = st.text_input('Sports',value=0)
        with col4:
            strategy = st.text_input('Strategy',value=0)
            early = st.text_input('Early Acess',value=0)
            violent = st.text_input('Violent',value=0)
            utilities = st.text_input('Utilities',value=0)
            design = st.text_input('Design',value=0)
            animation = st.text_input('Animation',value=0)
            video = st.text_input('Video Production',value=0)
        with col5:
            audio = st.text_input('Audio Production',value=0)
            photo = st.text_input('Photo Editing',value=0)
            web = st.text_input('Web Publishing',value=0)
            education = st.text_input('Education',value=0)
            development = st.text_input('Game Development',value=0)
            racing = st.text_input('Racing',value=0)
            accounting= st.text_input('Accounting',value=0)
            
        
        diagnosis = ''
        if st.button('Test Result'):
            try:
                # Call price_prediction() and assign its return value to prediction
                prediction = price_prediction([rage, windows, mac, linux, controller, positive, negative, owners, averagetime,
                                                rentability, review, action, adventure, casual, indie, multiplayer, rpg,
                                                simulation, sports, strategy, early, violent, utilities, design, animation,
                                                video, audio, photo, web, education, development, racing, accounting, dlc])
                diagnosis = str(prediction)
            except:
                diagnosis = 'Error: could not make prediction'

        st.success(diagnosis)

    if __name__ == '__main__':
        xgb()






# Run the app
if __name__ == '__main__':
    home()