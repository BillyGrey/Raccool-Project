# Raccool-Project
Raccol is a Web App designed to help, optimize and organize the game searching experience. It was my Final Project for the Data Analytics Course at IronHack and won Best Project!

In this project, I used 5 API endpoints and did 2 Web Scrapes in order to colect and organize all the information about videogames: 
- Name, Genre, Publisher, Developer, Price and Description
- Year of Release, Categories, Languages and System Requirements (both for Windows, Mac and Linux)
- Price history (since early 2014) (in a plot format)
- Number of Players history (since early 2012) (in a plot format)
- Estimated number of owners, Lowest & Highest number of Players, Mean number of players, Current number of online players and Average Playthrough Time
- Additional Information (Age Restrictions, Warnings, Achievements, Controller Support, Subtitles, Virtual Reality)

With some Machine Learning (XGBoost), I also optimized a model that would predict the price of game when given its properties (with a F-Score of 0,97 and root mean square error of only 1,75€)

Then, I created a database on SQL to store all the important data I had collected, cleaned and processed;

In Visual Studio Code, with Streamlit, I created the Web App that allowed the user to interact with the previous constructed database and either search for a game, get a game recommendation or calculate the price of a game with certain characteristics.
