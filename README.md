# [Spytify](https://spytify.streamlit.app/)

## Introduction

This README provides a detailed description of the internal workings of the Spytify page, as well as a step-by-step guide on how to use its various functionalities.

## Process Description

The Spytify page is designed to help you discover and explore new music in a personalized way. Below is a detailed explanation of each step of the process:

### 1. User Input

#### Step 1: Enter Preferences

Fill out the form with your musical preferences, including the type of query (song or artist), the query itself, the desired number of results, and the musical genres of interest.

#### Step 2: Processing Preferences

Once you submit your preferences, the page will process the information and make calls to the Spotify API using the Spotipy library. This will generate two sets of data: found songs and recommended playlists + songs.

#### Step 3: Results

Explore the results of your search, including the found songs and the recommended playlists based on your selected musical genres.

### 2. Clustering Recommendations

Once the results are obtained, the clustering model with DBSCAN is launched. This algorithm clusters songs based on their local density, providing more accurate and personalized results.

### 3. Exploratory Data Analysis

This section showcases visualizations created using the Plotly package to analyze playlist and song data.

### 4. Playlist Export

After the clustering process, users have the option to create a personalized playlist by combining their selected songs with those from the majority group in the clustering results.

## How to Use the Page

To use the Spytify page, follow these steps:

1. Complete the user input form with your musical preferences.
2. Explore the results of your search and select the songs for the clustering process.
3. Configure the export of your personalized playlist.
4. Enjoy your new playlist on Spotify.

## Contact

If you have any questions or comments about the operation of the Spytify page, feel free to contact us.

---
# Our Amazing Development Team! :rocket:

Hey there! We're a passionate team of developers working together to reach new heights in the tech world :earth_americas:. Focused on innovation and performance, let us introduce you to our awesome team! :computer:

## José Dos Reis

- LinkedIn: [José Angel Dos Reis Zapata](https://www.linkedin.com/in/jose-dosreis/?locale=en_US)
- GitHub: [José Dos Reis - josedosr](https://github.com/josedosr)
- Email: [josedosr@hotmail.com](mailto:josedosr@hotmail.com)

José is our coding rockstar!🤘🏽 With undeniable talent and skills, top-notch ideas and a passion for challenges makes him one of our team's keys to success :key:. He brings wonderful energy and never gives up!:sparkles:
                
## Patricia García

- LinkedIn: [Patricia G-R Palombi](https://www.linkedin.com/in/patricia-g-r-palombi-269b78183/)
- GitHub: [Name on GitHub](link_to_github_profile_2)
- Email: [patgace@gmail.com](mailto:patgace@gmail.com)
                
Patricia is our wizard of creativity, transforming our ideas with vibrant colors and captivating design, making our team stand out in style! :art::rainbow:

## Pamela Colman

- LinkedIn: [Pamela J. Colman V.](https://www.linkedin.com/in/pamela-j-colman-v/)
- GitHub: [Pamela Colman - pamve](https://github.com/pamve)
- Email: [pamvecol@gmail.com](mailto:pamvecol@gmail.com)
                
Pamela is our data guru, providing brilliant insights and in-depth analyses that guide our team's success, turning data into impactful decisions! :chart_with_upwards_trend::bulb:

## Daniel Muñoz

- LinkedIn: [Daniel Muñoz Monte](https://www.linkedin.com/in/dmunoz-m/)
- GitHub: [Daniel Muñoz - devmunoz](https://github.com/devmunoz)
- Email: [daniel.munoz.monte@gmail.com](mailto:daniel.munoz.monte@gmail.com)

Daniel is our web development maestro, with exceptional experience, fearlessness to tweak any code, and brilliant ideas that elevate our team to the next level! :computer::rocket:
