# Movie Recommendation System

A beginner-friendly Python recommendation system that uses the real MovieLens dataset. It recommends movies based on the genres entered by the user and combines genre similarity with available user ratings.

## Dataset

Download **MovieLens Latest Small** from the official GroupLens website:

https://grouplens.org/datasets/movielens/latest/

Download `ml-latest-small.zip` (about 1 MB), then extract it. It contains `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`.

The code uses:
- `movies.csv` for movie titles and genres.
- `ratings.csv` to calculate average ratings and rating counts.

The official dataset README describes the small dataset as containing around 9,700 movies and 100,000 ratings. See:
https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

Review the dataset's README and terms of use before redistributing the data.

## Project structure

Recommended layout:

```text
movie_recommendation_system/
├── main.py
├── README.md
├── requirements.txt
└── ml-latest-small/
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv
    └── links.csv
```

The script also supports putting `movies.csv` and `ratings.csv` directly beside `main.py`.

## Requirements

- Python 3.10 or newer
- pandas
- scikit-learn

## Installation

Open a terminal in the project directory and run:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the program

```bash
python main.py
```

Enter genres when prompted, for example:

```text
Action, Adventure, Sci-Fi
```

Then enter how many recommendations you want.

The program prints the recommended movie titles, genres, average rating, rating count, and genre-match score. It also saves the results to `recommendations.csv`.

## How it works

1. Loads movie titles and genres from `movies.csv`.
2. Loads user ratings from `ratings.csv`, when available.
3. Uses TF-IDF vectorization to represent movie genres as numeric vectors.
4. Calculates cosine similarity between the user's genre preferences and each movie's genres.
5. Combines the genre match score (80%) with the average rating scaled to five stars (20%).
6. Ranks the results and saves them to a CSV file.

This is a content-based recommendation baseline, not a personalized collaborative-filtering model. The ratings component is a simple popularity/quality signal; it does not learn an individual user's rating history.

## Troubleshooting

- **`Could not find movies.csv`:** extract `ml-latest-small.zip` and put the `ml-latest-small` folder beside `main.py`, or place `movies.csv` and `ratings.csv` directly beside `main.py`.
- **No matching movies:** use genre names shown by the program, such as `Comedy`, `Drama`, `Action`, `Romance`, or `Sci-Fi`.
- **Missing ratings file:** the recommender still works from genres, but it cannot use rating information.
