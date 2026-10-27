from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("ml-latest-small")
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"


def load_data():
    movies_path = MOVIES_FILE if MOVIES_FILE.exists() else Path("movies.csv")
    ratings_path = RATINGS_FILE if RATINGS_FILE.exists() else Path("ratings.csv")

    if not movies_path.exists():
        raise FileNotFoundError(
            "Could not find movies.csv. Extract the MovieLens ZIP and place "
            "movies.csv in the project folder or keep the ml-latest-small folder."
        )

    movies = pd.read_csv(movies_path)

    if "title" not in movies.columns or "genres" not in movies.columns:
        raise ValueError("movies.csv must contain 'title' and 'genres' columns.")

    movies["genres"] = movies["genres"].fillna("").replace("(no genres listed)", "")

    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)

        if {"movieId", "rating"}.issubset(ratings.columns):
            stats = ratings.groupby("movieId")["rating"].agg(
                average_rating="mean",
                rating_count="count"
            ).reset_index()

            movies = movies.merge(stats, on="movieId", how="left")
        else:
            movies["average_rating"] = 0.0
            movies["rating_count"] = 0
    else:
        movies["average_rating"] = 0.0
        movies["rating_count"] = 0

    movies["average_rating"] = movies["average_rating"].fillna(0.0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    return movies


def build_recommender(movies):
    vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")
    genre_matrix = vectorizer.fit_transform(movies["genres"])
    return vectorizer, genre_matrix


def recommend_movies(preferences, movies, vectorizer, genre_matrix, count=10):
    user_vector = vectorizer.transform([preferences.replace(",", " ")])
    similarity_scores = cosine_similarity(user_vector, genre_matrix).flatten()

    results = movies.copy()
    results["similarity"] = similarity_scores

    max_rating = results["average_rating"].max()
    if max_rating > 0:
        results["rating_score"] = results["average_rating"] / 5.0
    else:
        results["rating_score"] = 0.0

    results["recommendation_score"] = (
        0.80 * results["similarity"] + 0.20 * results["rating_score"]
    )

    results = results[results["similarity"] > 0].copy()
    results = results.sort_values(
        ["recommendation_score", "rating_count"],
        ascending=False
    )

    return results.head(count)


def main():
    try:
        movies = load_data()
        vectorizer, genre_matrix = build_recommender(movies)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Dataset error: {error}")
        return

    print("=" * 50)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 50)
    print(f"Movies loaded: {len(movies):,}")
    print("\nAvailable genres:")
    all_genres = sorted({
        genre
        for genre_list in movies["genres"].dropna()
        for genre in genre_list.split("|")
        if genre and genre != "(no genres listed)"
    })
    print(", ".join(all_genres))

    preferences = input(
        "\nEnter your preferred genres (e.g. Action, Comedy, Sci-Fi): "
    ).strip()

    if not preferences:
        print("Please enter at least one genre or interest.")
        return

    try:
        count = int(input("How many recommendations do you want? ").strip())
        if count < 1:
            raise ValueError
    except ValueError:
        print("Invalid number. Showing 10 recommendations.")
        count = 10

    recommendations = recommend_movies(
        preferences,
        movies,
        vectorizer,
        genre_matrix,
        count
    )

    print("\nRECOMMENDED MOVIES")
    print("-" * 50)

    if recommendations.empty:
        print("No matching movies found. Try genres from the available list.")
        return

    for rank, (_, movie) in enumerate(recommendations.iterrows(), start=1):
        rating_text = (
            f"{movie['average_rating']:.1f}/5 from {movie['rating_count']} ratings"
            if movie["rating_count"] > 0
            else "No rating data available"
        )
        print(f"{rank}. {movie['title']}")
        print(f"   Genres: {movie['genres']}")
        print(f"   Rating: {rating_text}")
        print(f"   Match score: {movie['similarity']:.1%}")
        print()

    output_path = Path("recommendations.csv")
    recommendations[
        ["title", "genres", "average_rating", "rating_count", "similarity", "recommendation_score"]
    ].to_csv(output_path, index=False)
    print(f"Recommendations saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
