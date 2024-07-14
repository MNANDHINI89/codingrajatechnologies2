Python 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9320d8d2-6129-4426-a8d3-7b594420f495",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy import stats\n",
    "from ast import literal_eval\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer\n",
    "from sklearn.metrics.pairwise import linear_kernel, cosine_similarity\n",
    "from nltk.stem.snowball import SnowballStemmer\n",
    "from nltk.stem.wordnet import WordNetLemmatizer\n",
    "from nltk.corpus import wordnet\n",
    "\n",
    "import warnings; warnings.simplefilter('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ac3822bd-2786-4987-81ff-7ea79b52d822",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Sarala\\AppData\\Local\\Temp\\ipykernel_22820\\3520735879.py:1: DtypeWarning: Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.\n",
      "  md = pd. read_csv('movies_metadata.csv')\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>adult</th>\n",
       "      <th>belongs_to_collection</th>\n",
       "      <th>budget</th>\n",
       "      <th>genres</th>\n",
       "      <th>homepage</th>\n",
       "      <th>id</th>\n",
       "      <th>imdb_id</th>\n",
       "      <th>original_language</th>\n",
       "      <th>original_title</th>\n",
       "      <th>overview</th>\n",
       "      <th>...</th>\n",
       "      <th>release_date</th>\n",
       "      <th>revenue</th>\n",
       "      <th>runtime</th>\n",
       "      <th>spoken_languages</th>\n",
       "      <th>status</th>\n",
       "      <th>tagline</th>\n",
       "      <th>title</th>\n",
       "      <th>video</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>\n",
       "      <td>30000000</td>\n",
       "      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>\n",
       "      <td>http://toystory.disney.com/toy-story</td>\n",
       "      <td>862</td>\n",
       "      <td>tt0114709</td>\n",
       "      <td>en</td>\n",
       "      <td>Toy Story</td>\n",
       "      <td>Led by Woody, Andy's toys live happily in his ...</td>\n",
       "      <td>...</td>\n",
       "      <td>1995-10-30</td>\n",
       "      <td>373554033.0</td>\n",
       "      <td>81.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Toy Story</td>\n",
       "      <td>False</td>\n",
       "      <td>7.7</td>\n",
       "      <td>5415.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>65000000</td>\n",
       "      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8844</td>\n",
       "      <td>tt0113497</td>\n",
       "      <td>en</td>\n",
       "      <td>Jumanji</td>\n",
       "      <td>When siblings Judy and Peter discover an encha...</td>\n",
       "      <td>...</td>\n",
       "      <td>1995-12-15</td>\n",
       "      <td>262797249.0</td>\n",
       "      <td>104.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>\n",
       "      <td>Released</td>\n",
       "      <td>Roll the dice and unleash the excitement!</td>\n",
       "      <td>Jumanji</td>\n",
       "      <td>False</td>\n",
       "      <td>6.9</td>\n",
       "      <td>2413.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>\n",
       "      <td>0</td>\n",
       "      <td>[{'id': 10749, 'name': 'Romance'}, {'id': 35, ...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>15602</td>\n",
       "      <td>tt0113228</td>\n",
       "      <td>en</td>\n",
       "      <td>Grumpier Old Men</td>\n",
       "      <td>A family wedding reignites the ancient feud be...</td>\n",
       "      <td>...</td>\n",
       "      <td>1995-12-22</td>\n",
       "      <td>0.0</td>\n",
       "      <td>101.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Still Yelling. Still Fighting. Still Ready for...</td>\n",
       "      <td>Grumpier Old Men</td>\n",
       "      <td>False</td>\n",
       "      <td>6.5</td>\n",
       "      <td>92.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>16000000</td>\n",
       "      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>31357</td>\n",
       "      <td>tt0114885</td>\n",
       "      <td>en</td>\n",
       "      <td>Waiting to Exhale</td>\n",
       "      <td>Cheated on, mistreated and stepped on, the wom...</td>\n",
       "      <td>...</td>\n",
       "      <td>1995-12-22</td>\n",
       "      <td>81452156.0</td>\n",
       "      <td>127.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Friends are the people who let you be yourself...</td>\n",
       "      <td>Waiting to Exhale</td>\n",
       "      <td>False</td>\n",
       "      <td>6.1</td>\n",
       "      <td>34.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>\n",
       "      <td>0</td>\n",
       "      <td>[{'id': 35, 'name': 'Comedy'}]</td>\n",
       "      <td>NaN</td>\n",
       "      <td>11862</td>\n",
       "      <td>tt0113041</td>\n",
       "      <td>en</td>\n",
       "      <td>Father of the Bride Part II</td>\n",
       "      <td>Just when George Banks has recovered from his ...</td>\n",
       "      <td>...</td>\n",
       "      <td>1995-02-10</td>\n",
       "      <td>76578911.0</td>\n",
       "      <td>106.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Just When His World Is Back To Normal... He's ...</td>\n",
       "      <td>Father of the Bride Part II</td>\n",
       "      <td>False</td>\n",
       "      <td>5.7</td>\n",
       "      <td>173.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   adult                              belongs_to_collection    budget  \\\n",
       "0  False  {'id': 10194, 'name': 'Toy Story Collection', ...  30000000   \n",
       "1  False                                                NaN  65000000   \n",
       "2  False  {'id': 119050, 'name': 'Grumpy Old Men Collect...         0   \n",
       "3  False                                                NaN  16000000   \n",
       "4  False  {'id': 96871, 'name': 'Father of the Bride Col...         0   \n",
       "\n",
       "                                              genres  \\\n",
       "0  [{'id': 16, 'name': 'Animation'}, {'id': 35, '...   \n",
       "1  [{'id': 12, 'name': 'Adventure'}, {'id': 14, '...   \n",
       "2  [{'id': 10749, 'name': 'Romance'}, {'id': 35, ...   \n",
       "3  [{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...   \n",
       "4                     [{'id': 35, 'name': 'Comedy'}]   \n",
       "\n",
       "                               homepage     id    imdb_id original_language  \\\n",
       "0  http://toystory.disney.com/toy-story    862  tt0114709                en   \n",
       "1                                   NaN   8844  tt0113497                en   \n",
       "2                                   NaN  15602  tt0113228                en   \n",
       "3                                   NaN  31357  tt0114885                en   \n",
       "4                                   NaN  11862  tt0113041                en   \n",
       "\n",
       "                original_title  \\\n",
       "0                    Toy Story   \n",
       "1                      Jumanji   \n",
       "2             Grumpier Old Men   \n",
       "3            Waiting to Exhale   \n",
       "4  Father of the Bride Part II   \n",
       "\n",
       "                                            overview  ... release_date  \\\n",
       "0  Led by Woody, Andy's toys live happily in his ...  ...   1995-10-30   \n",
       "1  When siblings Judy and Peter discover an encha...  ...   1995-12-15   \n",
       "2  A family wedding reignites the ancient feud be...  ...   1995-12-22   \n",
       "3  Cheated on, mistreated and stepped on, the wom...  ...   1995-12-22   \n",
       "4  Just when George Banks has recovered from his ...  ...   1995-02-10   \n",
       "\n",
       "       revenue runtime                                   spoken_languages  \\\n",
       "0  373554033.0    81.0           [{'iso_639_1': 'en', 'name': 'English'}]   \n",
       "1  262797249.0   104.0  [{'iso_639_1': 'en', 'name': 'English'}, {'iso...   \n",
       "2          0.0   101.0           [{'iso_639_1': 'en', 'name': 'English'}]   \n",
       "3   81452156.0   127.0           [{'iso_639_1': 'en', 'name': 'English'}]   \n",
       "4   76578911.0   106.0           [{'iso_639_1': 'en', 'name': 'English'}]   \n",
       "\n",
       "     status                                            tagline  \\\n",
       "0  Released                                                NaN   \n",
       "1  Released          Roll the dice and unleash the excitement!   \n",
       "2  Released  Still Yelling. Still Fighting. Still Ready for...   \n",
       "3  Released  Friends are the people who let you be yourself...   \n",
       "4  Released  Just When His World Is Back To Normal... He's ...   \n",
       "\n",
       "                         title  video vote_average vote_count  \n",
       "0                    Toy Story  False          7.7     5415.0  \n",
       "1                      Jumanji  False          6.9     2413.0  \n",
       "2             Grumpier Old Men  False          6.5       92.0  \n",
       "3            Waiting to Exhale  False          6.1       34.0  \n",
       "4  Father of the Bride Part II  False          5.7      173.0  \n",
       "\n",
       "[5 rows x 24 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "md = pd. read_csv('movies_metadata.csv')\n",
    "md.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b7174080-3713-44be-9d9d-611a4fd458f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b5f0ab4e-e264-482b-95db-3b70f1da20b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.244896612406511"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')\n",
    "vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')\n",
    "C = vote_averages.mean()\n",
    "C"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "be27ed80-8d70-4e04-884e-016534ce439f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "434.0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "m = vote_counts.quantile(0.95)\n",
    "m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "45f75170-0485-495b-8fb3-efb070f17233",
   "metadata": {},
   "outputs": [],
   "source": [
    "md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0129fbda-15c2-4aaa-bb97-c50e989814b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2274, 6)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]\n",
    "qualified['vote_count'] = qualified['vote_count'].astype('int')\n",
    "qualified['vote_average'] = qualified['vote_average'].astype('int')\n",
    "qualified.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cb26794d-cc4a-4cb5-ab09-3c0b905d46f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def weighted_rating(x):\n",
    "    v = x['vote_count']\n",
    "    R = x['vote_average']\n",
    "    return (v/(v+m) * R) + (m/(m+v) * C)\n",
    "qualified['wr'] = qualified.apply(weighted_rating, axis=1)\n",
    "qualified = qualified.sort_values('wr', ascending=False).head(250)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0e56bb67-79d7-4084-9c0a-f1a5102900a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>year</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>popularity</th>\n",
       "      <th>genres</th>\n",
       "      <th>wr</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>15480</th>\n",
       "      <td>Inception</td>\n",
       "      <td>2010</td>\n",
       "      <td>14075</td>\n",
       "      <td>8</td>\n",
       "      <td>29.108149</td>\n",
       "      <td>[Action, Thriller, Science Fiction, Mystery, A...</td>\n",
       "      <td>7.917588</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12481</th>\n",
       "      <td>The Dark Knight</td>\n",
       "      <td>2008</td>\n",
       "      <td>12269</td>\n",
       "      <td>8</td>\n",
       "      <td>123.167259</td>\n",
       "      <td>[Drama, Action, Crime, Thriller]</td>\n",
       "      <td>7.905871</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22879</th>\n",
       "      <td>Interstellar</td>\n",
       "      <td>2014</td>\n",
       "      <td>11187</td>\n",
       "      <td>8</td>\n",
       "      <td>32.213481</td>\n",
       "      <td>[Adventure, Drama, Science Fiction]</td>\n",
       "      <td>7.897107</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2843</th>\n",
       "      <td>Fight Club</td>\n",
       "      <td>1999</td>\n",
       "      <td>9678</td>\n",
       "      <td>8</td>\n",
       "      <td>63.869599</td>\n",
       "      <td>[Drama]</td>\n",
       "      <td>7.881753</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4863</th>\n",
       "      <td>The Lord of the Rings: The Fellowship of the Ring</td>\n",
       "      <td>2001</td>\n",
       "      <td>8892</td>\n",
       "      <td>8</td>\n",
       "      <td>32.070725</td>\n",
       "      <td>[Adventure, Fantasy, Action]</td>\n",
       "      <td>7.871787</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>292</th>\n",
       "      <td>Pulp Fiction</td>\n",
       "      <td>1994</td>\n",
       "      <td>8670</td>\n",
       "      <td>8</td>\n",
       "      <td>140.950236</td>\n",
       "      <td>[Thriller, Crime]</td>\n",
       "      <td>7.868660</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>314</th>\n",
       "      <td>The Shawshank Redemption</td>\n",
       "      <td>1994</td>\n",
       "      <td>8358</td>\n",
       "      <td>8</td>\n",
       "      <td>51.645403</td>\n",
       "      <td>[Drama, Crime]</td>\n",
       "      <td>7.864000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7000</th>\n",
       "      <td>The Lord of the Rings: The Return of the King</td>\n",
       "      <td>2003</td>\n",
       "      <td>8226</td>\n",
       "      <td>8</td>\n",
       "      <td>29.324358</td>\n",
       "      <td>[Adventure, Fantasy, Action]</td>\n",
       "      <td>7.861927</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>351</th>\n",
       "      <td>Forrest Gump</td>\n",
       "      <td>1994</td>\n",
       "      <td>8147</td>\n",
       "      <td>8</td>\n",
       "      <td>48.307194</td>\n",
       "      <td>[Comedy, Drama, Romance]</td>\n",
       "      <td>7.860656</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5814</th>\n",
       "      <td>The Lord of the Rings: The Two Towers</td>\n",
       "      <td>2002</td>\n",
       "      <td>7641</td>\n",
       "      <td>8</td>\n",
       "      <td>29.423537</td>\n",
       "      <td>[Adventure, Fantasy, Action]</td>\n",
       "      <td>7.851924</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>256</th>\n",
       "      <td>Star Wars</td>\n",
       "      <td>1977</td>\n",
       "      <td>6778</td>\n",
       "      <td>8</td>\n",
       "      <td>42.149697</td>\n",
       "      <td>[Adventure, Action, Science Fiction]</td>\n",
       "      <td>7.834205</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1225</th>\n",
       "      <td>Back to the Future</td>\n",
       "      <td>1985</td>\n",
       "      <td>6239</td>\n",
       "      <td>8</td>\n",
       "      <td>25.778509</td>\n",
       "      <td>[Adventure, Comedy, Science Fiction, Family]</td>\n",
       "      <td>7.820813</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>834</th>\n",
       "      <td>The Godfather</td>\n",
       "      <td>1972</td>\n",
       "      <td>6024</td>\n",
       "      <td>8</td>\n",
       "      <td>41.109264</td>\n",
       "      <td>[Drama, Crime]</td>\n",
       "      <td>7.814847</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1154</th>\n",
       "      <td>The Empire Strikes Back</td>\n",
       "      <td>1980</td>\n",
       "      <td>5998</td>\n",
       "      <td>8</td>\n",
       "      <td>19.470959</td>\n",
       "      <td>[Adventure, Action, Science Fiction]</td>\n",
       "      <td>7.814099</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>46</th>\n",
       "      <td>Se7en</td>\n",
       "      <td>1995</td>\n",
       "      <td>5915</td>\n",
       "      <td>8</td>\n",
       "      <td>18.45743</td>\n",
       "      <td>[Crime, Mystery, Thriller]</td>\n",
       "      <td>7.811669</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   title  year  vote_count  \\\n",
       "15480                                          Inception  2010       14075   \n",
       "12481                                    The Dark Knight  2008       12269   \n",
       "22879                                       Interstellar  2014       11187   \n",
       "2843                                          Fight Club  1999        9678   \n",
       "4863   The Lord of the Rings: The Fellowship of the Ring  2001        8892   \n",
       "292                                         Pulp Fiction  1994        8670   \n",
       "314                             The Shawshank Redemption  1994        8358   \n",
       "7000       The Lord of the Rings: The Return of the King  2003        8226   \n",
       "351                                         Forrest Gump  1994        8147   \n",
       "5814               The Lord of the Rings: The Two Towers  2002        7641   \n",
       "256                                            Star Wars  1977        6778   \n",
       "1225                                  Back to the Future  1985        6239   \n",
       "834                                        The Godfather  1972        6024   \n",
       "1154                             The Empire Strikes Back  1980        5998   \n",
       "46                                                 Se7en  1995        5915   \n",
       "\n",
       "       vote_average  popularity  \\\n",
       "15480             8   29.108149   \n",
       "12481             8  123.167259   \n",
       "22879             8   32.213481   \n",
       "2843              8   63.869599   \n",
       "4863              8   32.070725   \n",
       "292               8  140.950236   \n",
       "314               8   51.645403   \n",
       "7000              8   29.324358   \n",
       "351               8   48.307194   \n",
       "5814              8   29.423537   \n",
       "256               8   42.149697   \n",
       "1225              8   25.778509   \n",
       "834               8   41.109264   \n",
       "1154              8   19.470959   \n",
       "46                8    18.45743   \n",
       "\n",
       "                                                  genres        wr  \n",
       "15480  [Action, Thriller, Science Fiction, Mystery, A...  7.917588  \n",
       "12481                   [Drama, Action, Crime, Thriller]  7.905871  \n",
       "22879                [Adventure, Drama, Science Fiction]  7.897107  \n",
       "2843                                             [Drama]  7.881753  \n",
       "4863                        [Adventure, Fantasy, Action]  7.871787  \n",
       "292                                    [Thriller, Crime]  7.868660  \n",
       "314                                       [Drama, Crime]  7.864000  \n",
       "7000                        [Adventure, Fantasy, Action]  7.861927  \n",
       "351                             [Comedy, Drama, Romance]  7.860656  \n",
       "5814                        [Adventure, Fantasy, Action]  7.851924  \n",
       "256                 [Adventure, Action, Science Fiction]  7.834205  \n",
       "1225        [Adventure, Comedy, Science Fiction, Family]  7.820813  \n",
       "834                                       [Drama, Crime]  7.814847  \n",
       "1154                [Adventure, Action, Science Fiction]  7.814099  \n",
       "46                            [Crime, Mystery, Thriller]  7.811669  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qualified.head(15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "c882360a-5cd2-4bd8-96f8-2c19753c2b90",
   "metadata": {},
   "outputs": [],
   "source": [
    "s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)\n",
    "s.name = 'genre'\n",
    "gen_md = md.drop('genres', axis=1).join(s)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a19020c0-9a23-4dc4-acfc-a69906366524",
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_chart(genre, percentile=0.85):\n",
    "    df = gen_md[gen_md['genre'] == genre]\n",
    "    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')\n",
    "    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')\n",
    "    C = vote_averages.mean()\n",
    "    m = vote_counts.quantile(percentile)\n",
    "    \n",
    "    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]\n",
    "    qualified['vote_count'] = qualified['vote_count'].astype('int')\n",
    "    qualified['vote_average'] = qualified['vote_average'].astype('int')\n",
    "    \n",
    "    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)\n",
    "    qualified = qualified.sort_values('wr', ascending=False).head(250)\n",
    "    \n",
    "    return qualified"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "55f2df7b-ad11-4929-bdbd-03bcc8ffa652",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9097, 25)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "links_small = pd.read_csv('links_small.csv')\n",
    "links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')\n",
    "md = md.drop([8257, 8364, 8879])\n",
    "#Check EDA Notebook for how and why I got these indices.\n",
    "md['id'] = md['id'].astype('int')\n",
    "smd = md[md['id'].isin(links_small)]\n",
    "smd.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "959ae331-b60d-47cf-8c21-913cdf9ff2a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(45460, 25)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "credits = pd.read_csv('credits.csv')\n",
    "keywords = pd.read_csv('keywords.csv')\n",
    "keywords['id'] = keywords['id'].astype('int')\n",
    "credits['id'] = credits['id'].astype('int')\n",
    "md['id'] = md['id'].astype('int')\n",
    "md.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "c1147d4c-e455-4a08-a71e-12da055666e2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9217, 28)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "md = md.merge(credits, on='id')\n",
    "md = md.merge(keywords, on='id')\n",
    "smd = md[md['id'].isin(links_small)]\n",
    "smd.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f6ea9252-080a-47dd-9b21-e19c4e34cd61",
   "metadata": {},
   "outputs": [],
   "source": [
    "smd['cast'] = smd['cast'].apply(literal_eval)\n",
    "smd['crew'] = smd['crew'].apply(literal_eval)\n",
    "smd['keywords'] = smd['keywords'].apply(literal_eval)\n",
    "smd['cast_size'] = smd['cast'].apply(lambda x: len(x))\n",
    "smd['crew_size'] = smd['crew'].apply(lambda x: len(x))\n",
    "def get_director(x):\n",
    "    for i in x:\n",
    "        if i['job'] == 'Director':\n",
    "            return i['name']\n",
    "    return np.nan\n",
    "smd['director'] = smd['crew'].apply(get_director)\n",
    "smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])\n",
    "smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)\n",
    "smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "fb3fe7dc-3a34-4eb5-ac62-7cc52dd7de54",
   "metadata": {},
...    "outputs": [],
...    "source": [
...     "smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(\" \", \"\")) for i in x])\n",
...     "smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(\" \", \"\")))\n",
...     "smd['director'] = smd['director'].apply(lambda x: [x,x, x])"
...    ]
...   },
...   {
...    "cell_type": "code",
...    "execution_count": 33,
...    "id": "06797d7b-2fe6-4ecd-971c-6ad4d42b297d",
...    "metadata": {},
...    "outputs": [
...     {
...      "data": {
...       "text/plain": [
...        "keyword\n",
...        "independent film        610\n",
...        "woman director          550\n",
...        "murder                  398\n",
...        "duringcreditsstinger    327\n",
...        "based on novel          318\n",
...        "Name: count, dtype: int64"
...       ]
...      },
...      "execution_count": 33,
...      "metadata": {},
...      "output_type": "execute_result"
...     }
...    ],
...    "source": [
...     "s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)\n",
...     "s.name = 'keyword'\n",
...     "s = s.value_counts()\n",
...     "s[:5]"
...    ]
...   },
...   {
...    "cell_type": "code",
...    "execution_count": 34,
...    "id": "059fd509-c6f3-40a8-80e6-4fe6bb175a5c",
...    "metadata": {},
...    "outputs": [
...     {
...      "data": {
...       "text/plain": [
...        "'dog'"
...       ]
...      },
...      "execution_count": 34,
...      "metadata": {},
...      "output_type": "execute_result"
...     }
...    ],
...    "source": [
...     "s = s[s > 1]\n",
...     "stemmer = SnowballStemmer('english')\n",
...     "stemmer.stem('dogs')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "7b24c95a-af40-4fb6-a390-92055750a1b2",
   "metadata": {},
   "outputs": [
    {
     "ename": "InvalidParameterError",
     "evalue": "The 'min_df' parameter of CountVectorizer must be a float in the range [0.0, 1.0] or an int in the range [1, inf). Got 0 instead.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mInvalidParameterError\u001b[0m                     Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[35], line 13\u001b[0m\n\u001b[0;32m     11\u001b[0m smd[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124msoup\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m smd[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124msoup\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m.\u001b[39mapply(\u001b[38;5;28;01mlambda\u001b[39;00m x: \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin(x))\n\u001b[0;32m     12\u001b[0m count \u001b[38;5;241m=\u001b[39m CountVectorizer(analyzer\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mword\u001b[39m\u001b[38;5;124m'\u001b[39m,ngram_range\u001b[38;5;241m=\u001b[39m(\u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m2\u001b[39m),min_df\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0\u001b[39m, stop_words\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124menglish\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m---> 13\u001b[0m count_matrix \u001b[38;5;241m=\u001b[39m \u001b[43mcount\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit_transform\u001b[49m\u001b[43m(\u001b[49m\u001b[43msmd\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43msoup\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     14\u001b[0m cosine_sim \u001b[38;5;241m=\u001b[39m cosine_similarity(count_matrix, count_matrix)\n\u001b[0;32m     15\u001b[0m smd \u001b[38;5;241m=\u001b[39m smd\u001b[38;5;241m.\u001b[39mreset_index()\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:1466\u001b[0m, in \u001b[0;36m_fit_context.<locals>.decorator.<locals>.wrapper\u001b[1;34m(estimator, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1461\u001b[0m partial_fit_and_fitted \u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m   1462\u001b[0m     fit_method\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpartial_fit\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mand\u001b[39;00m _is_fitted(estimator)\n\u001b[0;32m   1463\u001b[0m )\n\u001b[0;32m   1465\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m global_skip_validation \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m partial_fit_and_fitted:\n\u001b[1;32m-> 1466\u001b[0m     \u001b[43mestimator\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_validate_params\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1468\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[0;32m   1469\u001b[0m     skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[0;32m   1470\u001b[0m         prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[0;32m   1471\u001b[0m     )\n\u001b[0;32m   1472\u001b[0m ):\n\u001b[0;32m   1473\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m fit_method(estimator, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:666\u001b[0m, in \u001b[0;36mBaseEstimator._validate_params\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    658\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_validate_params\u001b[39m(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    659\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Validate types and values of constructor parameters\u001b[39;00m\n\u001b[0;32m    660\u001b[0m \n\u001b[0;32m    661\u001b[0m \u001b[38;5;124;03m    The expected type and values must be defined in the `_parameter_constraints`\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    664\u001b[0m \u001b[38;5;124;03m    accepted constraints.\u001b[39;00m\n\u001b[0;32m    665\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 666\u001b[0m     \u001b[43mvalidate_parameter_constraints\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    667\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_parameter_constraints\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    668\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_params\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdeep\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    669\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcaller_name\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[38;5;18;43m__class__\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[38;5;18;43m__name__\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m    670\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\utils\\_param_validation.py:95\u001b[0m, in \u001b[0;36mvalidate_parameter_constraints\u001b[1;34m(parameter_constraints, params, caller_name)\u001b[0m\n\u001b[0;32m     89\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m     90\u001b[0m     constraints_str \u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m     91\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m, \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin([\u001b[38;5;28mstr\u001b[39m(c)\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mfor\u001b[39;00m\u001b[38;5;250m \u001b[39mc\u001b[38;5;250m \u001b[39m\u001b[38;5;129;01min\u001b[39;00m\u001b[38;5;250m \u001b[39mconstraints[:\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]])\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m or\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     92\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mconstraints[\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     93\u001b[0m     )\n\u001b[1;32m---> 95\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m InvalidParameterError(\n\u001b[0;32m     96\u001b[0m     \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThe \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mparam_name\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[38;5;124m parameter of \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mcaller_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m must be\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     97\u001b[0m     \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mconstraints_str\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m. Got \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mparam_val\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[38;5;124m instead.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     98\u001b[0m )\n",
      "\u001b[1;31mInvalidParameterError\u001b[0m: The 'min_df' parameter of CountVectorizer must be a float in the range [0.0, 1.0] or an int in the range [1, inf). Got 0 instead."
     ]
    }
   ],
   "source": [
    "def filter_keywords(x):\n",
    "    words = []\n",
    "    for i in x:\n",
    "        if i in s:\n",
    "            words.append(i)\n",
    "    return words\n",
    "smd['keywords'] = smd['keywords'].apply(filter_keywords)\n",
    "smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])\n",
    "smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(\" \", \"\")) for i in x])\n",
    "smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']\n",
    "smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))\n",
    "count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')\n",
    "count_matrix = count.fit_transform(smd['soup'])\n",
    "cosine_sim = cosine_similarity(count_matrix, count_matrix)\n",
    "smd = smd.reset_index()\n",
    "titles = smd['title']\n",
    "indices = pd.Series(smd.index, index=smd['title'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "6f0a7292-5731-468b-a301-626615a23d51",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'cosine_sim' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[36], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mget_recommendations\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mThe Dark Knight\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\u001b[38;5;241m.\u001b[39mhead(\u001b[38;5;241m10\u001b[39m)\n",
      "Cell \u001b[1;32mIn[26], line 6\u001b[0m, in \u001b[0;36mget_recommendations\u001b[1;34m(title)\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mget_recommendations\u001b[39m(title):\n\u001b[0;32m      5\u001b[0m     idx \u001b[38;5;241m=\u001b[39m indices[title]\n\u001b[1;32m----> 6\u001b[0m     sim_scores \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(\u001b[38;5;28menumerate\u001b[39m(\u001b[43mcosine_sim\u001b[49m[idx]))\n\u001b[0;32m      7\u001b[0m     sim_scores \u001b[38;5;241m=\u001b[39m \u001b[38;5;28msorted\u001b[39m(sim_scores, key\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mlambda\u001b[39;00m x: x[\u001b[38;5;241m1\u001b[39m], reverse\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[0;32m      8\u001b[0m     sim_scores \u001b[38;5;241m=\u001b[39m sim_scores[\u001b[38;5;241m1\u001b[39m:\u001b[38;5;241m31\u001b[39m]\n",
      "\u001b[1;31mNameError\u001b[0m: name 'cosine_sim' is not defined"
     ]
    }
   ],
   "source": [
    "get_recommendations('The Dark Knight').head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "6893855a-0402-4202-8dcf-9a2bddf6064b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def improved_recommendations(title):\n",
    "    idx = indices[title]\n",
    "    sim_scores = list(enumerate(cosine_sim[idx]))\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
    "    sim_scores = sim_scores[1:26]\n",
    "    movie_indices = [i[0] for i in sim_scores]\n",
    "    \n",
    "    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]\n",
    "    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')\n",
    "    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')\n",
    "    C = vote_averages.mean()\n",
    "    m = vote_counts.quantile(0.60)\n",
    "    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]\n",
    "    qualified['vote_count'] = qualified['vote_count'].astype('int')\n",
    "    qualified['vote_average'] = qualified['vote_average'].astype('int')\n",
    "    qualified['wr'] = qualified.apply(weighted_rating, axis=1)\n",
    "    qualified = qualified.sort_values('wr', ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "55e56deb-af8f-4f58-9b42-dbccc69b7d9f",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'cosine_sim' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[39], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mimproved_recommendations\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mThe Dark Knight\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n",
      "Cell \u001b[1;32mIn[38], line 3\u001b[0m, in \u001b[0;36mimproved_recommendations\u001b[1;34m(title)\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mimproved_recommendations\u001b[39m(title):\n\u001b[0;32m      2\u001b[0m     idx \u001b[38;5;241m=\u001b[39m indices[title]\n\u001b[1;32m----> 3\u001b[0m     sim_scores \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(\u001b[38;5;28menumerate\u001b[39m(\u001b[43mcosine_sim\u001b[49m[idx]))\n\u001b[0;32m      4\u001b[0m     sim_scores \u001b[38;5;241m=\u001b[39m \u001b[38;5;28msorted\u001b[39m(sim_scores, key\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mlambda\u001b[39;00m x: x[\u001b[38;5;241m1\u001b[39m], reverse\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[0;32m      5\u001b[0m     sim_scores \u001b[38;5;241m=\u001b[39m sim_scores[\u001b[38;5;241m1\u001b[39m:\u001b[38;5;241m26\u001b[39m]\n",
      "\u001b[1;31mNameError\u001b[0m: name 'cosine_sim' is not defined"
     ]
    }
   ],
   "source": [
    "improved_recommendations('The Dark Knight')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a187bec0-55c7-444f-ada1-5a803b22e0f1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
