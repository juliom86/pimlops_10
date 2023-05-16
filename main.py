from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = FastAPI()


@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    df = pd.read_csv('peli_mes.csv')
    d = df.loc[df.month == mes]
    d = d.original_title.to_list()
    return {'mes': mes, 'cantidad': d[0]}


@app.get('/peliculas_dia/{dia}')
def peliculas_dia(dia: str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrebaron ese dia historicamente'''
    df = pd.read_csv('peli_dia.csv')
    d = df.loc[df.day == dia]
    d = d.original_title.to_list()
    return {'dia': dia, 'cantidad': d[0]}


@app.get('/franquicia/{franquicia}')
def franquicia(franquicia: str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    df = pd.read_csv('franq.csv')
    d = df.loc[df.belongs_to_collection == franquicia]
    c = d['count'].to_list()[0]
    m = d['mean'].to_list()[0]
    s = d['sum'].to_list()[0]
    return {
        'franquicia': franquicia,
        'cantidad': c,
        'ganancia_total': s,
        'ganancia_promedio': m
    }


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais: str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    df = pd.read_csv('paispeli.csv')
    d = df.loc[df.country == pais]
    d = d.num_movies.to_list()
    return {'pais': pais, 'cantidad': d[0]}


@app.get('/productoras/{productora}')
def productoras(productora: str):
    '''Ingresas la productora, retornando la ganancia toal y la cantidad de peliculas que produjeron'''
    df = pd.read_csv('prod.csv')
    d = df.loc[df.companies == productora]
    c = d['Number'].to_list()[0]
    m = d['Average'].to_list()[0]
    s = d['Total'].to_list()[0]
    return {'productora': productora, 'ganancia_total': s, 'cantidad': c}


@app.get('/retorno/{pelicula}')
def retorno(pelicula: str):
    '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo'''
    df = pd.read_csv('peliret.csv')
    d = df.loc[df.title == pelicula]
    b = d['budget'].to_list()[0]
    g = d['revenue'].to_list()[0]
    r = d['return'].to_list()[0]
    a = d['year'].to_list()[0]
    return {
        'pelicula': pelicula,
        'inversion': b,
        'ganancia': g,
        'retorno': r,
        'anio': a
    }

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    '''Ingresasun nombre de pelicula y te recomienda las similares en una lista'''
    i = pd.read_csv("titulos.csv").iloc[:5000]
    tfidf = TfidfVectorizer(stop_words="english")
    i["overview"] = i["overview"].fillna("")

    tfidf_matriz = tfidf.fit_transform(i["overview"])
    coseno_sim = linear_kernel(tfidf_matriz, tfidf_matriz)
    
    indices = pd.Series(i.index, index=i["title"]).drop_duplicates()
    idx = indices[titulo]
    simil = list(enumerate(coseno_sim[idx]))
    simil = sorted(simil, key=lambda x: x[1], reverse=True)
    simil = simil[1:11]
    movie_index = [i[0] for i in simil]

    lista = i["title"].iloc[movie_index].to_list()[:5]
    
    return {'lista recomendada': lista}
