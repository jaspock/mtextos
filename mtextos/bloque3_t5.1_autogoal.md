
T5.1 AutoGOAL
====================================

```{image} /images/bloque3/t5/t5_autogoal_challenge.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 500px
:align: center
```


```{image} /images/bloque3/t5/t5_autogoal_example.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 500px
:align: center
```


```{image} /images/bloque3/t5/t5_autogoal_flow.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 500px
:align: center
```

```{image} /images/bloque3/t5/t5_autogoal_arq.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 500px
:align: center
```

```{image} /images/bloque3/t5/t5_autogoal_datatypes.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 500px
:align: center
```

````
>>> from autogoal .ml import AutoML
>>> from autogoal . datasets import haha
>>> from autogoal .kb import List , Sentence , CategoricalVector
>>> automl = AutoML (
      input = List (Sentences ()), # **tipos de entrada**
      output = CategoricalVector (), # **tipo de salida**
      score_metric=balanced_accuracy_score # **métrica a optimizar (Función objetivo)**
    )
>>> X, y = haha . load () # cargar datos del dominio especifico
>>> automl.fit(X, y) # ejecutar optimizacion
````
Ejemplo de cóodigo fuente para ejecutar AutoGOAL en un
conjunto de datos específico, en este caso, un problema de PLN.


````
>>> from autogoal.kb import *
>>> from autogoal.ml import AutoML

>>> classifier = AutoML(
    input= List(Sentence()),
    output=CategoricalVector(),
    score_metric=balanced_accuracy_score,   
    registry=None,
    search_algorithm=PESearch,
    search_iterations=args.iterations,
    search_kwargs=dict(
        pop_size=args.popsize,
        search_timeout=args.global_timeout,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * 1024 ** 3,
    ),
    include_filter=".*",
    exclude_filter=None,
    validation_split=0.3,
    cross_validation="median",
    cross_validation_steps=3,
    random_state=None,
    metalearning_log=False,
    errors="warn",
)
>>> automl.fit(X, y)

````
Ejemplo de código fuente con más parámetros

````
class LR( sklearn . linear_model . LogisticRegression ):
def __init__ (
self ,
penalty : Categorical ("l1", "l2"),
C: Continuous (0.1 , 10)
):
super (). __init__ ( penalty = penalty , C=C)
def run(self , input : Tuple ( MatrixContinuous ,
CategoricalVector ))
-> CategoricalVector :
if self . training :
X, y = input
self . fit(X, y)
return y
else :
return self . predict (X)
````
Ejemplo de definición de adaptadores para algoritmos de scikitlearn.


````
class Word2VecEmbedding :
def __init__ ( self ):
# cargar modelo word2vec de la API de gensim
self . model = gensim . downloader . load ("glove - twitter -25")
def run(self , input : Word ) -> ContinuousVector :
try :
return self . model . get_vector ( input )
except :
return np. zeros (25)
````
Figura: Ejemplo de denicióon de un componente para el coomputo de
vectores de embedding.

````
class WikipediaSummary :
def run(self , input : Word ) -> Summary :
try :
return wikipedia . summary ( input )
except :
return ""
````
Figura: Ejemplo de denicion de un componente para extraer el resumen
de un ariculo de Wikipedia.
