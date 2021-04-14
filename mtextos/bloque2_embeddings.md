
Representaciones de palabras y oraciones
========================================

Un aspecto fundamental en el procesamiento del lenguaje natural es utilizar representaciones numéricas adecuadas de los textos en los diferentes niveles (subpalabras, palabras, frases,  párrafos, etc.). En este apartado nos centraremos especialmente en las palabras y en uno de los algoritmos más utilizados para obtener dichas representaciones conocidas como *word embeddings*.

## Introducción a los embeddings de palabras

Para comenzar con el tema vamos a seguir la [guía ilustrada][guía] sobre embeddings de palabras de Jay Alammar. Esta guía describe la familia de algoritmos conocidos como *word2vec*; este conjunto de algoritmos lo forman *contextual bag of words* (CBOW) y *skip-gram*. Aquí nos centraremos en el primero, pero las ideas en las que se basa el segundo son muy similares.

[guía]: https://jalammar.github.io/illustrated-word2vec/

```{admonition} Nota
:class: note
Los algoritmos de word2vec no son la única manera de obtener embeddings de palabras. Anteriormente, ya se habían propuesto otras formas de obtenerlos.

Se puede utilizar una red neuronal *feedforward* con una capa oculta que usa a la entrada los embeddings de las *n* palabras anteriores y emite a la salida la probabilidad de la siguiente palabra. En este caso, las activaciones de la capa oculta se podrían tomar como los embeddings de dicha palabra. A diferencia de CBOW, aquí habría una capa oculta y se usarían solo las palabras anteriores, aunque es trivial adaptar la red para que use a la entrada un contexto de palabras anteriores y posteriores.

Una opción más elaborada pasa por usar una red neuronal recurrente con una capa oculta entrenada para predecir la siguiente palabra. Una vez finalizado el entrenamiento, se podrían utilizar las representaciones de estado de la capa oculta como embeddings de las palabras. A diferencia de CBOW, los embeddings aquí no son estáticos (incontextuales): dada una frase en particular, se puede ir suministrando a la red recurrente los embeddings de las palabras anteriores y obtener una representación vectorial de la siguiente palabra en el contexto anterior concreto de dicha frase.</p>
<p>Más adelante, veremos arquitecturas aún más avanzadas (como BERT) que permiten obtener embeddings contextuales más profundos.
```

De los embeddings aprendidos con word2vec se pueden obtener ciertas analogías interesantes con simples operaciones aritméticas sobre los embeddings. Así, si consideramos el embedding más cercano al vector resultante de calcular embedding("France") - embedding("Paris") + embedding("Rome"), este resulta ser el correspondiente a "Italy". Otras analogías interesantes se pueden ver en la figura {numref}`analogia-word2vec`.

```{figure} images/mikolov-word2vec-analogies.png
---
height: 400px
name: analogia-word2vec
---
Analogías mostradas en el trabajo de [Mikolov et al.][mikolov2013] de 2013.

[mikolov2013]: https://arxiv.org/pdf/1301.3781.pdf
```



## Ecuaciones del modelo CBOW

Las ecuaciones que conforman el modelo CBOW están descritas en las secciones 4.1. y 4.2 de estas [notas de clase][notas] del curso [CS224n][cs224] (Natural Language Processing with Deep
Learning) de Stanford.

[notas]: https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf
[cs224]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/

### Análisis de las ecuaciones

Resumiendo lo que allí se explica, una vez fijado el vocabulario, tenemos dos embeddings diferentes para cada palabra. Estos embeddings se agrupan en las matrices $\cal{V}$ (para los embeddings de entrada) y $\cal{U}$ (embeddings de salida) en las que la fila $i$-esima o la columna $i$-ésima, respectivamente, son la representación vectorial de la palabra $i$-esima del vocabulario; esta diferencia en la organización de los valores por filas o columnas simplemente sirve para que los productos matriciales sean más cómodos de realizar, pero no tiene ninguna otra justificación. La representación de la palabra $i$-ésima según la matriz considerada se denota por $\boldsymbol{v}_i$ o bien $\boldsymbol{u}_i$. La otra dimensión de estas dos matrices es el tamaño de los embeddings $n$.

El modelo se entrena para predecir la palabra central en base a las palabras del contexto (por ejemplo, en base a las dos palabras a su izquierda y a las dos palabras a su derecha). Todas las palabras del contexto se representan con un único vector de tamaño $n$, denotado por $\hat{\boldsymbol{v}}$, resultado de promediar los embeddings correspondientes a cada una de ellas según $\cal{V}$. A continuación, el algoritmo CBOW calcula el producto $\boldsymbol{z} = \hat{\boldsymbol{v}} \, \cal{U}$. El resultado de esta multiplicación, $\boldsymbol{z}$, es un vector de tamaño $n$ donde el elemento $i$-ésimo es el producto escalar del vector representante del contexto de entrada $\hat{\boldsymbol{v}}$ y el embedding de la palabra $i$-ésima según $\cal{U}$. Es importante destacar que el producto escalar puede usarse como una medida de [similitud entre dos vectores][similitud]. Por ello, si $\boldsymbol{z}_i > \boldsymbol{z}_j$, podemos concluir que el embedding promediado que representa el contexto de la entrada es más parecido al embedding de salida (según la matriz $\cal{U}$) de la palabra $i$-ésima que al de la palabra $j$-ésima. Para poder interpretar los valores de $\boldsymbol{z}$ como probabilidades, solo queda normalizarlos mediante la función softmax para obtener $\hat{\boldsymbol{y}}$. A los valores de $\boldsymbol{z}$, previos a la normalización, se les conoce normalmente como *logits*.

[similitud]: https://math.stackexchange.com/a/689078

La función de error compara la salida producida por la red con el vector one-hot correspondiente a la palabra central. Para que este error sea lo menor posible, el elemento de $\hat{\boldsymbol{y}}$ correspondiente al índice de la palabra central ha de ser lo mayor posible, y esto solo se consigue si el vector representante de la entrada $\hat{\boldsymbol{v}}$ (y, por ende, los embeddings de las palabras del contexto) es parecido al embedding de salida de la palabra central. Vemos cómo de esta manera los embeddings de las palabras del contexto y de la palabra central se irán acercando durante el entrenamiento en ambas matrices.

Dado que la salida deseada $\boldsymbol{y}$ es un vector one-hot, si asumimos que la palabra central tiene por índice $c$, la entropia cruzada que define la función de pérdida entre este vector y la salida de la red es simplemente $J = - \text{log}(\hat{y}_c)$, que se puede expresar de forma que dependa de los embeddings contextuales de entrada y del embedding de salida $\boldsymbol{u}_c$ (para poder así calcular las derivadas parciales y actualizar esos embeddings mediante descenso por gradiente) sustituyendo $\hat{\boldsymbol{y}}_c$ y aplicando estas propiedades de los logaritmos:

$$
\log a/b &=& \log a - \log b \\[1.5ex]
\log e^a &=& a
$$

Tras el entrenamiento del modelo CBOW se suelen considerar los embeddings de $\cal{V}$ como los representantes de las palabras a usar en la tarea de aprendizaje principal, aunque en principio también podrían usarse los de $\cal{U}$.


```{admonition} Nota
:class: note
Es recomendable que amplíes tus conocimientos sobre la [entropia][entropía] y la [entropía cruzada][cruzada] siguiendo estos enlaces, ya que son conceptos fundamentales en aprendizaje automático, en general, y procesamiento del lenguaje natural, en particular.

[entropía]: https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
[cruzada]: https://datascience.stackexchange.com/a/20301
```

```{admonition} Nota
:class: note
Dado que en la función de error necesitamos calcular el logaritmo de una probabilidad, es habitual usar al final de la red neuronal la función log_softmax en lugar de softmax, que aunque conceptualmente podemos ver como la misma función con la aplicación final de un logaritmo, tiene una implementación más eficiente y numericamente estable.
```

## Visualización de embeddings

Mediante la herramienta [Embedding Projector][projector] vamos a explorar visualmente la distribución de las palabras en el espacio de embeddings. 

[projector]: https://projector.tensorflow.org/

La herramienta tiene varios paneles:

- El panel de datos en el que seleccionamos los datos a examinar; nos vamos a centrar en el conjunto *Word2Vec All* (embeddings de dimensión 200 para unas 71000 palabras).
- El panel de proyección en el que se selecciona la técnica de reducción de dimensionalidad utilizada para representar los datos en un espacio de bi o tridimensional; podemos empezar por seleccionar PCA (*principal component analysis*), que es más rápida.
- El panel de visualización en el que se muestran los embeddings.
- El panel de inspección en el que podemos introducir una palabra y ver la lista de sus vecinos más cercanos.

Para visualizar sesgos en las distribuciones podemos ir al panel de proyección y en *Custom* elegir el embedding de una palabra para el lado izquierdo y otro para el derecho. En el panel de *bookmarks* (abajo a la derecha) puedes encontrar un ejemplo ya creado. La expresión regular para una palabra exacta es /^bad$/, por ejemplo.

## Colecciones de embeddings para palabras y frases

Existen colecciones como [fastText][fasttext], que incluyen embeddings para dos millones de palabras en inglés (obtenidos procesando un corpus de 16.000 millones de palabras) o embeddings multilingües para más de cien idiomas. Aunque las representaciones a nivel de frase se pueden obtener mediante la integración de las representaciones individuales de las palabras, hay sistemas más avanzados como [LASER][laser] que ofrecen un codificador neuronal capaz de emitir embeddings de frases en 93 idiomas (23 alfabetos diferentes).

[fasttext]: https://fasttext.cc/
[laser]: https://github.com/facebookresearch/LASER
