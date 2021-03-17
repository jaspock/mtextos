
La arquitectura transformer
===========================

En 2017 las redes neuronales recurrentes basadas en unidades LSTM como las que hemos estudiado eran la arquitectura habitual para el procesamiento neuronal de secuencias, en general, y del lenguaje natural, en particular. Algunos investigadores comenzaban a obtener también buenos resultados en esta área con las redes neuronales convolucionales, tradicionalmente empleadas con imágenes. Por otro lado, los mecanismos de atención introducidos unos años antes en las redes recurrentes habían mejorado su capacidad para resolver ciertas tareas y abierto el abánico de posibilidades de estos modelos. Además, el modelo conocido como codificador-descodificador (*encoder-decoder* en inglés) se convertía en la piedra angular de los sistemas que transformaban una secuencia en otra (sistemas conocidos como *seq2seq* como, por ejemplo, los sistemas de traducción automática o de obtención de resúmenes). A mediados de 2017, sin embargo, aparece un artículo {cite}`allyouneed` que propone eliminar la recurrencia del modelo codificador-descodificador y sustituirla por lo que se denomina autoatención (*self-attention*); aunque el artículo se centra en la tarea de la traducción automática, en muy poco tiempo la aplicación de esta arquitectura, bautizada como *transformer*, en muchos otros campos se descubre altamente eficaz hasta el punto de relegar a las arquitecturas recurrentes a un segundo plano. El transformer sería, además, uno de los elementos fundamentales de los modelos preentrenados que estudiaremos más adelante y que comenzarían a aparecer en los meses o años siguientes.

En este apartado comenzaremos estudiando la arquitectura codificador-decodificador y el mecanismo de atención inicial propuesto para ellas, para pasar después a estudiar el transformer y su mecanismo de autoatención.

```{admonition} Nota
:class: note
Aunque el énfasis en esta asignatura no está en la traducción automática, los desarrollos que vamos a comentar surgieron inicialmente en esta tarea, por lo que la discusión girará en torno a esta aplicación concreta. En posteriores apartados veremos cómo los modelos pueden adaptarse sin excesivas modificaciones a otras tareas más específicas de la minería de textos.
```

## Arquitectura codificador-descodificador sobre redes reucrrentes y mecanismo de atención

Para comenzar con el tema, vamos a seguir la guía ilustrada de Jay Alammar sobre los modelos [seq2seq con atención][seq2seq]. Esta guía aborda inicialmente el estudio de la arquitectura codificador-descodificador sobre redes neuronales recurrentes y describe entonces en la última parte el mecanismo de atención.

[seq2seq]: https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/


## La arquitectura transformer

Introducida la arquitectura *seq2seq*, usaremos en este bloque otra guía ilustrada de Jay Alammar para presentar este vez el [transformer][transformer].

[transformer]: http://jalammar.github.io/illustrated-transformer/

Las representaciones aprendidas tras el entrenamiento por un transformer en cada una de sus capas para una nueva frase de entrada pueden considerarse (de la misma manera que con una red recurrente) como embeddings contextuales de los diferentes tokens de la entrada que pueden usarse a la hora de representarlos en otras tareas. En principio, cualquier capa puede ser adecuada para obtener estas representaciones, pero algunos trabajos han demostrado que ciertas capas son más adecuadas que otras para ciertas tareas. Las capas más cercanas a la entrada parecen representar información más relacionada con la morfología, mientras que las capas finales se relacionan más con la semántica.

### Las diferentes caras de la atención

En el siguiente análisis nos basaremos en la [discusión][dontloo] de *dontloo* en Cross Validated. Hay tres conceptos clave en el mecanismo de atención para cuya explicación se puede hacer un símil con los conceptos homónimos en los sistemas de extracción de información:

[dontloo]: https://stats.stackexchange.com/a/424127/240809

- las *consultas*, que serían equivalentes a los términos a buscar;
- las *claves*, que serían los valores del campo (por ejemplo, título) sobre el que se realiza la búsqueda;
- los *valores*, que serían aquello devuelto finalmente por el motor de búsqueda.

En el sistema *seq2seq*, la atención es una media ponderada de las claves:

$$
\boldsymbol{c} = \sum_{j} \alpha_j \boldsymbol{h}_j   \qquad \mathrm{con} \,\, \sum_j \alpha_j = 1
$$

Si $\alpha$ fuera un vector one-hot, la atención se reduciría a recuperar aquel elemento de entre los distintos $\boldsymbol{h}_j$ en base al correspondiente índice; pero sabemos que $\alpha$ difícilmente será un vector unitario, por lo que se tratará más bien de una recuperación ponderada. En este caso, $\boldsymbol{c}$ puede considerarse como el valor resultante.

Hay una diferencia importante en cómo este vector de pesos con suma 1 se obtiene en las arquitecturas de *seq2seq* y la del *transformer*. En el primer caso, se usa una red neuronal *feedforward*, representada mediante la función $a$, que determina la *compatibilidad* entre la representación del token $i$-ésimo del descodificador $\boldsymbol{s}_i$ y la representación del token $j$-ésimo del codificador $\boldsymbol{h}_j$:

$$
e_{ij} = a(\boldsymbol{s}_i,\boldsymbol{h}_j)
$$

y de aquí:

$$
\alpha_{ij} = \frac{\mathrm{exp}(e_{ij})}{\sum_k \mathrm{exp}(e_{ik})}
$$

Supongamos que la longitud de la secuencia de entrada es $m$ y la de la salida generada hasta esete momento es $n$. Un problema de este enfoque es que en cada paso del descodificador es necesario pasar por la red neuronal $a$ un total de $mn$ veces para computar todos los $e_{ij}$.

Existe una estrategia más eficiente que pasa por proyectar los $\boldsymbol{s}_i$ y los $\boldsymbol{h}_j$ a un espacio común (mediante, por ejemplo, sendas transformaciones lineales de una capa, $f$ y $g$) y usar entonces una medida de similitud (como el producto escalar) para obtener la puntuación $e_{ij}$:

$$
e_{ij} = f(\boldsymbol{s}_i) \cdot g(\boldsymbol{h}_j)^T
$$

Podemos considerar que el vector de proyección $f(\boldsymbol{s}_i)$ es la consulta realizada por el descodificador y el vector de proyección $g(\boldsymbol{h}_j)$ es la clave proveniente del descodificador. Ahora solo es necesario realizar $n$ llamadas a $f$ y $m$ llamadas a $g$, con lo que hemos reducido la complejidad a $m+n$. Además, hemos conseguido que los $e_{ij}$ puedan calcularse eficientemente mediante producto de matrices.

El mecanismo de atención del transformer establece las condiciones para que las proyecciones de consultas y claves y el cálculo de la similitud se puedan llevar a cabo. Cuando la atención se realiza desde y hacia vectores con el mismo origen (por ejemplo, dentro del codificador) se denomina *autoatención*. El transformer combina autoatención separada en codificador y descodificador con el otro mecanismo de atención *heterogénea* en el que $Q$ viene del descodificador y $K$ y $V$ vienen del codificador.

La ecuación básica del transformer es esta:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

En realidad, las ecuaciones finales del transformer son ligeramente más complejas que esta al considerar los múltiples cabezales.

## Visualización de embeddings contextuales

Mediante la herramienta [exBERT][exbert] vamos a explorar visualmente las representaciones intermedias obtenidas en el codificador de un transformer. 

[exbert]: https://huggingface.co/exbert/?model=bart-large&modelKind=bidirectional&sentence=The%20moon%20is%20shinning%20brightly%20tonight.

En la [herramienta] puedes seleccionar el modelo a utilizar (en estos momentos no hemos estudiado las diferencias entre ellas, por lo que con *BART* es suficiente), la frase de entrada, el grado de la atención, las capas y los cabezales a mostrar. La capa superior (la más alejada de la entrada) está a la izquierda. Para seleccionar o deseleccionar un cabezal puedes hacer clic en las columnas. Puedes seleccionar un token haciendo clic sobre él y ocultarlo con doble clic. Al colocarte sobre una palabra puedes ver la predicción que haría el modelo del token que corresponde al embedding obtenido por la red en esa posición, capa y cabezal; observa que si ocultas *moon* a la derecha, por ejemplo, a la izquierda el sistema tiende a predecir *sun* en esa posición.

[herramienta]: https://www.youtube.com/watch?v=e31oyfo_thY


## Para saber más

"[The annotated transformer][annotated]" es un documento que va mostrando paso a paso el artículo científico original del transformer y su *traslación* a código en Python. Este material es opcional y de una complejidad superior a la requerida en la asignatura, pero es la mejor manera de entender la arquitectura a bajo nivel.

[annotated]: https://nlp.seas.harvard.edu/2018/04/03/attention.html


## Referencias

```{bibliography} bloque2_transformer.bib
```
