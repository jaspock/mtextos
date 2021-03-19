
T5.1 AutoGOAL
====================================

## Problemas en los enfoques de AutoML clásicos

Las **propuestas de Auto-ML actuales** requieren de una **previa definición de los algoritmos** que serán considerados durante el **proceso de búsqueda**. **Sistemas** con un **diseño extensible** logran **separar** la **lógica** y el conjunto de **técnicas**, haciendo al segundo fácilmente modificable. Sin embargo, **no muchos logran cumplir este objetivo**, requiriendo cambios computacionales pertinentes a la hora de añadir enfoques de machine learning a ser considerados.

**Gran parte de los sistemas** que forman el estado del arte, **interpretan** el problema del **Auto-ML** como un problema de **selección combinada de modelos** y **optimización de hiperparámetros**. Este enfoque captura naturalmente el área del aprendizaje supervisado y, a la vez, **impide resolver** directamente **tareas de otros campos** del ML como el aprendizaje no supervisado.

El **Auto-ML heterogéneo** es una nueva variante del Auto-ML general que **consiste** en la **creación de flujos de machine learning más complejos** y con **estructuras dependientes de la tarea**. Abarcando áreas como el aprendizaje supervisado y no supervisado de manera natural, el problema heterogéneo requiere la creación de **sistemas más flexibles y generales**.

## ¿Qué tipo de AutoML es AutoGOAL?

**AutoGOAL** es un sistema diseñado específicamente para **Problemas de AutoML heterogéneo**.

### ¿Por qué utilizarlo?

**Permite** a los investigadores y profesionales **desarrollar rápidamente algoritmos de referencia optimizados** en diversos problemas de aprendizaje automático.

````
>>> from autogoal.datasets import cars
>>> from autogoal.ml import AutoML

>>> X, y = cars.load()
>>> automl = AutoML()
>>> automl.fit(X, y)
````

Figura 1. Ejemplo simple de uso de AutoGOAL

Sin embargo, los sistemas **AutoML no deben intentar reemplazar a los expertos humanos**, sino **servir como herramientas complementarias** que permitan a los investigadores **obtener rápidamente mejores prototipos** y conocimientos sobre las estrategias más prometedoras en un problema concreto. Las técnicas de AutoML abren las puertas a revolucionar la forma en que se realiza la investigación y el desarrollo del aprendizaje automático en la academia y la industria.

### ¿Qué diferencia a AutoGOAL del resto de bibliotecas?

A diferencia de los enfoques de AutoML existentes, **AutoGOAL** puede **combinar técnicas y algoritmos de diferentes bibliotecas y tecnologı́as**, incluidos algoritmos de aprendizaje de **máquina clásicos**, **extracción de caracterı́sticas**, **herramientas de procesamiento de lenguaje natural** y diversas arquitecturas de **redes neuronales**.

```{image} /images/bloque3/t5/t5_autogoal_challenge.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 600px
:align: center
```

Figura 2. Donde se centra generalmente el autoML

### ¿Qué es AutoGOAL?

AutoGOAL es un **marco de Python** para la **optimización automática**, **generación** y **aprendizaje de pipelines(flujos o tuberías)** de software.

Una **pipeline** se **define**, a los efectos de AutoGOAL, como **cualquier componente de software**, ya sea una **jerarquía de clases**, un conjunto de **funciones** o cualquier **combinación** de los mismos, que trabajan juntos para resolver un problema específico.

Con AutoGOAL puede definir un pipeline de muchas formas diferentes, de modo que ciertas partes de ella sean configurables o sintonizables, y luego usar algoritmos de búsqueda para encontrar la mejor manera de ajustarla o configurarla para un problema dado.

```{image} /images/bloque3/t5/t5_autogoal_flow.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 600px
:align: center
```

Figura 3. Visión general de Autogoal

```{image} /images/bloque3/t5/t5_autogoal_example.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 500px
:align: center
```

Figura 4. Gramática probabilística

#### Funciones

- [**Optimización de caja negra:**](https://autogoal.github.io/guide/blackbox/) un optimizador de caja negra que se puede aplicar a cualquier función.
- [**Pipelines predefinidos:**](https://autogoal.github.io/guide/predefined/) pre-empaquetados con pipelines basados en marcos de aprendizaje automático populares, que puede usar en pocas líneas de código para crear canales de aprendizaje automático altamente optimizados para una amplia gama de problemas.
- [**Flujos basados en clases:**](https://autogoal.github.io/guide/cfg/) la API basada en clases le permite convertir cualquier jerarquía de clases en un espacio optimizable. Usted define clases y anota los parámetros del constructor con atributos, y AutoGOAL construye automáticamente una gramática que genera todas las instancias posibles de su jerarquía.
- [**Canalizaciones basadas en grafos:**](https://autogoal.github.io/guide/graphs) la API basada en gráficos le permite explorar espacios definidos como gráficos. La gramática de un gráfico se define como un conjunto de reglas de reescritura de gráficos, que toman nodos existentes y los reemplazan por patrones más complejos. AutoGOAL luego se transforma en un objeto evaluable, por ejemplo, una red neuronal.
- [**Pipelines funcionales:**](https://autogoal.github.io/guide/functional/) la API funcional le permite convertir cualquier código de Python que resuelva alguna tarea en un pipeline optimizable. Escribe un método regular e introduce los parámetros de AutoGOAL en el flujo de código, que luego se optimizarán automáticamente para producir la salida óptima.

## Temas que podemos tratar con AutoGOAL

- [Tema 1. AutoGOAL para la resolución de problemas de alto nivel](#tema-1-autogoal-para-la-resolucion-de-problemas-de-alto-nivel)
- [Tema 2. Beneficios de la arquitectura de AutoGOAL](#tema-2-beneficions-de-la-arquitectura-de-autogoal)
- [Tema 3. Uso de componentes](#tema-3-uso-de-componentes)

### Tema 1. AutoGOAL para la resolución de problemas de alto nivel

¿Como definimos un problema con AutoGOAL?
Es necesario definir:

- **Entrada**,
- **Salida**, y
- **Métrica a optimizar** (Función objetivo)

Véase el siguiente ejemplo:

 ````
>>> from autogoal.ml import AutoML
>>> from autogoal. datasets import haha
>>> from autogoal.kb import List , Sentence , CategoricalVector
>>> automl = AutoML (
>>>      input = List (Sentences ()), # tipos de entrada: seleccionar el tipo de dato semántico
>>>      output = CategoricalVector (), # tipo de salida: seleccionar el tipo de dato semántico
>>>      score_metric=balanced_accuracy_score # métrica a optimizar (Función objetivo): Seleccionar la métrica objetivo
>>>      )
>>> X, y = haha.load () # cargar datos del dominio especifico
>>> automl.fit(X, y) # ejecutar optimizacion
````

Figura 5. Ejemplo de código fuente para ejecutar AutoGOAL en un conjunto de datos específico, en este caso, un problema de PLN.

Podemos considerando **más parámetros**:

````
>>> from autogoal.kb import *
>>> from autogoal.ml import AutoML

>>> classifier = AutoML(
>>>    input= List(Sentence()),
>>>    output=CategoricalVector(),
>>>    score_metric=balanced_accuracy_score, #función objetivo
>>>    registry=None,
>>>    search_algorithm=PESearch,
>>>    search_iterations=args.iterations,
>>>    search_kwargs=dict(
>>>        pop_size=args.popsize,
>>>        search_timeout=args.global_timeout,
>>>        evaluation_timeout=args.timeout,
>>>        memory_limit=args.memory * 1024 ** 3,
>>>    ),
>>>    include_filter=".*",
>>>    exclude_filter=None,
>>>    validation_split=0.3,
>>>    cross_validation="median",
>>>    cross_validation_steps=3,
>>>    random_state=None,
>>>    metalearning_log=False,
>>>    errors="warn",
>>>    )
>>> automl.fit(X, y)
````

Figura 6. Ejemplo de código fuente con más parámetros.

#### Tipos de datos semánticos en AutoGOAL

Los **tipos de datos semánticos** permiten identificar la **compatibilidad** entre la **entrada** y **salida** de los **distintos algoritmos** pertenecientes a los Pipelines. De este modo es posible, a **través** de los **algoritmos transformar** un **dato en otro** y conseguir que el flujo del Pipeline no se interrumpa.
Se aplican además **jerarquías de conceptos** los cuales **proporcionan interfaces comunes** para los distintos **tipos de datos**.
En el siguiente enlace podemos encontrar las especificaciones : <https://autogoal.github.io/api/autogoal.kb/#classes>

```{image} /images/bloque3/t5/t5_autogoal_datatypes.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 600px
:align: center
```

Figura 7. Datos semánticos

Problema de ejemplo clásicos con distintos tipos de datos (texto, imágenes, clasificación básica)

- Ejemplo [UCI](https://autogoal.github.io/examples/solving_uci_datasets/)
- Ejemplo [HAHA](https://autogoal.github.io/examples/solving_haha_2019/)
- Ejemplo [MEDDOCAN](https://autogoal.github.io/examples/solving_meddocan_2019/)

#### Definición de pipeline según AutoGOAL

##### ¿Cómo saber cuando dos algoritmos son conectables?
Si **analizamos** los **entradas**(inputs) y **salidas**(outputs) de los distintos **algoritmos** **podemos saber si estos son conectables** dentro de un pipeline. [Ver tabla de algoritmos](https://autogoal.github.io/guide/algorithms/) 

<span style="color:red">[@Suilan puedes añadir alguna explicación corta aquí y un ejemplo de código?]</span>

##### ¿Qué es un algoritmo para AutoGOAL?

Un algoritmo en AutoGOAL un clase que se **define con un tipo de entrada y salida**, y **contiene un método ````run````**. Veamos el siguiente ejemplo:

````
class Algorithm():
    def __init__(self, parameter1:Categorical ("l1", "l2"), parameter2:Continuous (0.1, 10),...): 
        ...
    def run(self, input: Tuple(MatrixContinuous ,CategoricalVector)) -> CategoricalVector #método run, entrada y salida
        if self.training:
            X, y = input
            self.fit(X, y)
            return y
        else:
            return self.predict(X)
````
Figura 9. Ejemplo nuevo de algoritmo

#### Proceso de construcción del grafo de algoritmos

La siguiente imagen muestra una **explicación de alto nivel** del proceso de **construcción del grafo de algoritmos** y cómo se generan las muestras de algoritmos.

```{image} /images/bloque3/t5/t5_autogoal_process.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 600px
:align: center
```

Figura 8. Proceso de muestreo y optimización de Pipelines.

**Ejemplo de código utilizando AutoGOAL desde la clase AutoML.**

<span style="color:red">[@Suilan puedes poner aquí el ejemplo? No sé a qué te refieres]</span>

Una vez definido el problema de esta forma,  se pueden resolver problemas clásicos utilizando las herramientas y algoritmos disponibles en AutoGOAL.

<span style="color:red">[@Suilan poner un ejemplo corto si es posible]</span>

#### Integración con otras librerías: Caso de estudio Sklearn

````
class LR(sklearn.linear_model.LogisticRegression):
    def __init__ (self, penalty:Categorical ("l1", "l2"), C:Continuous (0.1, 10)):
        super().__init__(penalty = penalty, C=C)

    def run(self, input : Tuple ( MatrixContinuous ,CategoricalVector )) -> CategoricalVector
        if self.training:
            X, y = input
            self.fit(X, y)
            return y
        else:
            return self.predict(X)
````

Figura 9. Ejemplo de definición de adaptadores para algoritmos de scikitlearn.

En el siguiente enlace encontraremos documentación de ejemplo donde se **integra la librería Sklearn** en **AutoGOAL**:

- <https://autogoal.github.io/examples/sklearn_simple_grammar/>

### Tema 2. Beneficions de la arquitectura de AutoGOAL

AutoGOAL defiende una **arquitectura dividida por capas y módulos**. Cada **capa se encarga** de la **agrupación** de distintos tipos de **módulos** responsables de los siguentes aspectos:

- registro de recuros y bibliotecas
- adaptación de algortimos a la plataforma AutoGOAL
- gestión de flujos de procesos pipelines, gestión de gramáticas y la optimización de pipelines
- definición conceptual de clases y tipos semánticos que se utilizan en los procesos de automatización

```{image} /images/bloque3/t5/t5_autogoal_arq.jpg
:alt: comic xkcd 2421
:class: bg-primary mb-1
:width: 600px
:align: center
```

Figura 10. Arquitectura de AutoGOAL.

#### Módulo de Gramática

Proporciona un conjunto de anotaciones de tipo que se utilizan para definir el espacio de hiperparámetros de una técnica o algoritmo arbitrario.
Cada técnica se representa como una clase de Python, y los hiperparámetros correspondientes se representan como argumentos anotados del método ````__init__````, ya sea valores primitivos (por ejemplo, numéricos, texto, etc.) o instancias de otras clases, anotadas recursivamente. Dada una colección de clases anotadas, este módulo infiere automáticamente una gramática libre de contexto que describe el espacio de todas las instancias posibles de esas clases.

##### ¿Qué es una gramática?

Un mecanismo formal para describir una estructura jerárquica a partir   de reglas que definen como se generan subestructuras. Esto sería una gramática libre del contexto. La estructura se define recursivamente partiendo de un concepto raíz (en este caso Pipeline) que se compone recursivamente de la concatenación de otros conceptos, que a su vez  pueden estar compuestos por más conceptos. Cuando un concepto no se define en función de otros se considera un Terminal de la gramática y de lo contrario un concepto No Terminal. Lo más interesante de las gramáticas es que nos permiten representar espacios infinitos de forma finita. Se representan de la siguiente forma:

````
                    Oración:  Sujeto Predicado | Predicado
                    Sujeto:  Articulo Sustantivo | Sustantivo | Sustantivo Adjetivo
                    Predicado:  Verbo Complementos 
                    artículos: el | la | los | las …
                     ……. 
````

normalmente los **No Terminales** se representan con **mayúsculas** y los **Terminales** con **minúsculas**

````
>>> class MyClass:
         def __init__(self, x: Discrete(1,3), y: Continuous(0,1)):
             pass

>>> grammar = generate_cfg(MyClass)
>>> print(grammar)

<MyClass>   := MyClass (x=<MyClass_x>, y=<MyClass_y>)
<MyClass_x> := discrete (min=1, max=3)
<MyClass_y> := continuous (min=0, max=1)
````

Figura 11. Ejemplo de gramática a partir de una clase. Tomado de [AutoGOAL](https://autogoal.github.io/api/autogoal.grammar.generate_cfg/).



````
>>> class Pipeline(SkPipeline):
>>>    def __init__(
>>>        self,
>>>        vectorizer: Union("Vectorizer", Count, TfIdf),
>>>        decomposer: Union("Decomposer", NoDec, SVD),
>>>        classifier: Union("Classifier", LR, SVM, DT),
>>>        ):
>>>        self.vectorizer = vectorizer
>>>        self.decomposer = decomposer
>>>        self.classifier = classifier

>>>        super().__init__(
>>>            [("vect", vectorizer), ("decomp", decomposer), ("class", classifier),]
>>>        )
...
>>> grammar =  generate_cfg(Pipeline)
>>> print(grammar)

<Pipeline>      := Pipeline (vectorizer=<Vectorizer>, decomposer=<Decomposer>, classifier=<Classifier>)
<Vectorizer>    := <Count> | <TfIdf>
<Count>         := Count (ngram=<Count_ngram>)
<Count_ngram>   := discrete (min=1, max=3)
<TfIdf>         := TfIdf (ngram=<TfIdf_ngram>, use_idf=<TfIdf_use_idf>)
<TfIdf_ngram>   := discrete (min=1, max=3)
<TfIdf_use_idf> := boolean ()
<Decomposer>    := <NoDec> | <SVD>
<NoDec>         := NoDec ()
<SVD>           := SVD (n=<SVD_n>)
<SVD_n>         := discrete (min=50, max=200)
<Classifier>    := <LR> | <SVM> | <DT>
<LR>            := LR (penalty=<LR_penalty>, reg=<LR_reg>)
<LR_penalty>    := categorical (options=['l1', 'l2'])
<LR_reg>        := continuous (min=0.1, max=10)
<SVM>           := SVM (kernel=<SVM_kernel>, reg=<SVM_reg>)
<SVM_kernel>    := categorical (options=['rbf', 'linear', 'poly'])
<SVM_reg>       := continuous (min=0.1, max=10)
<DT>            := DT (criterion=<DT_criterion>)
<DT_criterion>  := categorical (options=['gini', 'entropy'])
````

Figura 12. Ejemplo de gramática a partir de un pipeline de clasificación

#### Módulo de Optimización

Proporciona estrategias de muestreo sobre una gramática libre del contexto que construye recursivamente una instancia especı́fica basada en las anotaciones. Se implementan dos estrategias de optimización:

- [búsqueda aleatoria](https://autogoal.github.io/api/autogoal.search.RandomSearch/) y
- [evolución gramatical probabilı́stica](https://autogoal.github.io/api/autogoal.search.SearchAlgorithm/)

Esta última realiza un ciclo de muestreo/actualización que selecciona las instancias de mejor rendimiento de acuerdo con alguna métrica predefinida (p.e., precisión) y actualiza iterativamente el modelo probabilístico interno del algoritmo de muestreo.

En el siguiente enlace encontraremos la documentación correspondiente: [Ver más...](https://autogoal.github.io/api/autogoal.optimize/)

#### Módulo de Flujos (Pipelines)

Proporciona una abstracción para que los algoritmos se comuniquen entre sı́ a través de un patrón Facade, es decir, la implementación de un método ````run```` con anotaciones para los tipos de entrada y salida. Las clases que implementan este patrón se conectan automáticamente en un grafo de algoritmos donde cada ruta representa un posible flujo para resolver un problema, especificado por los tipos de datos de entrada y salida.

<span style="color:red"> [@Suilan podemos poner aquí algún fragmento de código o imagen que ayude a entender este módulo?]</span>

##### ¿Cómo añadir un algoritmo nuevo a AutoGOAL? 

1. Definir una clase
2. Anotar los hiperparámetros
3. Implementar el método run
4. Pasar a la clase AutoML una lista personalizada de algoritmos. Ver figura 14 variable ````registry````. 

````
>>> @nice_repr
>>> class WikipediaSummary:  # Definición de la clase
>>>    """This class find a word in Wikipedia and return a summary in Spanish.
>>>    """
>>>    def __init__(self, sentences_count:Discrete(5,20)):  # anotación de hiperpárametros
>>>        self.sentences_count=sentences_count
>>>
>>>    def run(self, input: Word(domain='general', language='english'))-> Summary():
>>>        
>>>        try:
>>>            return wikipedia.summary(input, sentences=self.sentences_count)
>>>        except:
>>>            return ""
````

Figura 13. Ejemplo de definición de un nuevo algoritmo para extraer el resumen
de un ariculo de Wikipedia.

````
>>> from autogoal.ml import AutoML
>>> from autogoal.datasets import haha
>>> from autogoal.kb import List, Sentence, CategoricalVector
>>> automl = AutoML (
>>>    input = List (Sentences ()), # tipos de entrada
>>>    output = CategoricalVector (), # tipo de salida
>>>    score_metric=balanced_accuracy_score, # métrica a optimizar (Función objetivo)
>>>    registry=find_Clases().extend(WikipediaSummary) # añadimos a todas los algoritmos registrados. No es necesario añadir el nuevo si este se crea dentro del módulo ````contrib```` el nuevo algoritmo siempre y cuando este no.
>>>    )
>>> X, y = haha.load() # cargar datos del dominio especifico
>>> automl.fit(X, y) # ejecutar optimizacion
````
Figura 14. Incorporación del nuevo algoritmo en el proceso de AutoML

En cada implementación de un algoritmo (ahora una clase) se define un contructor con argumentos semánticamente anotados que describen los posibles rangos de valores de sus hiperparámetros.
Los argumentos pueden ser valores:

- discretos,
- continuos,
- categóricos,
- booleanos,
- instancias de alguna otra clase.

Cada argumento provee los rangos válidos para el hiperparámetro correspondiente.
Por ejemplo, anotaciones discretas y continuas definen valores máximos y mı́nimos, mientras que las categóricas presentan una lista de posibles valores. En el caso de los hiperparámetros que son instancias de otros algoritmos, AutoGOAL es capaz de encontrar el conjunto de clases válidas por los que pueden ser reemplazados.

### Tema 3. Uso de componentes

#### Definición de espacio de búsqueda en AutoGOAL

Un espacio de búsqueda es el conjunto de todos los posibles Pipelines   para resolver un problema. Son todas las posibilidades entre las que se quiere elegir y sus combinaciones. Se representa utilizando una gramática, donde el nodo inicial es Pipeline. En implementación esta gramática se puede inferir de la jerarquía de clases que se diseñe, pasando la clase que represente el concepto raíz de la gramática y se cumpla que los tipos de datos estén anotados.
Pasos para utilizar AutoGOAL

1. [Definición de espacio](#definicion-de-espacio)
2. [Función de evaluación](#funcion-de-evaluacion)
3. [Generar gramática](#generar-gramatica)
4. [Instanciación de clase hija de SearchAlgorithm](#instanciacion-de-searchalgorithm)
5. [Ejecutar Funciones](#ejecutar-funciones)

##### Definición de espacio

Para definir un espacio de búsqueda propio hay que definir clases con sus parámetros del constructor anotados, similar a lo que se hace en la sección anterior. Lo que no es necesario es el método run porque al ser definido por el usuario no tiene que cumplir con la interfaz de AutoML.

````
>>> registry.extend(find_classes()) # explicitly build the graph of pipelines
>>> space = build_pipelines(
>>>    input=List(Tuple(Sentence(), Sentence())), # límite del espacio
>>>    output=List(Sentence()), # límite del espacio
>>>    registry=registry # listado de algoritmos a considerar en el espacio de búsqueda
>>>    )
...
>>> pipe = space.sample()
>>> print(pipe)

````

Figura 15. Definición de espacio de búsqueda

##### Función de evaluación

Otro requisito es contar con una función que evalúe cada Pipeline. Esta sería la función objetivo que se desea optimizar. Lo importante es esta función reciba un Pipeline y le asigna un número que sirve para comparar diferentes pipelines.

¡!!Lo que hace AutoML es ejecutar el Pipeline en un dataset y medir error de predicción)!!! [Ver ejemplo](#tema-1-autogoal-para-la-resolucion-de-problemas-de-alto-nivel)

##### Generar gramática

Una vez que se tiene el Espacio de Búsqueda y la Función Objetivo se genera la gramática utilizando el método generate_cfg, que está en autogoal.grammar. [Ver ejemplo](#que-es-una-gramatica)

##### Instanciación de SearchAlgorithm

Luego instancias alguna clase que herede de la clase [SearchAlgorithm](https://autogoal.github.io/api/autogoal.search.SearchAlgorithm/) y le pasas la Gramática, la Función Objetivo y restricciones de tiempo y memorias.
La documentación al respecto la podéis encontrar en el siguiente enlace: <https://autogoal.github.io/examples/comparing_search_strategies/>

##### Ejecutar Funciones

De esta forma se puede utilizar AutoGOAL para resolver problemas que no sean de AutoML. Una aplicación de esta forma uso es seleccionar el mejor ensemble dado un conjunto de algoritmos, donde el concepto de mejor se define en la función Objetivo.
La documentación al respecto la podéis encontrar en el siguiente enlace: <https://autogoal.github.io/examples/comparing_search_strategies/>

## Bibliogarfía

[1] Estévez-Velarde, S., Gutiérrez, Y., Almeida-Cruz, Y., & Montoyo, A. (2021). General-purpose hierarchical optimisation of machine learning pipelines with grammatical evolution. Information Sciences, 543, 58-71.

[2] Estevez-Velarde, S., Piad-Morffis, A., Gutiérrez, Y., Montoyo, A., Muñoz, R., & Almeida-Cruz, Y. (2020). Demo Application for the AutoGOAL Framework. Association for Computational Linguistics.

[3] Estevanell-Valladares, E. L., Estevez-Velarde, S., Piad-Morffis, A., Gutiérrez, Y., Montoyo, A., & Almeida-Cruz, Y. (2021). Hacia la democratización del aprendizaje de máquinas. Revista Cubana de Transformación Digital, 2(1), 130-143.

[4] Estevez-Velarde, S., Gutiérrez, Y., Montoyo, A., & Cruz, Y. A. (2020, December). Automatic Discovery of Heterogeneous Machine Learning Pipelines: An Application to Natural Language Processing. In Proceedings of the 28th International Conference on Computational Linguistics (pp. 3558-3568).

[5] Estevez-Velarde, S., Gutiérrez, Y., Montoyo, A., & Almeida-Cruz, Y. (2019, July). AutoML strategy based on grammatical evolution: A case study about knowledge discovery from text. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 4356-4365).