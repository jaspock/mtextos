
Práctica. Lectura del código de un detector de entidades nombradas
==================================================================

En esta práctica leerás el código de un programa escrito por otros y lo documentarás para demostrar que entiendes lo que hace. La documentación resultante es por lo que se te evaluará.

Desde diversos frentes se [defiende][defiende] la lectura de código escrito por otros como un enfoque adecuado para aprender a programar o mejorar las habilidades como programador. Esta asignatura incluye algunas prácticas en las que utilizarás modelos preentrenados para ciertas tareas de procesamiento del lenguaje natural, pero tanto la programación de los modelos subyacentes como el procesamiento de los datos quedarán normalmente ocultos tras la interfaz de los programas correspondientes. Partiendo de la premisa de que entender cómo funcionan las cosas ayuda a sacarles provecho y dado que la asignatura no incluye entre sus objetivos la escritura desde cero de un programa completo, esta práctica propone un acercamiento intermedio al estudiar el código escrito por terceros como preparación necesaria para poder más adelante crear tus propios programas o modificar los existentes. En el desarrollo de la profesión de científico de datos te encontrarás en situaciones en las que no existe un modelo que resuelva exactamente tu problema y deberás plantearte modificar el funcionamiento de un sistema base.

[defiende]: https://www.stevejgordon.co.uk/become-a-better-developer-by-reading-source-code

Algunos estudios asemejan la labor de leer código a la de leer un texto escrito en lenguaje natural; otros lo equiparan más bien al procesamiento mental de ecuaciones matemáticas; otros [estudios][estudios] sugieren que el cerebro usa mecanismo distintos para la asimilación del código fuente.

[estudios]: https://www.sciencedaily.com/releases/2020/12/201215131236.htm


## Objetivo

En esta práctica vas a estudiar un programa que permite entrenar y probar un sistema neuronal de reconocimiento de entidades nombradas. El programa se compone de varios módulos escritos en Python y que usan la librería Pytorch para la programación del modelo neuronal. El código está en [esta carpeta][ner] de un repositorio que tendrás que clonar. A efectos de reconocimiento de la autoría, el código está tomado del que utilizan como inspiración para sus proyectos los estudiantes del curso [CS230][cs230], «Deep Learning», de la Universidad de Stanford.

[cs230]: https://github.com/cs230-stanford/cs230-code-examples/
[ner]: https://github.com/jaspock/cs230-code-examples/tree/master/pytorch/nlp

El código original tiene algunos comentarios que puedes traducir al español o ignorar, pero tienes que añadir muchos más para explicar con cierto nivel de detalle qué hacen las principales líneas del código. 

Para hacerte una idea de cómo funciona el reconocedor de entidades nombradas, ejecuta en primer lugar el programa tal y como se explica en el README saltando la parte opcional que descarga datos de Kaggle.

## Comentarios en Python

Los comentarios en Python pueden ser de [dos tipos][tipos] principalmente: comentarios de tipo *docstring* (que pueden aparecer en cualquier lugar del código, aunque suelen colocarse justo después de la definición de una clase o una función para documentarlas) y comentarios de una línea. Los primeros aparecen rodeados por una secuencia de tres comillas al principio y al final, y pueden ocupar más de una línea; los segundos se extienden desde el carácter de la almohadilla hasta el final de la línea. Dado que el propósito de esta práctica es escribir una documentación elaborada, es recomendable que uses más los comentarios de tipo *docstring*.

[tipos]: https://realpython.com/documenting-python-code/

## Uso de Pycco para generar la documentación

En lugar de presentar el código fuente documentado, vas a generar una vista en HTML de los comentarios y el código que facilita la lectura. Para ello, vas a usar la herramienta [Pycco][pycco]. Para aprender sobre su funcionamiento basta con estudiar un pequeño [tutorial][tutorial]. Estudia también uno de los ficheros del código fuente de Pycco (que se usa a sí mismo para generar su documentación) como, por ejemplo, [main.py][pyccomain], y observa después como se muestran sus comentarios en la [página web generada][pyccoejemplo]. Como ves, se puede utilizar *markdown* para dar un formato sencillo al texto de los comentarios.

Para generar en la carpeta *docs* la documentación en HTML del código del detector de entidades nombradas:

```{code-block} python
  pip install pycco
  pycco --generate-index *.py model/*.py
```

[pycco]: https://github.com/pycco-docs/pycco
[pyccomain]: https://github.com/pycco-docs/pycco/blob/master/pycco/main.py
[pyccoejemplo]: https://pycco-docs.github.io/pycco/
[tutorial]: https://realpython.com/generating-code-documentation-with-pycco/

## Orden sugerido para el estudio del código

PENDIENTE DE FINALIZACIÓN

