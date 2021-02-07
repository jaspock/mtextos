
Materiales de Minería de Textos
===============================

Universitat d'Alacant, curso 2020–2021
--------------------------------------

*«This is an exciting time to be working in speech and language processing», Daniel Jurafsky, James H. Martin, 2009*

*«You shall know a word by the company it keeps», John Rupert Firth, 1957*


% comentario
 
Novedades
---------

`````{list-table}
:header-rows: 0
:widths: 10 90
:class: tablita

* - 23 Ene 
  - El curso comienza el día 10 de febrero. Por cuestiones organizativas, habrá algunas pequeñas diferencias entre el tipo de sesión (teórica o práctica) que figura en la ficha de la asignatura y la que se impartirá en cada momento. El número total de horas de teoría y prácticas impartido a final de curso se corresponderá en cualquier caso con el correcto.
* - 22 Ene 
  - Se ha publicado la primera versión de los materiales de la asignatura. Estos materiales pueden pueden ir cambiando antes de la clase en la que se impartan.

`````

Guía docente y normas del curso
-------------------------------

Estos son los materiales de clase de la asignatura Minería de Textos, coordinada por el profesor [Juan Antonio Pérez Ortiz][japerez_url] ([@japer3z][japerez_twitter]) de la Universitat d'Alacant e impartida también por los profesores [Francisco de Borja Navarro Colorado][borja url] y [Yoan Gutiérrez Vázquez][yoan url]. 

Para obtener información sobre la evaluación de la asignatura puedes consultar la [guía docente][guía]. Algunos aspectos adicionales que no están recogidos en la guía son los siguientes:

[japerez_url]: https://cvnet.cpd.ua.es/curriculum-breve/es/perez-ortiz-juan-antonio/15404
[borja url]: https://cvnet.cpd.ua.es/curriculum-breve/es/navarro-colorado-francisco-de-borja/9307
[yoan url]: https://cvnet.cpd.ua.es/curriculum-breve/es/gutierrez-vazquez-yoan/49618
[japerez_twitter]: https://twitter.com/japer3z
[guía]: http://cv1.cpd.ua.es/ConsPlanesEstudio/cvFichaAsiEEES.asp?wCodEst=C203&wcodasi=34063&wLengua=C&scaca=2020-21

- La asistencia a las clases en principio no es obligatoria (en el sentido de que se vaya a requerir un número mínimo de asistencias para poder aprobar la asignatura), pero sí que es recomendable una presencia constante. La asistencia la puedes realizar tanto presencialmente (en las semanas que te asigne la universidad a través de la aplicación de Docencia Dual) o telemáticamente (en cualquiera de las semanas de curso, incluso en las que en principio la aplicación te hubiera permitido asistir presencialmente). Si tienes alguna ocupación que te impide asistir a todas o gran parte de las clases, incluso online, envía un justificante escaneado al profesor coordinador a través del sistema de tutoría de UACloud.
- La visita (virtual en este curso) a los profesores durante sus horas de tutoría no puede ser obligatoria por cuestiones normativas, pero es también muy recomendable, ya que es la oportunidad de recibir supervisión sobre tus conocimientos de la materia o la calidad de los trabajos que has desarrollado. Reserva turno a través de UACloud con anterioridad y conéctate a la sala virtual allí indicada. Si el horario no es compatible con tu agenda, escribe al profesor e intentará encontrar un hueco fuera de dicho horario para atenderte.
- Las prácticas se realizan individualmente. Cada uno de los tres bloques de la asignatura tendrá uno o más trabajos prácticos. Los trabajos del primer bloque contarán un 15% en la nota final de las prácticas, los del segundo un 35% y un 50% los del tercero.

El [código fuente][fuente] de estas páginas, escrito en MyST para Jupyter Book, está disponible en Github.

[fuente]: https://github.com/jaspock/mtextos

Puedes obtener una copia local de estas páginas (por ejemplo, para poder consultarlas sin conexión) ejecutando::

```
  wget --mirror --no-parent --convert-links --page-requisites https://jaspock.github.io/mtextos/
```
Pero ten en cuenta que su contenido irá cambiando a lo largo del curso.

Presentación de la asignatura
-----------------------------

La diferencia entre los términos *minería de textos* y *procesamiento del lenguaje natural* es un tanto difusa. Podríamos decir que la minería de textos es el proceso de descubrimiento de patrones y obtención de información relevante de grandes colecciones de textos. Al igual que en el concepto más amplio de minería de datos, el énfasis no es en la extracción (minado) de los datos en sí, sino en la extracción de patrones y conocimiento relevante. En el caso de la minería de textos, las fuentes de información son textos digitalizados, pero no se suele considerar otro tipo de información más estructurada como las bases de datos. El procesamiento del lenguaje natural, por otro lado, aplica técnicas lingüísticas, computacionales y de aprendizaje automático a datos en lenguaje natural, habitualmente en forma de voz o texto, de manera que el ordenador adquiera cierta *comprensión* sobre su contenido. El procesamiento de voz no se suele considerar parte de la minería de textos, pero esta se vale muy frecuentemente de las técnicas de procesamiento del lenguaje natural para conseguir sus objetivos.

La asignatura se centra en presentar los fundamentos, características y aplicaciones de las técnicas actuales para el procesamiento del lenguaje natural, pero no pretende entrar en excesivos detalles sobre modelos muy recientes, ya que el ritmo de aparición de sistemas que mejoran los resultados de los anteriores es extremadamente rápido. Como ejemplo, el conjunto de *benchmarks* llamado SuperGLUE quedó prácticamente [obsoleto en menos de dos años][superglue] por la sucesiva aparición de arquitecturas que conseguían cada vez mejores resultados en los problemas que incluía.

[superglue]: https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/

La asignatura tiene tres bloques: el primero ("{ref}`label_introduccion`") introduce los fundamentos de la lingüística computacional; en el segundo ("{ref}`label_tecnicas`") se estudia con cierto nivel de detalle las arquitecturas neuronales más empleadas en el área del procesamiento del lenguaje natural; por último, el tercer bloque ("{ref}`label_aplicaciones`") discute algunas de las aplicaciones más importantes, apoyándose para ello en lo estudiado en los dos bloques anteriores. 

