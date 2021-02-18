

Fundamentos de PLN. Práctica 2. Análisis semántico.
====================================================

## Actividad 1 - Simultación de un sistema de WSD

Para comprender el proceso de WSD, en esta práctica vamos a simular que eres un sistema de WSD.

Dado el [este corpus](https://www.dlsi.ua.es/~borja/a2.txt) y WordNet en español (accesible desde el [Multilingual Central Repository](https://adimen.si.ehu.es/cgi-bin/wei/public/wei.consult.perl)), ve seleccionando una a una cada palabra del texto (solo nombres, verbos o adjetivos), busca sus *synsets* en WN y elige el más apropiado para el contexto donde aparece la palabra.

Esta actividad se realizará en grupo durante la hora de clase.


## Actividad 2. Análisis semántico de un corpus con Freeling.

El objetivo de esta práctica es realizar análisis semántico con [Freeling](http://nlp.lsi.upc.edu/freeling/) y enteder la salida que ofrece.

Por pasos:

1. Instala Freeling. Ver aquí:

[https://freeling-user-manual.readthedocs.io/en/latest/installation/installation-packages/](https://freeling-user-manual.readthedocs.io/en/latest/installation/installation-packages/)

2. Con la opción _analyze_, analiza el corpus anterior con sentidos (WSD) y roles semánticos. Crear un script SH o BASH para automatizar la consulta. Ver aquí:

[https://freeling-user-manual.readthedocs.io/en/v4.2/analyzer/#using-analyzer-program-to-process-corpora](https://freeling-user-manual.readthedocs.io/en/v4.2/analyzer/#using-analyzer-program-to-process-corpora)

3. Analiza la salida y crea un script (en python, por ejemplo) que incorpore esa información dos fichero ordenados por columnas, uno con la información semántica léxica

    token | lema | sentido

y otro con la información de roles semánticos:

    evento | role A0 | role A1 | role A2 | role A3
 | role A4 | ArgM | ...

## Para saber más...

- [http://nlp.lsi.upc.edu/freeling/](http://nlp.lsi.upc.edu/freeling/)
- [https://freeling-user-manual.readthedocs.io/en/v4.2/](https://freeling-user-manual.readthedocs.io/en/v4.2/)

