

# Fundamentos de PLN. Práctica 1.

## Objeto

Crear un *script* para anotar un pequeño corpus a diferentes niveles de descripción lingüística con SPaCy. Este tipo de pre-procesamiento del corpus es común en diferentes tareas de minería de textos.

Análisis a realizar:

- Tokenización
- Lematización
- Análisis categorial y morfológico
- Análisis sintáctico de dependencias
- Extracción de entidades

Con esta práctica aprenderás los fundamentos de SpaCy como herramienta de PLN para preprocesar corpus para minería de textos. Se asume que se conoce cómo funciona COLAB. Si no es así, consulta al profesor.

## Herramientas

- Cuaderno COLAB para crear el código: [https://colab.research.google.com](https://colab.research.google.com)
- Python (ya instalado en COLAB)
- SpaCy: [https://spacy.io/](https://spacy.io/)

## Recursos

- Corpus: un fichero de texto que puedes descargar [aquí](https://www.dlsi.ua.es/~borja/a2.txt). Codificación: UTF-8.
- Documentación básica SpaCy. Aquí está explicado todo lo necesario para hacer esta páctica:

[https://spacy.io/usage/spacy-101](https://spacy.io/usage/spacy-101)

### Por pasos:

1. Crear un cuaderno COLAB vacío.
2. Importar SpaCy (Esto es fácil :)

    import spacy

2. Descargar los módulos SpaCy para español (en COLAB), importar y asignar a una variable para su uso posterior.
3. Subir el corpus a COLAB, abrir el documento y asignarlo a una variable.
4. Analizar el corpus entero con SpaCy.
5. Extraer "palabra | lema | categoria_gramatical" y descargar en un fichero CSV.
6. Extraer los grupos nominales según el análisis de dependencias y descargar en un fichero .txt
7. Extraer las entidades nombradas y descargar en un fichero .txt
8. (Opcional, solo si sobra tiempo en clase) Crear un gráfico en COLAB donde se muestren la cantidad de nombres, adjetivos, verbos y adverbios.
9. Entrega: enviar enlace del cuaderno COLAB al profesor.

## Para saber más

Curso avanzado de SpaCy (en español):

[https://course.spacy.io/es/](https://course.spacy.io/es/)

