# Detecção de objetos de nanopartículas com YOLO

Este repositório contém a implementação da detecção de objetos de nanopartículas usando YOLO (You Only Look Once). O objetivo deste projeto é detectar e localizar nanopartículas em imagens de microscopia usando técnicas de aprendizado de máquina.

## Visão geral

Neste projeto, aplicamos o YOLO para detectar nanopartículas em imagens de microscopia. O projeto visa fornecer um método robusto e eficiente para detecção de nanopartículas.

### Principais recursos

- Scripts para tratamento de dados brutos do dataset criado.
- Modelo YOLO pré-treinado e ajustado para detecção de nanopartículas.
- Script de inferência e benchmarking.

## Configuração de ambiente

Para executar o código localmente, deve-se instalar o módulo ultralytics em python.

```bash
pip install ultralytics
```

Clonar esse repositório.

```bash
git clone https://github.com/izaias-saturnino/yolo-nanoparticle
cd yolo-nanoparticle
```

Após isso, deve-se configurar o localização de datasets para a pasta do projeto clonado.

## Execução do treinamento

Para executar o código em *yolo_experiment.py* corretamente deve-se modificar as variáveis *data_file*, *local_model_name*, *base_model_name* e *oriented_bb* de acordo com os valores necessários. A variável *data_file* deve possuir o arquivo *.yaml* necessário para que o dataset seja encontrado; a variável *local_model_name* deve possuir o nome do modelo que deve ser treinado; a variável *base_model_name* deve possuir o nome do modelo base, o qual será baixado caso não exista (útil para fazer retreinamento e/ou baixar menos modelos); finalmente, a variável *oriented_bb* deve informar se o modelo treinado será orientado ou não.

Para mais informações, veja a [documentação do YOLOv8](https://docs.ultralytics.com/pt/models/yolov8/).

## Medição das métricas

Para obter as métricas, deve-se executar *test_generated_models.py* com as variáveis com valores corretos. A variável *data_file* deve possuir o arquivo *.yaml* necessário para que o dataset seja encontrado; a variável *model_names* deve conter uma lista com o caminho relativo dos modelos; a variável *confidence* deve conter o valor do limiar de confiança desejado; a variável *metric_sets* deve conter os conjuntos de dados a ser testados; se a opção *save_results* ser ativada, então o script salvará os resultados das predições; *dir-path* deve conter o diretório do conjunto de dados; finalmente, *oriented_bb* deve informar se o modelo a ser testado é orientado ou não.